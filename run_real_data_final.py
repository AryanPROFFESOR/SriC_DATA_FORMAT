"""
run_real_data_final.py — SriC real-data benchmark with plots
"""

import sys
import os
import time
import tempfile
import gc

import numpy as np
import scipy.sparse as sp
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import sric

FILES = {
    "cge_interneuron": r"C:\Users\Aryan\Documents\Research\GENETICS_BIOPHYSICS\regulatory-capability\data\cge_interneuron.h5ad",
    #"mge_interneuron": r"C:\Users\Aryan\Documents\Research\GENETICS_BIOPHYSICS\regulatory-capability\data\mge_interneuron.h5ad",
}

EXTERNAL_DRIVE = "D:/"
if not os.path.exists(EXTERNAL_DRIVE):
    print(f"WARNING: External drive {EXTERNAL_DRIVE} not found. Falling back to {tempfile.gettempdir()}")
    EXTERNAL_DRIVE = tempfile.gettempdir()


def mb(n):
    return f"{n / 1024**2:.2f} MB"


def plot_summary(results):
    if not results:
        return

    names = [r["name"] for r in results]
    ratios = [r["ratio"] for r in results]
    h5ad_times = [r["h5ad_load_s"] for r in results]
    write_times = [r["sric_write_s"] for r in results]
    load_times = [r["sric_load_s"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(names, ratios, color="steelblue")
    axes[0].set_title("Compression ratio (h5ad / SriC)")
    axes[0].set_ylabel("Ratio")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    x = np.arange(len(names))
    axes[1].bar(x - 0.25, write_times, width=0.25, label="SriC write")
    axes[1].bar(x, load_times, width=0.25, label="SriC load")
    axes[1].bar(x + 0.25, h5ad_times, width=0.25, label="h5ad load")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_title("Runtime comparison")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "run_real_data_final_summary.png")
    fig.savefig(out_path, dpi=200)
    print(f"\n  Plot saved to: {out_path}")
    plt.close(fig)


print("=" * 75)
print("  SriC Real-Data Benchmark — Aryan's scRNA datasets")
print("=" * 75)

results = []

for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"\n  {name}: FILE NOT FOUND at {path}")
        continue

    print(f"\n{'─' * 75}")
    print(f"  Dataset: {name}")
    print(f"  Path: {path}")

    # ── Load h5ad (standard in-memory) ────────────────────────────────────────
    t0 = time.perf_counter()
    adata = ad.read_h5ad(path)
    t_h5ad_load = time.perf_counter() - t0

    h5ad_size = os.path.getsize(path)
    n_cells, n_genes = adata.shape
    print(f"  Shape: {n_cells:,} cells × {n_genes:,} genes")
    print(f"  h5ad size on disk:  {mb(h5ad_size)}")
    print(f"  h5ad load time:     {t_h5ad_load:.2f}s")
    print(f"  X dtype:            {adata.X.dtype}")
    print(f"  Layers:             {list(adata.layers.keys())}")

    if sp.issparse(adata.X):
        nnz = adata.X.nnz
    else:
        nnz = int(np.count_nonzero(adata.X))
    print(f"  NNZ: {nnz:,}  |  sparsity: {1 - nnz / (n_cells * n_genes):.1%}")

    # ── Build SriC container ──────────────────────────────────────────────────
    print("\n  Converting to SriC...")
    c = sric.from_anndata(adata)

    is_integer = np.issubdtype(adata.X.dtype, np.integer)
    layer_key = "X_raw" if is_integer else "X_norm"
    print(f"  Primary layer dtype: {adata.X.dtype} → {'raw counts (integer)' if is_integer else 'normalised (float)'}")

    if is_integer and "X_raw" in c.layers:
        c.add_derived_layer("X_norm", "log1p", "X_raw")

    ct_col = "cell_type" if "cell_type" in adata.obs.columns else None

    # Step C: Write SriC directly to External D:\ using a permanent name
    sric_path = os.path.join(EXTERNAL_DRIVE, f"{name}_benchmark.sric")

    fit_zm = ct_col is not None and is_integer
    t0 = time.perf_counter()
    sric.SriCWriter(
        c,
        fit_zeromodel=fit_zm,
        cell_type_col=ct_col or "cell_type",
    ).write(sric_path)
    t_sric_write = time.perf_counter() - t0
    sric_size = os.path.getsize(sric_path)

    # ── Cache validation data before purging RAM ──────────────────────────────
    test_gene_idx = c.n_vars // 2
    test_gene = c.var_names[test_gene_idx]

    ref_layer = c.layers[layer_key]
    if sp.issparse(ref_layer):
        X_orig_slice = ref_layer[:, test_gene_idx].toarray().ravel()
    else:
        X_orig_slice = np.asarray(ref_layer[:, test_gene_idx]).ravel()

    print("  Purging write-caches and cleaning RAM...")
    del adata
    del c
    gc.collect()

    # ── Load SriC ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    c2 = sric.SriCReader(sric_path, verify_checksums=True).load()
    t_sric_load = time.perf_counter() - t0

    # ── Verify round-trip (slice comparison) ──────────────────────────────────
    raw_key = "X_raw" if "X_raw" in c2.layers else list(c2.layers.keys())[0]
    X_rec_slice = c2.layers[raw_key][:, test_gene_idx].toarray().ravel()

    if is_integer:
        match = np.array_equal(X_orig_slice.astype(np.int32), X_rec_slice.astype(np.int32))
        match_str = "✓ BIT-EXACT" if match else "✗ MISMATCH"
    else:
        err = np.max(np.abs(X_orig_slice.astype(np.float64) - X_rec_slice.astype(np.float64)))
        match = err < 1e-4
        match_str = f"✓ max_err={err:.2e}" if match else f"✗ max_err={err:.2e}"

    # ── Gene query ────────────────────────────────────────────────────────────
    N = 2
    t0 = time.perf_counter()
    for _ in range(N):
        sric.SriCReader(sric_path).query_gene(test_gene, layer=raw_key)
    t_query = (time.perf_counter() - t0) / N * 1000

    # ── Print results ─────────────────────────────────────────────────────────
    ratio = h5ad_size / sric_size
    print(f"\n  RESULTS:")
    print(f"  h5ad size:          {mb(h5ad_size)}")
    print(f"  SriC size:          {mb(sric_size)}")
    print(f"  Compression ratio:  {ratio:.2f}× smaller than h5ad")
    print(f"  h5ad load time:     {t_h5ad_load:.2f}s")
    print(f"  SriC write time:    {t_sric_write:.2f}s")
    print(f"  SriC load time:     {t_sric_load:.2f}s")
    print(f"  Gene query:         {t_query:.1f}ms  (gene: {test_gene})")
    print(f"  Round-trip:         {match_str}")

    results.append({
        "name": name,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "h5ad_mb": h5ad_size / 1024**2,
        "sric_mb": sric_size / 1024**2,
        "ratio": ratio,
        "match": match,
        "h5ad_load_s": t_h5ad_load,
        "sric_write_s": t_sric_write,
        "sric_load_s": t_sric_load,
    })

    # os.unlink(sric_path)

# ── Summary table ─────────────────────────────────────────────────────────────
if results:
    print(f"\n\n{'='*75}")
    print("  SUMMARY — Real Data Results")
    print(f"{'='*75}")
    print(f"  {'Dataset':<20} {'Cells':>8} {'Genes':>8} {'h5ad':>10} {'SriC':>10} {'Ratio':>8} {'Match':>8}")
    print(f"  {'─'*75}")
    for r in results:
        print(f"  {r['name']:<20} {r['n_cells']:>8,} {r['n_genes']:>8,} "
              f"  {r['h5ad_mb']:>7.2f}MB {r['sric_mb']:>7.2f}MB "
              f"  {r['ratio']:>6.2f}×  {'✓' if r['match'] else '✗':>6}")

    print(f"\n  Average compression ratio: {sum(r['ratio'] for r in results) / len(results):.2f}×")
    plot_summary(results)
