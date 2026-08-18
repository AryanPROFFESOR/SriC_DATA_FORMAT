"""
run_real_data.py — Benchmark SriC on Aryan's real scRNA datasets
"""

import sys, os, time, tempfile, zlib
sys.path.insert(0, '.')

import numpy as np
import scipy.sparse as sp
import anndata as ad
import sric

FILES = {
    "cge_interneuron": r"C:\Users\Aryan\Documents\Research\GENETICS_BIOPHYSICS\regulatory-capability\data\cge_interneuron.h5ad",
    "mge_interneuron": r"C:\Users\Aryan\Documents\Research\GENETICS_BIOPHYSICS\regulatory-capability\data\mge_interneuron.h5ad",
}

def mb(n): return f"{n/1024**2:.2f} MB"

print("=" * 70)
print("  SriC Real-Data Benchmark — Aryan's scRNA datasets")
print("=" * 70)

results = []

for name, path in FILES.items():
    if not os.path.exists(path):
        print(f"\n  {name}: FILE NOT FOUND at {path}")
        continue

    print(f"\n{'─'*70}")
    print(f"  Dataset: {name}")
    print(f"  Path: {path}")

    # ── Load h5ad (Standard In-Memory) ────────────────────────────────────────
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
    print(f"\n  Converting to SriC...")
    c = sric.from_anndata(adata)

    is_integer = np.issubdtype(adata.X.dtype, np.integer)
    layer_key = "X_raw" if is_integer else "X_norm"
    print(f"  Primary layer dtype: {adata.X.dtype} → {'raw counts (integer)' if is_integer else 'normalised (float)'}")

    if is_integer and 'X_raw' in c.layers:
        c.add_derived_layer('X_norm', 'log1p', 'X_raw')

    ct_col = 'cell_type' if 'cell_type' in adata.obs.columns else None

    # ── Write SriC to External D:\ ────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(dir="D:/", suffix='.sric', delete=False) as t:
        sric_path = t.name

    fit_zm = ct_col is not None and is_integer
    t0 = time.perf_counter()
    sric.SriCWriter(
        c,
        fit_zeromodel=fit_zm,
        cell_type_col=ct_col or 'cell_type'
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
    import gc; gc.collect()

    # ── Load SriC ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    c2 = sric.SriCReader(sric_path, verify_checksums=True).load()
    t_sric_load = time.perf_counter() - t0

    # ── Verify round-trip (Slice Comparison) ──────────────────────────────────
    raw_key = 'X_raw' if 'X_raw' in c2.layers else list(c2.layers.keys())[0]
    X_rec_slice = c2.layers[raw_key][:, test_gene_idx].toarray().ravel()

    if is_integer:
        match = np.array_equal(X_orig_slice.astype(np.int32), X_rec_slice.astype(np.int32))
        match_str = f"✓ BIT-EXACT" if match else "✗ MISMATCH"
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
    print(f"  Compression ratio:  {ratio:.2f}× smaller than h5ad")
    print(f"  h5ad load time:     {t_h5ad_load:.2f}s")
    print(f"  SriC write time:    {t_sric_write:.2f}s")
    print(f"  SriC load time:     {t_sric_load:.2f}s")
    print(f"  Gene query:         {t_query:.1f}ms  (gene: {test_gene})")
    print(f"  Round-trip:         {match_str}")

    results.append({
        'name': name,
        'n_cells': n_cells, 'n_genes': n_genes,
        'h5ad_mb': h5ad_size/1024**2,
        'sric_mb': sric_size/1024**2,
        'ratio': ratio,
        'match': match,
    })

    # Clean up the 1.5GB temp file
    os.unlink(sric_path)

# ── Summary table ─────────────────────────────────────────────────────────────
if results:
    print(f"\n\n{'='*70}")
    print("  SUMMARY — Real Data Results")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'Cells':>8} {'Genes':>8} {'h5ad':>10} {'SriC':>10} {'Ratio':>8}")
    print(f"  {'─'*70}")
    for r in results:
        print(f"  {r['name']:<20} {r['n_cells']:>8,} {r['n_genes']:>8,} "
              f"  {r['h5ad_mb']:>7.2f}MB {r['sric_mb']:>7.2f}MB "
              f"  {r['ratio']:>6.2f}×")
