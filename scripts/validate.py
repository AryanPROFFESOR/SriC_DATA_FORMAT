"""
validate.py — ZeroModel Statistical Validation

This is the core scientific validation needed for Nature Methods.

We simulate datasets with KNOWN dropout probabilities (true_pi),
run our MoM-ZINB estimator, and measure:

  1. Estimation accuracy: Pearson r and MSE between estimated and true pi
  2. Bias: whether the estimator systematically over/under-estimates
  3. Sample-size dependence: accuracy vs cells per cell-type
  4. Expression-level dependence: accuracy vs gene mean expression
  5. Comparison: our MoM estimator vs naive (observed zero fraction)

The naive estimator: pi_naive_gc = (n_zeros_gc / n_cells_c)
This ignores that NB also produces zeros (not all zeros are dropout).
Our ZINB MoM corrects for NB-expected zeros.

If our estimator is better than naive, that is a real scientific contribution.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from dataclasses import dataclass
from typing import List, Tuple

from simulate import simulate_zinb, SimulatedDataset
import sric
from sric.zeromodel import fit_zero_model


# ── Naive estimator (baseline to beat) ───────────────────────────────────────

def naive_dropout_estimate(X: sp.csr_matrix,
                            cell_type_ids: np.ndarray,
                            n_celltypes: int) -> np.ndarray:
    """
    Naive estimator: pi_gc = observed_zero_fraction per (gene, celltype).

    This overcounts dropout because NB distributions also produce zeros
    even without technical dropout. Our ZINB MoM corrects for this.
    """
    n_cells, n_genes = X.shape
    pi_naive = np.zeros((n_genes, n_celltypes), dtype=np.float64)
    n_ct = np.bincount(cell_type_ids, minlength=n_celltypes)

    csr = X.tocsr()
    for c in range(n_celltypes):
        cell_mask = cell_type_ids == c
        X_ct = csr[cell_mask, :]
        n_expressed = np.array((X_ct > 0).sum(axis=0)).ravel()
        n_c = n_ct[c]
        if n_c > 0:
            pi_naive[:, c] = (n_c - n_expressed) / n_c

    return pi_naive


# ── Validation metrics ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    n_cells:          int
    n_genes:          int
    n_celltypes:      int
    # MoM estimator
    mom_pearson_r:    float
    mom_spearman_r:   float
    mom_mse:          float
    mom_bias:         float    # mean(estimated - true)
    # Naive estimator
    naive_pearson_r:  float
    naive_mse:        float
    naive_bias:       float
    # Stratified by expression level
    mom_r_low_expr:   float    # genes with true_mu < 1
    mom_r_high_expr:  float    # genes with true_mu >= 1


def validate_single(ds: SimulatedDataset,
                    verbose: bool = False) -> ValidationResult:
    """
    Run ZeroModel estimation on one simulated dataset and compare to ground truth.
    """
    n_cells, n_genes = ds.X.shape
    n_ct = len(ds.cell_type_names)
    ct_labels = np.array([ds.cell_type_names[i] for i in ds.cell_type_ids])

    # Our MoM-ZINB estimator
    zm = fit_zero_model(ds.X, ct_labels)
    if zm is None:
        raise RuntimeError("ZeroModel estimation failed")

    # Align cell type order
    est_ct_names = zm['cell_type_names']
    true_ct_names = ds.cell_type_names

    # Map estimated columns to true columns
    col_map = [est_ct_names.index(name) if name in est_ct_names else -1
               for name in true_ct_names]

    est_pi_aligned = np.zeros_like(ds.true_pi)
    for tc, ec in enumerate(col_map):
        if ec >= 0:
            est_pi_aligned[:, tc] = zm['pi'][:, ec].astype(np.float64)

    # Naive estimator
    naive_pi = naive_dropout_estimate(ds.X, ds.cell_type_ids, n_ct)

    # Flatten for correlation (only genes with variation in true_pi)
    true_flat  = ds.true_pi.ravel()
    mom_flat   = est_pi_aligned.ravel()
    naive_flat = naive_pi.ravel()

    # Filter out entries with zero true mean (undefined)
    valid = ds.true_mu.ravel() > 0
    t = true_flat[valid]; m = mom_flat[valid]; n = naive_flat[valid]

    mom_r,   _ = pearsonr(t, m)
    mom_sr,  _ = spearmanr(t, m)
    naive_r, _ = pearsonr(t, n)

    mom_mse   = float(np.mean((m - t)**2))
    naive_mse = float(np.mean((n - t)**2))
    mom_bias  = float(np.mean(m - t))
    naive_bias = float(np.mean(n - t))

    # Stratify by expression level
    mean_expr_flat = ds.true_mu.ravel()[valid]
    low_mask  = mean_expr_flat < 1.0
    high_mask = mean_expr_flat >= 1.0

    mom_r_low  = pearsonr(t[low_mask],  m[low_mask])[0]  if low_mask.sum()  > 10 else np.nan
    mom_r_high = pearsonr(t[high_mask], m[high_mask])[0] if high_mask.sum() > 10 else np.nan

    if verbose:
        print(f"  n_cells={n_cells:,}  n_genes={n_genes:,}  n_ct={n_ct}")
        print(f"  MoM  : r={mom_r:.3f}  MSE={mom_mse:.4f}  bias={mom_bias:+.4f}")
        print(f"  Naive: r={naive_r:.3f}  MSE={naive_mse:.4f}  bias={naive_bias:+.4f}")
        print(f"  MoM wins: {mom_mse < naive_mse}")

    return ValidationResult(
        n_cells=n_cells, n_genes=n_genes, n_celltypes=n_ct,
        mom_pearson_r=mom_r, mom_spearman_r=mom_sr,
        mom_mse=mom_mse, mom_bias=mom_bias,
        naive_pearson_r=naive_r, naive_mse=naive_mse, naive_bias=naive_bias,
        mom_r_low_expr=mom_r_low, mom_r_high_expr=mom_r_high,
    )


def validate_sample_size_curve(
    n_cells_list: List[int] = [200, 500, 1000, 2000, 5000],
    n_genes: int = 5000,
    n_groups: int = 4,
    n_seeds: int = 3,
) -> List[ValidationResult]:
    """
    Measure estimation accuracy vs cells per cell-type.
    This is the key plot for the paper: shows when our estimator
    becomes reliable enough to be useful.
    """
    results = []
    for n_cells in n_cells_list:
        seed_results = []
        for seed in range(n_seeds):
            print(f"  n_cells={n_cells:,}  seed={seed}...", end="", flush=True)
            ds = simulate_zinb(n_cells=n_cells, n_genes=n_genes,
                               n_groups=n_groups, seed=seed*100+42)
            r  = validate_single(ds, verbose=False)
            seed_results.append(r)
            print(f" mom_r={r.mom_pearson_r:.3f}  naive_r={r.naive_pearson_r:.3f}")
        # Average over seeds
        avg = ValidationResult(
            n_cells=n_cells, n_genes=n_genes, n_celltypes=n_groups,
            mom_pearson_r  = float(np.mean([x.mom_pearson_r  for x in seed_results])),
            mom_spearman_r = float(np.mean([x.mom_spearman_r for x in seed_results])),
            mom_mse        = float(np.mean([x.mom_mse        for x in seed_results])),
            mom_bias       = float(np.mean([x.mom_bias       for x in seed_results])),
            naive_pearson_r= float(np.mean([x.naive_pearson_r for x in seed_results])),
            naive_mse      = float(np.mean([x.naive_mse      for x in seed_results])),
            naive_bias     = float(np.mean([x.naive_bias     for x in seed_results])),
            mom_r_low_expr = float(np.nanmean([x.mom_r_low_expr  for x in seed_results])),
            mom_r_high_expr= float(np.nanmean([x.mom_r_high_expr for x in seed_results])),
        )
        results.append(avg)
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("  ZeroModel Statistical Validation")
    print("=" * 60)
    print()

    # Single dataset validation
    print("1. Single dataset (3000 cells, 5000 genes, 6 cell types):")
    ds = simulate_zinb(n_cells=3000, n_genes=5000, n_groups=6, seed=42)
    r  = validate_single(ds, verbose=True)
    print()

    # Sample-size curve
    print("2. Sample-size dependence:")
    results = validate_sample_size_curve(
        n_cells_list=[300, 600, 1000, 2000],
        n_genes=3000, n_groups=4, n_seeds=3)

    print()
    print(f"  {'n_cells':>8}  {'MoM r':>8}  {'Naive r':>8}  "
          f"{'MoM MSE':>10}  {'MoM wins':>10}")
    print(f"  {'-'*60}")
    for r in results:
        wins = "YES" if r.mom_mse < r.naive_mse else "no"
        print(f"  {r.n_cells:>8,}  {r.mom_pearson_r:>8.3f}  "
              f"{r.naive_pearson_r:>8.3f}  {r.mom_mse:>10.5f}  {wins:>10}")

    print()
    best = results[-1]
    print(f"At {best.n_cells:,} cells:")
    print(f"  MoM  Pearson r = {best.mom_pearson_r:.3f}")
    print(f"  Naive Pearson r = {best.naive_pearson_r:.3f}")
    print(f"  MoM improvement: {(best.naive_mse - best.mom_mse)/best.naive_mse*100:.1f}% lower MSE")
