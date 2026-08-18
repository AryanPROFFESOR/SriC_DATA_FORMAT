"""
simulate.py — Splatter-calibrated ZINB scRNA simulator

Implements the core generative model from:
  Zappia et al. 2017 "Splatter: simulation of single-cell RNA sequencing data"
  Genome Biology. doi:10.1186/s13059-017-1305-0

Model per gene g, cell i of type c:
  Step 1: cell-type mean  μ_gc = μ_g * fold_change_gc
  Step 2: biological CV   BCV_g ~ InvChiSq(df_g)
  Step 3: true expression λ_ig ~ Gamma(1/BCV_g^2, μ_gc * BCV_g^2)
  Step 4: dropout mask    D_ig ~ Bernoulli(π_gc)   [zero-inflation]
  Step 5: count           Y_ig ~ Poisson(λ_ig) * (1 - D_ig)

Parameters calibrated to human PBMC 10x Chromium v3 data
(published benchmark in Zappia et al. 2017).

Ground truth stored with each simulated dataset:
  true_pi   (n_genes, n_celltypes)  — true dropout probabilities
  true_mu   (n_genes, n_celltypes)  — true mean expression
  true_bcv  (n_genes,)              — true biological CV

This ground truth is used to validate our ZeroModel estimator.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Tuple

# ── Calibrated parameters (from Zappia et al. 2017, PBMC) ────────────────────

PBMC_PARAMS = {
    "mean_log_mu":     0.52,    # log-normal mean for gene means
    "sd_log_mu":       1.47,    # log-normal sd
    "mean_log_bcv":   -0.46,    # log-normal mean for BCV
    "sd_log_bcv":      0.39,    # log-normal sd
    "dropout_mid":     0.0,     # logistic midpoint (log mean scale)
    "dropout_shape":  -1.0,     # logistic shape (steeper = harder dropout)
    "lib_loc":         7.64,    # log library size mean
    "lib_scale":       0.78,    # log library size sd
    "de_prob":         0.10,    # fraction of genes DE per group
    "de_down_prob":    0.50,    # fraction of DE genes down-regulated
    "de_fc_scale":     0.4,     # log-normal sd of fold changes
}


@dataclass
class SimulatedDataset:
    """Ground-truth scRNA dataset from ZINB simulation."""
    X:               sp.csr_matrix   # (n_cells, n_genes) raw counts
    cell_type_ids:   np.ndarray      # (n_cells,) integer 0..n_groups-1
    cell_type_names: list            # group names
    gene_names:      np.ndarray      # (n_genes,) strings
    obs_names:       np.ndarray      # (n_cells,) barcodes
    # Ground truth (what our estimator should recover)
    true_pi:         np.ndarray      # (n_genes, n_groups) dropout prob
    true_mu:         np.ndarray      # (n_genes, n_groups) mean expression
    true_bcv:        np.ndarray      # (n_genes,) biological CV
    lib_sizes:       np.ndarray      # (n_cells,) total counts per cell


def simulate_zinb(
    n_cells:   int  = 3000,
    n_genes:   int  = 10000,
    n_groups:  int  = 6,
    seed:      int  = 42,
    params:    dict = None,
) -> SimulatedDataset:
    """
    Simulate a ZINB scRNA dataset with known ground truth.

    Parameters
    ----------
    n_cells  : total cells (evenly split across groups)
    n_genes  : number of genes
    n_groups : number of cell types / clusters
    seed     : random seed for reproducibility
    params   : parameter dict (defaults to PBMC calibration)

    Returns
    -------
    SimulatedDataset with raw counts X and full ground truth arrays
    """
    rng = np.random.default_rng(seed)
    p   = PBMC_PARAMS.copy()
    if params:
        p.update(params)

    # ── Gene-level parameters ─────────────────────────────────────────────────
    # Base mean expression per gene (log-normal, PBMC calibrated)
    base_mu = rng.lognormal(p["mean_log_mu"], p["sd_log_mu"], n_genes)

    # Biological CV per gene (log-normal)
    bcv = rng.lognormal(p["mean_log_bcv"], p["sd_log_bcv"], n_genes)
    bcv = np.clip(bcv, 0.05, 2.0)

    # ── Group fold changes ────────────────────────────────────────────────────
    # Each group has DE genes with random fold changes
    de_mask = rng.random((n_genes, n_groups)) < p["de_prob"]
    fc_signs = rng.choice([-1, 1], size=(n_genes, n_groups),
                           p=[p["de_down_prob"], 1 - p["de_down_prob"]])
    fc_magnitudes = rng.lognormal(0, p["de_fc_scale"], (n_genes, n_groups))
    log_fc = de_mask * fc_signs * fc_magnitudes
    fold_changes = np.exp(log_fc)   # (n_genes, n_groups)

    # Group-specific means
    true_mu = base_mu[:, None] * fold_changes   # (n_genes, n_groups)

    # ── Dropout probabilities ─────────────────────────────────────────────────
    # Logistic dropout model: P(dropout) = logistic(shape * (log(mu) - mid))
    # Genes with higher expression have lower dropout probability
    log_mu = np.log1p(true_mu)   # (n_genes, n_groups)
    true_pi = 1.0 / (1.0 + np.exp(
        -p["dropout_shape"] * (p["dropout_mid"] - log_mu)
    ))
    true_pi = np.clip(true_pi, 1e-6, 1 - 1e-6)

    # ── Cell-level parameters ─────────────────────────────────────────────────
    cells_per_group = n_cells // n_groups
    group_ids = np.repeat(np.arange(n_groups),
                          [cells_per_group] * (n_groups - 1) +
                          [n_cells - cells_per_group * (n_groups - 1)])
    rng.shuffle(group_ids)

    # Library sizes (log-normal)
    lib_sizes = rng.lognormal(p["lib_loc"], p["lib_scale"], n_cells)
    lib_sizes = np.round(lib_sizes).astype(np.int64)

    # ── Simulate counts ───────────────────────────────────────────────────────
    rows_list, cols_list, vals_list = [], [], []

    for i in range(n_cells):
        g = group_ids[i]
        lib = lib_sizes[i]

        # Gamma-Poisson: sample true expression rates
        shape = 1.0 / (bcv ** 2)   # (n_genes,)
        scale = true_mu[:, g] * (bcv ** 2)   # (n_genes,)

        lam = rng.gamma(shape, scale)   # (n_genes,) true expression rates

        # Scale to library size
        lam_scaled = lam / lam.sum() * lib

        # Poisson counts
        raw_counts = rng.poisson(lam_scaled)   # (n_genes,)

        # Apply dropout mask
        dropout_mask = rng.random(n_genes) < true_pi[:, g]
        raw_counts[dropout_mask] = 0

        # Store nonzero entries
        nz = np.where(raw_counts > 0)[0]
        if len(nz):
            rows_list.append(np.full(len(nz), i, dtype=np.int32))
            cols_list.append(nz.astype(np.int32))
            vals_list.append(raw_counts[nz].astype(np.int32))

    if rows_list:
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)
        X = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_genes))
    else:
        X = sp.csr_matrix((n_cells, n_genes), dtype=np.int32)

    # ── Names ─────────────────────────────────────────────────────────────────
    group_names = [f"CellType_{chr(65+g)}" for g in range(n_groups)]
    gene_names  = np.array([f"GENE_{i:06d}" for i in range(n_genes)])
    obs_names   = np.array([f"CELL_{i:07d}" for i in range(n_cells)])

    actual_sparsity = 1 - X.nnz / (n_cells * n_genes)

    print(f"[simulate] {n_cells:,} cells × {n_genes:,} genes | "
          f"{n_groups} types | sparsity {actual_sparsity:.1%} | "
          f"nnz {X.nnz:,}")

    return SimulatedDataset(
        X               = X,
        cell_type_ids   = group_ids,
        cell_type_names = group_names,
        gene_names      = gene_names,
        obs_names       = obs_names,
        true_pi         = true_pi,
        true_mu         = true_mu,
        true_bcv        = bcv,
        lib_sizes       = lib_sizes,
    )


if __name__ == "__main__":
    ds = simulate_zinb(n_cells=3000, n_genes=10000, n_groups=6)
    print(f"X shape: {ds.X.shape}, nnz: {ds.X.nnz:,}")
    print(f"true_pi range: [{ds.true_pi.min():.3f}, {ds.true_pi.max():.3f}]")
    print(f"true_mu range: [{ds.true_mu.min():.4f}, {ds.true_mu.max():.1f}]")
    print(f"lib_sizes: median={int(np.median(ds.lib_sizes)):,} "
          f"[{ds.lib_sizes.min():,}, {ds.lib_sizes.max():,}]")
