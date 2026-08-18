"""
celltype_stats.py — Per-(gene, cell-type) Summary Statistics Block

Replaces the misnamed "ZeroModel" with statistically honest quantities.

CRITICAL SCIENTIFIC NOTE:
The ZINB dropout probability π_gc is NOT identifiable from observed count
data alone. This is a known result (Risso et al. 2018 Nat Commun; van Dijk
et al. 2018 Cell). Without additional constraints (e.g. spike-ins, known
expression in bulk), one cannot distinguish:
  (A) gene with low mean + high technical dropout
  (B) gene with very low mean + no dropout
Both produce identical observed zero fractions.

Therefore we store OBSERVED, IDENTIFIABLE statistics only:

  obs_zero_frac  (n_genes, n_celltypes)  — observed fraction of zeros
  mean_expr      (n_genes, n_celltypes)  — mean UMI count (including zeros)
  mean_expr_nz   (n_genes, n_celltypes)  — mean UMI among expressing cells
  frac_expressing(n_genes, n_celltypes)  — fraction of cells expressing gene

These are:
  - Directly computable from data (no model assumptions)
  - Identifiable (no ambiguity in their definition)
  - Useful for downstream tools (MAGIC, kNN-smoothing, QC reports)
  - Novel as embedded format metadata (no existing format stores these)

Novel architectural contribution:
  A researcher opening a .SriC file gets pre-computed per-celltype
  gene statistics without re-scanning the full matrix. For large atlases
  (1M+ cells), this saves hours of recomputation.
"""

from __future__ import annotations
import json, zlib
import numpy as np
import scipy.sparse as sp
from typing import Optional


def compute_celltype_stats(
    layer:             sp.spmatrix,
    cell_type_labels:  np.ndarray,
) -> Optional[dict]:
    """
    Compute per-(gene, cell-type) summary statistics.

    All quantities are directly observable from the data.
    No model assumptions, no latent variable estimation.

    Parameters
    ----------
    layer            : CSR integer count matrix (n_cells, n_genes)
    cell_type_labels : string or integer cell-type label per cell

    Returns
    -------
    dict with float16 arrays shaped (n_genes, n_celltypes):
        obs_zero_frac   — fraction of cells with zero count
        mean_expr       — mean count (all cells, including zeros)
        mean_expr_nz    — mean count among expressing cells only
        frac_expressing — fraction of cells with count > 0
        n_cells_ct      — number of cells per cell-type (1D)
        cell_type_names — list of cell-type name strings
    """
    if not sp.issparse(layer):
        return None

    csr = layer.tocsr().astype(np.float64)
    n_cells, n_genes = csr.shape

    labels      = np.asarray(cell_type_labels, dtype=str)
    unique_ct, ct_ids = np.unique(labels, return_inverse=True)
    n_ct        = len(unique_ct)
    n_cells_ct  = np.bincount(ct_ids, minlength=n_ct)

    obs_zero_frac    = np.zeros((n_genes, n_ct), np.float64)
    mean_expr        = np.zeros((n_genes, n_ct), np.float64)
    mean_expr_nz     = np.zeros((n_genes, n_ct), np.float64)
    frac_expressing  = np.zeros((n_genes, n_ct), np.float64)

    for c in range(n_ct):
        mask = ct_ids == c
        nc   = int(n_cells_ct[c])
        if nc == 0:
            continue

        X_ct = csr[mask, :]

        # Sum of counts per gene
        sum_counts  = np.asarray(X_ct.sum(axis=0)).ravel()

        # Number of expressing cells (nnz per column)
        n_expressing = np.asarray((X_ct > 0).sum(axis=0)).ravel().astype(float)

        mean_expr[:,c]       = sum_counts / nc
        frac_expressing[:,c] = n_expressing / nc
        obs_zero_frac[:,c]   = 1.0 - n_expressing / nc

        # Mean among expressing cells (avoid divide-by-zero)
        safe_nexp = np.where(n_expressing > 0, n_expressing, 1.0)
        mean_expr_nz[:,c] = sum_counts / safe_nexp
        mean_expr_nz[n_expressing == 0, c] = 0.0

    return {
        'obs_zero_frac':   obs_zero_frac.astype(np.float16),
        'mean_expr':       mean_expr.astype(np.float16),
        'mean_expr_nz':    mean_expr_nz.astype(np.float16),
        'frac_expressing': frac_expressing.astype(np.float16),
        'n_cells_ct':      n_cells_ct.tolist(),
        'cell_type_names': unique_ct.tolist(),
        'n_genes':         n_genes,
        'n_celltypes':     n_ct,
    }


def serialize_celltype_stats(stats: dict) -> bytes:
    """Compact serialisation: JSON header + raw float16 arrays."""
    if stats is None:
        return b''

    header = {
        'cell_type_names': stats['cell_type_names'],
        'n_cells_ct':      stats['n_cells_ct'],
        'n_genes':         stats['n_genes'],
        'n_celltypes':     stats['n_celltypes'],
        'arrays':          ['obs_zero_frac','mean_expr',
                            'mean_expr_nz','frac_expressing'],
    }
    header_b = json.dumps(header).encode()

    shape = (stats['n_genes'], stats['n_celltypes'])
    arrays = b''.join(
        stats[k].astype(np.float16).tobytes()
        for k in header['arrays']
    )

    return zlib.compress(header_b + b'\x00' + arrays, level=6)


def deserialize_celltype_stats(data: bytes) -> Optional[dict]:
    """Deserialise a celltype stats block."""
    if not data:
        return None
    raw      = zlib.decompress(data)
    null_pos = raw.index(b'\x00')
    header   = json.loads(raw[:null_pos])
    n_genes  = header['n_genes']
    n_ct     = header['n_celltypes']
    n        = n_genes * n_ct
    pos      = null_pos + 1
    result   = {'cell_type_names': header['cell_type_names'],
                'n_cells_ct':      header['n_cells_ct']}
    for key in header['arrays']:
        arr = np.frombuffer(raw[pos:pos + n*2], np.float16).reshape(n_genes, n_ct)
        result[key] = arr
        pos += n * 2
    return result


def validate_stats_accuracy(
    stats: dict,
    layer: sp.spmatrix,
    cell_type_labels: np.ndarray,
) -> dict:
    """
    Verify stored statistics against directly recomputed values.
    Returns MSE for each statistic (should be ~0 up to float16 precision).
    """
    recomputed = compute_celltype_stats(layer, cell_type_labels)
    if recomputed is None or stats is None:
        return {}

    results = {}
    for key in ['obs_zero_frac', 'mean_expr', 'frac_expressing']:
        s = stats[key].astype(np.float64)
        r = recomputed[key].astype(np.float64)
        results[key] = {
            'mse':      float(np.mean((s - r)**2)),
            'max_err':  float(np.max(np.abs(s - r))),
            'pearson_r': float(np.corrcoef(s.ravel(), r.ravel())[0, 1]),
        }
    return results
