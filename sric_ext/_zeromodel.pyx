# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""
sric_ext._zeromodel  —  Zero-inflation parameter estimation (Cython)

Original contribution: per-(gene, cell-type) dropout probability estimation
using a zero-inflated negative binomial model with Method-of-Moments.

Model per (gene g, cell type c):
    Y_{i,g} | cell_type(i)=c  ~  ZINB(π_gc, μ_gc, r_gc)
    P(Y=0)  = π_gc + (1-π_gc) * NB(0; μ_gc, r_gc)
    P(Y=k)  = (1-π_gc) * NB(k; μ_gc, r_gc)    k > 0

Parameters estimated via MoM:
    μ_gc   = E[Y] = sum_i(Y_{ig} * 1[type=c]) / n_c
    r_gc   = μ²/(Var[Y]-μ)  when Var > μ, else ∞ (Poisson)
    π_gc   = max(0, (observed_zeros - NB_expected_zeros) / n_c)

No library dependency. Pure Cython inner loop.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t
from libc.math   cimport pow as c_pow

cnp.import_array()


def estimate_zinb_params(
    cnp.ndarray X_data    not None,
    cnp.ndarray X_indptr  not None,
    cnp.ndarray X_indices not None,
    tuple shape,
    cnp.ndarray ct_ids    not None,
    int n_ct,
) -> dict:
    """
    Estimate ZINB parameters per (gene, cell-type) from a CSR count matrix.

    Parameters
    ----------
    X_data    : int32 nonzero values
    X_indptr  : int32 CSR row pointers
    X_indices : int32 CSR column indices (gene indices)
    shape     : (n_cells, n_genes)
    ct_ids    : int32 cell-type integer label per cell (0-indexed)
    n_ct      : number of cell types

    Returns
    -------
    dict with float64 arrays:
        mu          (n_genes, n_ct)  — mean expression per (gene, cell-type)
        r           (n_genes, n_ct)  — NB dispersion (∞ → Poisson limit)
        pi          (n_genes, n_ct)  — dropout probability in [0, 1]
        n_cells_ct  (n_ct,)          — cells per cell-type
    """
    cdef cnp.ndarray[cnp.int32_t, ndim=1] data    = np.asarray(X_data,    np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indptr  = np.asarray(X_indptr,  np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] indices = np.asarray(X_indices, np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim=1] ct      = np.asarray(ct_ids,    np.int32)

    n_cells, n_genes = shape

    # Accumulators
    cdef cnp.ndarray[cnp.float64_t, ndim=2] S1  = np.zeros((n_genes, n_ct), np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] S2  = np.zeros((n_genes, n_ct), np.float64)
    cdef cnp.ndarray[cnp.int32_t,   ndim=2] NE  = np.zeros((n_genes, n_ct), np.int32)
    cdef cnp.ndarray[cnp.int32_t,   ndim=1] NC  = np.zeros(n_ct, np.int32)

    cdef int64_t row, s, e, i
    cdef int32_t g, c
    cdef double  v

    for row in range(n_cells):
        c = ct[row]; NC[c] += 1
        s = indptr[row]; e = indptr[row + 1]
        for i in range(s, e):
            g = indices[i]; v = <double>data[i]
            S1[g, c] += v
            S2[g, c] += v * v
            NE[g, c] += 1

    # Output arrays
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mu  = np.zeros((n_genes, n_ct), np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] r   = np.full ((n_genes, n_ct), 1e9,  np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] pi  = np.zeros((n_genes, n_ct), np.float64)

    cdef double s1, s2, nc_d, mu_gc, var_gc, r_gc, nb_zero, pi_gc, excess
    cdef int32_t nc_i, ne_i

    for g in range(n_genes):
        for c in range(n_ct):
            nc_i = NC[c]
            ne_i = NE[g, c]
            if nc_i == 0:
                pi[g, c] = 1.0
                continue

            nc_d  = <double>nc_i
            s1    = S1[g, c]
            s2    = S2[g, c]
            mu_gc = s1 / nc_d

            # MoM: variance estimated over all cells (including zeros)
            var_gc = (s2 - s1 * s1 / nc_d) / (nc_d - 1.0) if nc_i > 1 else 0.0

            # NB dispersion r: μ²/(Var-μ)
            if var_gc > mu_gc and mu_gc > 0.0:
                r_gc = mu_gc * mu_gc / (var_gc - mu_gc)
            else:
                r_gc = 1e9   # Poisson limit

            # Dropout probability: excess zeros above NB expectation
            if mu_gc > 0.0:
                # NB zero probability: (r/(r+μ))^r
                nb_zero = c_pow(r_gc / (r_gc + mu_gc), r_gc)
                excess  = <double>(nc_i - ne_i) - nc_d * nb_zero
                pi_gc   = excess / nc_d if excess > 0.0 else 0.0
            else:
                pi_gc = <double>(nc_i - ne_i) / nc_d

            mu [g, c] = mu_gc
            r  [g, c] = r_gc
            pi [g, c] = min(1.0, max(0.0, pi_gc))

    return {
        'mu':         mu,
        'r':          r,
        'pi':         pi,
        'n_cells_ct': np.asarray(NC),
    }
