"""sric.convert — interoperability with AnnData, scipy sparse, dense numpy."""
from __future__ import annotations
import numpy as np, scipy.sparse as sp
from typing import Optional, List
from .container import SriCContainer

def from_scipy_sparse(X, obs_names=None, var_names=None,
                      layer_name="X_raw", modality="RNA") -> SriCContainer:
    n, m = X.shape
    obs = np.array(obs_names or [f"cell_{i:06d}" for i in range(n)])
    var = np.array(var_names or [f"gene_{i:06d}" for i in range(m)])
    c = SriCContainer(obs, var)
    c.add_layer(layer_name, X, modality=modality)
    c.log_operation("from_scipy_sparse", shape=list(X.shape))
    return c

def from_dense(X, obs_names=None, var_names=None,
               layer_name="X_raw", modality="RNA", sparsify=True):
    M = sp.csr_matrix(X) if sparsify else X
    return from_scipy_sparse(M, obs_names, var_names, layer_name, modality)

def from_anndata(adata, copy_all_layers=True) -> SriCContainer:
    try: import anndata
    except ImportError: raise ImportError("pip install anndata")
    c = SriCContainer(np.array(adata.obs_names), np.array(adata.var_names))
    if adata.X is not None:
        dt = adata.X.dtype if not sp.issparse(adata.X) else adata.X.dtype
        key = "X_raw" if np.issubdtype(dt, np.integer) else "X_norm"
        c.add_layer(key, adata.X)
    if copy_all_layers:
        for k, v in adata.layers.items(): c.add_layer(k, v)
    for col in adata.obs.columns:
        try: c.obs[col] = np.array(adata.obs[col])
        except Exception: pass
    for col in adata.var.columns:
        try: c.var[col] = np.array(adata.var[col])
        except Exception: pass
    for k, v in adata.obsm.items(): c.obsm[k] = np.asarray(v, np.float64)
    c.uns = dict(adata.uns)
    if "spatial" in adata.obsm:
        c.set_spatial(np.asarray(adata.obsm["spatial"], np.float64))
    c.log_operation("from_anndata", n_obs=c.n_obs, n_vars=c.n_vars)
    return c

def to_anndata(c: SriCContainer):
    try: import anndata as ad, pandas as pd
    except ImportError: raise ImportError("pip install anndata pandas")
    from .container import DerivedLayerRef
    X = None
    for k in ("X_norm","X_raw"):
        if k in c.layers:
            layer = c.layers[k]
            if isinstance(layer, DerivedLayerRef):
                X = c.get_layer(k)
            else:
                X = layer
            break
    if X is None and c.layers:
        first = next(v for v in c.layers.values()
                     if not isinstance(v, DerivedLayerRef))
        X = first
    obs_df = pd.DataFrame(c.obs, index=c.obs_names)
    var_df = pd.DataFrame(c.var, index=c.var_names)
    adata  = ad.AnnData(X=X, obs=obs_df, var=var_df,
                        obsm=dict(c.obsm), uns=dict(c.uns))
    for k, v in c.layers.items():
        if k not in ("X_norm","X_raw"):
            adata.layers[k] = v if not isinstance(v, DerivedLayerRef) else c.get_layer(k)
    if c.spatial is not None:
        adata.obsm["spatial"] = c.spatial
    return adata
