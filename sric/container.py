"""
sric.container — In-memory data container  v3.0

SriCContainer holds all biological data for one dataset.
Key design decisions vs AnnData:
  - obs_names / var_names stored in dedicated file blocks (not JSON header)
  - Multi-modal: each modality has its own var_names (RNA genes ≠ ATAC peaks)
  - Derived layers: log1p and sqrt of integer layers stored as descriptors
  - Provenance: append-only, frozen dataclass entries
  - ZeroModel: dropout parameters stored as optional metadata block
"""

from __future__ import annotations
import datetime, numpy as np, scipy.sparse as sp
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class ProvenanceEntry:
    timestamp:        str
    operation:        str
    parameters:       Dict[str, Any]
    software_version: str = "sric-3.0.0"

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "operation": self.operation,
                "parameters": self.parameters, "software_version": self.software_version}

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceEntry":
        return cls(**d)


class ModalityGroup:
    """
    One omics modality: its own var_names and layers.
    All groups share obs_names (same cells).
    """
    def __init__(self, name: str, var_names: np.ndarray, n_obs: int):
        self.name      = name
        self.var_names = np.asarray(var_names, dtype=str)
        self.n_obs     = n_obs
        self.layers:   Dict[str, Any] = {}

    @property
    def n_vars(self) -> int: return len(self.var_names)

    def add_layer(self, key: str, data):
        if hasattr(data, 'shape') and data.shape != (self.n_obs, self.n_vars):
            raise ValueError(
                f"[{self.name}] Layer '{key}': shape {data.shape} ≠ "
                f"({self.n_obs}, {self.n_vars})")
        self.layers[key] = sp.csr_matrix(data) if sp.issparse(data) else np.asarray(data)


class DerivedLayerRef:
    """
    Reference to a layer that is computed from another rather than stored.
    Stored in the file as a ~50-byte descriptor; reconstructed at load time.
    """
    def __init__(self, transform: str, source_key: str,
                 size_factors: Optional[np.ndarray] = None,
                 scale: float = 1e4):
        self.transform   = transform
        self.source_key  = source_key
        self.size_factors = size_factors
        self.scale        = scale

    def __repr__(self):
        return f"DerivedLayerRef({self.transform!r} of '{self.source_key}')"


class SriCContainer:
    """
    In-memory single-cell data container.

    Primary access
    --------------
    container.layers["X_raw"]               — sparse int32 count matrix
    container.layers["X_norm"]              — DerivedLayerRef OR sparse float64
    container.modalities["ATAC"].layers[..] — ATAC modality
    container.obs["cell_type"]              — cell metadata
    container.obsm["X_umap"]               — embeddings
    container.spatial                       — (n_obs, D) float64
    container.zero_model                    — dropout parameters (optional)
    container.provenance                    — List[ProvenanceEntry], append-only
    """

    def __init__(self, obs_names: np.ndarray, var_names: np.ndarray):
        self._obs_names = np.asarray(obs_names, dtype=str)
        self._var_names = np.asarray(var_names, dtype=str)

        self.layers:     Dict[str, Any]          = {}
        self.modalities: Dict[str, ModalityGroup] = {}

        self.obs:  Dict[str, np.ndarray] = {}
        self.var:  Dict[str, np.ndarray] = {}
        self.obsm: Dict[str, np.ndarray] = {}
        self.varm: Dict[str, np.ndarray] = {}
        self.uns:  Dict[str, Any]        = {}

        self.spatial:    Optional[np.ndarray] = None
        self.zero_model: Optional[dict]       = None
        self.ontology_map: Dict[str, str]     = {}

        self._provenance: List[ProvenanceEntry] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def obs_names(self) -> np.ndarray: return self._obs_names
    @property
    def var_names(self) -> np.ndarray: return self._var_names
    @property
    def n_obs(self) -> int: return len(self._obs_names)
    @property
    def n_vars(self) -> int: return len(self._var_names)
    @property
    def shape(self): return (self.n_obs, self.n_vars)
    @property
    def provenance(self) -> List[ProvenanceEntry]:
        return list(self._provenance)

    # ── Layer management ──────────────────────────────────────────────────────

    def add_layer(self, key: str, data, modality: str = "RNA"):
        """Add a data layer. Shape must match (n_obs, n_vars)."""
        if isinstance(data, DerivedLayerRef):
            self.layers[key] = data
            self._log("add_derived_layer", key=key, transform=data.transform,
                      source=data.source_key)
            return

        exp = (self.n_obs, self.n_vars)
        if hasattr(data, 'shape') and data.shape != exp:
            raise ValueError(f"Layer '{key}': shape {data.shape} ≠ {exp}")

        self.layers[key] = sp.csr_matrix(data) if sp.issparse(data) else np.asarray(data)
        self._log("add_layer", key=key, modality=modality,
                  shape=list(exp), dtype=str(
                      data.dtype if not sp.issparse(data) else data.dtype))

    def add_derived_layer(self, key: str, transform: str,
                          source_key: str,
                          size_factors: Optional[np.ndarray] = None,
                          scale: float = 1e4):
        """
        Register a derived (recomputable) layer instead of storing float values.

        Example
        -------
        container.add_derived_layer("X_norm", "log1p", "X_raw")
        # X_norm will be recomputed as log1p(X_raw) at load time — exact.
        """
        from .layer_codec import DERIVABLE
        if transform not in DERIVABLE:
            raise ValueError(f"'{transform}' not derivable. Use {DERIVABLE}.")
        if source_key not in self.layers:
            raise KeyError(f"Source layer '{source_key}' not found.")
        ref = DerivedLayerRef(transform, source_key, size_factors, scale)
        self.layers[key] = ref
        self._log("add_derived_layer", key=key, transform=transform,
                  source_key=source_key)

    def get_layer(self, key: str):
        """
        Get a layer. Derived layers are recomputed on first access
        then cached — subsequent calls have zero recomputation cost.
        """
        if key not in self.layers:
            raise KeyError(f"Layer '{key}' not found. Available: {list(self.layers)}")
        layer = self.layers[key]
        if isinstance(layer, DerivedLayerRef):
            # Lazy recompute: compute once, cache as concrete matrix
            from .layer_codec import recompute_derived
            src = self.layers.get(layer.source_key)
            if src is None:
                raise KeyError(f"Source layer '{layer.source_key}' not found.")
            computed = sp.csr_matrix(recompute_derived(
                {"transform": layer.transform,
                 "size_factors": layer.size_factors,
                 "scale": layer.scale},
                src
            ))
            # Cache it (keep the DerivedLayerRef as a sentinel key too)
            self._layer_cache = getattr(self, '_layer_cache', {})
            self._layer_cache[key] = computed
            return computed
        # Check cache
        cache = getattr(self, '_layer_cache', {})
        if key in cache:
            return cache[key]
        return layer

    # ── Multi-modal ───────────────────────────────────────────────────────────

    def add_modality(self, name: str, var_names: np.ndarray) -> ModalityGroup:
        if name in self.modalities:
            raise ValueError(f"Modality '{name}' already registered.")
        mg = ModalityGroup(name, var_names, self.n_obs)
        self.modalities[name] = mg
        self._log("add_modality", name=name, n_vars=len(var_names))
        return mg

    # ── Spatial ───────────────────────────────────────────────────────────────

    def set_spatial(self, coords: np.ndarray):
        coords = np.asarray(coords, np.float64)
        if coords.ndim != 2 or coords.shape[0] != self.n_obs:
            raise ValueError(f"coords must be ({self.n_obs}, D), got {coords.shape}")
        if coords.shape[1] not in (2, 3, 4):
            raise ValueError(f"D must be 2, 3, or 4 (got {coords.shape[1]})")
        self.spatial = coords
        self._log("set_spatial", shape=list(coords.shape))

    # ── Ontology ──────────────────────────────────────────────────────────────

    def set_ontology(self, obs_col: str, mapping: Dict[str, str]):
        self.ontology_map.update(mapping)
        self._log("set_ontology", obs_col=obs_col, n_mapped=len(mapping))

    # ── Provenance ────────────────────────────────────────────────────────────

    def _log(self, op: str, **kw):
        self._provenance.append(ProvenanceEntry(
            timestamp=datetime.datetime.now(datetime.timezone.utc)
                        .isoformat().replace("+00:00", "Z"),
            operation=op, parameters=kw))

    def log_operation(self, op: str, **kw):
        self._log(op, **kw)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        layer_desc = []
        for k, v in self.layers.items():
            if isinstance(v, DerivedLayerRef):
                layer_desc.append(f"{k}[derived:{v.transform}]")
            else:
                layer_desc.append(k)
        sp_str = f"spatial={self.spatial.shape}" if self.spatial is not None else "no spatial"
        zm_str = "zero_model=YES" if self.zero_model else "zero_model=no"
        return (
            f"SriCContainer v3.0\n"
            f"  shape      : {self.n_obs:,} obs × {self.n_vars:,} vars\n"
            f"  layers     : {layer_desc}\n"
            f"  modalities : {list(self.modalities)}\n"
            f"  obs cols   : {list(self.obs)}\n"
            f"  {sp_str}  |  {zm_str}\n"
            f"  provenance : {len(self._provenance)} entries"
        )
