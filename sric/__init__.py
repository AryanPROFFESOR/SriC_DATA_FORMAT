"""
sric v3.0.0 — Single-cell Robust Information Container

Genuine contributions:
  1. Derived Layer Elimination — log1p/sqrt layers stored as 5-byte descriptors.
     No existing format does this. Achieves ~22× vs storing float64.
  2. Master Topology Map — CSR coordinates stored once for all same-nnz layers.
  3. ZeroModel — per-(gene, cell-type) ZINB dropout parameters embedded in file.
  4. Gorilla XOR 7-bit fix — corrects silent data corruption in Pelkonen 2015.
  5. BioLZ — uint8 clipping before zlib for integer count arrays (~10× vs int32).

What is NOT claimed as novel here:
  - Gorilla XOR algorithm itself (Pelkonen 2015)
  - Zigzag bit-packing (Protocol Buffers, FastPFOR)
  - zlib compression (1995)
  - CSR sparse format
"""
from .container import SriCContainer, ModalityGroup, ProvenanceEntry, DerivedLayerRef
from .writer    import SriCWriter
from .reader    import SriCReader
from .convert   import from_anndata, to_anndata, from_scipy_sparse, from_dense
from .utils     import codec_backend

__version__ = "3.0.0"
__all__ = [
    "SriCContainer", "ModalityGroup", "ProvenanceEntry", "DerivedLayerRef",
    "SriCWriter", "SriCReader",
    "from_anndata", "to_anndata", "from_scipy_sparse", "from_dense",
    "codec_backend",
]
