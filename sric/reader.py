"""
sric.reader — Binary decoder v3.0

Uses OS-level mmap for zero-copy file mapping.
Each block is decompressed JIT (Just-In-Time) only when accessed.
Derived layers (log1p etc.) are recomputed at load, not decoded.
"""

from __future__ import annotations
import io, json, zlib, struct, mmap
import numpy as np, scipy.sparse as sp
from typing import Optional, Dict

from .container import SriCContainer, ModalityGroup, ProvenanceEntry, DerivedLayerRef
from .layer_codec import (decode_int_values, decode_float_values,
                          parse_derived_descriptor, recompute_derived,
                          TAG_DERIVED, _is_integer_dtype)
from .utils import bitpack_decode, reconstruct_deltas, sha256_block
from .celltype_stats import deserialize_celltype_stats

MAGIC    = b"SriC"
ALIGN    = 64
PREAMBLE = 44   # 4+3+1+4+32

TAG_OBSN = b"OBSN"; TAG_VARN = b"VARN"; TAG_TOPO = b"TOPO"
TAG_LAYR = b"LAYR"; TAG_DLYR = b"DLYR"; TAG_MGRP = b"MGRP"
TAG_OBSM = b"OBSM"; TAG_OBSC = b"OBSC"; TAG_VARC = b"VARC"
TAG_SPAT = b"SPAT"; TAG_ZMDL = b"CTST"; TAG_BTRI = b"BTRI"


class SriCReader:
    """
    Read a .SriC v3 file into a SriCContainer.

    File is memory-mapped; blocks are read via seek, not full load.
    Derived layers are recomputed exactly from their source integer layers.
    """

    def __init__(self, path: str, verify_checksums: bool = True):
        self.path = path
        self.verify = verify_checksums
        self._mm: Optional[mmap.mmap] = None
        self._fh = None

    def _open(self):
        if self._mm is None:
            self._fh = open(self.path, 'rb')
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __enter__(self):  self._open(); return self
    def __exit__(self, *_):
        if self._mm: self._mm.close(); self._mm = None
        if self._fh: self._fh.close(); self._fh = None

    def _mm_slice(self, start: int, end: int) -> bytes:
        self._open()
        return bytes(self._mm[start:end])

    # ── Block IO ──────────────────────────────────────────────────────────────

    def _read_block(self, offset: int):
        self._open()
        mm  = self._mm
        tag = bytes(mm[offset:offset+4])
        dlen = struct.unpack('<I', mm[offset+4:offset+8])[0]
        chk  = bytes(mm[offset+8:offset+40])
        data = bytes(mm[offset+40:offset+40+dlen])
        if self.verify:
            if sha256_block(data) != chk:
                raise ValueError(f"SHA-256 mismatch at offset {offset} tag={tag!r}")
        return tag, data

    def _next_offset(self, offset: int) -> int:
        self._open()
        dlen = struct.unpack('<I', self._mm[offset+4:offset+8])[0]
        end  = offset + 40 + dlen
        return end + (-end % ALIGN)

    # ── Preamble ──────────────────────────────────────────────────────────────

    def _parse_preamble(self) -> dict:
        self._open()
        mm = self._mm
        if bytes(mm[0:4]) != MAGIC:
            raise ValueError(f"Not a .SriC file (magic={bytes(mm[0:4])!r})")
        major = mm[4]
        if major != 3:
            raise ValueError(
                f"Unsupported major version {major} (this reader supports v3.x)")
        hlen = struct.unpack('<I', mm[8:12])[0]
        hsha = bytes(mm[12:44])
        hraw = bytes(mm[44:44+hlen])
        if self.verify and sha256_block(hraw) != hsha:
            raise ValueError("Header SHA-256 mismatch — file corrupted")
        hdr = json.loads(zlib.decompress(hraw))
        hdr['_hdr_end'] = 44 + hlen
        return hdr

    def _footer_offset(self) -> int:
        self._open()
        sz = len(self._mm)
        if bytes(self._mm[sz-4:sz]) != b"SRIC":
            raise ValueError("Missing end sentinel — file truncated")
        return struct.unpack('<Q', self._mm[sz-12:sz-4])[0]

    def _parse_footer(self) -> dict:
        _, data = self._read_block(self._footer_offset())
        return json.loads(zlib.decompress(data))

    # ── String arrays ─────────────────────────────────────────────────────────

    def _decode_str_block(self, data: bytes) -> np.ndarray:
        p = json.loads(zlib.decompress(data))
        idx = np.frombuffer(zlib.decompress(bytes.fromhex(p['idx'])),
                            dtype=p['dtype'])
        return np.array(p['labels'])[idx]

    # ── Topology ──────────────────────────────────────────────────────────────

    def _decode_topo(self, data: bytes) -> dict:
        pos = 0
        ml  = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        meta = json.loads(data[pos:pos+ml]); pos += ml
        ipl = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        ip_d = bitpack_decode(data[pos:pos+ipl]); pos += ipl
        idl = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        id_d = bitpack_decode(data[pos:pos+idl])

        indptr = reconstruct_deltas(ip_d).astype(np.int32)
        nnz    = meta['nnz']; n_rows = meta['n_rows']

        # OOM FIX: Use disk-backed memmap for massive arrays, targeted to D Drive
        if nnz > 100_000_000:
            import tempfile
            _tmp = tempfile.TemporaryFile(dir="D:/")  # Force scratch onto external SSD
            indices = np.memmap(_tmp, dtype=np.int32, mode='w+', shape=(nnz,))
        else:
            indices = np.empty(nnz, np.int32)

        flat = 0
        for r in range(n_rows):
            rn = int(indptr[r+1]-indptr[r])
            if rn == 0: continue
            indices[flat:flat+rn] = reconstruct_deltas(id_d[flat:flat+rn]).astype(np.int32)
            flat += rn
        return {"indptr": indptr, "indices": indices,
                "shape": (meta['n_rows'], meta['n_cols']), "nnz": nnz}

    # ── Layer ─────────────────────────────────────────────────────────────────

    def _decode_layer(self, data: bytes, shared_topo: Optional[dict]):
        pos = 0
        ml  = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        meta = json.loads(data[pos:pos+ml]); pos += ml

        if meta.get('owns_topo') and meta.get('is_sparse'):
            tl = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
            topo = self._decode_topo(data[pos:pos+tl]) if tl else shared_topo
            pos += tl
        elif meta.get('owns_topo'):
            tl = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
            pos += tl; topo = None
        else:
            topo = shared_topo

        vl   = struct.unpack('<I', data[pos:pos+4])[0]; pos += 4
        vdat = data[pos:pos+vl]

        # Detect encoding by looking at dtype and tag byte
        is_int = meta.get('dtype','').startswith(('int','uint'))
        if is_int:
            vals = decode_int_values(vdat)
        else:
            vals = decode_float_values(vdat)

        if meta.get('is_sparse') and topo:
            return sp.csr_matrix((vals, topo['indices'], topo['indptr']),
                                 shape=topo['shape'])
        if not meta.get('is_sparse'):
            return vals.reshape(tuple(meta.get('shape', [len(vals)])))
        return vals

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _decode_meta(self, data: bytes) -> dict:
        raw = json.loads(zlib.decompress(data))
        out: dict = {}
        for k, v in raw.items():
            t = v.get('t', v.get('type'))
            if t == 'dict':
                idx = np.frombuffer(zlib.decompress(bytes.fromhex(v['idx'])),
                                    dtype=v['dt'])
                out[k] = np.array(v['labels'])[idx]
            elif t == 'arr' or t == 'ndarray':
                out[k] = np.array(v['data'], dtype=v.get('dtype', v.get('dtype')))
            else:
                out[k] = v.get('data', v.get('data'))
        return out

    # ── Full load ─────────────────────────────────────────────────────────────

    def load(self, recompute_derived: bool = True) -> SriCContainer:
        """
        Load all blocks. Derived layers are recomputed from source integers.

        Parameters
        ----------
        recompute_derived : if True (default), derived layers (log1p etc.)
            are computed and stored as sparse float matrices.
            If False, layers are stored as DerivedLayerRef objects.
        """
        hdr    = self._parse_preamble()
        footer = self._parse_footer()

        hdr_end = hdr['_hdr_end']
        offset  = hdr_end + (-hdr_end % ALIGN)
        footer_off = self._footer_offset()

        container: Optional[SriCContainer] = None
        shared_topo: Optional[dict]        = None
        obs_names: Optional[np.ndarray]    = None
        var_names: Optional[np.ndarray]    = None
        current_mg:   Optional[ModalityGroup] = None
        mg_topo:   Optional[dict]          = None

        while offset < footer_off:
            tag, data = self._read_block(offset)

            if tag == TAG_OBSN:
                obs_names = self._decode_str_block(data)

            elif tag == TAG_VARN:
                vn = self._decode_str_block(data)
                if current_mg is not None:
                    current_mg.var_names = vn
                elif var_names is None:
                    var_names = vn
                if container is None and obs_names is not None and var_names is not None:
                    container = SriCContainer(obs_names, var_names)
                    container.modality_map = hdr.get('modalities', {})
                    container.ontology_map = hdr.get('ontology_map', {})
                    container.uns          = footer.get('uns', {})
                    container._provenance  = [ProvenanceEntry.from_dict(p)
                                              for p in hdr.get('provenance', [])]

            elif tag == TAG_TOPO:
                topo = self._decode_topo(data)
                if current_mg is not None: mg_topo = topo
                else:                      shared_topo = topo

            elif tag == TAG_MGRP:
                mg_meta = json.loads(zlib.decompress(data))
                name = mg_meta['name']
                current_mg = ModalityGroup(name, np.array([]), hdr['n_obs'])
                mg_topo = None
                if container is not None:
                    container.modalities[name] = current_mg

            elif tag == TAG_LAYR:
                layer = self._decode_layer(data,
                                           mg_topo if current_mg else shared_topo)
                ml   = struct.unpack('<I', data[:4])[0]
                meta = json.loads(data[4:4+ml])
                if current_mg is not None:
                    current_mg.layers[meta['key']] = layer
                elif container is not None:
                    container.layers[meta['key']] = layer

            elif tag == TAG_DLYR:
                ml   = struct.unpack('<I', data[:4])[0]
                meta = json.loads(data[4:4+ml])
                desc = parse_derived_descriptor(data[4+ml:])
                if container is not None:
                    key = meta['key']
                    if recompute_derived:
                        src = container.layers.get(desc['source_key'])
                        if src is not None:
                            from .layer_codec import recompute_derived as _recomp
                            container.layers[key] = sp.csr_matrix(
                                _recomp(desc, src))
                        else:
                            # Source not yet loaded; store ref
                            container.layers[key] = DerivedLayerRef(
                                desc['transform'], desc['source_key'],
                                desc.get('size_factors'), desc.get('scale', 1e4))
                    else:
                        container.layers[key] = DerivedLayerRef(
                            desc['transform'], desc['source_key'],
                            desc.get('size_factors'), desc.get('scale', 1e4))

            elif tag == TAG_OBSM:
                ml   = struct.unpack('<I', data[:4])[0]
                meta = json.loads(data[4:4+ml])
                arr  = decode_float_values(data[4+ml:]).reshape(meta['shape'])
                if container: container.obsm[meta['key']] = arr

            elif tag == TAG_OBSC:
                if container: container.obs = self._decode_meta(data)

            elif tag == TAG_VARC:
                if container: container.var = self._decode_meta(data)

            elif tag == TAG_SPAT:
                ml   = struct.unpack('<I', data[:4])[0]
                meta = json.loads(data[4:4+ml])
                arr  = decode_float_values(data[4+ml:]).reshape(meta['shape'])
                if container: container.spatial = arr

            elif tag == TAG_ZMDL:
                ct_stats = deserialize_celltype_stats(data)
                if container: container.zero_model = ct_stats

            offset = self._next_offset(offset)

        if container is None:
            raise ValueError("File missing OBSN/VARN blocks")

        # Resolve any DerivedLayerRef that needed source loaded after it
        if recompute_derived:
            for key, val in list(container.layers.items()):
                if isinstance(val, DerivedLayerRef):
                    src = container.layers.get(val.source_key)
                    if src is not None and not isinstance(src, DerivedLayerRef):
                        from .layer_codec import recompute_derived as _rc
                        container.layers[key] = sp.csr_matrix(_rc(
                            {"transform": val.transform,
                             "size_factors": val.size_factors,
                             "scale": val.scale}, src))

        return container

    # ── Direct seeks ─────────────────────────────────────────────────────────

    def query_gene(self, gene: str, layer: str = "X_raw",
                   modality: str = "RNA") -> np.ndarray:
        """Return expression of one gene across all cells (B-Tree seek)."""
        footer = self._parse_footer()
        bi     = footer['block_index']
        gi     = footer['gene_index']
        if gene not in gi:
            raise KeyError(f"Gene '{gene}' not found.")
        col = gi[gene]
        lkey = f"layer:{modality}:{layer}"
        if lkey not in bi:
            raise KeyError(f"Layer '{layer}' not found. Available: {list(bi)}")
        tkey = f"__topo_{modality}__"
        topo = None
        if tkey in bi:
            _, td = self._read_block(bi[tkey]); topo = self._decode_topo(td)
        _, ld = self._read_block(bi[lkey])
        mat = self._decode_layer(ld, topo)
        if sp.issparse(mat):
            return np.asarray(mat[:, col].todense()).ravel()
        return mat[:, col]

    def query_cell(self, barcode: str, layer: str = "X_raw",
                   modality: str = "RNA") -> np.ndarray:
        """Return full expression profile of one cell (B-Tree seek)."""
        footer = self._parse_footer()
        bi     = footer['block_index']
        if '__obsn__' not in bi:
            raise ValueError("No OBSN block in index.")
        _, od = self._read_block(bi['__obsn__'])
        obs   = self._decode_str_block(od)
        hits  = np.where(obs == barcode)[0]
        if not len(hits):
            raise KeyError(f"Barcode '{barcode}' not found.")
        idx  = int(hits[0])
        tkey = f"__topo_{modality}__"
        topo = None
        if tkey in bi:
            _, td = self._read_block(bi[tkey]); topo = self._decode_topo(td)
        lkey = f"layer:{modality}:{layer}"
        _, ld = self._read_block(bi[lkey])
        mat = self._decode_layer(ld, topo)
        if sp.issparse(mat):
            return np.asarray(mat[idx, :].todense()).ravel()
        return mat[idx, :]
