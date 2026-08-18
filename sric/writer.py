"""
sric.writer — Binary encoder v3.0

FILE LAYOUT
───────────
 0     4   Magic "SriC"
 4     3   Version [3, 0, 0]
 7     1   Flags
 8     4   Header JSON length
 12   32   Header SHA-256
 44   N    Header JSON (zlib)
 [64-byte cache-line aligned from here]
 OBSN     obs_names (dict-encoded strings)
 VARN     var_names (dict-encoded strings)
 TOPO     Master Topology Map (shared CSR coords for same-nnz layers)
 LAYR*    per layer (values only if shared topo; else topo+values)
 DLYR*    derived layer descriptors (transform name + source key)
 MGRP*    additional modality groups (each with own VARN+TOPO+LAYR)
 OBSM*    embedding arrays
 OBSC     obs metadata (dict-encoded strings + numeric arrays)
 VARC     var metadata
 SPAT     spatial coordinates (optional)
 ZMDL     ZeroModel parameters (optional)
 BTRI     B-Tree index footer
 [8]      footer byte offset
 [4]      end sentinel "SRIC"

Block structure: [4 tag][4 length][32 SHA-256][N data][pad to 64-byte boundary]
"""

from __future__ import annotations
import io, json, zlib, struct, os, tempfile, datetime
import numpy as np, scipy.sparse as sp
from typing import Dict, Any, Optional

from .container import SriCContainer, ModalityGroup, DerivedLayerRef
from .layer_codec import (encode_int_values, encode_float_values,
                          make_derived_descriptor, TAG_DERIVED,
                          _is_integer_dtype)
from .utils import bitpack_encode, compute_deltas, sha256_block
from .celltype_stats import (compute_celltype_stats, serialize_celltype_stats)

MAGIC   = b"SriC"
VERSION = (3, 0, 0)
ALIGN   = 64

TAG_OBSN = b"OBSN"; TAG_VARN = b"VARN"; TAG_TOPO = b"TOPO"
TAG_LAYR = b"LAYR"; TAG_DLYR = b"DLYR"; TAG_MGRP = b"MGRP"
TAG_OBSM = b"OBSM"; TAG_OBSC = b"OBSC"; TAG_VARC = b"VARC"
TAG_SPAT = b"SPAT"; TAG_ZMDL = b"CTST"; TAG_BTRI = b"BTRI"


def _align(buf: io.BytesIO):
    p = buf.tell(); pad = (-p) % ALIGN
    if pad: buf.write(b'\x00' * pad)


def _write_block(buf: io.BytesIO, tag: bytes, data: bytes) -> int:
    _align(buf); offset = buf.tell()
    buf.write(tag)
    buf.write(struct.pack('<I', len(data)))
    buf.write(sha256_block(data))
    buf.write(data)
    _align(buf)
    return offset


def _encode_str_array(arr: np.ndarray) -> bytes:
    """Dictionary-encode a string array."""
    labels, idx = np.unique(arr.astype(str), return_inverse=True)
    n = len(labels)
    dt = 'uint8' if n <= 256 else ('uint16' if n <= 65536 else 'uint32')
    ic = zlib.compress(idx.astype(dt).tobytes(), level=6)
    return zlib.compress(json.dumps(
        {"n": len(arr), "labels": labels.tolist(), "dtype": dt, "idx": ic.hex()}
    ).encode(), level=6)


def _encode_meta(d: Dict[str, np.ndarray]) -> bytes:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype.kind in ('U','S','O'):
            labels, idx = np.unique(v.astype(str), return_inverse=True)
            n = len(labels)
            dt = 'uint8' if n<=256 else ('uint16' if n<=65536 else 'uint32')
            ic = zlib.compress(idx.astype(dt).tobytes(), level=6)
            out[k] = {"t":"dict","n":len(v),"labels":labels.tolist(),"dt":dt,"idx":ic.hex()}
        elif isinstance(v, np.ndarray):
            out[k] = {"t":"arr","dtype":str(v.dtype),"data":v.tolist()}
        else:
            out[k] = {"t":"scalar","data":v}
    return zlib.compress(json.dumps(out).encode(), level=6)


def _encode_topology(csr) -> bytes:
    meta = {"n_rows": csr.shape[0], "n_cols": csr.shape[1], "nnz": int(csr.nnz)}
    mb   = json.dumps(meta).encode()
    
    # 1. Encode indptr (usually small enough for RAM)
    ip_d = compute_deltas(csr.indptr.astype(np.int64))
    ip_b = bitpack_encode(ip_d)

    # 2. Chunked encoding for indices to avoid massive RAM spikes
    n_values = csr.nnz
    n_blocks = (n_values + 127) // 128
    
    # Manually build the bitpack header: [4: n_values][4: n_blocks]
    idx_header = struct.pack('<II', n_values, n_blocks)
    payload_bytes = [idx_header]

    # 10 MB buffer (must be a multiple of 128 to align with Cython blocks)
    BUFFER_CAPACITY = 128 * 10000 
    buffer = np.zeros(BUFFER_CAPACITY, dtype=np.int64)
    buf_pos = 0

    for r in range(csr.shape[0]):
        s, e = int(csr.indptr[r]), int(csr.indptr[r+1])
        if e > s:
            # Compute deltas for this specific row
            rd = compute_deltas(csr.indices[s:e].astype(np.int64))
            rd_len = len(rd)
            rd_pos = 0

            # Stream into buffer
            while rd_pos < rd_len:
                space = BUFFER_CAPACITY - buf_pos
                take = min(space, rd_len - rd_pos)
                
                buffer[buf_pos : buf_pos + take] = rd[rd_pos : rd_pos + take]
                buf_pos += take
                rd_pos += take

                # Flush buffer to bitpack_encode when full
                if buf_pos == BUFFER_CAPACITY:
                    chunk_enc = bitpack_encode(buffer)
                    # Strip the 8-byte header from the chunk, append raw packed blocks
                    payload_bytes.append(chunk_enc[8:]) 
                    buf_pos = 0

    # Encode remaining tail
    if buf_pos > 0:
        chunk_enc = bitpack_encode(buffer[:buf_pos])
        payload_bytes.append(chunk_enc[8:])

    # Calculate total size of the chunked indices
    idx_b_len = sum(len(chunk) for chunk in payload_bytes)

    # Pre-assemble all parts into a single list to avoid sequential '+' allocations
    final_parts = [
        struct.pack('<I', len(mb)), mb,
        struct.pack('<I', len(ip_b)), ip_b,
        struct.pack('<I', idx_b_len)
    ]
    final_parts.extend(payload_bytes)

    # A single join allocates the exact required memory just once
    return b''.join(final_parts)


def _integer_dtype(arr) -> bool:
    dt = arr.dtype if not sp.issparse(arr) else arr.dtype
    return dt.kind in ('i', 'u')


def _encode_layer(buf, key, data, ref_nnz, modality="RNA", idx=None) -> int:
    """Encode one layer block and return its byte offset."""
    if sp.issparse(data):
        csr = data.tocsr()
        vals = csr.data
        owns_topo = (csr.nnz != ref_nnz)
    else:
        vals = np.asarray(data).ravel()
        owns_topo = True
        csr = None

    if _integer_dtype(data):
        val_bytes = encode_int_values(vals.astype(np.int64))
    else:
        val_bytes = encode_float_values(vals.astype(np.float64))

    meta = {"key": key, "modality": modality,
            "is_sparse": sp.issparse(data), "owns_topo": owns_topo,
            "dtype": str(data.dtype if not sp.issparse(data) else data.dtype)}
    if not sp.issparse(data):
        meta["shape"] = list(np.asarray(data).shape)
    mb = json.dumps(meta).encode()

    topo_sec = b""
    if owns_topo and csr is not None:
        tb = _encode_topology(csr)
        topo_sec = struct.pack('<I', len(tb)) + tb
    elif owns_topo:
        topo_sec = struct.pack('<I', 0)

    payload = (struct.pack('<I', len(mb)) + mb + topo_sec +
               struct.pack('<I', len(val_bytes)) + val_bytes)
    return _write_block(buf, TAG_LAYR, payload)


def _encode_derived_layer(buf, key, ref: DerivedLayerRef, idx=None) -> int:
    """Encode a derived layer descriptor (no float values stored)."""
    desc_b = make_derived_descriptor(
        ref.transform, ref.source_key, ref.size_factors, ref.scale)
    meta   = json.dumps({"key": key, "transform": ref.transform,
                          "source_key": ref.source_key}).encode()
    payload = struct.pack('<I', len(meta)) + meta + desc_b
    return _write_block(buf, TAG_DLYR, payload)


class SriCWriter:
    """Encode a SriCContainer to .SriC binary format v3."""

    def __init__(self, container: SriCContainer,
                 fit_zeromodel: bool = True,
                 cell_type_col: str = "cell_type"):
        self.c             = container
        self.fit_zeromodel = fit_zeromodel
        self.cell_type_col = cell_type_col
        self._idx: Dict[str, int] = {}

    def write(self, path: str):
        """Atomic write: builds to temp file, renames on success."""
        dir_ = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".sric.tmp")
        os.close(fd)
        try:
            self._write_to(tmp)
            os.replace(tmp, path)
        except Exception:
            try: os.unlink(tmp)
            except OSError: pass
            raise

        size_mb = os.path.getsize(path) / 1024**2
        print(f"[SriC] Written : {path}")
        print(f"[SriC] Size    : {size_mb:.3f} MB")
        print(f"[SriC] Shape   : {self.c.n_obs:,} obs × {self.c.n_vars:,} vars")
        layer_desc = []
        for k, v in self.c.layers.items():
            layer_desc.append(f"{k}[derived]" if isinstance(v, DerivedLayerRef) else k)
        print(f"[SriC] Layers  : {layer_desc}")

    def _write_to(self, path: str):
        """Streams the container components directly to disk."""
        with open(path, 'wb') as buf:
            # ── Preamble ──────────────────────────────────────────────────────────
            flags = 0
            if self.c.spatial is not None:         flags |= 0x01
            if self.c.modalities:                  flags |= 0x02
            if self.c.zero_model is not None or (
                    self.fit_zeromodel and
                    self.cell_type_col in self.c.obs and
                    'X_raw' in self.c.layers and
                    not isinstance(self.c.layers['X_raw'], DerivedLayerRef)):
                flags |= 0x04

            hdr = {
                "format": "SriC", "version": list(VERSION),
                "created": datetime.datetime.now(datetime.timezone.utc)
                                      .isoformat().replace("+00:00", "Z"),
                "n_obs": self.c.n_obs, "n_vars": self.c.n_vars,
                "layers": [k for k, v in self.c.layers.items()
                           if not isinstance(v, DerivedLayerRef)],
                "derived_layers": [k for k, v in self.c.layers.items()
                                   if isinstance(v, DerivedLayerRef)],
                "modalities": {k: {"n_vars": mg.n_vars, "layers": list(mg.layers)}
                               for k, mg in self.c.modalities.items()},
                "obsm_keys": list(self.c.obsm),
                "ontology_map": self.c.ontology_map,
                "provenance": [p.to_dict() for p in self.c.provenance],
            }
            hdr_b   = zlib.compress(json.dumps(hdr).encode(), level=6)
            hdr_sha = sha256_block(hdr_b)

            buf.write(MAGIC)
            buf.write(bytes(VERSION))
            buf.write(struct.pack('<B', flags))
            buf.write(struct.pack('<I', len(hdr_b)))
            buf.write(hdr_sha)
            buf.write(hdr_b)

            # ── String name blocks ────────────────────────────────────────────────
            self._idx['__obsn__'] = _write_block(buf, TAG_OBSN,
                                                  _encode_str_array(self.c.obs_names))
            self._idx['__varn__'] = _write_block(buf, TAG_VARN,
                                                  _encode_str_array(self.c.var_names))

            # ── Primary topology + layers ─────────────────────────────────────────
            ref_csr = next((v.tocsr() for v in self.c.layers.values()
                            if sp.issparse(v)), None)
            ref_nnz = ref_csr.nnz if ref_csr is not None else 0

            if ref_csr is not None:
                self._idx['__topo_RNA__'] = _write_block(
                    buf, TAG_TOPO, _encode_topology(ref_csr))

            for key, data in self.c.layers.items():
                if isinstance(data, DerivedLayerRef):
                    off = _encode_derived_layer(buf, key, data)
                    self._idx[f'dlyr:RNA:{key}'] = off
                else:
                    off = _encode_layer(buf, key, data, ref_nnz)
                    self._idx[f'layer:RNA:{key}'] = off

            # ── Additional modality groups ────────────────────────────────────────
            for mname, mg in self.c.modalities.items():
                mg_meta = zlib.compress(json.dumps({
                    "name": mname, "n_obs": mg.n_obs,
                    "n_vars": mg.n_vars, "layers": list(mg.layers)
                }).encode(), level=6)
                self._idx[f'__mgrp_{mname}__'] = _write_block(
                    buf, TAG_MGRP, mg_meta)
                self._idx[f'__varn_{mname}__'] = _write_block(
                    buf, TAG_VARN, _encode_str_array(mg.var_names))

                mg_ref = next((v.tocsr() for v in mg.layers.values()
                               if sp.issparse(v)), None)
                mg_nnz = mg_ref.nnz if mg_ref is not None else 0
                if mg_ref is not None:
                    self._idx[f'__topo_{mname}__'] = _write_block(
                        buf, TAG_TOPO, _encode_topology(mg_ref))
                for key, data in mg.layers.items():
                    self._idx[f'layer:{mname}:{key}'] = _encode_layer(
                        buf, key, data, mg_nnz, modality=mname)

            # ── Embeddings ────────────────────────────────────────────────────────
            for key, arr in self.c.obsm.items():
                a    = arr.astype(np.float64)
                genc = encode_float_values(a.ravel())
                meta = json.dumps({"key": key, "shape": list(a.shape)}).encode()
                self._idx[f'obsm:{key}'] = _write_block(
                    buf, TAG_OBSM,
                    struct.pack('<I', len(meta)) + meta + genc)

            # ── Obs / Var metadata ────────────────────────────────────────────────
            if self.c.obs:
                self._idx['__obsc__'] = _write_block(
                    buf, TAG_OBSC, _encode_meta(self.c.obs))
            if self.c.var:
                self._idx['__varc__'] = _write_block(
                    buf, TAG_VARC, _encode_meta(self.c.var))

            # ── Spatial ───────────────────────────────────────────────────────────
            if self.c.spatial is not None:
                meta = json.dumps({"shape": list(self.c.spatial.shape)}).encode()
                genc = encode_float_values(self.c.spatial.ravel())
                self._idx['__spatial__'] = _write_block(
                    buf, TAG_SPAT,
                    struct.pack('<I', len(meta)) + meta + genc)

            # ── CellType Statistics block ─────────────────────────────────────────
            cts = self.c.zero_model   # reuse field, now stores ct_stats
            if cts is None and self.fit_zeromodel:
                if (self.cell_type_col in self.c.obs and
                        'X_raw' in self.c.layers and
                        not isinstance(self.c.layers['X_raw'], DerivedLayerRef)):
                    cts = compute_celltype_stats(
                        self.c.layers['X_raw'],
                        self.c.obs[self.cell_type_col])
            if cts is not None:
                cts_bytes = serialize_celltype_stats(cts)
                if cts_bytes:
                    self._idx['__zmdl__'] = _write_block(buf, TAG_ZMDL, cts_bytes)

            # ── B-Tree footer ─────────────────────────────────────────────────────
            def _sanitize_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.generic):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: _sanitize_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(_sanitize_for_json(v) for v in obj)
                return obj

            gene_idx = {g: int(i) for i, g in enumerate(self.c.var_names)}
            footer   = {"block_index": self._idx,
                        "gene_index":  gene_idx,
                        "uns":         _sanitize_for_json(self.c.uns)}
            _align(buf)
            footer_offset = buf.tell()

            # Write the final footer blocks directly to the active file handle
            _write_block(buf, TAG_BTRI,
                         zlib.compress(json.dumps(footer).encode(), level=9))
            buf.write(struct.pack('<Q', footer_offset))
            buf.write(b"SRIC")

            # Function ends here. The `with` block will automatically close the file safely.
