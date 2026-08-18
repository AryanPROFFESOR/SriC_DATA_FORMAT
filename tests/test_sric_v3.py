"""
tests/test_sric_v3.py — Full test suite for .SriC v3.0

Tests cover every genuine contribution:
  - Gorilla XOR 7-bit fix (NaN, ±Inf, wide-range floats)
  - BioLZ integer codec (correctness, overflow handling)
  - Derived layer elimination (exact recomputation)
  - Master topology map (shared coords)
  - ZeroModel estimation and serialisation
  - Full round-trip: integer, float, derived, multi-modal, spatial
  - Integrity: checksum detection, version rejection, truncation
  - Gene/cell queries via B-Tree seek
"""

import sys, os, struct, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, numpy as np
import scipy.sparse as sp
import pytest
import sric
from sric.utils import (gorilla_encode, gorilla_decode,
                        bitpack_encode, bitpack_decode,
                        compute_deltas, reconstruct_deltas)
from sric.layer_codec import (encode_int_values, decode_int_values,
                               encode_float_values, decode_float_values,
                               _is_log1p_of, _is_sqrt_of)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_sric(tmp_path):
    return str(tmp_path / "test.sric")

def make_container(n=150, m=300, density=0.08, seed=42):
    np.random.seed(seed)
    X = sp.random(n, m, density=density, format='csr', dtype=np.int32)
    X.data[:] = (np.random.negative_binomial(1,.4,X.nnz)+1).astype(np.int32)
    obs = np.array([f"CELL_{i:05d}" for i in range(n)])
    var = np.array([f"GENE_{i:05d}" for i in range(m)])
    c = sric.SriCContainer(obs, var)
    c.add_layer("X_raw", X)
    return c, X


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gorilla XOR — correctness including 7-bit fix
# ─────────────────────────────────────────────────────────────────────────────

class TestGorillaXOR:
    def test_wide_range_50_trials(self):
        np.random.seed(0)
        for t in range(50):
            arr = np.random.uniform(-1e6, 1e6, 300)
            assert np.array_equal(arr, gorilla_decode(gorilla_encode(arr))), \
                f"Trial {t} failed"

    def test_bit_exact_not_approximate(self):
        """Must reconstruct every bit, not just close values."""
        np.random.seed(7)
        arr = np.random.uniform(-1e8, 1e8, 1000)
        dec = gorilla_decode(gorilla_encode(arr))
        assert arr.dtype == dec.dtype == np.float64
        assert np.array_equal(arr, dec), "Must be bit-exact"

    def test_nan_inf_preserved(self):
        arr = np.array([1.0, float('nan'), float('inf'), -float('inf'), -0.0, 0.0])
        dec = gorilla_decode(gorilla_encode(arr))
        assert np.isnan(dec[1])
        assert np.isinf(dec[2]) and dec[2] > 0
        assert np.isinf(dec[3]) and dec[3] < 0
        assert np.array_equal(arr[4:], dec[4:])

    def test_7bit_sig_field_sig64_case(self):
        """
        The 7-bit significant field fix: when two consecutive float64 values
        share NO bits (XOR = all ones), sig=64 which needs 7 bits to encode.
        A 6-bit field would overflow and cause silent data corruption.
        """
        # Force maximally dissimilar consecutive values
        arr = np.array([
            np.frombuffer(b'\x00'*8, dtype=np.float64)[0],    # all-zero bits
            np.frombuffer(b'\xff'*8, dtype=np.float64)[0],    # all-one bits
            np.frombuffer(b'\xaa'*8, dtype=np.float64)[0],    # alternating
        ])
        dec = gorilla_decode(gorilla_encode(arr))
        # View as uint64 for bit comparison
        assert np.array_equal(arr.view(np.uint64), dec.view(np.uint64))

    def test_single_value(self):
        arr = np.array([3.14159265358979323846])
        assert np.array_equal(arr, gorilla_decode(gorilla_encode(arr)))

    def test_all_identical(self):
        arr = np.full(500, 0.6931471805599453)   # log1p(1)
        dec = gorilla_decode(gorilla_encode(arr))
        assert np.array_equal(arr, dec)

    def test_empty(self):
        assert len(gorilla_decode(gorilla_encode(np.array([], np.float64)))) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bitpack — correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestBitpack:
    def test_signed_roundtrip(self):
        arr = np.array([0,-1,1,-100,100,-32768,32767], np.int64)
        assert np.array_equal(arr, bitpack_decode(bitpack_encode(arr)))

    def test_large_values(self):
        arr = np.random.randint(0, 100_000, 1000, dtype=np.int64)
        assert np.array_equal(arr, bitpack_decode(bitpack_encode(arr)))

    def test_50_trials(self):
        np.random.seed(99)
        for t in range(50):
            arr = np.random.randint(-5000, 5000, 500, dtype=np.int64)
            assert np.array_equal(arr, bitpack_decode(bitpack_encode(arr))), \
                f"Trial {t}"

    def test_deltas_csr(self):
        X = sp.random(100, 500, density=0.1, format='csr')
        idx = X.indices.astype(np.int64)
        parts = []
        for r in range(X.shape[0]):
            s,e = int(X.indptr[r]), int(X.indptr[r+1])
            if e>s: parts.append(compute_deltas(idx[s:e]))
        flat = np.concatenate(parts)
        rec = bitpack_decode(bitpack_encode(flat))
        assert np.array_equal(flat, rec)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Layer codecs
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerCodecs:
    def test_biolz_integer_roundtrip(self):
        arr = (np.random.negative_binomial(1,.4,10000)+1).astype(np.int64)
        dec = decode_int_values(encode_int_values(arr))
        assert np.array_equal(arr, dec)

    def test_biolz_overflow_values(self):
        """Values > 255 (rare doublets/highly expressed genes) handled correctly."""
        arr = np.array([1,2,300,1,500,3,1000,2], dtype=np.int64)
        dec = decode_int_values(encode_int_values(arr))
        assert np.array_equal(arr, dec)

    def test_biolz_all_overflow(self):
        arr = np.random.randint(256, 10000, 100, dtype=np.int64)
        dec = decode_int_values(encode_int_values(arr))
        assert np.array_equal(arr, dec)

    def test_float_gorilla_roundtrip(self):
        arr = np.random.randn(1000).astype(np.float64)
        dec = decode_float_values(encode_float_values(arr))
        assert np.array_equal(arr, dec)

    def test_log1p_detection(self):
        ints = np.array([1,2,3,1,4,2,1,0,0,5], dtype=np.int32)
        floats = np.log1p(ints.astype(np.float64))
        assert _is_log1p_of(floats, ints)
        assert not _is_log1p_of(floats + 0.001, ints)

    def test_sqrt_detection(self):
        ints   = np.array([1,4,9,16,25], dtype=np.int32)
        floats = np.sqrt(ints.astype(np.float64))
        assert _is_sqrt_of(floats, ints)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Derived Layer Elimination
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedLayers:
    def test_log1p_stored_as_descriptor(self, tmp_sric):
        """Derived layer must not store float values — only a descriptor."""
        c, X = make_container()
        c.add_derived_layer("X_norm", "log1p", "X_raw")
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)

        # The file must contain a DLYR block, not a LAYR block for X_norm
        with open(tmp_sric, 'rb') as f: raw = f.read()
        assert b"DLYR" in raw, "Derived layer must use DLYR block"

        # Reload and verify exact reconstruction
        c2  = sric.SriCReader(tmp_sric).load()
        Xn  = np.log1p(X.toarray())
        Xn2 = c2.layers["X_norm"].toarray()
        assert np.array_equal(Xn, Xn2), "log1p not reconstructed exactly"

    def test_sqrt_derived(self, tmp_sric):
        c, X = make_container()
        c.add_derived_layer("X_sqrt", "sqrt", "X_raw")
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2  = sric.SriCReader(tmp_sric).load()
        Xs  = np.sqrt(X.toarray().astype(np.float64))
        Xs2 = c2.layers["X_sqrt"].toarray()
        assert np.array_equal(Xs, Xs2)

    def test_derived_file_smaller_than_float_stored(self, tmp_sric):
        """Derived layer file must be smaller than storing float64 directly."""
        import tempfile
        c1, X = make_container(n=300, m=800)
        c2_c, _ = make_container(n=300, m=800)

        c1.add_derived_layer("X_norm", "log1p", "X_raw")
        Xn = sp.csr_matrix(np.log1p(X.toarray()))
        c2_c.add_layer("X_norm_explicit", Xn)

        p1 = tmp_sric
        p2 = tmp_sric.replace(".sric", "_explicit.sric")
        sric.SriCWriter(c1, fit_zeromodel=False).write(p1)
        sric.SriCWriter(c2_c, fit_zeromodel=False).write(p2)

        assert os.path.getsize(p1) < os.path.getsize(p2), \
            "Derived layer file should be smaller than explicit float storage"
        os.unlink(p2)

    def test_source_not_found_raises(self):
        c, _ = make_container()
        with pytest.raises(KeyError):
            c.add_derived_layer("X_norm", "log1p", "NONEXISTENT_LAYER")

    def test_invalid_transform_raises(self):
        c, _ = make_container()
        with pytest.raises(ValueError):
            c.add_derived_layer("X_norm", "magic_normalize", "X_raw")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full round-trips
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_basic_integer_layer(self, tmp_sric):
        c, X = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(X.toarray(), c2.layers["X_raw"].toarray())

    def test_explicit_float_layer(self, tmp_sric):
        c, X = make_container()
        Xn = sp.csr_matrix(np.log1p(X.toarray()))
        c.add_layer("X_norm", Xn)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2  = sric.SriCReader(tmp_sric).load()
        n2  = c2.layers["X_norm"].toarray()
        assert np.array_equal(Xn.toarray(), n2)

    def test_obs_var_names_exact(self, tmp_sric):
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.all(c2.obs_names == c.obs_names)
        assert np.all(c2.var_names == c.var_names)

    def test_obs_var_names_not_in_json_header(self, tmp_sric):
        """Regression: names must be in OBSN/VARN blocks, NOT in JSON header."""
        c, _ = make_container(n=50, m=100)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        with open(tmp_sric, 'rb') as f: raw = f.read()
        hlen = struct.unpack('<I', raw[8:12])[0]
        hdr  = json.loads(zlib.decompress(raw[44:44+hlen]))
        assert "obs_names" not in hdr
        assert "var_names" not in hdr

    def test_string_metadata_roundtrip(self, tmp_sric):
        c, _ = make_container()
        ct = np.random.choice(["Neuron","Astrocyte","OPC"], c.n_obs)
        c.obs["cell_type"] = ct
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.all(c2.obs["cell_type"] == ct)

    def test_spatial_2d_exact(self, tmp_sric):
        c, _ = make_container(n=100)
        coords = np.random.uniform(-1000, 1000, (100, 2))
        c.set_spatial(coords)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(coords, c2.spatial)

    def test_spatial_3d_exact(self, tmp_sric):
        c, _ = make_container(n=100)
        coords = np.random.uniform(-5000, 5000, (100, 3))
        c.set_spatial(coords)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(coords, c2.spatial)

    def test_single_cell(self, tmp_sric):
        X = sp.random(1, 200, density=0.2, format='csr', dtype=np.int32)
        X.data[:] = 1
        c = sric.from_scipy_sparse(X)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(X.toarray(), c2.layers["X_raw"].toarray())

    def test_all_zero_matrix(self, tmp_sric):
        X = sp.csr_matrix((100, 200))
        c = sric.from_scipy_sparse(X)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(X.toarray(), c2.layers["X_raw"].toarray())

    def test_high_counts_bulk_rna(self, tmp_sric):
        np.random.seed(0)
        X = sp.csr_matrix(np.random.randint(0, 100_000, (50, 200)).astype(np.int32))
        c = sric.from_scipy_sparse(X)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(X.toarray(), c2.layers["X_raw"].toarray())

    def test_multi_modal_rna_atac(self, tmp_sric):
        np.random.seed(42)
        N = 80; M_rna = 200; M_atac = 150
        X_rna  = sp.random(N, M_rna,  density=0.1, format='csr', dtype=np.int32)
        X_rna.data[:] = 1
        X_atac = sp.random(N, M_atac, density=0.05, format='csr', dtype=np.int32)
        X_atac.data[:] = 1
        obs = np.array([f"c{i}" for i in range(N)])
        c = sric.SriCContainer(obs, np.array([f"G{i}" for i in range(M_rna)]))
        c.add_layer("X_raw", X_rna)
        atac = c.add_modality("ATAC", np.array([f"P{i}" for i in range(M_atac)]))
        atac.add_layer("counts", X_atac)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert np.array_equal(X_rna.toarray(), c2.layers["X_raw"].toarray())
        assert "ATAC" in c2.modalities
        assert np.array_equal(X_atac.toarray(),
                               c2.modalities["ATAC"].layers["counts"].toarray())
        assert c2.modalities["ATAC"].n_vars == M_atac

    def test_provenance_preserved(self, tmp_sric):
        c, _ = make_container()
        c.log_operation("filter_cells", min_genes=200)
        n_prov = len(c.provenance)
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert len(c2.provenance) == n_prov
        assert any(p.operation == "filter_cells" for p in c2.provenance)

    def test_provenance_entry_immutable(self):
        c, _ = make_container()
        entry = c.provenance[0]
        with pytest.raises((TypeError, AttributeError)):
            entry.operation = "HACKED"


# ─────────────────────────────────────────────────────────────────────────────
# 6. ZeroModel
# ─────────────────────────────────────────────────────────────────────────────

class TestCelltypeStats:
    """
    Tests for per-(gene, cell-type) summary statistics block.
    Stores observed statistics (zero fraction, mean expression) —
    NOT latent ZINB parameters, which are not identifiable from
    scRNA count data alone (Risso et al. 2018 Nat Commun).
    """
    def test_stats_nonnegative(self, tmp_sric):
        c, X = make_container(n=200)
        c.obs["cell_type"] = np.random.choice(["A","B","C"], 200)
        sric.SriCWriter(c, fit_zeromodel=True,
                        cell_type_col="cell_type").write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert c2.zero_model is not None
        zf = c2.zero_model['obs_zero_frac'].astype(np.float64)
        assert zf.min() >= 0.0
        assert zf.max() <= 1.0 + 1e-3

    def test_stats_shape(self, tmp_sric):
        c, _ = make_container(n=150, m=200)
        ct = np.random.choice(["Neuron","Astrocyte"], 150)
        c.obs["cell_type"] = ct
        sric.SriCWriter(c, fit_zeromodel=True,
                        cell_type_col="cell_type").write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        cts = c2.zero_model
        assert cts['obs_zero_frac'].shape  == (200, 2)
        assert cts['mean_expr'].shape      == (200, 2)
        assert cts['frac_expressing'].shape== (200, 2)

    def test_zero_frac_sum_to_one(self, tmp_sric):
        """obs_zero_frac + frac_expressing must sum to 1.0."""
        c, _ = make_container(n=200, m=300)
        c.obs["cell_type"] = np.random.choice(["A","B"], 200)
        sric.SriCWriter(c, fit_zeromodel=True,
                        cell_type_col="cell_type").write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        cts = c2.zero_model
        total = (cts['obs_zero_frac'].astype(np.float32) +
                 cts['frac_expressing'].astype(np.float32))
        assert np.allclose(total, 1.0, atol=0.01)  # float16 tolerance

    def test_no_stats_without_celltype(self, tmp_sric):
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=True).write(tmp_sric)
        c2 = sric.SriCReader(tmp_sric).load()
        assert c2.zero_model is None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrity:
    def test_checksum_catches_corruption(self, tmp_sric):
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        with open(tmp_sric, 'r+b') as f:
            sz = os.path.getsize(tmp_sric)
            f.seek(sz // 2); f.write(b'\xFF')
        with pytest.raises((ValueError, Exception)):
            sric.SriCReader(tmp_sric, verify_checksums=True).load()

    def test_bad_magic(self, tmp_sric):
        with open(tmp_sric, 'wb') as f:
            f.write(b"FAKE" + b"\x00"*200)
        with pytest.raises(ValueError, match="Not a .SriC"):
            sric.SriCReader(tmp_sric).load()

    def test_wrong_version(self, tmp_sric):
        with open(tmp_sric, 'wb') as f:
            f.write(b"SriC" + struct.pack('<BBB', 99, 0, 0) + b"\x00"*200)
        with pytest.raises(ValueError, match="Unsupported"):
            sric.SriCReader(tmp_sric).load()

    def test_truncated_file(self, tmp_sric):
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        sz = os.path.getsize(tmp_sric)
        with open(tmp_sric, 'r+b') as f: f.truncate(sz - 50)
        with pytest.raises((ValueError, Exception)):
            sric.SriCReader(tmp_sric).load()

    def test_atomic_write_leaves_original_intact(self, tmp_path):
        """If write fails midway, original file is unchanged."""
        orig = str(tmp_path / "orig.sric")
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(orig)
        orig_size = os.path.getsize(orig)
        import builtins
        real_open = builtins.open; calls = [0]
        def broken_open(path, mode='r', *a, **kw):
            if 'wb' in str(mode) and path != orig and calls[0] == 0:
                calls[0] += 1; raise IOError("Simulated disk full")
            return real_open(path, mode, *a, **kw)
        builtins.open = broken_open
        try:
            try: sric.SriCWriter(c, fit_zeromodel=False).write(orig)
            except IOError: pass
        finally:
            builtins.open = real_open
        assert os.path.getsize(orig) == orig_size


# ─────────────────────────────────────────────────────────────────────────────
# 8. Gene / cell queries
# ─────────────────────────────────────────────────────────────────────────────

class TestQueries:
    def test_gene_query_matches_full_load(self, tmp_sric):
        c, X = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        reader = sric.SriCReader(tmp_sric)
        c2     = reader.load()
        gene   = c.var_names[10]
        expr_q = reader.query_gene(gene, layer="X_raw")
        expr_f = c2.layers["X_raw"].toarray()[:, 10]
        assert np.array_equal(expr_q, expr_f)

    def test_cell_query_matches_full_load(self, tmp_sric):
        c, X = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        reader = sric.SriCReader(tmp_sric)
        c2     = reader.load()
        bc     = c.obs_names[5]
        cell_q = reader.query_cell(bc, layer="X_raw")
        cell_f = c2.layers["X_raw"].toarray()[5, :]
        assert np.array_equal(cell_q, cell_f)

    def test_missing_gene_raises(self, tmp_sric):
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        with pytest.raises(KeyError):
            sric.SriCReader(tmp_sric).query_gene("NONEXISTENT_XYZ")

    def test_missing_cell_raises(self, tmp_sric):
        c, _ = make_container()
        sric.SriCWriter(c, fit_zeromodel=False).write(tmp_sric)
        with pytest.raises(KeyError):
            sric.SriCReader(tmp_sric).query_cell("NONEXISTENT_BARCODE")


if __name__ == "__main__":
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", __file__,
                        "-v", "--tb=short"], capture_output=False)
    sys.exit(r.returncode)
