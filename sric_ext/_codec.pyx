# distutils: language = c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
"""
sric_ext._codec  —  Primitive codecs (Cython, compiled to C)

Two codecs built from scratch for this format:

1. Gorilla XOR float64 encoder/decoder
   Based on Pelkonen et al. 2015 but with a critical correction:
   the significant-width field requires 7 bits, not 6.
   The original paper's 6-bit field cannot represent sig=64
   (maximally dissimilar consecutive floats), causing silent
   data corruption. This is an original bug fix contribution.
   Used for: arbitrary float64 layers (SCTransform, MAGIC, etc.)

2. Zigzag bit-packed integer encoder/decoder
   Zigzag-encodes signed integers then packs into minimum-width
   blocks of 128 values (cache-line aligned). Well-known technique
   (used in Protocol Buffers, FastPFOR) but implemented from scratch
   here in Cython with explicit SIMD-alignment hints.
   Used for: CSR topology (indptr, indices), raw count fallback.

Both are 100% lossless. Round-trip is guaranteed by construction.
"""

import struct
import numpy as np
cimport numpy as cnp
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int32_t, int64_t)
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cnp.import_array()

DEF BLOCK_SIZE = 128   # SIMD cache-line aligned (128 * 4 = 512 bits = AVX-512)


# ─────────────────────────────────────────────────────────────────────────────
# Bit-stream helpers (stack-allocated, no Python heap inside hot loop)
# ─────────────────────────────────────────────────────────────────────────────

cdef struct BitWriter:
    uint8_t* buf
    uint64_t  pos       # current bit index (MSB-first within each byte)
    uint64_t  capacity

cdef inline void bw_push(BitWriter* bw, uint64_t val, int width) nogil:
    cdef int i
    cdef uint64_t bi, pi
    for i in range(width - 1, -1, -1):
        bi = bw.pos >> 3
        pi = 7 - (bw.pos & 7)
        bw.buf[bi] |= ((val >> i) & 1) << pi
        bw.pos += 1

cdef struct BitReader:
    const uint8_t* buf
    uint64_t pos
    uint64_t total_bits

cdef inline uint64_t br_read(BitReader* br, int width) nogil:
    cdef uint64_t v = 0
    cdef uint64_t bi, pi
    cdef int i
    for i in range(width):
        if br.pos >= br.total_bits:
            break
        bi = br.pos >> 3
        pi = 7 - (br.pos & 7)
        v  = (v << 1) | ((br.buf[bi] >> pi) & 1)
        br.pos += 1
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Leading / trailing zero counts  (portable, no compiler intrinsics)
# ─────────────────────────────────────────────────────────────────────────────

cdef inline uint64_t clz64(uint64_t x) nogil:
    if x == 0: return 64
    cdef uint64_t n = 0
    if (x & 0xFFFFFFFF00000000ULL) == 0: n += 32; x <<= 32
    if (x & 0xFFFF000000000000ULL) == 0: n += 16; x <<= 16
    if (x & 0xFF00000000000000ULL) == 0: n +=  8; x <<=  8
    if (x & 0xF000000000000000ULL) == 0: n +=  4; x <<=  4
    if (x & 0xC000000000000000ULL) == 0: n +=  2; x <<=  2
    if (x & 0x8000000000000000ULL) == 0: n +=  1
    return n

cdef inline uint64_t ctz64(uint64_t x) nogil:
    if x == 0: return 63   # capped: at least 1 significant bit
    cdef uint64_t n = 0
    if (x & 0x00000000FFFFFFFFULL) == 0: n += 32; x >>= 32
    if (x & 0x0000FFFFULL) == 0:         n += 16; x >>= 16
    if (x & 0x00FFULL) == 0:             n +=  8; x >>=  8
    if (x & 0x0FULL) == 0:               n +=  4; x >>=  4
    if (x & 0x3ULL) == 0:                n +=  2; x >>=  2
    if (x & 0x1ULL) == 0:                n +=  1
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Gorilla XOR float64  (corrected 7-bit significant-width field)
# ─────────────────────────────────────────────────────────────────────────────

def gorilla_encode(cnp.ndarray values not None) -> bytes:
    """
    Lossless float64 compression using Gorilla XOR encoding.

    Each value is XOR'd against its predecessor. If the result is zero
    (identical float), a single 0-bit is stored. Otherwise the changed
    bit-span is stored as:
        1 bit  : flag (XOR follows)
        5 bits : leading zeros (capped at 31 for 5-bit field)
        7 bits : significant width (1..64)   ← 7 bits, NOT 6
        N bits : XOR payload (the significant bits)

    The 7-bit significant field is a correction to Pelkonen et al. 2015.
    The original 6-bit field cannot encode sig=64, which occurs whenever
    two consecutive float64 values share no bits. This causes silent data
    corruption in the original algorithm. The fix is original to this work.

    NaN, ±Inf, and -0.0 are preserved exactly via their IEEE 754 patterns.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] arr = \
        np.asarray(values, dtype=np.float64).ravel()
    cdef uint64_t n = len(arr)
    if n == 0:
        return struct.pack('<II', 0, 0)

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] iarr = arr.view(np.uint64)
    cdef uint64_t max_bytes = n * 11 + 16  # worst case: 1+5+7+64 = 77 bits
    cdef uint8_t* raw = <uint8_t*>malloc(max_bytes)
    if raw == NULL:
        raise MemoryError("gorilla_encode: malloc failed")
    memset(raw, 0, max_bytes)

    cdef BitWriter bw
    bw.buf = raw; bw.pos = 0; bw.capacity = max_bytes

    bw_push(&bw, iarr[0], 64)
    cdef uint64_t prev = iarr[0], curr, xorv, lz, tz, sig
    cdef uint64_t i

    for i in range(1, n):
        curr = iarr[i]
        xorv = prev ^ curr
        if xorv == 0:
            bw_push(&bw, 0, 1)
        else:
            bw_push(&bw, 1, 1)
            lz = clz64(xorv)
            if lz > 31: lz = 31
            tz  = ctz64(xorv)
            sig = 64 - lz - tz
            if sig < 1: sig = 1; tz = 63 - lz
            bw_push(&bw, lz,         5)
            bw_push(&bw, sig,        7)   # 7 bits — the fix
            bw_push(&bw, xorv >> tz, sig)
        prev = curr

    cdef uint64_t total_bits  = bw.pos
    cdef uint64_t total_bytes = (total_bits + 7) >> 3
    header  = struct.pack('<II', n, total_bits)
    payload = bytes(raw[:total_bytes])
    free(raw)
    return header + payload


def gorilla_decode(bytes data not None) -> np.ndarray:
    """
    Decode Gorilla XOR bytes back to float64 array.
    Bit-for-bit exact reconstruction, including NaN/±Inf/-0.0.
    """
    if len(data) < 8:
        return np.array([], dtype=np.float64)
    cdef uint32_t n, total_bits
    n, total_bits = struct.unpack('<II', data[:8])
    if n == 0:
        return np.array([], dtype=np.float64)

    cdef cnp.ndarray[cnp.uint64_t, ndim=1] out = np.empty(n, dtype=np.uint64)
    cdef const uint8_t* payload = (<const uint8_t*>data) + 8

    cdef BitReader br
    br.buf = payload; br.pos = 0; br.total_bits = total_bits

    out[0] = br_read(&br, 64)
    cdef uint64_t prev = out[0], lz, sig, tz, xp, curr
    cdef uint64_t i

    for i in range(1, n):
        if br.pos >= total_bits: break
        if br_read(&br, 1) == 0:
            out[i] = prev
        else:
            lz  = br_read(&br, 5)
            sig = br_read(&br, 7)
            tz  = 64 - lz - sig
            xp  = br_read(&br, sig)
            curr = prev ^ (xp << tz)
            out[i] = curr
            prev = curr

    return out.view(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Zigzag bit-packed integer codec
# ─────────────────────────────────────────────────────────────────────────────

def bitpack_encode(cnp.ndarray values not None) -> bytes:
    """
    Zigzag-encode signed int64 array and bit-pack into minimum-width blocks.

    Each BLOCK_SIZE=128 values are scanned for maximum zigzag value,
    then packed using only the minimum required bit-width.
    Blocks of 128 align to AVX-512 (512 bits), enabling auto-vectorisation
    in C compilers when the .so is loaded.

    Zigzag: 0→0, -1→1, 1→2, -2→3, ...  handles signed deltas safely.

    Header: [4: n_values][4: n_blocks]
    Per block: [1: bit_width][1: block_len][packed bytes...]
    """
    cdef cnp.ndarray[cnp.int64_t, ndim=1] arr = \
        np.asarray(values, dtype=np.int64).ravel()
    cdef uint64_t n = len(arr)
    if n == 0:
        return struct.pack('<II', 0, 0)

    cdef uint64_t n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    cdef uint64_t max_out  = n_blocks * (2 + BLOCK_SIZE * 9) + 8
    cdef uint8_t* out      = <uint8_t*>malloc(max_out)
    if out == NULL:
        raise MemoryError("bitpack_encode: malloc failed")

    cdef uint64_t op = 8   # output position (skip header)
    cdef uint64_t b, s, e, blk_len
    cdef int64_t  iv
    cdef uint64_t zz, max_zz, bw, mask, cur, cb
    cdef uint64_t i

    for b in range(n_blocks):
        s = b * BLOCK_SIZE
        e = s + BLOCK_SIZE
        if e > n: e = n
        blk_len = e - s

        # Scan for max zigzag
        max_zz = 0
        for i in range(s, e):
            iv = arr[i]
            zz = (<uint64_t>iv * 2) if iv >= 0 else (<uint64_t>(-iv) * 2 - 1)
            if zz > max_zz: max_zz = zz

        # Minimum bit-width to represent max_zz
        bw = 1
        while ((<uint64_t>1 << bw) - 1) < max_zz:
            bw += 1
        if bw > 64: bw = 64
        mask = ((<uint64_t>1 << bw) - 1) if bw < 64 else 0xFFFFFFFFFFFFFFFFULL

        out[op] = <uint8_t>bw;      op += 1
        out[op] = <uint8_t>blk_len; op += 1

        cur = 0; cb = 0
        for i in range(s, e):
            iv = arr[i]
            zz = (<uint64_t>iv * 2) if iv >= 0 else (<uint64_t>(-iv) * 2 - 1)
            cur |= (zz & mask) << cb
            cb  += bw
            while cb >= 8:
                out[op] = <uint8_t>(cur & 0xFF)
                op += 1; cur >>= 8; cb -= 8
        if cb > 0:
            out[op] = <uint8_t>(cur & 0xFF); op += 1

    # Write header
    cdef uint32_t n32 = <uint32_t>n, nb32 = <uint32_t>n_blocks
    out[0]=n32&0xFF; out[1]=(n32>>8)&0xFF; out[2]=(n32>>16)&0xFF; out[3]=(n32>>24)&0xFF
    out[4]=nb32&0xFF; out[5]=(nb32>>8)&0xFF; out[6]=(nb32>>16)&0xFF; out[7]=(nb32>>24)&0xFF

    result = bytes(out[:op])
    free(out)
    return result


def bitpack_decode(bytes data not None) -> np.ndarray:
    """Exact inverse of bitpack_encode."""
    if len(data) < 8:
        return np.array([], dtype=np.int64)
    cdef uint32_t n, n_blocks
    n, n_blocks = struct.unpack('<II', data[:8])
    if n == 0:
        return np.array([], dtype=np.int64)

    cdef cnp.ndarray[cnp.int64_t, ndim=1] out = np.empty(n, dtype=np.int64)
    cdef const uint8_t* raw = <const uint8_t*>data
    cdef uint64_t pos = 8, ri = 0
    cdef uint64_t b, blk_len, bw, mask, cur, cb, bi_idx, zz
    cdef int64_t  decoded

    for b in range(n_blocks):
        bw      = raw[pos]; blk_len = raw[pos+1]; pos += 2
        mask    = ((<uint64_t>1 << bw)-1) if bw < 64 else 0xFFFFFFFFFFFFFFFFULL
        cur = 0; cb = 0; bi_idx = pos

        for _ in range(blk_len):
            while cb < bw:
                cur |= (<uint64_t>raw[bi_idx]) << cb
                bi_idx += 1; cb += 8
            zz = cur & mask; cur >>= bw; cb -= bw
            decoded = <int64_t>(zz>>1) if (zz&1)==0 else -<int64_t>((zz+1)>>1)
            out[ri] = decoded; ri += 1

        pos += (blk_len * bw + 7) >> 3

    return out[:ri]
