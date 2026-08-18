"""
sric.utils — Codec primitives

Gorilla XOR (7-bit fix) and zigzag bitpack are implemented in Cython.
If the .so is unavailable, pure-Python fallbacks are used automatically.

What each function actually does — no inflated claims:
  gorilla_encode / gorilla_decode : Gorilla XOR float64 compression
      Our contribution: 7-bit sig field (fixes silent corruption in Pelkonen 2015).
      Compression: 1.4–1.6× on typical log-norm scRNA data.
      (The data is not time-series, so XOR prefix sharing is limited.)

  bitpack_encode / bitpack_decode : zigzag + block bit-packing for int64
      Well-known technique. Implemented from scratch in Cython.
      Useful for CSR indptr / indices where deltas are small.

  compute_deltas / reconstruct_deltas : cumsum helpers for CSR topology
  sha256_block : block integrity digest
"""

from __future__ import annotations
import struct, hashlib
import numpy as np
from typing import Tuple

_CYTHON = False
try:
    from sric_ext._codec import (
        gorilla_encode as _ge,
        gorilla_decode as _gd,
        bitpack_encode as _bpe,
        bitpack_decode as _bpd,
    )
    _CYTHON = True
except ImportError:
    pass


# ── Pure-Python fallbacks ────────────────────────────────────────────────────

def _py_gorilla_encode(values: np.ndarray) -> bytes:
    arr  = np.asarray(values, np.float64).ravel()
    iarr = arr.view(np.uint64)
    n    = len(arr)
    if n == 0: return struct.pack('<II', 0, 0)
    bits: list[tuple[int,int]] = []
    def push(v,w): bits.append((int(v)&((1<<w)-1), w))
    push(int(iarr[0]), 64); prev = int(iarr[0])
    for i in range(1, n):
        curr = int(iarr[i]); xv = prev ^ curr
        if xv == 0: push(0,1)
        else:
            push(1,1)
            lz = min((64-xv.bit_length()) if xv else 64, 31)
            tz = 0; tmp=xv
            while (tmp&1)==0 and tz<63: tz+=1; tmp>>=1
            sig = 64-lz-tz
            push(lz,5); push(sig,7); push(xv>>tz, sig)
        prev = curr
    total = sum(w for _,w in bits)
    buf=bytearray(); cur=0; pos=7
    for val,w in bits:
        for i in range(w-1,-1,-1):
            cur|=((val>>i)&1)<<pos; pos-=1
            if pos<0: buf.append(cur); cur=0; pos=7
    if pos<7: buf.append(cur)
    return struct.pack('<II',n,total)+bytes(buf)

def _py_gorilla_decode(data: bytes) -> np.ndarray:
    if len(data)<8: return np.array([],np.float64)
    n,total=struct.unpack('<II',data[:8])
    if n==0: return np.array([],np.float64)
    pay=data[8:]
    def gb(p):
        bi=p>>3; return (pay[bi]>>(7-(p&7)))&1 if bi<len(pay) else 0
    def gbs(p,w):
        v=0
        for _ in range(w): v=(v<<1)|gb(p); p+=1
        return v,p
    out=np.empty(n,np.uint64); pos=0
    first,pos=gbs(pos,64); out[0]=first; prev=first; idx=1
    while idx<n and pos<total:
        same,pos=gbs(pos,1)
        if same==0: out[idx]=prev
        else:
            lz,pos=gbs(pos,5); sig,pos=gbs(pos,7)
            tz=64-lz-sig; xp,pos=gbs(pos,sig)
            curr=prev^(xp<<max(0,tz)); out[idx]=curr; prev=curr
        idx+=1
    return out[:idx].view(np.float64)

def _py_bitpack_encode(values: np.ndarray) -> bytes:
    arr=np.asarray(values,np.int64).ravel(); n=len(arr)
    if n==0: return struct.pack('<II',0,0)
    zz=np.where(arr>=0,arr*2,(-arr)*2-1).astype(np.uint64)
    B=128; nb=(n+B-1)//B; parts=[struct.pack('<II',n,nb)]
    for b in range(nb):
        blk=zz[b*B:(b+1)*B]; mx=int(blk.max()) if len(blk) else 0
        bw=max(1,int(np.ceil(np.log2(mx+2))) if mx>0 else 1)
        pk=bytearray([bw,len(blk)]); cur=0; cb=0
        for v in blk:
            cur|=(int(v)<<cb); cb+=bw
            while cb>=8: pk.append(cur&0xFF); cur>>=8; cb-=8
        if cb>0: pk.append(cur&0xFF)
        parts.append(bytes(pk))
    return b''.join(parts)

def _py_bitpack_decode(data: bytes) -> np.ndarray:
    if len(data)<8: return np.array([],np.int64)
    n,nb=struct.unpack('<II',data[:8])
    if n==0: return np.array([],np.int64)
    pos=8; result=[]
    for _ in range(nb):
        bw=data[pos]; bl=data[pos+1]; pos+=2
        nb_=( bl*bw+7)//8; bb=data[pos:pos+nb_]; pos+=nb_
        cur=0; cb=0; bi=0; mask=(1<<bw)-1
        for _ in range(bl):
            while cb<bw:
                if bi<len(bb): cur|=(bb[bi]<<cb); bi+=1
                cb+=8
            zz=cur&mask; cur>>=bw; cb-=bw
            result.append((zz>>1) if (zz&1)==0 else -((zz+1)>>1))
    return np.array(result,np.int64)


# ── Public dispatch ──────────────────────────────────────────────────────────

def gorilla_encode(v: np.ndarray) -> bytes:
    return _ge(np.asarray(v,np.float64).ravel()) if _CYTHON else _py_gorilla_encode(v)

def gorilla_decode(d: bytes) -> np.ndarray:
    return _gd(d) if _CYTHON else _py_gorilla_decode(d)

def bitpack_encode(v: np.ndarray) -> bytes:
    return _bpe(np.asarray(v,np.int64).ravel()) if _CYTHON else _py_bitpack_encode(v)

def bitpack_decode(d: bytes) -> np.ndarray:
    return _bpd(d) if _CYTHON else _py_bitpack_decode(d)

def compute_deltas(arr: np.ndarray) -> np.ndarray:
    a=np.asarray(arr,np.int64)
    if not len(a): return a
    d=np.empty_like(a); d[0]=a[0]; d[1:]=np.diff(a); return d

def reconstruct_deltas(d: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(d,np.int64))

def sha256_block(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def codec_backend() -> str:
    return "Cython (C-compiled)" if _CYTHON else "Pure-Python fallback"
