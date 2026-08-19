"""
cli.py — Resumable CLI for .SriC
Downloads and converts massive scRNA-seq datasets with crash-recovery.
"""

import os
import sys
import argparse
import tempfile
import time
import requests
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import anndata as ad
import sric

def download_resumable(url: str, dest_path: str, chunk_size: int = 8 * 1024 * 1024):
    """Streams a remote file to disk, resuming automatically if broken."""
    print(f"[*] Target URL: {url}")
    
    headers = {}
    mode = 'wb'
    existing_size = 0
    
    # Check if a partial file already exists to resume
    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        headers['Range'] = f'bytes={existing_size}-'
        mode = 'ab'
        print(f"[*] Partial file found. Resuming from {existing_size / (1024**2):.2f} MB...")

    try:
        with requests.get(url, stream=True, headers=headers) as r:
            # If the server doesn't support partial ranges, it returns 200 instead of 206
            if r.status_code == 200 and existing_size > 0:
                print("[!] Server does not support resume. Restarting download...")
                mode = 'wb'
                existing_size = 0
            elif r.status_code not in (200, 206):
                r.raise_for_status()

            # Calculate total size
            content_length = int(r.headers.get("content-length", 0))
            total_size = content_length + existing_size if r.status_code == 206 else content_length
            
            progress = tqdm(
                initial=existing_size,
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc="Downloading",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            with open(dest_path, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
                        
            progress.close()
    except requests.exceptions.RequestException as e:
        print(f"\n[!] Network error: {e}")
        print("[!] Download interrupted. Re-run the exact same command to resume.")
        sys.exit(1)

def convert_h5ad_to_sric(h5ad_path: str, output_sric_path: str):
    """Out-of-core conversion pipeline for standard .h5ad files."""
    print(f"\n[*] Initializing out-of-core conversion to .SriC...")
    t0 = time.perf_counter()
    
    adata = ad.read_h5ad(h5ad_path, backed="r")
    n_cells, n_genes = adata.shape
    print(f"    Dimensions: {n_cells:,} cells × {n_genes:,} genes")
    
    c = sric.SriCContainer(np.array(adata.obs_names), np.array(adata.var_names))
    X = adata.X
    dtype = X.dtype
    is_integer = np.issubdtype(dtype, np.integer)
    layer_key = "X_raw" if is_integer else "X_norm"
    
    is_backed = not sp.issparse(X) and (hasattr(X, "unbacked") or "h5py" in str(type(X)) or "_CSRDataset" in str(type(X)))
    
    if is_backed:
        h5_data = X._data if hasattr(X, "_data") else X.data
        h5_indices = X._indices if hasattr(X, "_indices") else X.indices
        h5_indptr = X._indptr if hasattr(X, "_indptr") else X.indptr
        
        nnz = h5_data.shape[0]
        fd, fi, fp = tempfile.TemporaryFile(), tempfile.TemporaryFile(), tempfile.TemporaryFile()
        mm_data = np.memmap(fd, dtype=dtype, mode="w+", shape=(nnz,))
        mm_indices = np.memmap(fi, dtype=np.int32, mode="w+", shape=(nnz,))
        mm_indptr = np.memmap(fp, dtype=np.int32, mode="w+", shape=(n_cells + 1,))
        
        CHUNK = 50_000_000
        for i in range(0, nnz, CHUNK):
            mm_data[i:i+CHUNK] = h5_data[i:i+CHUNK]
            mm_indices[i:i+CHUNK] = h5_indices[i:i+CHUNK].astype(np.int32)
        for i in range(0, n_cells + 1, CHUNK):
            mm_indptr[i:i+CHUNK] = h5_indptr[i:i+CHUNK].astype(np.int32)
            
        X_matrix = sp.csr_matrix((mm_data, mm_indices, mm_indptr), shape=(n_cells, n_genes))
        c.add_layer(layer_key, X_matrix)
    else:
        c.add_layer(layer_key, X)
        
    for col in adata.obs.columns:
        c.obs[col] = np.array(adata.obs[col])
    for col in adata.var.columns:
        c.var[col] = np.array(adata.var[col])
    c.uns = dict(adata.uns)
    
    if is_integer and "X_raw" in c.layers:
        c.add_derived_layer("X_norm", "log1p", "X_raw")
        
    ct_col = "cell_type" if "cell_type" in adata.obs.columns else None
    sric.SriCWriter(c, fit_zeromodel=(ct_col is not None and is_integer), cell_type_col=ct_col or "cell_type").write(output_sric_path)
    
    t_total = time.perf_counter() - t0
    sric_size = os.path.getsize(output_sric_path) / (1024**2)
    print(f"[✓] Successfully generated: {output_sric_path} ({sric_size:.2f} MB) in {t_total:.2f}s")

def handle_fetch(args):
    url = args.url
    out_name = args.output
    
    raw_filename = url.split("?")[0].split("/")[-1]
    if not out_name:
        out_name = f"{os.path.splitext(raw_filename)[0]}.sric"
    elif not out_name.endswith(".sric"):
        out_name = f"{out_name}.sric"
        
    if raw_filename.endswith(".sric"):
        download_resumable(url, out_name)
    else:
        tmp_source_path = f"{out_name}.tmp.h5ad"
        try:
            download_resumable(url, tmp_source_path)
            convert_h5ad_to_sric(tmp_source_path, out_name)
        finally:
            if os.path.exists(tmp_source_path):
                os.remove(tmp_source_path)

def main():
    parser = argparse.ArgumentParser(prog="sric", description="SriC CLI: Resumable download and fast scRNA-seq converter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch", help="Download and convert dataset URL to .SriC")
    fetch_parser.add_argument("url", type=str, help="Direct URL to dataset")
    fetch_parser.add_argument("-o", "--output", type=str, default=None, help="Output .sric filename")
    
    args = parser.parse_args()
    if args.command == "fetch":
        handle_fetch(args)

if __name__ == "__main__":
    main()