"""
download_dataset.py — Automated Downloader and Extractor for DCID Dataset
Downloads DCID.zip directly from Hugging Face Hub (3.8 GB) and extracts to DCID/DCID.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/168sir/drill-core-image-dataset/resolve/main/DCID.zip"
EXPECTED_SIZE = 3819926174  # ~3.56 GB compressed

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {mb_down:.1f} MB / {mb_total:.1f} MB [{percent:.1f}%]")
    else:
        sys.stdout.write(f"\rDownloading: {downloaded / (1024 * 1024):.1f} MB")
    sys.stdout.flush()

def main():
    root = Path(__file__).resolve().parent
    dest_zip = root / "DCID.zip"
    dest_dir = root / "DCID"
    final_root = dest_dir / "DCID"

    if (final_root / "DCID-512-7").is_dir() and (final_root / "DCID-512-35").is_dir():
        print(f"[OK] DCID dataset is already extracted and present at:\n  {final_root}")
        print("Everything is ready to proceed!")
        return

    if not dest_zip.exists() or dest_zip.stat().st_size < EXPECTED_SIZE:
        print("=" * 70)
        print("  Downloading Drill Core Image Dataset (DCID) [3.8 GB]")
        print(f"  Source: {DATASET_URL}")
        print("=" * 70)
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(dest_zip, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", EXPECTED_SIZE))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                percent = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB [{percent:.1f}%]")
                sys.stdout.flush()
        print("\n[OK] Download completed.")

    print(f"\nExtracting {dest_zip.name} to {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(dest_dir)
    print("[OK] Extraction completed successfully!")
    print(f"DCID Root is located at: {final_root}")

if __name__ == "__main__":
    main()
