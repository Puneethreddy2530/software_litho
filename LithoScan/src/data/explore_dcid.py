"""
explore_dcid.py — Dataset reconnaissance script for LithoScan.

Performs:
  1.  Auto-discovery of the four DCID directories.
  2.  Verification that all expected directories are present.
  3.  Recursive structure inspection.
  4.  Identification of all lithology classes.
  5.  Per-class image count (total, train, test).
  6.  Image dimension and format reporting.
  7.  Corrupt / unreadable image detection.
  8.  Duplicate filename detection within each split.
  9.  Cross-split duplicate detection via MD5 hash.
  10. Manifest generation  →  data/manifests/dcid_manifest.csv
  11. Summary report       →  reports/dataset_report.md
  12. Terminal summary.

Usage (from project root):
  python src/data/explore_dcid.py

Environment variable override for DCID root:
  DCID_ROOT=<path>  python src/data/explore_dcid.py
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("litho.explore")

# ---------------------------------------------------------------------------
# Configurable paths (override via env vars or edit here)
# ---------------------------------------------------------------------------
# Root of the extracted DCID archive
_DEFAULT_DCID_ROOT = Path(
    os.environ.get(
        "DCID_ROOT",
        r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID",
    )
)
# Project root (LithoScan/)
_DEFAULT_PROJECT_ROOT = Path(
    os.environ.get(
        "LITHO_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],   # src/data/ -> src/ -> LithoScan/
    )
)

EXPECTED_VERSIONS = [
    "DCID-512-7",
    "DCID-512-35",
    "noise-512-7",
    "noise-512-35",
]
SPLITS = ["train", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class ImageRecord:
    """Holds metadata for a single image file."""
    __slots__ = (
        "filepath", "dataset_version", "class_name", "class_id",
        "split", "width", "height", "channels", "file_size",
        "md5", "is_corrupt",
    )

    def __init__(
        self,
        filepath: str,
        dataset_version: str,
        class_name: str,
        class_id: int,
        split: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        channels: Optional[int] = None,
        file_size: int = 0,
        md5: str = "",
        is_corrupt: bool = False,
    ) -> None:
        self.filepath = filepath
        self.dataset_version = dataset_version
        self.class_name = class_name
        self.class_id = class_id
        self.split = split
        self.width = width
        self.height = height
        self.channels = channels
        self.file_size = file_size
        self.md5 = md5
        self.is_corrupt = is_corrupt

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


# ---------------------------------------------------------------------------
# Step 1 — Verify directory structure
# ---------------------------------------------------------------------------

def verify_structure(dcid_root: Path) -> List[str]:
    """
    Check that all expected version directories exist.

    Returns
    -------
    List[str]
        Names of any missing directories.
    """
    logger.info("Verifying DCID directory structure in: %s", dcid_root)
    missing = []
    for version in EXPECTED_VERSIONS:
        version_dir = dcid_root / version
        if not version_dir.is_dir():
            missing.append(version)
            logger.warning("MISSING: %s", version_dir)
        else:
            for split in SPLITS:
                split_dir = version_dir / split
                status = "OK" if split_dir.is_dir() else "MISSING split"
                logger.info("  %-25s %s/ — %s", version, split, status)
    return missing


# ---------------------------------------------------------------------------
# Step 2 — Collect classes
# ---------------------------------------------------------------------------

def _sort_key(name: str) -> int:
    try:
        return int(name.split(".")[0])
    except ValueError:
        return 9999


def collect_classes(dcid_root: Path, version: str) -> List[str]:
    """Return sorted class names for a dataset version (reads from train/)."""
    train_dir = dcid_root / version / "train"
    if not train_dir.is_dir():
        return []
    classes = sorted(
        [d.name for d in train_dir.iterdir() if d.is_dir()],
        key=_sort_key,
    )
    return classes


# ---------------------------------------------------------------------------
# Step 3 — MD5 helper
# ---------------------------------------------------------------------------

def _md5(filepath: str, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Step 4 — Build records for a single version + split
# ---------------------------------------------------------------------------

def scan_split(
    dcid_root: Path,
    version: str,
    split: str,
    classes: List[str],
    class_to_idx: Dict[str, int],
) -> List[ImageRecord]:
    """
    Walk a single version/split and return one ImageRecord per image file.
    """
    split_dir = dcid_root / version / split
    if not split_dir.is_dir():
        logger.warning("Split directory missing: %s", split_dir)
        return []

    records: List[ImageRecord] = []
    for cls_name in classes:
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            logger.warning("Class directory missing: %s", cls_dir)
            continue
        cls_id = class_to_idx[cls_name]

        img_files = sorted(
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for img_path in tqdm(img_files, desc=f"{version}/{split}/{cls_name}", leave=False, unit="img"):
            filepath = str(img_path.resolve())
            file_size = img_path.stat().st_size
            # Image metadata
            width = height = channels = None
            is_corrupt = False
            try:
                with Image.open(filepath) as im:
                    width, height = im.size
                    channels = len(im.getbands())
                    im.verify()  # detect truncated / corrupt files
            except (OSError, UnidentifiedImageError, Exception) as exc:
                logger.debug("Corrupt: %s — %s", filepath, exc)
                is_corrupt = True
                # Re-open to get dimensions even if verify fails
                try:
                    with Image.open(filepath) as im2:
                        width, height = im2.size
                        channels = len(im2.getbands())
                except Exception:
                    pass

            md5_val = _md5(filepath)
            records.append(ImageRecord(
                filepath=filepath,
                dataset_version=version,
                class_name=cls_name,
                class_id=cls_id,
                split=split,
                width=width,
                height=height,
                channels=channels,
                file_size=file_size,
                md5=md5_val,
                is_corrupt=is_corrupt,
            ))

    return records


# ---------------------------------------------------------------------------
# Step 5 — Duplicate checks
# ---------------------------------------------------------------------------

def find_filename_duplicates(records: List[ImageRecord]) -> Dict[str, List[str]]:
    """
    Detect duplicate filenames (basename only) within a list of records.

    Returns dict of {filename: [filepath1, filepath2, ...]} for duplicates.
    """
    name_map: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        name_map[Path(rec.filepath).name].append(rec.filepath)
    return {k: v for k, v in name_map.items() if len(v) > 1}


def find_hash_duplicates(
    train_records: List[ImageRecord],
    test_records: List[ImageRecord],
) -> List[Tuple[str, str]]:
    """
    Find images with the same MD5 hash appearing in both train and test sets.

    Returns list of (train_filepath, test_filepath) pairs.
    """
    train_hash_map: Dict[str, str] = {
        rec.md5: rec.filepath for rec in train_records if rec.md5
    }
    duplicates = []
    for rec in test_records:
        if rec.md5 and rec.md5 in train_hash_map:
            duplicates.append((train_hash_map[rec.md5], rec.filepath))
    return duplicates


# ---------------------------------------------------------------------------
# Step 6 — Write manifest CSV
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS = [
    "filepath", "dataset_version", "class_name", "class_id",
    "split", "width", "height", "channels", "file_size", "md5", "is_corrupt",
]


def write_manifest(records: List[ImageRecord], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())
    logger.info("Manifest written → %s  (%d rows)", manifest_path, len(records))


# ---------------------------------------------------------------------------
# Step 7 — Generate markdown report
# ---------------------------------------------------------------------------

def write_report(
    all_records: List[ImageRecord],
    missing_dirs: List[str],
    filename_dups: Dict[str, Dict[str, List[str]]],
    hash_dups: Dict[str, List[Tuple[str, str]]],
    report_path: Path,
    elapsed: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    corrupt = [r for r in all_records if r.is_corrupt]

    # Count per version per split
    version_split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        version_split_counts[r.dataset_version][r.split] += 1

    # Class counts per version per split
    class_counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for r in all_records:
        class_counts[r.dataset_version][r.split][r.class_name] += 1

    lines = [
        "# LithoScan — DCID Dataset Reconnaissance Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Elapsed: {elapsed:.1f}s",
        "",
        "> **License**: DCID is released under [CC BY-NC 4.0]"
        "(https://creativecommons.org/licenses/by-nc/4.0/).  "
        "Academic / non-commercial use only.",
        "",
        "---",
        "",
        "## 1  Structure Verification",
        "",
    ]

    if missing_dirs:
        lines.append(f"⚠️  **Missing directories**: {missing_dirs}")
    else:
        lines.append("✅  All four expected DCID directories found.")

    lines += [
        "",
        "| Directory | train | test | Total |",
        "|-----------|------:|-----:|------:|",
    ]
    for version in EXPECTED_VERSIONS:
        tr = version_split_counts[version].get("train", 0)
        te = version_split_counts[version].get("test", 0)
        lines.append(f"| `{version}` | {tr:,} | {te:,} | {tr+te:,} |")

    # Per-class tables
    lines += ["", "---", "", "## 2  Class Distributions", ""]
    for version in EXPECTED_VERSIONS:
        num_classes = len(class_counts[version].get("train", {}))
        lines.append(f"### {version}  ({num_classes} classes)")
        lines.append("")
        lines.append("| Class | Train | Test | Total |")
        lines.append("|-------|------:|-----:|------:|")
        all_classes = sorted(
            class_counts[version].get("train", {}).keys(),
            key=_sort_key,
        )
        for cls in all_classes:
            tr = class_counts[version]["train"].get(cls, 0)
            te = class_counts[version]["test"].get(cls, 0)
            lines.append(f"| {cls} | {tr:,} | {te:,} | {tr+te:,} |")
        lines.append("")

    # Image dimensions
    dims = set(
        (r.width, r.height, r.channels)
        for r in all_records
        if not r.is_corrupt and r.width
    )
    lines += [
        "---",
        "",
        "## 3  Image Dimensions",
        "",
        "| Width | Height | Channels |",
        "|------:|-------:|---------:|",
    ]
    for w, h, c in sorted(dims):
        lines.append(f"| {w} | {h} | {c} |")

    # Corruption
    lines += [
        "",
        "---",
        "",
        "## 4  Data Quality",
        "",
        f"- **Corrupt / unreadable images**: {len(corrupt)}",
    ]
    if corrupt:
        lines.append("")
        lines.append("| Filepath | Version | Class | Split |")
        lines.append("|----------|---------|-------|-------|")
        for r in corrupt[:50]:
            lines.append(f"| `{r.filepath}` | {r.dataset_version} | {r.class_name} | {r.split} |")

    # Filename duplicates
    total_fn_dups = sum(len(v) for v in filename_dups.values())
    lines += [
        "",
        f"- **Filename duplicates** (within splits): {total_fn_dups}",
    ]
    if filename_dups:
        for version, dups in list(filename_dups.items())[:5]:
            lines.append(f"  - `{version}`: {len(dups)} duplicate filenames")

    # Cross-split hash duplicates
    total_hash_dups = sum(len(v) for v in hash_dups.values())
    lines += [
        f"- **Cross-split hash duplicates** (train ↔ test leakage): {total_hash_dups}",
    ]
    if total_hash_dups == 0:
        lines.append("  ✅  No leakage detected.")
    else:
        lines.append("  ⚠️  Leakage detected — investigate before training.")

    lines += [
        "",
        "---",
        "",
        "## 5  Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total images scanned | {len(all_records):,} |",
        f"| Corrupt images | {len(corrupt)} |",
        f"| Unique dimension combinations | {len(dims)} |",
        f"| Cross-split hash duplicates | {total_hash_dups} |",
        f"| Scan time | {elapsed:.1f}s |",
        "",
        "---",
        "",
        "## 6  Citation",
        "",
        "```",
        "Jia-Yu Li et al. A large-scale, high-quality dataset for lithology identification:",
        "Construction and applications. Petroleum Science, 2025.",
        "https://doi.org/10.1016/j.petsci.2025.04.013",
        "```",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Report written → %s", report_path)


# ---------------------------------------------------------------------------
# Step 8 — Terminal summary
# ---------------------------------------------------------------------------

def print_terminal_summary(
    all_records: List[ImageRecord],
    missing_dirs: List[str],
    corrupt_count: int,
    total_hash_dups: int,
    manifest_path: Path,
    report_path: Path,
    elapsed: float,
) -> None:
    # Use ASCII-safe output so the summary renders correctly on Windows
    # consoles using cp1252 or similar single-byte encodings.
    sep = "=" * 70
    print(f"\n{sep}")
    print("  LithoScan - DCID Reconnaissance Summary")
    print(sep)

    if missing_dirs:
        print(f"\n  [WARN] MISSING directories: {missing_dirs}")
    else:
        print("\n  [OK]   All 4 DCID directories verified.")

    version_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in all_records:
        version_totals[r.dataset_version][r.split] += 1

    print()
    print(f"  {'Dataset':<22} {'Train':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for v in EXPECTED_VERSIONS:
        tr = version_totals[v].get("train", 0)
        te = version_totals[v].get("test", 0)
        print(f"  {v:<22} {tr:>8,} {te:>8,} {tr+te:>8,}")

    print()
    print(f"  Total images scanned  : {len(all_records):,}")
    print(f"  Corrupt images        : {corrupt_count}")
    print(f"  Train<->Test leakage  : {total_hash_dups} hash duplicates")
    print(f"  Scan time             : {elapsed:.1f}s")
    print()
    print(f"  Manifest  -> {manifest_path}")
    print(f"  Report    -> {report_path}")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_reconnaissance(
    dcid_root: Path = _DEFAULT_DCID_ROOT,
    project_root: Path = _DEFAULT_PROJECT_ROOT,
) -> None:
    t0 = time.perf_counter()
    logger.info("DCID root     : %s", dcid_root)
    logger.info("Project root  : %s", project_root)

    # -----------------------------------------------------------------------
    # 1. Verify directories
    # -----------------------------------------------------------------------
    if not dcid_root.is_dir():
        logger.error(
            "DCID root directory not found: %s\n"
            "Set the DCID_ROOT environment variable or edit _DEFAULT_DCID_ROOT.",
            dcid_root,
        )
        sys.exit(1)

    missing_dirs = verify_structure(dcid_root)

    # -----------------------------------------------------------------------
    # 2. Collect all records across all versions and splits
    # -----------------------------------------------------------------------
    all_records: List[ImageRecord] = []
    filename_dups_by_version: Dict[str, Dict[str, List[str]]] = {}
    hash_dups_by_version: Dict[str, List[Tuple[str, str]]] = {}

    for version in EXPECTED_VERSIONS:
        version_dir = dcid_root / version
        if not version_dir.is_dir():
            continue

        classes = collect_classes(dcid_root, version)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        logger.info("Scanning %s — %d classes", version, len(classes))

        train_records: List[ImageRecord] = []
        test_records: List[ImageRecord] = []

        for split in SPLITS:
            split_records = scan_split(
                dcid_root, version, split, classes, class_to_idx
            )
            if split == "train":
                train_records = split_records
            else:
                test_records = split_records
            all_records.extend(split_records)

        # Filename duplicates
        fn_dups: Dict[str, List[str]] = {}
        for split, recs in [("train", train_records), ("test", test_records)]:
            d = find_filename_duplicates(recs)
            if d:
                fn_dups[split] = d
        if fn_dups:
            filename_dups_by_version[version] = fn_dups

        # Cross-split hash duplicates
        hd = find_hash_duplicates(train_records, test_records)
        if hd:
            hash_dups_by_version[version] = hd
            logger.warning(
                "%s: %d cross-split hash duplicates found!", version, len(hd)
            )
        else:
            logger.info("%s: no cross-split hash duplicates.", version)

    # -----------------------------------------------------------------------
    # 3. Write manifest
    # -----------------------------------------------------------------------
    manifest_path = project_root / "data" / "manifests" / "dcid_manifest.csv"
    write_manifest(all_records, manifest_path)

    # -----------------------------------------------------------------------
    # 4. Write markdown report
    # -----------------------------------------------------------------------
    report_path = project_root / "reports" / "dataset_report.md"
    elapsed = time.perf_counter() - t0
    write_report(
        all_records=all_records,
        missing_dirs=missing_dirs,
        filename_dups=filename_dups_by_version,
        hash_dups=hash_dups_by_version,
        report_path=report_path,
        elapsed=elapsed,
    )

    # -----------------------------------------------------------------------
    # 5. Terminal summary
    # -----------------------------------------------------------------------
    corrupt_count = sum(1 for r in all_records if r.is_corrupt)
    total_hash_dups = sum(len(v) for v in hash_dups_by_version.values())
    print_terminal_summary(
        all_records=all_records,
        missing_dirs=missing_dirs,
        corrupt_count=corrupt_count,
        total_hash_dups=total_hash_dups,
        manifest_path=manifest_path,
        report_path=report_path,
        elapsed=elapsed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LithoScan — DCID Dataset Reconnaissance"
    )
    parser.add_argument(
        "--dcid-root",
        type=Path,
        default=_DEFAULT_DCID_ROOT,
        help="Path to the extracted DCID archive root directory.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_DEFAULT_PROJECT_ROOT,
        help="LithoScan project root (contains data/, reports/, etc.).",
    )
    args = parser.parse_args()
    run_reconnaissance(dcid_root=args.dcid_root, project_root=args.project_root)
