"""
generate_paperB_manifest.py — Generate SHA-256 artifact inventory and release manifest for Paper B.
"""

import hashlib
import json
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "paperB_uncertainty"

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    tracked_patterns = [
        "configs/paperB_uncertainty/*",
        "reports/paperB_uncertainty/*.md",
        "results/paperB_uncertainty/metrics/*",
        "results/paperB_uncertainty/figures/*",
        "src/calibration/paperB_methods.py",
        "src/scripts/build_paperB_splits.py",
        "src/scripts/run_paperB_experiments.py",
        "src/scripts/generate_paperB_figures.py",
        "src/scripts/generate_paperB_tables.py",
        "src/scripts/inspect_results.py",
        "manuscript/paperB_uncertainty/tables/*",
        "manuscript/paperB_uncertainty/figures/*"
    ]

    records = []
    import glob
    for pat in tracked_patterns:
        full_pat = str(PROJECT_ROOT / pat)
        for f_path in glob.glob(full_pat):
            p = Path(f_path)
            if p.is_file():
                rel = p.relative_to(PROJECT_ROOT).as_posix()
                size = p.stat().st_size
                sha = hash_file(p)
                category = rel.split("/")[0]
                records.append({
                    "relative_path": rel,
                    "category": category,
                    "size_bytes": size,
                    "sha256": sha
                })

    df = pd.DataFrame(records)
    csv_out = REPORTS_DIR / "paperB_artifact_inventory.csv"
    df.to_csv(csv_out, index=False)
    print(f"Artifact inventory written: {csv_out} ({len(df)} items)")

    manifest_data = {
        "release_version": "1.0",
        "project": "LithoScan Paper B",
        "title": "When Should a Lithology Classifier Abstain? Calibration, Selective Risk, and Human Referral Under Defective Drill-Core Imaging",
        "timestamp": "2026-10-22T21:00:00Z",
        "total_artifacts": len(records),
        "artifacts": records
    }
    json_out = REPORTS_DIR / "release_manifest.json"
    with open(json_out, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Release manifest written: {json_out}")

if __name__ == "__main__":
    main()
