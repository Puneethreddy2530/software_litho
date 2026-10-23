"""
build_paperB_splits.py — Generate deterministic split manifests for Paper B experiments.
Creates disjoint calibration and evaluation partitions for DCID-35 and DCID-7.
Saves manifest JSON files to configs/paperB_uncertainty/.
"""

import json
import hashlib
import os
import random
from pathlib import Path

DCID_ROOT = r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID"
CONFIG_DIR = Path(r"C:\Users\punee\Desktop\Software_lithology\LithoScan\configs\paperB_uncertainty")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026

def get_class_images(dir_path):
    classes = sorted(os.listdir(dir_path))
    class_map = {}
    for c in classes:
        c_dir = os.path.join(dir_path, c)
        imgs = sorted(os.listdir(c_dir))
        class_map[c] = [os.path.join(c, img).replace("\\", "/") for img in imgs]
    return class_map

def build_manifest_dcid35():
    random.seed(SEED)
    clean_test_dir = os.path.join(DCID_ROOT, "DCID-512-35", "test")
    noise_train_dir = os.path.join(DCID_ROOT, "noise-512-35", "train")
    noise_test_dir = os.path.join(DCID_ROOT, "noise-512-35", "test")

    clean_test_map = get_class_images(clean_test_dir)
    noise_train_map = get_class_images(noise_train_dir)
    noise_test_map = get_class_images(noise_test_dir)

    cal_clean = []
    eval_clean = []
    for c, imgs in clean_test_map.items():
        shuffled = list(imgs)
        random.shuffle(shuffled)
        cal_clean.extend([f"DCID-512-35/test/{p}" for p in shuffled[:50]])
        eval_clean.extend([f"DCID-512-35/test/{p}" for p in shuffled[50:]])

    cal_def = []
    dev_def_unused = []
    for c, imgs in noise_train_map.items():
        shuffled = list(imgs)
        random.shuffle(shuffled)
        cal_def.extend([f"noise-512-35/train/{p}" for p in shuffled[:100]])
        dev_def_unused.extend([f"noise-512-35/train/{p}" for p in shuffled[100:]])

    eval_def = []
    for c, imgs in noise_test_map.items():
        eval_def.extend([f"noise-512-35/test/{p}" for p in imgs])

    # Sort all lists for determinism
    cal_clean.sort()
    eval_clean.sort()
    cal_def.sort()
    eval_def.sort()
    dev_def_unused.sort()

    manifest = {
        "benchmark": "DCID-35",
        "seed": SEED,
        "dcid_root": DCID_ROOT.replace("\\", "/"),
        "partitions": {
            "cal_clean": {
                "count": len(cal_clean),
                "per_class": 50,
                "files": cal_clean
            },
            "eval_clean": {
                "count": len(eval_clean),
                "per_class": 150,
                "files": eval_clean
            },
            "cal_def": {
                "count": len(cal_def),
                "per_class": 100,
                "files": cal_def
            },
            "eval_def": {
                "count": len(eval_def),
                "per_class": 80,
                "files": eval_def
            },
            "dev_def_reserved": {
                "count": len(dev_def_unused),
                "per_class": 220,
                "files": dev_def_unused
            }
        }
    }

    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["manifest_sha256"] = sha256

    out_path = CONFIG_DIR / "paperB_split_manifest_dcid35.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"DCID-35 manifest created: {out_path}")
    print(f"  cal_clean: {len(cal_clean)} images (50/class)")
    print(f"  eval_clean: {len(eval_clean)} images (150/class)")
    print(f"  cal_def: {len(cal_def)} images (100/class)")
    print(f"  eval_def: {len(eval_def)} images (80/class)")
    print(f"  SHA-256: {sha256}")
    return manifest

def build_manifest_dcid7():
    random.seed(SEED)
    clean_test_dir = os.path.join(DCID_ROOT, "DCID-512-7", "test")
    noise_train_dir = os.path.join(DCID_ROOT, "noise-512-7", "train")
    noise_test_dir = os.path.join(DCID_ROOT, "noise-512-7", "test")

    clean_test_map = get_class_images(clean_test_dir)
    noise_train_map = get_class_images(noise_train_dir)
    noise_test_map = get_class_images(noise_test_dir)

    cal_clean = []
    eval_clean = []
    for c, imgs in clean_test_map.items():
        shuffled = list(imgs)
        random.shuffle(shuffled)
        cal_clean.extend([f"DCID-512-7/test/{p}" for p in shuffled[:250]])
        eval_clean.extend([f"DCID-512-7/test/{p}" for p in shuffled[250:]])

    cal_def = []
    dev_def_unused = []
    for c, imgs in noise_train_map.items():
        shuffled = list(imgs)
        random.shuffle(shuffled)
        cal_def.extend([f"noise-512-7/train/{p}" for p in shuffled[:400]])
        dev_def_unused.extend([f"noise-512-7/train/{p}" for p in shuffled[400:]])

    eval_def = []
    for c, imgs in noise_test_map.items():
        eval_def.extend([f"noise-512-7/test/{p}" for p in imgs])

    cal_clean.sort()
    eval_clean.sort()
    cal_def.sort()
    eval_def.sort()
    dev_def_unused.sort()

    manifest = {
        "benchmark": "DCID-7",
        "seed": SEED,
        "dcid_root": DCID_ROOT.replace("\\", "/"),
        "partitions": {
            "cal_clean": {
                "count": len(cal_clean),
                "per_class": 250,
                "files": cal_clean
            },
            "eval_clean": {
                "count": len(eval_clean),
                "per_class": 750,
                "files": eval_clean
            },
            "cal_def": {
                "count": len(cal_def),
                "per_class": 400,
                "files": cal_def
            },
            "eval_def": {
                "count": len(eval_def),
                "per_class": 400,
                "files": eval_def
            },
            "dev_def_reserved": {
                "count": len(dev_def_unused),
                "per_class": 1200,
                "files": dev_def_unused
            }
        }
    }

    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["manifest_sha256"] = sha256

    out_path = CONFIG_DIR / "paperB_split_manifest_dcid7.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"DCID-7 manifest created: {out_path}")
    print(f"  cal_clean: {len(cal_clean)} images (250/class)")
    print(f"  eval_clean: {len(eval_clean)} images (750/class)")
    print(f"  cal_def: {len(cal_def)} images (400/class)")
    print(f"  eval_def: {len(eval_def)} images (400/class)")
    print(f"  SHA-256: {sha256}")
    return manifest

if __name__ == "__main__":
    build_manifest_dcid35()
    build_manifest_dcid7()
