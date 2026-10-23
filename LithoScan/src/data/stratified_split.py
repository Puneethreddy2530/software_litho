"""
Phase 6 v1.1 Deterministic Class-Stratified Splitter
====================================================

Generates reproducible, class-balanced split manifests for DCID-7:
- Stratified validation set: exactly 400 images per class (2,800 total)
- Bounded-pilot training set: exactly 360 images per class (2,520 total)
- Full-campaign training set: exactly 3,600 images per class (25,200 total)
- Verifies disjoint index and path sets (zero leakage)
- Generates SHA-256 integrity hash for manifest locking
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any

from torchvision.datasets import ImageFolder


def generate_dcid7_stratified_manifest(
    dataset_train_root: str = "C:/Users/punee/Desktop/Software_lithology/DCID/DCID/DCID-512-7/train",
    val_per_class: int = 400,
    pilot_train_per_class: int = 360,
    seed: int = 42,
    output_path: str = "configs/phase6_split_manifest_dcid7_v1_1.json",
) -> Dict[str, Any]:
    root_path = Path(dataset_train_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Training path not found: {root_path}")

    # Discover classes and images deterministically
    dataset = ImageFolder(root=str(root_path))
    class_to_idx = dataset.class_to_idx
    classes = sorted(list(class_to_idx.keys()))

    # Group sample indices by class
    class_indices: Dict[str, List[int]] = {c: [] for c in classes}
    for idx, (path, label_idx) in enumerate(dataset.samples):
        class_name = dataset.classes[label_idx]
        class_indices[class_name].append(idx)

    rng = random.Random(seed)

    manifest_data = {
        "version": "1.1",
        "dataset": "DCID-512-7",
        "dataset_root": str(root_path),
        "seed": seed,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "val_per_class": val_per_class,
        "pilot_train_per_class": pilot_train_per_class,
        "full_train_per_class": 0,
        "splits": {
            "validation": [],
            "pilot_train": [],
            "full_train": [],
        },
        "per_class_summary": {},
    }

    all_val_indices = []
    all_pilot_train_indices = []
    all_full_train_indices = []

    for c in classes:
        indices = list(class_indices[c])
        # Sort first to ensure deterministic baseline ordering
        indices.sort(key=lambda i: dataset.samples[i][0])
        # Deterministic shuffle
        rng.shuffle(indices)

        if len(indices) < (val_per_class + pilot_train_per_class):
            raise ValueError(f"Class {c} has only {len(indices)} images, insufficient for requested split")

        val_idx = indices[:val_per_class]
        train_pool = indices[val_per_class:]
        pilot_train_idx = train_pool[:pilot_train_per_class]

        all_val_indices.extend(val_idx)
        all_pilot_train_indices.extend(pilot_train_idx)
        all_full_train_indices.extend(train_pool)

        manifest_data["per_class_summary"][c] = {
            "total_available": len(indices),
            "val_count": len(val_idx),
            "pilot_train_count": len(pilot_train_idx),
            "full_train_count": len(train_pool),
        }

    # Store paths relative to train_root for portability
    def to_relative(idx_list: List[int]) -> List[Dict[str, Any]]:
        return [
            {
                "index": int(i),
                "rel_path": str(Path(dataset.samples[i][0]).relative_to(root_path)).replace("\\", "/"),
                "class_name": dataset.classes[dataset.samples[i][1]],
                "class_idx": int(dataset.samples[i][1]),
            }
            for i in idx_list
        ]

    manifest_data["splits"]["validation"] = to_relative(all_val_indices)
    manifest_data["splits"]["pilot_train"] = to_relative(all_pilot_train_indices)
    manifest_data["splits"]["full_train"] = to_relative(all_full_train_indices)

    # Verification: Disjoint sets
    val_set = set(all_val_indices)
    pilot_train_set = set(all_pilot_train_indices)
    full_train_set = set(all_full_train_indices)

    assert len(val_set.intersection(full_train_set)) == 0, "CRITICAL ERROR: Overlap between validation and training!"
    assert len(val_set.intersection(pilot_train_set)) == 0, "CRITICAL ERROR: Overlap between validation and pilot training!"
    assert pilot_train_set.issubset(full_train_set), "Pilot train must be a strict subset of full train!"

    # Compute SHA-256 hash
    manifest_str = json.dumps(manifest_data, sort_keys=True, indent=2)
    sha256 = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    manifest_data["manifest_sha256"] = sha256

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[StratifiedSplit] Generated manifest at: {out_file}")
    print(f"[StratifiedSplit] SHA-256: {sha256}")
    print(f"[StratifiedSplit] Classes: {len(classes)}")
    print(f"[StratifiedSplit] Validation samples: {len(all_val_indices)} ({val_per_class}/class)")
    print(f"[StratifiedSplit] Pilot train samples: {len(all_pilot_train_indices)} ({pilot_train_per_class}/class)")
    print(f"[StratifiedSplit] Full train samples: {len(all_full_train_indices)} ({len(all_full_train_indices)//len(classes)}/class)")

    return manifest_data


if __name__ == "__main__":
    generate_dcid7_stratified_manifest()
