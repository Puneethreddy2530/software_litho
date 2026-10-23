import os
import sys
import torch
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset

def validate_split_reproducibility():
    config_path = PROJECT_ROOT / "configs" / "phase4_dcid7_server.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print("==================================================")
    print("SPLIT VALIDATION SCRIPT")
    print("==================================================")
    
    full_train = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["train_split"],
    )
    
    total_len = len(full_train)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    
    def construct_indices():
        gen = torch.Generator().manual_seed(config["dataloader"]["seed"])
        indices = torch.randperm(total_len, generator=gen)
        return indices[:train_len].tolist(), indices[train_len:].tolist()

    train1, val1 = construct_indices()
    train2, val2 = construct_indices()
    
    print("Checking Reproducibility...")
    assert train1 == train2, "Train indices are not reproducible!"
    assert val1 == val2, "Validation indices are not reproducible!"
    print("PASS: Two consecutive constructions yielded identical exact splits.")
    
    print("\nChecking Disjointness...")
    assert set(train1).isdisjoint(set(val1)), "Train and Validation sets OVERLAP!"
    print(f"PASS: Sets are completely disjoint. (Train: {len(train1)}, Val: {len(val1)})")
    
    print("\nFirst 10 Train Indices:", train1[:10])
    print("First 10 Val Indices:  ", val1[:10])
    
    # Path level verification
    test_ds = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["test_split"],
    )
    
    train_paths = set(full_train.samples[i][0] for i in train1)
    val_paths = set(full_train.samples[i][0] for i in val1)
    test_paths = set(s[0] for s in test_ds.samples)
    
    print("\nPath-Level Disjointness:")
    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0
    print("PASS: All file paths are strictly mutually exclusive.")
    
    print("\nSPLIT VALIDATION: PERFECT")

if __name__ == "__main__":
    validate_split_reproducibility()
