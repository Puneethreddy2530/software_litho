import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.server_fusion import LithoScanServerFusion
from src.data.dcid_dataset import DCIDDataset

def test_model(architecture_mode, device):
    print(f"  Testing architecture: {architecture_mode}")
    try:
        model = LithoScanServerFusion(num_classes=35, architecture_mode=architecture_mode).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        dummy_input = torch.randn(4, 3, 512, 512).to(device)
        dummy_target = torch.randint(0, 35, (4,)).to(device)
        
        optimizer.zero_grad()
        logits = model(dummy_input)
        
        if logits.shape != (4, 35):
            print(f"    FAIL: Logits shape {logits.shape} != (4, 35)")
            return False
            
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("    FAIL: Logits contain NaN/Inf")
            return False
            
        loss = nn.CrossEntropyLoss()(logits, dummy_target)
        if torch.isnan(loss) or torch.isinf(loss):
            print("    FAIL: Loss is NaN/Inf")
            return False
            
        loss.backward()
        optimizer.step()
        print("    PASS: Forward/Backward/Optimizer OK")
        return True
    except Exception as e:
        print(f"    FAIL: Exception occurred: {e}")
        return False

def verify_configs():
    configs = ["configs/phase5_dcid35_server.yaml", "configs/phase5_dcid35_cnn_only.yaml"]
    loaded = {}
    for c in configs:
        p = PROJECT_ROOT / c
        if not p.exists():
            print(f"FAIL: Missing config {c}")
            return False
        with open(p, "r") as f:
            # Load YAML
            content = f.read()
            # Basic manual check for duplicate top-level keys
            lines = [l.strip() for l in content.split('\n') if l.strip() and not l.startswith('#')]
            top_keys = [l.split(':')[0] for l in lines if not l.startswith(' ') and ':' in l]
            if len(top_keys) != len(set(top_keys)):
                print(f"FAIL: Duplicate top-level keys found in {c}")
                return False
                
        with open(p, "r") as f:
            data = yaml.safe_load(f)
            loaded[c] = data
            
    # Verify contents
    c1 = loaded["configs/phase5_dcid35_server.yaml"]
    c2 = loaded["configs/phase5_dcid35_cnn_only.yaml"]
    
    if c1["model"]["architecture_mode"] != "cnn_transformer":
        print("FAIL: Server config architecture_mode mismatch")
        return False
    if c1["dataset"]["num_classes"] != 35:
        print("FAIL: Server config num_classes mismatch")
        return False
        
    if c2["model"]["architecture_mode"] != "cnn_only":
        print("FAIL: CNN-only config architecture_mode mismatch")
        return False
    if c2["dataset"]["num_classes"] != 35:
        print("FAIL: CNN-only config num_classes mismatch")
        return False
        
    print(f"  PASS: Validated {configs[0]}")
    print(f"    -> mode: {c1['model']['architecture_mode']}, classes: {c1['dataset']['num_classes']}, epochs: {c1['training']['epochs']}")
    print(f"  PASS: Validated {configs[1]}")
    print(f"    -> mode: {c2['model']['architecture_mode']}, classes: {c2['dataset']['num_classes']}, epochs: {c2['training']['epochs']}")
    
    return True

def main():
    print("==========================================")
    print("PHASE 5A FINAL PRE-FLIGHT HARDENING")
    print("==========================================")
    
    # 1. Verify Configs
    print("\n[1] Verifying Config Files...")
    if not verify_configs():
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    # 2. Verify CUDA & Models
    print("\n[2] Verifying Architectures (CUDA/Forward/Backward)...")
    if not torch.cuda.is_available():
        print("FAIL: CUDA is not available.")
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    device = torch.device("cuda:0")
    if not test_model("cnn_transformer", device):
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
    if not test_model("cnn_only", device):
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    # 3. Dataset Fact & Splitting Verification
    print("\n[3] Verifying DCID-35 Dataset and Split Paths...")
    dataset_root = "C:/Users/punee/Desktop/Software_lithology/DCID/DCID"
    
    train_ds = DCIDDataset(dataset_root, "DCID-512-35", "train")
    test_ds = DCIDDataset(dataset_root, "DCID-512-35", "test")
    
    total_train = len(train_ds)
    total_test = len(test_ds)
    
    print(f"  Total official training images: {total_train}")
    print(f"  Total official test images: {total_test}")
    
    if total_train != 27999:
        print(f"FAIL: Expected 27999 training images, got {total_train}")
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    if total_test != 7000:
        print(f"FAIL: Expected 7000 test images, got {total_test}")
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    # Verify class 28 count
    class_name = "28.Banded iron formation"
    cls_idx = train_ds.classes.index(class_name)
    cls_count = sum(1 for _, target in train_ds.samples if target == cls_idx)
    
    print(f"  Class 28.Banded iron formation training count: {cls_count}")
    if cls_count != 799:
        print(f"FAIL: Expected 799 for Banded iron formation, got {cls_count}")
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    # Reproduce Exact Train/Val Split from train_server.py
    seed = 42
    gen = torch.Generator().manual_seed(seed)
    val_len = int(0.1 * total_train)
    train_len = total_train - val_len
    
    indices = torch.randperm(total_train, generator=gen)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]
    
    # Path extraction
    train_paths = {train_ds.samples[i][0] for i in train_idx.tolist()}
    val_paths = {train_ds.samples[i][0] for i in val_idx.tolist()}
    test_paths = {s[0] for s in test_ds.samples}
    
    print(f"  Train split count: {len(train_paths)}")
    print(f"  Val split count: {len(val_paths)}")
    
    train_val_int = train_paths.intersection(val_paths)
    train_test_int = train_paths.intersection(test_paths)
    val_test_int = val_paths.intersection(test_paths)
    
    print(f"  train INT val = {len(train_val_int)}")
    print(f"  train INT official_test = {len(train_test_int)}")
    print(f"  val INT official_test = {len(val_test_int)}")
    
    if len(train_val_int) > 0 or len(train_test_int) > 0 or len(val_test_int) > 0:
        print("FAIL: Split intersection is non-zero.")
        print("\nPHASE 5A FINAL PRE-FLIGHT: FAIL")
        sys.exit(1)
        
    print("\nPHASE 5A FINAL PRE-FLIGHT: PASS")
    
if __name__ == "__main__":
    main()
