import os
import sys
import time
import json
import yaml
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset, get_train_transforms, get_eval_transforms
from src.models.server_fusion import LithoScanServerFusion
from src.evaluation.artifact_intervention import mask_image_border

logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=7, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric_value):
        if self.best_score is None:
            self.best_score = metric_value
            return True
            
        is_better = metric_value > self.best_score if self.mode == "max" else metric_value < self.best_score
        
        if is_better:
            self.best_score = metric_value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def evaluate_dataset(model, dataloader, device, amp, apply_mask=False):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if apply_mask:
                # Apply 40-px border mask using ImageNet-normalized black (RGB=0)
                images = mask_image_border(images, border_width=40, is_normalized=True)['intervened_image']
            
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(images)
                preds = logits.argmax(dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    out_dir = Path(config["output"]["log_dir"])
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # Setup Logging
    # ---------------------------------------------------------
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(out_dir / "training.log", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Save config used
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 1. Prepare Datasets & Splitting Logic
    # ---------------------------------------------------------
    dcid_root_cfg = os.environ.get("DCID_ROOT", config["dataset"].get("dcid_root"))
    if not dcid_root_cfg or not Path(dcid_root_cfg).exists():
        fallback = PROJECT_ROOT.parent / "DCID" / "DCID"
        if fallback.exists():
            dcid_root_cfg = str(fallback)
    config["dataset"]["dcid_root"] = str(dcid_root_cfg)

    full_train = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["train_split"],
        transform=get_train_transforms(image_size=config["preprocessing"]["image_size"])
    )
    val_raw = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["train_split"],
        transform=get_eval_transforms(image_size=config["preprocessing"]["image_size"])
    )
    
    # Create single deterministic permutation
    gen = torch.Generator().manual_seed(config["dataloader"]["seed"])
    total_len = len(full_train)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    
    indices = torch.randperm(total_len, generator=gen)
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]
    
    # Verify correctness structurally
    assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist())), "Train/Val indices are not disjoint!"
    assert len(train_idx) == train_len
    assert len(val_idx) == val_len
    
    train_ds = torch.utils.data.Subset(full_train, train_idx.tolist())
    val_ds = torch.utils.data.Subset(val_raw, val_idx.tolist())
    
    # Test Data (Clean)
    test_ds = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["test_split"],
        transform=get_eval_transforms(image_size=config["preprocessing"]["image_size"])
    )
    
    # Test Data (Noise)
    noise_test_ds = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["noise_version"],
        split=config["dataset"]["test_split"],
        transform=get_eval_transforms(image_size=config["preprocessing"]["image_size"])
    )
    
    # Verify path-level disjointness
    train_paths = set(full_train.samples[i][0] for i in train_idx.tolist())
    val_paths = set(val_raw.samples[i][0] for i in val_idx.tolist())
    test_paths = set(s[0] for s in test_ds.samples)
    
    logger.info("==================================================")
    logger.info("DATASET SPLIT VERIFICATION")
    logger.info("==================================================")
    logger.info(f"Train samples: {len(train_paths)}")
    logger.info(f"Validation samples: {len(val_paths)}")
    logger.info(f"Test samples: {len(test_paths)}")
    logger.info(f"Train INT Validation: {len(train_paths & val_paths)}")
    logger.info(f"Train INT Test: {len(train_paths & test_paths)}")
    logger.info(f"Validation INT Test: {len(val_paths & test_paths)}")
    
    assert len(train_paths & val_paths) == 0, "Leakage detected between Train and Validation!"
    assert len(train_paths & test_paths) == 0, "Leakage detected between Train and Test!"
    assert len(val_paths & test_paths) == 0, "Leakage detected between Validation and Test!"
    
    train_loader = DataLoader(
        train_ds, batch_size=config["dataloader"]["batch_size"], 
        shuffle=True, num_workers=config["dataloader"]["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["dataloader"]["batch_size"], 
        shuffle=False, num_workers=config["dataloader"]["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=config["dataloader"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    noise_test_loader = DataLoader(noise_test_ds, batch_size=config["dataloader"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # ---------------------------------------------------------
    # 2. Build Model
    # ---------------------------------------------------------
    model = LithoScanServerFusion(
        num_classes=config["dataset"]["num_classes"],
        architecture_mode=config["model"]["architecture_mode"],
        pretrained=config["model"]["pretrained"],
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"]
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if config["training"]["amp"] else None
    
    early_stopper = EarlyStopping(patience=config["training"]["early_stopping_patience"])
    grad_accum = config["training"]["gradient_accumulation"]
    
    # Logging objects
    history = []
    best_val_f1 = 0.0
    best_epoch = 0
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 3. Training Loop
    # ---------------------------------------------------------
    logger.info("Starting Training Loop...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        all_train_preds, all_train_targets = [], []
        
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=config["training"]["amp"]):
                logits = model(images)
                loss = criterion(logits, labels)
                loss = loss / grad_accum
                
            scaler.scale(loss).backward()
            
            all_train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_train_targets.extend(labels.cpu().numpy())
            
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * grad_accum
            
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        
        val_acc, val_f1, _, _ = evaluate_dataset(model, val_loader, device, config["training"]["amp"])
        
        logger.info(f"Epoch {epoch+1:02d}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1": val_f1
        })
        
        if early_stopper(val_f1):
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            logger.info("  >> Saved new best model!")
            
        if early_stopper.early_stop:
            logger.info("Early stopping triggered.")
            break
            
    # Save final model just in case
    torch.save(model.state_dict(), ckpt_dir / "final_model.pt")
    
    total_time = time.time() - start_time
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    
    # ---------------------------------------------------------
    # 4. Final Evaluation (Best Model)
    # ---------------------------------------------------------
    logger.info("Loading best model for evaluation...")
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", weights_only=True))
    
    # A. Clean Test
    clean_acc, clean_f1, preds, targets = evaluate_dataset(model, test_loader, device, config["training"]["amp"], apply_mask=False)
    
    # B. Noisy Test
    noise_acc, noise_f1, _, _ = evaluate_dataset(model, noise_test_loader, device, config["training"]["amp"], apply_mask=False)
    
    # C. Masked Border Test
    logger.info("Running Artifact Masked Evaluation (border width = 40px, normalized black equivalent)")
    mask_acc, mask_f1, _, _ = evaluate_dataset(model, test_loader, device, config["training"]["amp"], apply_mask=True)
    
    # Reports
    target_names = test_ds.classes
    cls_report = classification_report(targets, preds, target_names=target_names, output_dict=True)
    pd.DataFrame(cls_report).transpose().to_csv(out_dir / "classification_report.csv")
    
    cm = confusion_matrix(targets, preds)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(out_dir / "confusion_matrix.csv")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    
    metrics = {
        "best_epoch": best_epoch,
        "total_epochs_run": epoch + 1,
        "training_time_seconds": total_time,
        "peak_cuda_memory_mb": peak_mem,
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        
        "validation_best_acc": float(history[best_epoch-1]["val_acc"]),
        "validation_best_f1": float(best_val_f1),
        
        "test_clean_accuracy": float(clean_acc),
        "test_clean_macro_f1": float(clean_f1),
        
        "test_noise_accuracy": float(noise_acc),
        "test_noise_macro_f1": float(noise_f1),
        "noise_absolute_drop": float(clean_acc - noise_acc),
        "noise_relative_drop": float((clean_acc - noise_acc) / clean_acc) if clean_acc > 0 else 0,
        
        "test_masked_accuracy": float(mask_acc),
        "test_masked_macro_f1": float(mask_f1),
        "mask_absolute_drop": float(clean_acc - mask_acc),
        "mask_relative_drop": float((clean_acc - mask_acc) / clean_acc) if clean_acc > 0 else 0,
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info("==================================================")
    logger.info("Evaluation Complete.")
    logger.info(f"Clean Test Acc: {clean_acc:.4f} | F1: {clean_f1:.4f}")
    logger.info(f"Noise Test Acc: {noise_acc:.4f} (Drop: {metrics['noise_absolute_drop']:.4f})")
    logger.info(f"Mask  Test Acc: {mask_acc:.4f} (Drop: {metrics['mask_absolute_drop']:.4f})")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
