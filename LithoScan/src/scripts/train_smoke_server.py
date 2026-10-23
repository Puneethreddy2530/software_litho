import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset, get_train_transforms, get_eval_transforms
from src.models.server_fusion import LithoScanServerFusion

def main():
    config_path = PROJECT_ROOT / "configs" / "phase4a_smoke.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loading DCID-512-7 train dataset...")
    full_train = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["train_split"],
        transform=get_train_transforms(image_size=512)
    )
    
    val_ds_raw = DCIDDataset(
        dcid_root=config["dataset"]["dcid_root"],
        version=config["dataset"]["version"],
        split=config["dataset"]["train_split"],
        transform=get_eval_transforms(image_size=512)
    )
    
    # Same deterministic 90/10 split as Phase 2
    gen = torch.Generator().manual_seed(config["dataloader"]["seed"])
    total_len = len(full_train)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    
    train_ds, _ = torch.utils.data.random_split(full_train, [train_len, val_len], generator=gen)
    _, val_ds = torch.utils.data.random_split(val_ds_raw, [train_len, val_len], generator=gen)
    
    # We will only use a small subset of the training and validation sets for a quick SMOKE TEST (e.g. 5 batches)
    train_ds = torch.utils.data.Subset(train_ds, list(range(20)))
    val_ds = torch.utils.data.Subset(val_ds, list(range(20)))
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"]["pin_memory"]
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if config["training"]["amp"] else None
    grad_accum = config["training"]["gradient_accumulation"]
    
    print("Starting 1-Epoch Smoke Training...")
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        with torch.amp.autocast('cuda', enabled=config["training"]["amp"]):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / grad_accum
            
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss += loss.item() * grad_accum
        
    print(f"Train Loss: {total_loss / len(train_loader):.4f}")
    
    print("Evaluating...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=config["training"]["amp"]):
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
    print(f"Validation Accuracy: {correct/total:.4f}")
    print("SMOKE TRAINING COMPLETE. The architecture supports standard PyTorch loop.")

if __name__ == "__main__":
    main()
