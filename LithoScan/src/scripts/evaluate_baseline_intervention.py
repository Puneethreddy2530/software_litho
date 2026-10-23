import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset, get_eval_transforms
from src.evaluation.artifact_intervention import mask_image_border

def evaluate_dataset(model, dataloader, device, apply_mask=False):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if apply_mask:
                # Same exact mask as Phase 4 Server
                images = mask_image_border(images, border_width=40, is_normalized=True)['intervened_image']
            
            # Phase 2 didn't strictly use AMP for eval, but standard forward pass
            logits = model(images)
            preds = logits.argmax(dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset
    image_size = 512
    test_ds = DCIDDataset(
        dcid_root="C:/Users/punee/Desktop/Software_lithology/DCID/DCID",
        version="DCID-512-7",
        split="test",
        transform=get_eval_transforms(image_size=image_size)
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    ckpt_path = PROJECT_ROOT / "checkpoints" / "exp_A" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    print("Model loaded successfully.")
    
    # 3. Evaluate Clean
    print("Evaluating Clean Test Set...")
    clean_acc, clean_f1 = evaluate_dataset(model, test_loader, device, apply_mask=False)
    
    # 4. Evaluate Masked
    print("Evaluating Masked Border Test Set...")
    mask_acc, mask_f1 = evaluate_dataset(model, test_loader, device, apply_mask=True)
    
    print("\n==================================================")
    print("RESNET-50 BASELINE INTERVENTION RESULTS")
    print("==================================================")
    print(f"Clean Acc:  {clean_acc:.4f} | F1: {clean_f1:.4f}")
    print(f"Masked Acc: {mask_acc:.4f} | F1: {mask_f1:.4f}")
    
    metrics = {
        "clean_acc": clean_acc,
        "clean_f1": clean_f1,
        "masked_acc": mask_acc,
        "masked_f1": mask_f1,
        "absolute_drop_acc": clean_acc - mask_acc,
        "relative_drop_acc": (clean_acc - mask_acc) / clean_acc if clean_acc > 0 else 0
    }
    
    out_dir = PROJECT_ROOT / "reports" / "phase4" / "dcid7_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "resnet50_intervention.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
