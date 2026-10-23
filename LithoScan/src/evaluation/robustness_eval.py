import os
import sys
import json
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision.models import resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset

def get_test_loader(dcid_root, version, batch_size=32):
    ds = DCIDDataset(dcid_root=dcid_root, version=version, split="test", image_size=512, skip_corrupt=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader, ds.classes

def evaluate_loader(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device.type):
                outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    cr = classification_report(
        all_targets, all_preds, 
        labels=list(range(len(class_names))),
        target_names=class_names, 
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    return cr, cm, all_targets, all_preds

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load loaders
    clean_loader, clean_classes = get_test_loader(args.dcid_root, args.clean_version)
    noise_loader, noise_classes = get_test_loader(args.dcid_root, args.noise_version)
    
    if clean_classes != noise_classes:
        raise ValueError(f"Class alignment mismatch! Clean: {clean_classes}, Noise: {noise_classes}")
    
    class_names = clean_classes
    num_classes = len(class_names)
    
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    
    print(f"Evaluating Clean Dataset: {args.clean_version}")
    clean_cr, clean_cm, _, _ = evaluate_loader(model, clean_loader, device, class_names)
    
    print(f"Evaluating Noise Dataset: {args.noise_version}")
    noise_cr, noise_cm, _, _ = evaluate_loader(model, noise_loader, device, class_names)
    
    # Save confusion matrices
    for name, cm in [("clean", clean_cm), ("noise", noise_cm)]:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix ({name})')
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_matrix_{name}.png")
        plt.close()
        
    # Calculate drops
    clean_acc = clean_cr['accuracy']
    noise_acc = noise_cr['accuracy']
    acc_drop = clean_acc - noise_acc
    
    clean_f1 = clean_cr['macro avg']['f1-score']
    noise_f1 = noise_cr['macro avg']['f1-score']
    f1_drop = clean_f1 - noise_f1
    
    # Save reports
    pd.DataFrame(clean_cr).transpose().to_csv(out_dir / "classification_report_clean.csv")
    pd.DataFrame(noise_cr).transpose().to_csv(out_dir / "classification_report_noise.csv")
    
    summary = {
        "clean_accuracy": clean_acc,
        "clean_macro_f1": clean_f1,
        "clean_weighted_f1": clean_cr['weighted avg']['f1-score'],
        "noise_accuracy": noise_acc,
        "noise_macro_f1": noise_f1,
        "noise_weighted_f1": noise_cr['weighted avg']['f1-score'],
        "accuracy_drop": acc_drop,
        "macro_f1_drop": f1_drop
    }
    
    with open(out_dir / "robustness_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Robustness evaluation complete. Drops: Acc {acc_drop*100:.2f}%, F1 {f1_drop*100:.2f}%")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clean-version", default="DCID-512-7")
    parser.add_argument("--noise-version", default="noise-512-7")
    parser.add_argument("--dcid-root", default=r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")
    args = parser.parse_args()
    main(args)
