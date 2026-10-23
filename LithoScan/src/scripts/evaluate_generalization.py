import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import Counter
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset, get_eval_transforms
from src.models.server_fusion import LithoScanServerFusion

def evaluate_subset(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # Both architectures just take the image in eval mode
            logits = model(images)
            preds = logits.argmax(dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    if len(all_targets) == 0:
        return 0.0, 0.0, None
        
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(7)))
    return acc, f1, cm

def get_excluded_test_paths(df, k):
    # Filter for distance <= k
    df_k = df[df['distance'] <= k]
    
    # We want Test paths from Train-Test pairs
    excluded = set()
    for _, row in df_k.iterrows():
        s1, s2 = row['split_1'], row['split_2']
        p1, p2 = row['path_1'], row['path_2']
        if s1 == 'Train' and s2 == 'Test':
            excluded.add(Path(p2).resolve())
        elif s1 == 'Test' and s2 == 'Train':
            excluded.add(Path(p1).resolve())
    return excluded

def build_subsets(test_ds, excluded_2, excluded_5, excluded_10):
    indices_all = []
    indices_2 = []
    indices_5 = []
    indices_10 = []
    
    labels_all = []
    labels_2 = []
    labels_5 = []
    labels_10 = []
    
    for idx, (path_str, target) in enumerate(test_ds.samples):
        p = Path(path_str).resolve()
        indices_all.append(idx)
        labels_all.append(target)
        
        if p not in excluded_2:
            indices_2.append(idx)
            labels_2.append(target)
            
        if p not in excluded_5:
            indices_5.append(idx)
            labels_5.append(target)
            
        if p not in excluded_10:
            indices_10.append(idx)
            labels_10.append(target)
            
    subsets = {
        "TEST_ALL": (Subset(test_ds, indices_all), labels_all, len(excluded_all:=set())),
        "TEST_FILTERED_2": (Subset(test_ds, indices_2), labels_2, len(excluded_2)),
        "TEST_FILTERED_5": (Subset(test_ds, indices_5), labels_5, len(excluded_5)),
        "TEST_FILTERED_10": (Subset(test_ds, indices_10), labels_10, len(excluded_10))
    }
    return subsets

def get_class_stats(labels, total_all):
    count = len(labels)
    c = Counter(labels)
    if count == 0:
        return count, 0.0, 0, 0, True
        
    pct = (count / total_all) * 100
    min_c = min([c[i] for i in range(7)]) if len(c) == 7 else 0
    max_c = max(c.values())
    disappears = len(c) < 7
    return count, pct, min_c, max_c, disappears

def load_resnet50(device):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    ckpt = PROJECT_ROOT / "checkpoints" / "exp_A" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device)

def load_server_model(architecture_mode, ckpt_dir, device):
    model = LithoScanServerFusion(
        num_classes=7,
        architecture_mode=architecture_mode,
        pretrained=False,
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    )
    ckpt = PROJECT_ROOT / "checkpoints" / "phase4" / ckpt_dir / "best_model.pt"
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = PROJECT_ROOT / "reports" / "phase4" / "generalization_stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(exist_ok=True)
    
    print("Loading test dataset...")
    dcid_root = Path(os.environ.get("DCID_ROOT", PROJECT_ROOT.parent / "DCID" / "DCID"))
    test_ds = DCIDDataset(
        dcid_root=dcid_root,
        version="DCID-512-7",
        split="test",
        transform=get_eval_transforms(image_size=512)
    )
    total_test_images = len(test_ds)
    
    print("Loading Phase 3 dHash results...")
    csv_path = PROJECT_ROOT / "reports" / "phase3" / "near_duplicate_pairs.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "reports" / "phase3" / "near_duplicate_pairs.csv.gz"
    df = pd.read_csv(csv_path)
    df_dcid7 = df[df['dataset'] == 'DCID-512-7']
    
    excluded_2 = get_excluded_test_paths(df_dcid7, 2)
    excluded_5 = get_excluded_test_paths(df_dcid7, 5)
    excluded_10 = get_excluded_test_paths(df_dcid7, 10)
    
    print("Building subsets...")
    subsets_data = build_subsets(test_ds, excluded_2, excluded_5, excluded_10)
    
    loaders = {}
    class_stats = []
    
    for name, (subset, labels, num_excluded) in subsets_data.items():
        loaders[name] = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)
        c, pct, min_c, max_c, dis = get_class_stats(labels, total_test_images)
        class_stats.append({
            "Subset": name,
            "Total_Images": c,
            "Retained_Pct": f"{pct:.2f}%",
            "Excluded_Count": num_excluded,
            "Min_Class_Count": min_c,
            "Max_Class_Count": max_c,
            "Any_Class_Disappears": dis
        })
        
    pd.DataFrame(class_stats).to_csv(out_dir / "class_counts.csv", index=False)
    
    # Generate class_distribution.csv
    class_dist = []
    classes = test_ds.classes
    for i in range(7):
        c_name = classes[i]
        count_all = subsets_data["TEST_ALL"][1].count(i)
        count_2 = subsets_data["TEST_FILTERED_2"][1].count(i)
        count_5 = subsets_data["TEST_FILTERED_5"][1].count(i)
        count_10 = subsets_data["TEST_FILTERED_10"][1].count(i)
        
        ret_2 = (count_2 / count_all * 100) if count_all > 0 else 0
        ret_5 = (count_5 / count_all * 100) if count_all > 0 else 0
        ret_10 = (count_10 / count_all * 100) if count_all > 0 else 0
        
        class_dist.append({
            "class_name": c_name,
            "all": count_all,
            "filtered_2": count_2,
            "retention_2_pct": f"{ret_2:.2f}%",
            "filtered_5": count_5,
            "retention_5_pct": f"{ret_5:.2f}%",
            "filtered_10": count_10,
            "retention_10_pct": f"{ret_10:.2f}%"
        })
    pd.DataFrame(class_dist).to_csv(out_dir / "class_distribution.csv", index=False)
    
    models_to_eval = [
        ("ResNet-50", lambda: load_resnet50(device)),
        ("CNN-only", lambda: load_server_model("cnn_only", "dcid7_cnn_only", device)),
        ("CNN-Transformer", lambda: load_server_model("cnn_transformer", "dcid7_server", device))
    ]
    
    expected_acc = {
        "ResNet-50": 1.0000,
        "CNN-only": 0.9996,
        "CNN-Transformer": 0.9994
    }
    
    results = []
    
    for model_name, loader_fn in models_to_eval:
        print(f"Loading {model_name}...")
        model = loader_fn()
        
        # Pre-flight check
        model.eval()
        dummy = torch.randn(4, 3, 512, 512).to(device)
        with torch.no_grad():
            logits = model(dummy)
        if logits.shape != (4,7) or torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"FAILED PRE-FLIGHT FOR {model_name}")
            sys.exit(1)
        
        for subset_name, loader in loaders.items():
            print(f"  Evaluating on {subset_name}...")
            acc, f1, cm = evaluate_subset(model, loader, device)
            
            if subset_name == "TEST_ALL":
                exp = expected_acc[model_name]
                if abs(acc - exp) > 0.001:
                    print(f"FAILED REPRODUCTION for {model_name}: expected {exp}, got {acc}")
                    sys.exit(1)
            
            results.append({
                "Model": model_name,
                "Subset": subset_name,
                "Accuracy": acc,
                "Macro_F1": f1
            })
            
            if cm is not None:
                cm_df = pd.DataFrame(cm, index=test_ds.classes, columns=test_ds.classes)
                cm_df.to_csv(out_dir / "confusion_matrices" / f"{model_name}_{subset_name}.csv")
                
        # Free memory
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    print("Evaluation completed! Generating comparison.md...")
    
    # Generate Comparison Markdown
    df_res = pd.DataFrame(results)
    
    with open(out_dir / "comparison.md", "w") as f:
        f.write("# Phase 4D: Generalization Stress Test\n\n")
        f.write("## 1. Subset Class Composition\n\n")
        f.write(pd.DataFrame(class_stats).to_markdown(index=False) + "\n\n")
        f.write("## 2. Evaluation Results\n\n")
        
        # Pivot the table for easy viewing
        pivot_df = df_res.pivot(index='Model', columns='Subset', values='Accuracy')
        # Reorder columns
        cols = ['TEST_ALL', 'TEST_FILTERED_2', 'TEST_FILTERED_5', 'TEST_FILTERED_10']
        pivot_df = pivot_df[cols]
        f.write("### Accuracy\n")
        f.write(pivot_df.to_markdown() + "\n\n")
        
        pivot_f1 = df_res.pivot(index='Model', columns='Subset', values='Macro_F1')[cols]
        f.write("### Macro F1\n")
        f.write(pivot_f1.to_markdown() + "\n\n")
        
    print("Done.")

if __name__ == "__main__":
    main()
