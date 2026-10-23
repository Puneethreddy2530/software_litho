import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset

def compute_dhash_bytes(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    return np.packbits(diff.flatten())

def get_hashes_for_loader(dcid_root, version, split):
    ds = DCIDDataset(dcid_root=dcid_root, version=version, split=split, image_size=512, skip_corrupt=True)
    paths = []
    hashes = []
    
    print(f"Computing hashes for {version} {split}...")
    for i in tqdm(range(len(ds))):
        path = ds.samples[i][0]
        h = compute_dhash_bytes(path)
        if h is not None:
            paths.append(path)
            hashes.append(h)
    
    return np.array(paths), np.array(hashes)

def find_similar_pairs(h1, h2, paths1, paths2, name1, name2, max_threshold=10):
    print(f"Comparing {name1} vs {name2}...")
    chunk_size = 1000
    similar_pairs = []
    
    for i in tqdm(range(0, len(h2), chunk_size)):
        h2_chunk = h2[i:i+chunk_size]
        xor = np.bitwise_xor(h1[:, None, :], h2_chunk[None, :, :])
        unpacked = np.unpackbits(xor, axis=2)
        distances = unpacked.sum(axis=2)
        
        y, x = np.where(distances <= max_threshold)
        for yi, xi in zip(y, x):
            similar_pairs.append({
                "split_1": name1,
                "split_2": name2,
                "path_1": str(paths1[yi]),
                "path_2": str(paths2[i + xi]),
                "distance": distances[yi, xi]
            })
            
    return similar_pairs

def audit_dataset(dcid_root, version, out_dir):
    train_paths, train_hashes = get_hashes_for_loader(dcid_root, version, "train")
    test_paths, test_hashes = get_hashes_for_loader(dcid_root, version, "test")
    
    gen = torch.Generator().manual_seed(42)
    total_len = len(train_paths)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    
    indices = torch.randperm(total_len, generator=gen).numpy()
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]
    
    train_split_paths = train_paths[train_idx]
    train_split_hashes = train_hashes[train_idx]
    
    val_split_paths = train_paths[val_idx]
    val_split_hashes = train_hashes[val_idx]
    
    results = []
    
    results.extend(find_similar_pairs(train_split_hashes, val_split_hashes, train_split_paths, val_split_paths, "Train", "Validation", max_threshold=10))
    results.extend(find_similar_pairs(train_split_hashes, test_hashes, train_split_paths, test_paths, "Train", "Test", max_threshold=10))
    results.extend(find_similar_pairs(val_split_hashes, test_hashes, val_split_paths, test_paths, "Validation", "Test", max_threshold=10))
    
    return results

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    res7 = audit_dataset(args.dcid_root, "DCID-512-7", out_dir)
    for r in res7: r["dataset"] = "DCID-512-7"
    all_results.extend(res7)
    
    res35 = audit_dataset(args.dcid_root, "DCID-512-35", out_dir)
    for r in res35: r["dataset"] = "DCID-512-35"
    all_results.extend(res35)
    
    df = pd.DataFrame(all_results)
    if len(df) > 0:
        df.to_csv(out_dir / "near_duplicate_pairs.csv", index=False)
    
    lines = [
        "# Phase 3: Perceptual Similarity Audit",
        "",
        "## Methodology",
        "- **Algorithm**: Perceptual dHash (Difference Hash, 64-bit)",
        "- **Thresholds Reported**: 0 (exact match), <=2 (very-high similarity), <=5 (high similarity), <=10",
        "",
        "> **Note on Train vs Validation**: This comparison is purely diagnostic. Because the validation set is deterministically derived from the official train directory, similarity here does not constitute test-set leakage.",
        "",
        "## Results",
        ""
    ]
    
    if len(df) == 0:
        lines.append("**VERIFIED**: Zero pairs found at distance <= 10 across all splits and datasets.")
    else:
        for ds_name in ["DCID-512-7", "DCID-512-35"]:
            d_df = df[df["dataset"] == ds_name]
            lines.append(f"### {ds_name}")
            
            for split_pair in [("Train", "Validation"), ("Train", "Test"), ("Validation", "Test")]:
                s_df = d_df[(d_df["split_1"] == split_pair[0]) & (d_df["split_2"] == split_pair[1])]
                
                c_0 = len(s_df[s_df["distance"] == 0])
                c_2 = len(s_df[s_df["distance"] <= 2])
                c_5 = len(s_df[s_df["distance"] <= 5])
                c_10 = len(s_df[s_df["distance"] <= 10])
                
                lines.append(f"#### {split_pair[0]} vs {split_pair[1]}")
                lines.append(f"- Exact perceptual match (dist == 0): {c_0}")
                lines.append(f"- Very-high similarity (dist <= 2): {c_2}")
                lines.append(f"- High similarity (dist <= 5): {c_5}")
                lines.append(f"- Total pairs (dist <= 10): {c_10}")
                
                if split_pair[1] == "Test" and c_10 > 0:
                    lines.append("")
                    lines.append(f"**OBSERVED**: Potential cross-boundary similarity between {split_pair[0]} and {split_pair[1]}.")
                    
                lines.append("")
        
        lines.append("### Closest Examples (Distance 0)")
        exact = df[df["distance"] == 0].head(10)
        for i, row in exact.iterrows():
            lines.append(f"- `{Path(row['path_1']).name}` ({row['split_1']}) <--> `{Path(row['path_2']).name}` ({row['split_2']})")
    
    with open(out_dir / "near_duplicate_audit.md", "w") as f:
        f.write("\\n".join(lines))
        
    print("Near duplicate audit complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/phase3")
    parser.add_argument("--dcid-root", default=r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")
    args = parser.parse_args()
    main(args)
