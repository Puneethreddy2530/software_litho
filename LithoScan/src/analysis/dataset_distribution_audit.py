import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset

def analyze_dataset(dcid_root, version):
    ds_train = DCIDDataset(dcid_root=dcid_root, version=version, split="train", skip_corrupt=True)
    ds_test = DCIDDataset(dcid_root=dcid_root, version=version, split="test", skip_corrupt=True)
    
    samples = ds_train.samples + ds_test.samples
    classes = ds_train.classes
    
    data = []
    
    print(f"Analyzing {version}...")
    for path, label in tqdm(samples):
        path_obj = Path(path)
        stat = path_obj.stat()
        file_size_kb = stat.st_size / 1024
        
        try:
            with Image.open(path_obj) as img:
                w, h = img.size
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_arr = np.array(img)
                mean_rgb = img_arr.mean(axis=(0,1))
                std_rgb = img_arr.std(axis=(0,1))
                brightness = img_arr.mean()
        except Exception as e:
            continue
            
        data.append({
            "class_name": classes[label],
            "file_name": path_obj.name,
            "width": w,
            "height": h,
            "file_size_kb": file_size_kb,
            "mean_r": mean_rgb[0],
            "mean_g": mean_rgb[1],
            "mean_b": mean_rgb[2],
            "std_r": std_rgb[0],
            "std_g": std_rgb[1],
            "std_b": std_rgb[2],
            "brightness": brightness
        })
        
    return pd.DataFrame(data)

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df7 = analyze_dataset(args.dcid_root, "DCID-512-7")
    df7["dataset"] = "DCID-512-7"
    
    df35 = analyze_dataset(args.dcid_root, "DCID-512-35")
    df35["dataset"] = "DCID-512-35"
    
    df = pd.concat([df7, df35], ignore_index=True)
    df.to_csv(out_dir / "class_visual_statistics.csv", index=False)
    
    # Generate MD report
    lines = [
        "# Phase 3: Class & Dataset Distribution Audit",
        "",
        "## Methodology",
        "Calculated exact image dimensions, file size in KB, and visual statistics (Mean RGB, Std RGB, Brightness) across all files in both DCID variants.",
        "",
        "## General Verified Facts",
    ]
    
    w_unique = df["width"].unique()
    h_unique = df["height"].unique()
    if len(w_unique) == 1 and len(h_unique) == 1:
        lines.append(f"- **Dimensions**: OBSERVED that 100% of images are {w_unique[0]}x{h_unique[0]}.")
    else:
        lines.append(f"- **Dimensions**: OBSERVED dimension variance. Widths: {w_unique}, Heights: {h_unique}.")
        
    for ds_name in ["DCID-512-7", "DCID-512-35"]:
        d_df = df[df["dataset"] == ds_name]
        lines.append(f"\\n## Class Level Aggregates ({ds_name})")
        
        agg = d_df.groupby("class_name").agg({
            "file_size_kb": ["mean", "std"],
            "brightness": ["mean", "std"],
            "file_name": "count"
        }).reset_index()
        
        agg.columns = ["Class", "Size KB (Mean)", "Size KB (Std)", "Brightness (Mean)", "Brightness (Std)", "Count"]
        
        brightness_std = agg["Brightness (Mean)"].std()
        size_std = agg["Size KB (Mean)"].std()
        
        lines.append(f"\\n**OBSERVED distributional differences:**")
        lines.append(f"- Inter-class Brightness Std: {brightness_std:.2f}")
        lines.append(f"- Inter-class File Size Std: {size_std:.2f} KB\\n")
        lines.append(f"*(Note: These statistics represent observed differences and do not prove shortcut learning.)*\\n")
        
        lines.append(agg.to_markdown(index=False))
        
    with open(out_dir / "class_distribution_audit.md", "w") as f:
        f.write("\\n".join(lines))
        
    print("Dataset distribution audit complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/phase3")
    parser.add_argument("--dcid-root", default=r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")
    args = parser.parse_args()
    main(args)
