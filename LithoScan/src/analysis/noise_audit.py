import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import hashlib
import random
from tqdm import tqdm

def md5_hash(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def audit_noise(dcid_root, clean_ver, noise_ver, out_dir):
    clean_dir = Path(dcid_root) / clean_ver
    noise_dir = Path(dcid_root) / noise_ver
    
    clean_files = sorted(list(clean_dir.rglob("*.jpg")))
    noise_files = sorted(list(noise_dir.rglob("*.jpg")))
    
    clean_rel = set([str(p.relative_to(clean_dir)) for p in clean_files])
    noise_rel = {str(p.relative_to(noise_dir)).replace("_noise_", "_"): str(p.relative_to(noise_dir)) for p in noise_files}
    
    intersection = sorted(list(clean_rel.intersection(set(noise_rel.keys()))))
    
    lines = [
        f"## {clean_ver} vs {noise_ver}",
        f"- **VERIFIED**: Clean File Count: {len(clean_files)}",
        f"- **VERIFIED**: Noise File Count: {len(noise_files)}",
        f"- **VERIFIED**: Mapped Filenames (Intersection): {len(intersection)}",
        ""
    ]
    
    if len(intersection) == 0:
        return lines
        
    gen = random.Random(42)
    sample_rels = gen.sample(intersection, min(100, len(intersection)))
    
    exact_hashes = 0
    total_abs_diff = 0
    total_changed_pixels = 0
    total_pixels = 0
    dims_match = True
    
    for rel in tqdm(sample_rels, desc=f"Analyzing {clean_ver}"):
        cp = clean_dir / rel
        np_p = noise_dir / noise_rel[rel]
        
        c_hash = md5_hash(cp)
        n_hash = md5_hash(np_p)
        if c_hash == n_hash:
            exact_hashes += 1
            
        c_img = Image.open(cp).convert("RGB")
        n_img = Image.open(np_p).convert("RGB")
        
        if c_img.size != n_img.size:
            dims_match = False
            
        c_arr = np.array(c_img).astype(np.float32)
        n_arr = np.array(n_img).astype(np.float32)
        
        diff = np.abs(c_arr - n_arr)
        total_abs_diff += diff.mean()
        
        changed = (diff > 0).any(axis=2)
        total_changed_pixels += changed.sum()
        total_pixels += changed.size
        
    mean_abs_diff = total_abs_diff / len(sample_rels)
    perc_changed = (total_changed_pixels / total_pixels) * 100
    
    lines.append(f"- **VERIFIED**: Sample Size for deep inspection: {len(sample_rels)}")
    lines.append(f"- **VERIFIED**: Exact MD5 hash matches: {exact_hashes}/{len(sample_rels)}")
    lines.append(f"- **VERIFIED**: Image dimensions remain identical: {dims_match}")
    lines.append(f"- **OBSERVED**: Mean absolute pixel difference per channel: {mean_abs_diff:.2f} (scale 0-255)")
    lines.append(f"- **OBSERVED**: Percentage of changed pixels: {perc_changed:.2f}%")
    lines.append("")
    
    vis_samples = sample_rels[:4]
    if vis_samples:
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle(f"Clean vs Noise Pairs ({clean_ver})", fontsize=16)
        
        for i, rel in enumerate(vis_samples):
            c_img = np.array(Image.open(clean_dir / rel).convert("RGB"))
            n_img = np.array(Image.open(noise_dir / noise_rel[rel]).convert("RGB"))
            diff_img = np.abs(c_img.astype(np.float32) - n_img.astype(np.float32)).mean(axis=2)
            
            axes[i, 0].imshow(c_img)
            axes[i, 0].set_title(f"Clean: {Path(rel).name}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(n_img)
            axes[i, 1].set_title(f"Noise Variant")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(diff_img, cmap='hot')
            axes[i, 2].set_title("Absolute Difference Map")
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        plt.savefig(out_dir / f"noise_sample_{clean_ver}.png")
        plt.close()
        
    return lines

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Phase 3: Noise Dataset Audit",
        "",
        "## Methodology",
        "Compared file counts, dimensions, exact MD5 hashes, and pixel-level differences between the clean dataset and the corresponding noise-injected variants. Analysis was strictly descriptive and diagnostic.",
        ""
    ]
    
    lines.extend(audit_noise(args.dcid_root, "DCID-512-7", "noise-512-7", out_dir))
    lines.extend(audit_noise(args.dcid_root, "DCID-512-35", "noise-512-35", out_dir))
    
    with open(out_dir / "noise_dataset_audit.md", "w") as f:
        f.write("\\n".join(lines))
        
    print("Noise audit complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/phase3")
    parser.add_argument("--dcid-root", default=r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")
    args = parser.parseargs() if hasattr(parser, 'parseargs') else parser.parse_args()
    main(args)
