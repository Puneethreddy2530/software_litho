import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset

def generate_border_grids(dcid_root, version, out_dir):
    ds = DCIDDataset(dcid_root=dcid_root, version=version, split="test", skip_corrupt=True)
    
    import random
    gen = random.Random(42)
    
    class_samples = {c: [] for c in range(len(ds.classes))}
    
    # Randomly shuffle dataset indices for sampling
    indices = list(range(len(ds.samples)))
    gen.shuffle(indices)
    
    for i in indices:
        path, label = ds.samples[i]
        if len(class_samples[label]) < 16:
            class_samples[label].append(path)
            
    margin = 40
    version_out = out_dir / ("dcid7" if "7" in version else "dcid35")
    version_out.mkdir(parents=True, exist_ok=True)
    
    for label, paths in class_samples.items():
        if len(paths) == 0:
            continue
            
        class_name = ds.classes[label]
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f"Border/Background Visual Audit - {class_name}", fontsize=20)
        
        for ax, path in zip(axes.flatten(), paths):
            img = np.array(Image.open(path).convert("RGB").resize((512, 512)))
            
            masked = img.copy()
            masked[margin:-margin, margin:-margin, :] = 0
            
            ax.imshow(masked)
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(version_out / f"background_{class_name.replace(' ', '_')}.png")
        plt.close()
        
    print(f"Background audit grids generated for {version}")

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    generate_border_grids(args.dcid_root, "DCID-512-7", out_dir)
    generate_border_grids(args.dcid_root, "DCID-512-35", out_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/phase3/background_audit")
    parser.add_argument("--dcid-root", default=r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")
    args = parser.parse_args()
    main(args)
