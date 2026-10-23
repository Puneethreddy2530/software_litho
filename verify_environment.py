"""
verify_environment.py — LithoScan Pre-Flight and Environment Diagnostic
Checks Python, PyTorch (CUDA), Git LFS checkpoint integrity, and DCID dataset status.
"""

import sys
import os
from pathlib import Path

def main():
    print("=" * 70)
    print("  LithoScan Environment & Checkpoint Integrity Audit")
    print("=" * 70)
    
    # 1. Python version
    py_ver = sys.version.split()[0]
    print(f"[*] Python Version: {py_ver}")
    if sys.version_info < (3, 10):
        print("    [!] Warning: Python 3.10+ is recommended.")
    else:
        print("    [OK] Python version OK.")
        
    # 2. PyTorch & CUDA
    try:
        import torch
        print(f"[*] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Acceleration: {'Available (' + torch.cuda.get_device_name(0) + ')' if cuda_avail else 'CPU Only (CUDA not detected)'}")
    except ImportError:
        print("    [!] PyTorch is not installed. Run: pip install -r requirements.txt")
        return

    # 3. Streamlit & Plotly
    try:
        import streamlit
        import plotly
        print("    [OK] Streamlit and Plotly are installed.")
    except ImportError as e:
        print(f"    [!] Missing dashboard library: {e}. Run: pip install -r requirements.txt")

    # 4. Checkpoints & Git LFS Integrity
    root = Path(__file__).resolve().parent
    litho_root = root / "LithoScan"
    ckpt_dir = litho_root / "checkpoints"
    
    key_models = [
        ckpt_dir / "phase5" / "dcid35_server" / "best_model.pt",
        ckpt_dir / "phase5" / "dcid35_cnn_only" / "best_model.pt",
        ckpt_dir / "phase4" / "dcid7_server" / "best_model.pt",
        ckpt_dir / "phase4" / "dcid7_cnn_only" / "best_model.pt",
        ckpt_dir / "exp_A" / "best_model.pt",
        ckpt_dir / "exp_B" / "best_model.pt",
        ckpt_dir / "phase6_snn_pilot" / "best_model.pt",
    ]
    
    print("\n[*] Auditing Pretrained Model Checkpoints:")
    all_ckpts_ok = True
    for p in key_models:
        rel = p.relative_to(litho_root) if p.is_relative_to(litho_root) else p
        if not p.exists():
            print(f"    [X] Missing: {rel}")
            all_ckpts_ok = False
        else:
            sz_mb = p.stat().st_size / (1024 * 1024)
            if sz_mb < 0.1:  # likely an unsmudged Git LFS pointer (<1KB)
                print(f"    [!] Git LFS Pointer (not downloaded): {rel} ({sz_mb*1024:.0f} bytes)")
                print("        Run: git lfs pull")
                all_ckpts_ok = False
            else:
                print(f"    [OK] {rel} ({sz_mb:.1f} MB)")
                
    if all_ckpts_ok:
        print("    [OK] All core model weights verified and ready for inference!")

    # 5. DCID Dataset check
    print("\n[*] Auditing DCID Dataset Root:")
    dcid_env = os.environ.get("DCID_ROOT")
    candidates = [
        Path(dcid_env) if dcid_env else None,
        root / "DCID" / "DCID",
        root / "DCID",
        root / "data" / "DCID",
    ]
    dcid_found = None
    for cand in candidates:
        if cand and cand.is_dir() and (cand / "DCID-512-7").is_dir():
            dcid_found = cand
            break
            
    if dcid_found:
        print(f"    [OK] Found valid DCID root at: {dcid_found}")
    else:
        print("    [-] DCID dataset not yet detected.")
        print("        To download the 3.8 GB dataset automatically, run:")
        print("            python download_dataset.py")
        print("        Or set DCID_ROOT environment variable to your existing dataset location.")
        
    print("\n" + "=" * 70)
    print("  Audit Summary: Environment check complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
