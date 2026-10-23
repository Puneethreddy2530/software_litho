import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet50
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.server_fusion import LithoScanServerFusion

def load_resnet50(device):
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 7)
    ckpt = PROJECT_ROOT / "checkpoints" / "exp_A" / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device), ckpt, sum(p.numel() for p in model.parameters())

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
    return model.to(device), ckpt, sum(p.numel() for p in model.parameters())

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_to_test = [
        ("ResNet-50", lambda: load_resnet50(device)),
        ("CNN-only", lambda: load_server_model("cnn_only", "dcid7_cnn_only", device)),
        ("CNN-Transformer", lambda: load_server_model("cnn_transformer", "dcid7_server", device))
    ]
    
    dummy_input = torch.randn(4, 3, 512, 512).to(device)
    
    for name, loader in models_to_test:
        print(f"\n--- Testing {name} ---")
        try:
            model, ckpt_path, param_count = loader()
            print(f"Checkpoint loaded successfully: {ckpt_path}")
            print(f"Parameter count: {param_count:,}")
            
            model.eval()
            with torch.no_grad():
                logits = model(dummy_input)
            
            print(f"Logits shape: {logits.shape}")
            if torch.isnan(logits).any():
                print("FAILED: NaN found in logits")
                sys.exit(1)
            if torch.isinf(logits).any():
                print("FAILED: Inf found in logits")
                sys.exit(1)
                
            if logits.shape != (4, 7):
                print(f"FAILED: Expected shape (4, 7) but got {logits.shape}")
                sys.exit(1)
                
            print("Status: PASS")
            
        except Exception as e:
            print(f"FAILED to load or run {name}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\nALL PRE-FLIGHT CHECKS PASSED")

if __name__ == "__main__":
    main()
