# Phase 4A Smoke Test Results (RTX 3050 Laptop / 6GB VRAM)

Date: 2026-09-14
Config: `configs/phase4a_smoke.yaml`

## Synthetic Validation
- **Gradients**: Non-zero gradients verified after synthetic backward pass.
- **Stability**: Zero NaNs, Zero Inf values detected.

## CUDA Memory Stress Test
- **Model Device**: `cuda:0`
- **Input/Target Device**: `cuda:0`
- **Batch Size**: 4
- **Gradient Accumulation**: 4
- **Execution Time (Single Step)**: 1.0755 seconds
- **CUDA Peak Allocated Memory**: 1143.04 MB
- **CUDA Total Reserved Memory**: 1280.00 MB

*The architecture fits comfortably within the strict 6.0 GB VRAM limit using `amp` and the specified gradient accumulation parameters.*

## Smoke Training Test
- **Dataset**: DCID-512-7 (Small 20-sample subsets for Smoke Test).
- **Setup**: Deterministic 90/10 split validation used (Seed 42). Official test set remained untouched.
- **Result**: Successfully executed 1 epoch.
- **Train Loss**: 1.8837
- **Validation Accuracy**: 0.1500 (Expected random baseline)

The model supports the standard PyTorch training loop without triggering OOM exceptions.
