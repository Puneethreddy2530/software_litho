# LithoScan — Overnight Execution Summary

Generated: 2026-09-14 03:45 (Local Time)

## Executive Summary
The unattended overnight session successfully completed both the Phase 2A (DCID-512-7) and Phase 2B (DCID-512-35) baseline experiments without interruption. All required evaluations—including clean/noise robustness degradation and Grad-CAM visual analyses—were completed. The scheduled jobs utilized approximately 4 hours of the 10-hour available window, completing well within safety margins.

---

## 1. Experiments Completed

### 🟢 Phase 2A: ResNet-50 on DCID-512-7 (Exp A)
- **Status**: ✅ Completed
- **Dataset Size**: 35,000 images (7 classes)
- **Best Epoch**: 4 (Early stopping triggered at Epoch 11)
- **Best Validation Accuracy**: 1.0000
- **Best Validation Macro F1**: 1.0000
- **Final Held-Out Test Accuracy**: 1.0000
- **Final Held-Out Test Macro F1**: 1.0000
- **Training Time**: 8224.76s (~2.28 hours)
- **OOM Events**: 0

### 🟢 Phase 2B: ResNet-50 on DCID-512-35 (Exp B)
- **Status**: ✅ Completed
- **Dataset Size**: 34,999 images (35 classes)
- **Best Epoch**: 3 (Early stopping triggered at Epoch 8)
- **Best Validation Accuracy**: 0.9982
- **Best Validation Macro F1**: 0.9982
- **Final Held-Out Test Accuracy**: 0.9969
- **Final Held-Out Test Macro F1**: 0.9969
- **Training Time**: 6089.36s (~1.69 hours)
- **OOM Events**: 0

### 🟢 Architecture Preparation
- **Status**: ✅ Completed
- **Details**: Basic class skeletons created for:
  - `src/models/server_fusion.py`
  - `src/models/field_snn.py`
  - `src/calibration/conformal.py`
  - `src/federated/aggregation.py`

*(No experiments were marked as "partially completed" or aborted)*

---

## 2. Robustness (Clean vs. Noise Degradation)

The baseline checkpoints were evaluated simultaneously against the `DCID-test` (clean) and `noise-test` sets to quantify real-world degradation.

| Experiment | Clean Test Acc | Noise Test Acc | Accuracy Drop | F1 Drop |
|------------|----------------|----------------|---------------|---------|
| **Exp A (7-class)** | 100.00% | 96.79% | **3.21%** | **3.21%** |
| **Exp B (35-class)** | 99.69% | 88.65% | **11.04%** | **10.47%** |

*Note: The degradation is notably sharper when operating in the 35-class regime, confirming the need for conformal triage or architecture improvements moving forward.*

---

## 3. Qualitative Artifact Observations

Qualitative Grad-CAM analysis (visualizing the final convolutional layer of Phase 2A) revealed no immediate, catastrophic artifact learning. 
- **Genuine Feature Usage**: Heatmaps successfully outline geological boundaries and rock textures.
- **Background Filtering**: The model accurately distinguishes core sample boundaries from background.
- **Artifacts**: Minor attention localized around physical cracks, but standard human-made artifacts (scale bars) are generally ignored.
*(For details, see `reports/exp_A/artifact_bias_observations.md`)*

---

## 4. Hardware and Performance

- **Environment**: NVIDIA GeForce RTX 3050 (6GB VRAM) Laptop GPU
- **OOM Events Detected & Handled**: 0 (The conservative batch size of 8 with gradient accumulation of 4 successfully stabilized memory)
- **Total GPU Training Time**: ~3.97 hours (14,314s)

## 5. Remaining Work

Now that the reproducible Baselines are completely established on both the 7-class and 35-class datasets, the subsequent phases can begin:
1. **Server Edition Development**: Build and train the actual Texture-aware CNN + Transformer Fusion (`server_fusion.py`).
2. **Field Edition Development**: Implement and train the lightweight SNN (`field_snn.py`).
3. **Conformal Prediction**: Train the calibration layer to calculate dynamic thresholds.
4. **Federated Learning**: Validate the Byzantine-robust aggregation strategy.
