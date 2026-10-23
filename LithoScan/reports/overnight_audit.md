# Strict Overnight Experiment Audit

Date: 2026-09-14
Objective: Independent verification of the unattended LithoScan Phase 2 overnight execution.

## 1. Directory Contents Audit
**Status: VERIFIED**
- `checkpoints/exp_A/`: Contains `best_model.pt` (90.0MB), `final_model.pt`, `config_used.yaml`, `metrics.json`, `training.log`, `training_history.csv`, `hardware.txt`.
- `checkpoints/exp_B/`: Contains `best_model.pt` (90.2MB), `final_model.pt`, `config_used.yaml`, `metrics.json`, `training.log`, `training_history.csv`, `hardware.txt`.
- `reports/exp_A/`: Contains `baseline_report.md`, `classification_report.csv`, `confusion_matrix.png`, `artifact_bias_observations.md`, and the `gradcam/` subdirectory (21 images).
- `reports/exp_B/`: Contains `baseline_report.md`, `classification_report.csv`, `confusion_matrix.png`.
- `reports/exp_A_robustness/`: Contains `robustness_summary.json`, clean/noise classification CSVs, and clean/noise confusion matrices.
- `reports/exp_B_robustness/`: Contains `robustness_summary.json`, clean/noise classification CSVs, and clean/noise confusion matrices.

## 2. Configuration & Execution Script Audit
**Status: VERIFIED**
The generated `config_used.yaml` files exactly match the required parameters:
- **Exp A**: `dataset: DCID-512-7`, `epochs: 30`, `batch_size: 8`, `gradient_accumulation: 4`, `lr: 0.0001`, `optimizer: AdamW`, `seed: 42`.
- **Exp B**: `dataset: DCID-512-35`, `epochs: 20`, `patience: 5`, `batch_size: 8`, `gradient_accumulation: 4`, `lr: 0.0001`, `optimizer: AdamW`, `seed: 42`.

## 3. Exp B Constraints Audit
**Status: VERIFIED**
- **Dataset**: Log explicitly confirms `DCID-512-35`.
- **Split**: 90/10 split on the training subset generated 25,200 train images and 2,799 validation images (exactly 90/10 of the 27,999 total).
- **Leakage Prevention**: Official test set was isolated with explicit validation (`Overlap T/V: 0, T/Test: 0, V/Test: 0`).
- **Limits**: `config_used.yaml` confirms 20 maximum epochs and patience 5.

## 4. Epochs Actually Completed
**Status: VERIFIED**
According to the independently parsed `training_history.csv` files:
- **Exp A**: Completed 11 epochs before triggering early stopping (best epoch = 4).
- **Exp B**: Completed 8 epochs before triggering early stopping (best epoch = 3).

## 5. Metrics Extraction
**Status: VERIFIED**
Independently parsed from `metrics.json`:

**Exp A (DCID-512-7)**
- Best validation accuracy: 1.0000
- Best validation macro F1: 1.0000
- Final test accuracy: 1.0000
- Final test macro F1: 1.0000

**Exp B (DCID-512-35)**
- Best validation accuracy: 0.9982
- Best validation macro F1: 0.9982
- Final test accuracy: 0.9969
- Final test macro F1: 0.9969

## 6. Robustness Summary Verification
**Status: VERIFIED**
Independently parsed from `robustness_summary.json`:
- **Exp A**: Clean Test Acc = 1.0000, Noise Test Acc = 0.9679. Accuracy Drop = 0.0321 (3.21%). Macro F1 Drop = 0.0321 (3.21%).
- **Exp B**: Clean Test Acc = 0.9969, Noise Test Acc = 0.8864. Accuracy Drop = 0.1104 (11.04%). Macro F1 Drop = 0.1047 (10.47%).

## 7. Clean/Noise Class Alignment Verification
**Status: VERIFIED**
The robustness evaluation script enforced strict equality (`clean_classes == noise_classes`) before proceeding, ensuring indices mathematically align.

## 8. Checkpoint Integrity
**Status: VERIFIED**
A standalone Python script successfully instantiated `resnet50(num_classes=7)` and `resnet50(num_classes=35)` and loaded `checkpoints/exp_A/best_model.pt` and `checkpoints/exp_B/best_model.pt` respectively onto the CPU without `state_dict` key mismatches.

## 9. Confusion Matrix Verification
**Status: VERIFIED**
The final held-out test evaluation successfully generated `confusion_matrix.png` in both `reports/exp_A/` and `reports/exp_B/`. Both files exist and correctly resolve as valid 1000x800 image arrays.

## 10. System Preservation
**Status: VERIFIED**
No experimental configurations, Phase 1 datasets, or original Phase 2A outputs were modified during this audit.
