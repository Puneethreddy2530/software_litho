# 01: CANONICAL EXPERIMENT INVENTORY — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Independent Research Auditor & Computer Vision Reviewer  
**Methodology:** Independent inspection of local filesystem checkpoints, training logs, CSV histories, and JSON metrics.

---

## 1. Inventory Summary

This canonical inventory tracks all experimental executions across Phases 1 through 5A. Each entry is evaluated against strict evidentiary standards:
- **Verified from source artifacts**: Physical checkpoint (`.pt`), raw log (`.log`), epoch history (`.csv`), and output metric artifact (`.json`) are present, non-empty, and mutually consistent.
- **Partially verified**: Checkpoint and logs exist, but one or more peripheral metrics (e.g., intervention metrics) were not generated.
- **Interrupted/incomplete**: Execution was stopped before convergence or target epochs; artifacts represent a partial state.
- **Contradictory**: Downstream summary reports assert values that conflict with raw training logs or source JSONs.
- **Summary-only/unverified**: Documented in markdown reports without corroborating checkpoint or training log.
- **Missing**: Claimed in project documentation but completely unexecuted.

---

## 2. Canonical Experiment Table

| Experiment ID | Phase | Dataset | Architecture | Training Status | Checkpoint File | Validation Metrics | Clean Test Metrics | Noise Metrics | Intervention Metrics | Runtime | Key Evidence Paths | Trust Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Phase 1 | DCID-7 / DCID-35 | N/A (Reconnaissance) | Completed | N/A | N/A | N/A | N/A | N/A | ~45 s | `data/manifests/dcid_manifest.csv`, `reports/dataset_report.md` | **Verified from source artifacts** |
| **EXP-02 (Exp A)** | Phase 2A | DCID-512-7 | ResNet-50 (Pretrained, Full Fine-tune) | Completed (Epoch 4 best, stopped Ep 11) | `checkpoints/exp_A/best_model.pt` (94.4 MB) | Val Acc: 1.0000<br>Val F1: 1.0000 | Test Acc: 1.0000<br>Test F1: 1.0000 | Clean Acc: 1.0000<br>Noise Acc: 0.9679<br>Drop: 0.0321 | N/A (Tested in Phase 4C) | 8,224.76 s (~2.28 h) | `checkpoints/exp_A/metrics.json`<br>`checkpoints/exp_A/training.log`<br>`reports/exp_A_robustness/robustness_summary.json` | **Verified from source artifacts** |
| **EXP-03 (Exp B)** | Phase 2B | DCID-512-35 | ResNet-50 (Pretrained, Full Fine-tune) | Completed (Epoch 3 best, stopped Ep 8) | `checkpoints/exp_B/best_model.pt` (94.6 MB) | Val Acc: 0.9982<br>Val F1: 0.9982 | Test Acc: 0.9969<br>Test F1: 0.9969 | Clean Acc: 0.9969<br>Noise Acc: 0.8864<br>Drop: 0.1104 | N/A (Not tested with border mask) | 6,089.36 s (~1.69 h) | `checkpoints/exp_B/metrics.json`<br>`checkpoints/exp_B/training.log`<br>`reports/exp_B_robustness/robustness_summary.json` | **Verified from source artifacts** *(Note: Misrepresented in Phase 5A)* |
| **EXP-04** | Phase 3 | DCID-512-7 | Frozen ResNet-50 + Logistic Regression | Completed (100 iters, conv. warning) | None saved | Val Acc: 0.9996<br>Val F1: 0.9996 | Test Acc: 0.9991<br>Test F1: 0.9991 | N/A | N/A | ~180 s | `reports/phase3/linear_baseline_metrics.csv`<br>`reports/phase3/linear_probe_test_DCID-512-7.csv` | **Verified from source artifacts** |
| **EXP-05** | Phase 3 | DCID-512-35 | Frozen ResNet-50 + Logistic Regression | Completed (100 iters, conv. warning) | None saved | Val Acc: 0.9885<br>Val F1: 0.9882 | Test Acc: 0.9928<br>Test F1: 0.9928 | N/A | N/A | ~420 s | `reports/phase3/linear_baseline_metrics.csv`<br>`reports/phase3/linear_probe_test_DCID-512-35.csv` | **Verified from source artifacts** *(Source of 0.9928 error)* |
| **EXP-06** | Phase 3 | DCID-7 / DCID-35 | dHash (64-bit Perceptual Audit) | Completed | N/A | N/A | N/A | N/A | Near-duplicate pairs table | ~950 s | `reports/phase3/near_duplicate_pairs.csv`<br>`reports/phase3/near_duplicate_audit.md` | **Verified from source artifacts** |
| **EXP-07** | Phase 4B | DCID-512-7 | CNN-only (ResNet-50 backbone + 1x1 conv + GAP) | Completed (Epoch 16 best, stopped Ep 23) | `checkpoints/phase4/dcid7_cnn_only/best_model.pt` | Val Acc: 1.0000<br>Val F1: 1.0000 | Test Acc: 0.9996<br>Test F1: 0.9996 | Noise Acc: 0.9600<br>Noise F1: 0.9605<br>Drop: 0.0396 | Masked Acc: 0.9960<br>Masked F1: 0.9960<br>Drop: 0.0036 | 18,338.87 s (~5.09 h) | `reports/phase4/dcid7_cnn_only/metrics.json`<br>`reports/phase4/dcid7_cnn_only/training.log`<br>`reports/phase4/dcid7_cnn_only/confusion_matrix.csv` | **Verified from source artifacts** |
| **EXP-08** | Phase 4B | DCID-512-7 | CNN-Transformer (ResNet-50 + 2-layer ViT Encoder) | Completed (Epoch 16 best, stopped Ep 23) | `checkpoints/phase4/dcid7_server/best_model.pt` | Val Acc: 1.0000<br>Val F1: 1.0000 | Test Acc: 0.9994<br>Test F1: 0.9994 | Noise Acc: 0.9664<br>Noise F1: 0.9667<br>Drop: 0.0330 | Masked Acc: 0.9984<br>Masked F1: 0.9984<br>Drop: 0.0010 | 20,532.48 s (~5.70 h) | `reports/phase4/dcid7_server/metrics.json`<br>`reports/phase4/dcid7_server/training.log`<br>`reports/phase4/dcid7_server/confusion_matrix.csv` | **Verified from source artifacts** |
| **EXP-09** | Phase 4C | DCID-512-7 | ResNet-50 Baseline (Inference Intervention) | Completed | `checkpoints/exp_A/best_model.pt` | N/A | Clean Acc: 1.0000<br>Clean F1: 1.0000 | N/A | Masked Acc: 0.9994<br>Masked F1: 0.9994<br>Drop: 0.0006 | ~110 s | `reports/phase4/dcid7_ablation/resnet50_intervention.json` | **Verified from source artifacts** |
| **EXP-10** | Phase 4D | DCID-512-7 | ResNet-50 / CNN-only / CNN-Transformer | Completed | All 3 Phase 4A/B checkpoints | N/A | Retained Acc at dHash $\le 10$:<br>ResNet: 1.0000<br>CNN-only: 0.9995<br>Trans: 0.9993 | N/A | Test filtered subsets: $k \in \{2, 5, 10\}$ | ~480 s | `reports/phase4/generalization_stress_test/metrics.csv`<br>`reports/phase4/generalization_stress_test/class_counts.csv` | **Verified from source artifacts** |
| **EXP-11 (Archived)** | Phase 5A | DCID-512-35 | CNN-Transformer (ResNet-50 + 2-layer ViT Encoder) | Interrupted (Killed during Epoch 4) | `checkpoints/phase5/archive_dcid35_server_interrupted_epoch4/best_model.pt` | Val Acc: 0.9914<br>Val F1: 0.9910 | N/A | N/A | N/A | ~9,914 s (partial) | `reports/phase5/archive_dcid35_server_interrupted_epoch4/training.log` | **Interrupted/incomplete** *(Archived cleanly)* |
| **EXP-12 (Archived)** | Phase 5A | DCID-512-35 | CNN-only (ResNet-50 backbone + 1x1 conv + GAP) | Interrupted (Stopped after Epoch 5) | `checkpoints/phase5/archive_dcid35_cnn_only_interrupted_epoch5/best_model.pt` | Val Acc: 0.9904<br>Val F1: 0.9899 | N/A | N/A | N/A | ~3,674 s (partial) | `reports/phase5/archive_dcid35_cnn_only_interrupted_epoch5/training.log` | **Interrupted/incomplete** *(Archived cleanly)* |
| **EXP-13** | Phase 5A | DCID-512-35 | CNN-only (ResNet-50 backbone + 1x1 conv + GAP) | Completed (Epoch 10 best, stopped Ep 17) | `checkpoints/phase5/dcid35_cnn_only/best_model.pt` (98.6 MB) | Val Acc: 0.9964<br>Val F1: 0.9962 | Test Acc: 0.9977<br>Test F1: 0.9977 | Noise Acc: 0.8750<br>Noise F1: 0.8811<br>Drop: 0.1227 | Masked Acc: 0.9937<br>Masked F1: 0.9937<br>Drop: 0.0040 | 12,215.48 s (~3.39 h) | `reports/phase5/dcid35_cnn_only/metrics.json`<br>`reports/phase5/dcid35_cnn_only/training.log`<br>`reports/phase5/dcid35_cnn_only/confusion_matrix.csv` | **Verified from source artifacts** |
| **EXP-14** | Phase 5A | DCID-512-35 | CNN-Transformer (ResNet-50 + 2-layer ViT Encoder) | Completed (Epoch 28 best, ran all 30 eps) | `checkpoints/phase5/dcid35_server/best_model.pt` (124.3 MB) | Val Acc: 0.9989<br>Val F1: 0.9989 | Test Acc: 0.9974<br>Test F1: 0.9974 | Noise Acc: 0.8893<br>Noise F1: 0.8913<br>Drop: 0.1081 | Masked Acc: 0.9951<br>Masked F1: 0.9951<br>Drop: 0.0023 | 23,874.27 s (~6.63 h) | `reports/phase5/dcid35_server/metrics.json`<br>`reports/phase5/dcid35_server/training.log`<br>`reports/phase5/dcid35_server/confusion_matrix.csv` | **Verified from source artifacts** |
| **EXP-15** | Phase 5B / Field | DCID-512-35 | SNN (Spiking Neural Network) | Unexecuted | None | N/A | N/A | N/A | N/A | 0 s | `src/models/field_snn.py` (empty stub) | **Missing** |
| **EXP-16** | Federated | DCID-512-35 | Federated Byzantine Simulation | Unexecuted | None | N/A | N/A | N/A | N/A | 0 s | `src/federated/aggregation.py` (empty stub) | **Missing** |
| **EXP-17** | Calibration | DCID-512-35 | Conformal Prediction Triage | Unexecuted | None | N/A | N/A | N/A | N/A | 0 s | `src/calibration/conformal.py` (empty stub) | **Missing** |

---

## 3. Forensic Deductions on Completed Runs

1. **Separation of Interrupted from Authoritative Runs:**
   - Both interrupted DCID-35 runs (EXP-11 and EXP-12) were moved into isolated subdirectories (`archive_dcid35_server_interrupted_epoch4` and `archive_dcid35_cnn_only_interrupted_epoch5`).
   - The authoritative runs (EXP-13 and EXP-14) were executed in clean directories with fresh logs, histories, and checkpoints. The reported training durations (12,215 s and 23,874 s) correspond strictly to the completed runs and correctly exclude the interrupted wall-clock time.

2. **Run Convergence Dynamics:**
   - On DCID-35, CNN-only triggered early stopping at epoch 17 (best epoch 10, patience 7).
   - On DCID-35, CNN-Transformer ran all 30 epochs without early stopping; its best validation F1 occurred at epoch 28 (val F1 = 0.998904), with epochs 29 and 30 registering slight declines (0.998509 and 0.998134).
