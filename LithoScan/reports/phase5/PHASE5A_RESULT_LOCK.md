# Phase 5A Result Lock

**Date of Result Lock:** 2026-09-16

Phase 5A results are frozen. No existing result artifact should be overwritten without an explicit new experiment/version.

## 1. Dataset Configuration
- **Dataset:** DCID-35 (512x512)
- **Official Train Split:** 27,999 samples
- **Official Test Split:** 7,000 samples
- **Train Subset (90%):** 25,200 samples
- **Validation Subset (10%):** 2,799 samples

## 2. Experimental Configurations
- **Batch Size:** 4
- **Gradient Accumulation:** 4
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Max Epochs:** 30
- **Early Stopping Patience:** 7
- **AMP:** True

## 3. Final Metrics
- **CNN-only**: 
  - Best Val F1: 0.9962 (Epoch 10)
  - Clean Test Acc/F1: 0.9977 / 0.9977
  - Noise Acc/F1: 0.8750 / 0.8811
  - Masked Acc/F1: 0.9937 / 0.9937
- **CNN-Transformer**: 
  - Best Val F1: 0.9989 (Epoch 28)
  - Clean Test Acc/F1: 0.9974 / 0.9974
  - Noise Acc/F1: 0.8893 / 0.8913
  - Masked Acc/F1: 0.9951 / 0.9951

## 4. Key Observations
- CNN-only clean test accuracy is 0.9977. CNN-Transformer clean test accuracy is 0.9974. Therefore CNN-only is 0.03 percentage points higher on clean accuracy.
- CNN-Transformer noise accuracy is 0.8893 versus 0.8750 for CNN-only. Therefore the CNN-Transformer advantage on official noise accuracy is 0.0143 = 1.43 percentage points.
- Masked accuracy difference is 0.9951 - 0.9937 = 0.0014 = 0.14 percentage points.
- For the tested DCID-35 configuration, CNN-only provides the more computationally efficient operating point, while CNN-Transformer provides modest robustness gains under the tested noise and border interventions.

## 5. Artifact Paths
- Comparison Report: `reports/phase5/phase5a_dcid35_comparison.md`
- Status Report: `reports/phase5/PHASE5A_STATUS.md`
- CNN-only Checkpoint: `checkpoints/phase5/dcid35_cnn_only/best_model.pt`
- CNN-Transformer Checkpoint: `checkpoints/phase5/dcid35_server/best_model.pt`
- CNN-only Metrics: `reports/phase5/dcid35_cnn_only/metrics.json`
- CNN-Transformer Metrics: `reports/phase5/dcid35_server/metrics.json`
- CNN-only Logs: `reports/phase5/dcid35_cnn_only/training.log`
- CNN-Transformer Logs: `reports/phase5/dcid35_server/training.log`

## 6. Important Limitations
- DCID-35 is still a benchmark dataset.
- The official noise and border interventions are specific test conditions.
- The Phase 4 similarity-filtered result was performed on DCID-7, not DCID-35.
- No conclusion from Phase 4D should be automatically transferred to DCID-35.
- Field/site generalization has not yet been demonstrated.
- No claim of universal architectural superiority should be made.
