# Phase 5A Status

Overall status: PASS

## CNN-Transformer
- training status: Completed (Early stopping not triggered before reaching late epochs, max 30)
- best epoch: 28
- best validation accuracy: 0.9989
- best validation Macro-F1: 0.9989
- clean test accuracy/F1: 0.9974 / 0.9974
- noise test accuracy/F1: 0.8893 / 0.8913
- noise degradation: 0.1081
- masked accuracy/F1: 0.9951 / 0.9951
- mask degradation: 0.0023
- training time: 23874 seconds (~6.6 hours)
- peak memory: 1504 MB
- parameters: 31,010,915

## CNN-only
- training status: Completed (Early stopping triggered at Epoch 17)
- best epoch: 10
- best validation accuracy: 0.9964
- best validation Macro-F1: 0.9962
- clean test accuracy/F1: 0.9977 / 0.9977
- noise test accuracy/F1: 0.8750 / 0.8811
- noise degradation: 0.1227
- masked accuracy/F1: 0.9937 / 0.9937
- mask degradation: 0.0040
- training time: 12215 seconds (~3.4 hours)
- peak memory: 1346 MB
- parameters: 24,575,075

## Three-way comparison
- **ResNet-50**: Val F1 0.9882, Test F1 0.9928.
- **CNN-only**: Val F1 0.9962, Test F1 0.9977 (Noise F1 0.8811, Masked F1 0.9937)
- **CNN-Transformer**: Val F1 0.9989, Test F1 0.9974 (Noise F1 0.8913, Masked F1 0.9951)

## Main findings
The experiments demonstrate that both customized architectures successfully fit the DCID-35 dataset to >99% clean test accuracy. The CNN-Transformer observed slightly less degradation under noise and border masking interventions, suggesting a marginal robustness benefit. However, the CNN-only model trained significantly faster (due to earlier convergence at Epoch 10), used fewer parameters, and achieved an equivalent clean test accuracy.

## Artifact inventory
- Comparison Report: `reports/phase5/phase5a_dcid35_comparison.md`
- CNN-Transformer Checkpoint: `checkpoints/phase5/dcid35_server/best_model.pt`
- CNN-only Checkpoint: `checkpoints/phase5/dcid35_cnn_only/best_model.pt`
- CNN-Transformer Metrics: `reports/phase5/dcid35_server/metrics.json`
- CNN-only Metrics: `reports/phase5/dcid35_cnn_only/metrics.json`
- CNN-Transformer Log: `reports/phase5/dcid35_server/training.log`
- CNN-only Log: `reports/phase5/dcid35_cnn_only/training.log`
- CNN-Transformer CM: `reports/phase5/dcid35_server/confusion_matrix.csv`
- CNN-only CM: `reports/phase5/dcid35_cnn_only/confusion_matrix.csv`

## Research recommendation
Based strictly on the observed results, **CNN-only** should be preferred for the server-side LithoScan pipeline. While the CNN-Transformer exhibited a ~1.4% advantage in noise robustness, the CNN-only model converges much faster (Epoch 10 vs 28), requires fewer parameters, and performs equally well on clean data. The minor robustness gain of the Transformer does not strongly justify its heavily increased training cost for this specific dataset configuration.
