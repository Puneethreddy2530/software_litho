# Phase 5A: DCID-35 Model Comparison

## 1. Overview
This report provides a scientifically cautious, three-way comparison between the existing Phase 2 ResNet-50 baseline, the newly trained CNN-only architecture, and the CNN-Transformer architecture on the DCID-35 (512px) geological dataset.

## 2. Experimental Setup
- **Training Epochs**: Max 30, Early Stopping Patience 7
- **Resolution**: 512x512
- **Interventions**: 
  - Standard noise application (`noise-512-35`)
  - 40px zero-padded border masking to test reliance on edge artifacts

## 3. Results Comparison

### Three-Way Summary Table
| Model | Best Val F1 | Clean Test Acc | Clean Test F1 | Noise Acc | Noise F1 | Noise Drop | Masked Acc | Masked F1 | Mask Drop | Parameters | Peak Memory | Training Time | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.9882 | 0.9928 | 0.9928 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CNN-only | 0.9962 | 0.9977 | 0.9977 | 0.8750 | 0.8811 | 0.1227 | 0.9937 | 0.9937 | 0.0040 | 24,575,075 | 1346 MB | 12215 s | 10 |
| CNN-Transformer | 0.9989 | 0.9974 | 0.9974 | 0.8893 | 0.8913 | 0.1081 | 0.9951 | 0.9951 | 0.0023 | 31,010,915 | 1504 MB | 23874 s | 28 |

## 4. Computational Trade-Off Analysis
- **Parameter Increase**: CNN-Transformer introduces 6,435,840 additional parameters compared to CNN-only.
- **Peak Memory Increase**: CNN-Transformer peak memory increased by 158 MB relative to CNN-only.
- **Training Time Increase**: CNN-Transformer training time increased by 11,659 seconds (~3.2 hours) relative to CNN-only.
- **Clean Test Accuracy Difference**: CNN-only clean test accuracy is 0.9977. CNN-Transformer clean test accuracy is 0.9974. Therefore CNN-only is 0.03 percentage points higher on clean accuracy.
- **Noise Test Accuracy Difference**: CNN-Transformer noise accuracy is 0.8893 versus 0.8750 for CNN-only. Therefore the CNN-Transformer advantage on official noise accuracy is 0.0143 = 1.43 percentage points.
- **Masked Accuracy Difference**: Masked accuracy difference is 0.9951 - 0.9937 = 0.0014 = 0.14 percentage points (a small difference).

## 5. Interpretation
- **Overall Clean Performance**: The observed clean test accuracy across all models remains exceptionally high (>99%). 
- **Robustness and Degradation**: Under the tested noise conditions, both new architectures experienced a substantial accuracy drop (~10.8% to 12.3%). The CNN-Transformer advantage on official noise accuracy is 1.43 percentage points.
- **Artifact Sensitivity**: When the 40px border intervention was applied, both architectures maintained >99% accuracy. The masked accuracy difference is 0.14 percentage points.
- **Deployment Interpretation**: For the tested DCID-35 configuration, CNN-only provides the more computationally efficient operating point, while CNN-Transformer provides modest robustness gains under the tested noise and border interventions.

## 6. Important Limitations
- DCID-35 is still a benchmark dataset.
- The official noise and border interventions are specific test conditions.
- The Phase 4 similarity-filtered result was performed on DCID-7, not DCID-35.
- No conclusion from Phase 4D should be automatically transferred to DCID-35.
- Field/site generalization has not yet been demonstrated.
- No claim of universal architectural superiority should be made.
