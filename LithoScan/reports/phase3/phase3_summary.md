# Phase 3: Dataset Difficulty, Shortcut, and Generalization Audit

Date: 2026-09-14
Objective: Diagnostic analysis of the unusually high baseline accuracy on DCID-7 (1.00) and DCID-35 (0.99) to identify underlying causes, shortcuts, or artifacts.

## 1. VERIFIED
- **Phase 1 Manifest**: `dcid_manifest.csv` correctly logged 97,999 rows (18.66 MB).
- **Integrity Status**: Dataset path/existence and current filesystem contents were read-only checked; no historical immutability snapshot was available.
- **Checkpoint Verification**: Both 7-class and 35-class baseline checkpoints loaded cleanly for inference verification.
- **Noise File-Level Observations**: Validated that clean and noise subsets map 1-to-1 via intersection (14,000 files). Image dimensions remain identical between clean and noise subsets.

## 2. OBSERVED
- **Noise Sampled Observations**: In a sample of 100 image pairs (clean vs noise), there were zero exact MD5 hash matches. The noise injection altered 99.98% of pixels with a mean absolute pixel difference per channel of ~36-38 (scale 0-255). *(Note: This observation on the 100-pair sample is not generalized as an absolute verification of the noise mechanism for all 14,000 images).*
- **Perceptual Similarity Audit**: 
  - **DCID-7**: 
    - Train vs Validation: 7 exact matches, 153 (dist<=2), 4,037 (dist<=5), 113,754 total similar pairs (dist<=10). *(Note: similarity here is expected because the validation set was derived from the training pool).*
    - Train vs Test: 18 exact matches, 492 (dist<=2), 11,537 (dist<=5), 292,375 total similar pairs (dist<=10).
    - Validation vs Test: 3 exact matches, 40 (dist<=2), 1,176 (dist<=5), 33,287 total similar pairs (dist<=10).
  - **DCID-35**:
    - Train vs Validation: 0 exact matches, 60 (dist<=2), 2,357 (dist<=5), 71,428 total similar pairs (dist<=10).
    - Train vs Test: 9 exact matches, 203 (dist<=2), 6,209 (dist<=5), 170,803 total similar pairs (dist<=10).
    - Validation vs Test: 0 exact matches, 20 (dist<=2), 600 (dist<=5), 18,569 total similar pairs (dist<=10).
  - *(Note: Cross-boundary similarity between train/validation and the official test set is treated as a diagnostic concern, not definitive proof of leakage).*
- **Linear Probe Audit**: 
  - **Method**: Logistic Regression (`solver='saga'`, `max_iter=100`, `n_jobs=-1`) on 2048-dimensional frozen ResNet-50 ImageNet features. Fitting completed but encountered convergence warnings ("max_iter was reached which means the coef_ did not converge") after 100 iterations.
  - **DCID-7**: Train Acc 0.9998 / Train Macro-F1 0.9998, Val Acc 0.9996 / Val Macro-F1 0.9996, Test Acc 0.9991 / Test Macro-F1 0.9991.
  - **DCID-35**: Train Acc 0.9988 / Train Macro-F1 0.9988, Val Acc 0.9885 / Val Macro-F1 0.9882, Test Acc 0.9928 / Test Macro-F1 0.9928.
- **Dataset Distribution**: Measurable inter-class distribution differences were observed. DCID-7 has an Inter-class Brightness Std of 25.25. DCID-35 has an Inter-class Brightness Std of 36.17. *(Note: File-size and brightness variance is not evidence of shortcut learning by itself).*
- **Qualitative Border/Background Audit**: Based on visually audited sample grids, prominent background elements like rulers, color bars, text, and standardized borders are visible in various classes. *(Note: This observation does not prove artifact bias globally across the dataset).*

## 3. HYPOTHESIS
- The baseline ResNet-50 may be leveraging the high linear separability of background artifacts and perceptual similarities across splits (shortcut learning) to artificially boost performance, rather than generalizing on complex geological features.

## 4. PENDING
- **Recommended Next Experiment (Phase 4)**: Phase 4 should test whether the proposed CNN-Transformer Server architecture improves classification while remaining robust under controlled artifact/background interventions.
- Execution of SNN Field Edition (Phase 5).
