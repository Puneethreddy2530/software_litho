# Phase 4D: Generalization Stress Test

## 1. Subset Class Composition

| Subset           |   Total_Images | Retained_Pct   |   Excluded_Count |   Min_Class_Count |   Max_Class_Count | Any_Class_Disappears   |
|:-----------------|---------------:|:---------------|-----------------:|------------------:|------------------:|:-----------------------|
| TEST_ALL         |           7000 | 100.00%        |                0 |              1000 |              1000 | False                  |
| TEST_FILTERED_2  |           6892 | 98.46%         |              108 |               947 |              1000 | False                  |
| TEST_FILTERED_5  |           6297 | 89.96%         |              703 |               689 |               999 | False                  |
| TEST_FILTERED_10 |           4155 | 59.36%         |             2845 |               115 |               932 | False                  |

## 2. Evaluation Results

### Accuracy
| Model           |   TEST_ALL |   TEST_FILTERED_2 |   TEST_FILTERED_5 |   TEST_FILTERED_10 |
|:----------------|-----------:|------------------:|------------------:|-------------------:|
| CNN-Transformer |   0.999429 |          0.99942  |          0.999365 |           0.999278 |
| CNN-only        |   0.999571 |          0.999565 |          0.999524 |           0.999519 |
| ResNet-50       |   1        |          1        |          1        |           1        |

### Macro F1
| Model           |   TEST_ALL |   TEST_FILTERED_2 |   TEST_FILTERED_5 |   TEST_FILTERED_10 |
|:----------------|-----------:|------------------:|------------------:|-------------------:|
| CNN-Transformer |   0.999429 |          0.999423 |          0.999386 |           0.999298 |
| CNN-only        |   0.999571 |          0.999561 |          0.999485 |           0.999027 |
| ResNet-50       |   1        |          1        |          1        |           1        |

## 3. Final Interpretation

**Outcome Category:** ROBUST GENERALIZATION

**Rationale:**
Performance remains approximately perfectly stable across all similarity-filtered subsets, even as the evaluation becomes extremely strict. At the `TEST_FILTERED_10` threshold, nearly 41% of the total test set is excluded, including almost 89% of all Basalt test images (dropping from 1,000 to 115). Despite this massive distortion in class composition and the aggressive removal of perceptually similar samples, the ResNet-50 baseline retained a perfect 1.0000 Accuracy, while the CNN-only and CNN-Transformer models dropped by less than 0.0003 Accuracy respectively. 

This indicates that the networks have genuinely learned robust distinguishing features for the lithology classes and exhibit virtually zero generalization sensitivity to perceptual similarity up to a dHash threshold of 10.
