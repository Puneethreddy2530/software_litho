# Phase 4C: DCID-7 Ablation Comparison

## 1. Experimental Results

| Metric | ResNet-50 Baseline | CNN-only (Ablation) | CNN-Transformer |
| :--- | :--- | :--- | :--- |
| **Clean Accuracy** | 1.0000 | 0.9996 | 0.9994 |
| **Official Noise Variant** | 0.9679 | 0.9600 | 0.9664 |
| **Border Mask Accuracy** | 0.9994 | 0.9960 | 0.9984 |
| **Parameter Count** | 23.5 M | 24.5 M | 31.0 M |
| **Training Time** | N/A | ~5.1 hrs (18339 s) | ~5.7 hrs (20532 s) |
| **Peak CUDA Memory**| N/A | 1346 MB | 1503 MB |

## 2. Pairwise Deltas

**Clean Accuracy**
* CNN-Transformer minus ResNet-50 = -0.0006
* CNN-only minus ResNet-50 = -0.0004
* CNN-Transformer minus CNN-only = -0.0002

**Noise Accuracy**
* CNN-Transformer minus ResNet-50 = -0.0015
* CNN-only minus ResNet-50 = -0.0079
* CNN-Transformer minus CNN-only = +0.0064

**Masked Accuracy (40px Border)**
* CNN-Transformer minus ResNet-50 = -0.0010
* CNN-only minus ResNet-50 = -0.0034
* CNN-Transformer minus CNN-only = +0.0024

## 3. Scientific Interpretation

**Transformer Utility:** 
The addition of the Transformer encoder adds ~6.5M parameters, increases training time by roughly 12%, and slightly increases memory footprint. In terms of clean performance, the Transformer provides no measurable benefit on DCID-7, slightly underperforming both the CNN-only model and the ResNet-50 baseline by microscopic margins (< 0.001). However, the Transformer *does* exhibit a measurable advantage in noise robustness (+0.0064 accuracy over CNN-only) and border-mask robustness (+0.0024 over CNN-only), suggesting the self-attention mechanism may marginally aid in localizing uncorrupted spatial features when the image is degraded.

**CNN-only Capability:**
The CNN-only architecture performs exceptionally well, achieving a near-perfect validation score (1.0000 F1) and clean test score (0.9996). It performs similarly to both other models on clean data, verifying that the spatial patch extraction backbone is highly capable of separating the lithology classes on its own.

**Sensitivity to Border Intervention:**
All three architectures demonstrated low sensitivity to the specific tested 40px border mask intervention. The ResNet-50 baseline dropped by only 0.0006, the CNN-only by 0.0036, and the CNN-Transformer by 0.0010. While this does *not* prove the models primarily rely on interior geological texture (they could be using shortcuts outside the 40px region), it conclusively shows that none of the models are destructively reliant on the specific outer 28.8% margin of the image.

## 4. Recommendation

**DO NOT PROCEED TO DCID-35 YET.**

**Rationale:**
The Phase 4 CNN-Transformer architecture fails to demonstrate a meaningful, generalizable research contribution beyond the existing ResNet-50 baseline on DCID-7. The baseline outperforms the CNN-Transformer on clean data, noisy data, and border-masked data. Furthermore, the CNN-Transformer requires more parameters, memory, and compute. 

Because the DCID-7 results show that the Transformer fusion does not yield a clear methodological leap, launching an expensive DCID-35 training run using this exact architecture is scientifically unjustified. The architecture or the problem statement must be iterated upon first (e.g., investigating why ResNet-50 is so robust, or redesigning the fusion block) before scaling up to 35 classes.
