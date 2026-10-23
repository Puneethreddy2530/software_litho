# 03: CODE AND METHODOLOGY FORENSIC AUDIT — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Methodology Auditor  
**Scope:** Complete line-by-line inspection of `src/models/`, `src/scripts/`, `src/data/`, `src/evaluation/`, `src/calibration/`, and `tests/`.

---

## 1. Learning-Rate Scheduler & Optimization Dynamics

### Source Inspection
Examined [train_baseline.py:L244-245, L365](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/models/train_baseline.py#L244-L245) and [train_server.py:L203-209, L256](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/scripts/train_server.py#L203-L209):

```python
# train_server.py:
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

for epoch in range(config["training"]["epochs"]):
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        ...
        if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    scheduler.step() # <-- OUTSIDE batch loop, ONCE per epoch
```

### Forensic Finding
- **Scheduler Cadence:** `scheduler.step()` is placed **outside** the batch iteration loop and executes exactly **once per epoch**.
- **T_max Consistency:** `T_max` is initialized to the total number of epochs (`T_max=30` in Phase 4/5; `T_max=20` or `30` in Phase 2). Because `scheduler.step()` fires once per epoch, the actual step frequency exactly matches `T_max`.
- **Verdict:** **NO PER-BATCH SCHEDULER BUG.** The learning rate undergoes a canonical cosine decay from $1.0\times 10^{-4}$ down to near zero over the full planned epoch budget.
- **Subtle Gotcha (Early Stopping with CosineAnnealingLR):** Because early stopping halts training before `epochs` is reached (e.g., stopping at epoch 17 of 30 for CNN-only on DCID-35, or epoch 8 of 20 for Exp B), the model never reaches the tail of the cosine schedule where learning rates drop below $10^{-5}$. This is standard behavior when combining early stopping with fixed-epoch cosine schedules, but means models terminated at different points along the cosine curve.

---

## 2. Gradient Accumulation and Automatic Mixed Precision (AMP)

### Control Flow
Examined [train_server.py:L236-254](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/scripts/train_server.py#L236-L254):
```python
loss = criterion(logits, labels)
loss = loss / grad_accum
scaler.scale(loss).backward()
...
if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
train_loss += loss.item() * grad_accum
```

### Forensic Findings
1. **Loss Scaling:** Loss is correctly divided by `grad_accum` before `.backward()`, ensuring the accumulated gradient represents the true batch average.
2. **Final Incomplete Accumulation Group:**
   - In **Phase 4 and Phase 5** (`batch_size=4`, `grad_accum=4`, `train_samples=25,200`):
     $$\frac{25,200}{4} = 6,300 \text{ batches}; \quad \frac{6,300}{4} = 1,575 \text{ accumulation cycles (remainder 0)}.$$
     Every single accumulation group is exactly full.
   - In **Phase 2 (Exp A and B)** (`batch_size=8`, `grad_accum=4`, `train_samples=25,200`):
     $$\frac{25,200}{8} = 3,150 \text{ batches}; \quad 3,150 \pmod 4 = 2 \text{ batches in final step}.$$
     On the final batch of each epoch in Phase 2, only 2 batches contributed to the gradient step while loss was divided by 4, scaling that single terminal step's gradient down by $0.5\times$. Severity is negligible ($1$ step out of $788$ per epoch).
3. **Effective Batch Size Inconsistency Across Phases:**
   - Phase 2 (ResNet Baseline): $8 \times 4 = \mathbf{32}$.
   - Phase 4 & Phase 5 (CNN-only and CNN-Transformer): $4 \times 4 = \mathbf{16}$.
   - Effective batch size was halved in Phases 4 and 5 to prevent OOM on the 6GB RTX 3050. While technically necessary, comparing models trained with effective batch size 32 vs 16 introduces an uncontrolled hyperparameter confounder that was not declared in downstream comparative reports.

---

## 3. Dataset Splitting, Determinism, and Checkpoint Selection

### Split Verification
Examined [train_baseline.py:L66-88](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/models/train_baseline.py#L66-L88) and [train_server.py:L126-177](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/scripts/train_server.py#L126-L177):
- Both scripts use an identical deterministic random seed (`seed=42`) applied to `torch.Generator()` on the official 28,000 (DCID-7) and 27,999 (DCID-35) training pools.
- Independently verified in execution that `train_subset` (25,200 samples) and `val_subset` (2,799 / 2,800 samples) contain **the exact same image indices** across all architectures.
- Both scripts enforce assertions confirming `len(train_paths & val_paths) == 0`, `len(train_paths & test_paths) == 0`, and `len(val_paths & test_paths) == 0`.
- **Transforms Separation:** `train_ds` applies data augmentation (`get_train_transforms`), whereas `val_ds` applies deterministic evaluation transforms (`get_eval_transforms`). This was implemented correctly.

### Checkpoint Selection & Test Isolation
- **Selection Metric:** `train_server.py` saves `best_model.pt` strictly on improvement of validation macro-F1 (`val_f1`).
- **Test Set Isolation:** The official test set (`test_ds`) is strictly loaded and evaluated **after** the training loop has finished and the best checkpoint has been re-loaded (`model.load_state_dict(torch.load(ckpt_dir / "best_model.pt"))`).
- **Verdict:** **NO LEAKAGE INTO MODEL SELECTION.** Test set metrics did not leak into training or checkpoint selection decisions.

---

## 4. Model Architecture & Complexity Deconstruction

### Source Inspection
Examined [server_fusion.py:L10-100](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/models/server_fusion.py#L10-L100):

```python
class LithoScanServerFusion(nn.Module):
    # Backbone: torchvision.models.resnet50(weights=IMAGENET1K_V1)
    # Layer 4 spatial map: [B, 2048, 16, 16] (at 512x512 input)
    # Projection: Conv2d(2048, 512, kernel_size=1) -> [B, 512, 16, 16]
    # Positional Embedding: Parameter(torch.randn(1, 256, 512))
    # Transformer: nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1), num_layers=2)
    # Pooling: transformed_tokens.mean(dim=1) -> [B, 512]
    # Classifier: Linear(512, num_classes)
```

### Architectural Reality vs. Claimed Novelty

| Property | Claimed in Documentation | Actual Code Reality | Scientific Novelty Status |
| :--- | :--- | :--- | :--- |
| **Texture Awareness** | "Texture-aware CNN + Transformer fusion" | Standard ResNet-50 backbone. No Gabor filters, no Local Binary Patterns, no multi-scale texture pyramids, no second branch. | **FABRICATED MARKETING LABEL** |
| **Backbone** | Proprietary feature extractor | Off-the-shelf Torchvision ResNet-50 truncated before average pooling. | **CONVENTIONAL ADAPTATION** |
| **Attention Module** | Advanced spatial fusion | Standard PyTorch `nn.TransformerEncoder` with 2 layers, 8 heads, $d=512$. | **STANDARD COMPONENT** |
| **Positional Encoding** | Custom spatial encoding | Standard learnable 1D Gaussian parameter tensor broadcast across the 256 patch sequence. | **STANDARD PRACTICE** |
| **Token Aggregation** | Multi-head pooling | `tokens.mean(dim=1)` (Simple global average pooling over the 256 tokens). | **CONVENTIONAL** |

### Parameter and Compute Accounting

| Architecture | Classes | Total Parameters | Trainable Parameters | Peak VRAM | Training Time | Clean Test Acc | Noise Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 (Baseline)** | 7 | 23,522,375 | 23,522,375 | ~1,340 MB | 8,225 s (~2.3 h) | **1.0000** | **0.9679** |
| **CNN-only** | 7 | 24,560,711 | 24,560,711 | 1,346 MB | 18,339 s (~5.1 h) | 0.9996 | 0.9600 |
| **CNN-Transformer** | 7 | 30,996,551 | 30,996,551 | 1,503 MB | 20,532 s (~5.7 h) | 0.9994 | 0.9664 |
| **ResNet-50 (Baseline)** | 35 | 23,582,691 | 23,582,691 | ~1,340 MB | 6,089 s (~1.7 h) | 0.9969 | 0.8864 |
| **CNN-only** | 35 | 24,575,075 | 24,575,075 | 1,346 MB | 12,215 s (~3.4 h) | **0.9977** | 0.8750 |
| **CNN-Transformer** | 35 | 31,010,915 | 31,010,915 | 1,504 MB | 23,874 s (~6.6 h) | 0.9974 | **0.8893** |

**Forensic Architecture Verdict:**
The CNN-Transformer adds **6,435,840 parameters (+26%)**, increases peak VRAM by ~158 MB, and nearly **doubles training time** over CNN-only (and quadruples training time over the ResNet baseline). In return, it is **0.03% worse than CNN-only on clean data**, and only **0.29% better than standard ResNet-50 on noisy data**. There is no scientific basis to claim architectural superiority.

---

## 5. Preprocessing & Augmentation Geological Critique

Examined [dcid_dataset.py:L44-60](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/src/data/dcid_dataset.py#L44-L60):
```python
transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Domain Discrepancies
1. **Vertical Flip in Borehole Photography:**  
   In cylindrical drill cores extracted from boreholes, the vertical axis represents stratigraphic depth (younging up / downhole sequence). Grading in turbidites, cross-bedding orientations, and sedimentary polarity are directional. Applying `RandomVerticalFlip(p=0.5)` physically inverts the stratigraphic vector. While acceptable for isotropic igneous rocks (granite, basalt), it is scientifically dubious for layered sedimentary facies (siltstone, shale, banded iron formation).
2. **Color Jitter (Hue Shift):**  
   Lithological naming frequently relies on subtle oxidation states (e.g., distinguishing *1.Red sandstone* from *2.Light sandstone*, or *20.Black Shale* from *21.Red shale*). Jittering hue by 0.05 risks shifting pixel values across color-diagnostic boundary definitions.

---

## 6. Audit of the Existing Software Test Suite

Examined [tests/test_dataset.py:L1-480](file:///C:/Users/punee/Desktop/Software_lithology/LithoScan/tests/test_dataset.py#L1-L480):
- **Coverage:** Tests check directory existence, class naming integrity (7 and 35 classes), ImageNet tensor normalization range, disjointness of file path strings, and `SplitViolationError` raising on invalid split names.
- **Scientific Limitation:** The test suite verifies **only data ingestion plumbing**.
- **What is NOT tested:**
  - Zero tests verify model forward passes or gradients.
  - Zero tests verify metric calculation correctness.
  - Zero tests verify loss functions or early stopping logic.
  - Zero tests verify calibration or uncertainty.
  - Zero tests verify geological validity or spatial group independence.
A 100% pass rate in `tests/test_dataset.py` provides no assurance of scientific validity.
