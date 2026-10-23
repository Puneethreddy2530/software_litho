# 02: Preregistered Experimental & Statistical Protocol — LithoScan Paper B

**Document ID:** `reports/paperB_uncertainty/02_preregistered_protocol.md`  
**Date:** October 2026  
**Status:** LOCKED & PREREGISTERED PRIOR TO EVALUATION  
**Research Track:** Paper B (*When Should a Lithology Classifier Abstain? Calibration, Selective Risk, and Human Referral Under Defective Drill-Core Imaging*)  
**Protocol Version:** 1.0  
**Configuration Root:** `configs/paperB_uncertainty/`  
**Manifest Hashes:**  
- `paperB_split_manifest_dcid35.json`: `897d1a76a7dcfe88e134e7f4c23422c3b42c219b08389d0e9dc94aad1b976b57`  
- `paperB_split_manifest_dcid7.json`: `7378969ee7c87ed77d2b8bf1c81edcfd373e865a05dbe5873abd317a79e5aee7`  

---

## 1. Experimental Scope & Architectural Baselines

### 1.1 Benchmark Scopes
- **Primary Benchmark (DCID-35):** 35 fine-grained lithological classes at $512 \times 512$ resolution. This represents a complex geological classification challenge where historical models exhibit a 12.3% performance collapse on defective cores.
- **Replication Benchmark (DCID-7):** 7 coarse lithological classes at $512 \times 512$ resolution, testing whether calibration and selective risk dynamics transfer across classification granularities.

### 1.2 Frozen Model Checkpoints
To prevent post-hoc architectural tuning or evaluation leakage, all models are evaluated strictly with frozen weights:
1. **ResNet-50 (`exp_B` for DCID-35, `exp_A` for DCID-7):** Standard torchvision ResNet-50 fine-tuned end-to-end from ImageNet weights.
   - Checkpoint (DCID-35): `checkpoints/exp_B/best_model.pt` (25,557,091 parameters).
   - Checkpoint (DCID-7): `checkpoints/exp_A/best_model.pt` (25,557,091 parameters).
2. **CNN-Only Bottleneck (`Phase 5A` for DCID-35, `Phase 4` for DCID-7):** ResNet-50 backbone with a 512-dimensional bottleneck projection (`LithoScanServerFusion`, mode `cnn_only`).
   - Checkpoint (DCID-35): `checkpoints/phase5/dcid35_cnn_only/best_model.pt` (24,575,075 parameters).
   - Checkpoint (DCID-7): `checkpoints/phase4/dcid7_cnn_only/best_model.pt` (24,575,075 parameters).

---

## 2. Partition Architecture & Data Independence

All evaluation cohorts are completely disjoint from calibration cohorts. Crucially, the defective calibration cohort is extracted from an untouched development pool that was never used during historical training.

### 2.1 Primary Benchmark Partition (DCID-35)
| Partition Code | Source Directory | Selection Rule | Total Images | Images/Class | Role in Protocol | Historical Exposure |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **$D_{\text{fit}}$** | `DCID-512-35/train` | Indices `:25200` | 25,200 | 720 | Model fitting | Historical (Phase 2/5A) |
| **$D_{\text{sel}}$** | `DCID-512-35/train` | Indices `25200:` | 2,799 | ~80 | Checkpoint selection | Historical (Phase 2/5A) |
| **$D_{\text{cal,clean}}$** | `DCID-512-35/test` | First 50/class (seed 2026) | **1,750** | 50 | In-distribution clean calibration | Retrospective benchmark |
| **$D_{\text{eval,clean}}$** | `DCID-512-35/test` | Remaining 150/class | **5,250** | 150 | Locked clean evaluation | Retrospective benchmark |
| **$D_{\text{cal,def}}$** | `noise-512-35/train` | First 100/class (seed 2026) | **3,500** | 100 | Domain-matched defective calibration | **Pristine / Untouched** |
| **$D_{\text{eval,def}}$** | `noise-512-35/test` | Full test cohort | **2,800** | 80 | Locked defective evaluation | Retrospective benchmark |
| **$D_{\text{dev,res}}$** | `noise-512-35/train` | Remaining 220/class | 7,700 | 220 | Reserved for future exploration | Pristine / Untouched |

### 2.2 Replication Benchmark Partition (DCID-7)
| Partition Code | Source Directory | Selection Rule | Total Images | Images/Class | Role in Protocol |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **$D_{\text{cal,clean}}$** | `DCID-512-7/test` | First 250/class (seed 2026) | **1,750** | 250 | Clean calibration |
| **$D_{\text{eval,clean}}$** | `DCID-512-7/test` | Remaining 750/class | **5,250** | 750 | Clean evaluation |
| **$D_{\text{cal,def}}$** | `noise-512-7/train` | First 400/class (seed 2026) | **2,800** | 400 | Defective calibration |
| **$D_{\text{eval,def}}$** | `noise-512-7/test` | Full test cohort | **2,800** | 400 | Defective evaluation |

---

## 3. Mathematical Formulations & Estimands

### 3.1 Probability Calibration
For model logits $z(x) \in \mathbb{R}^K$, uncalibrated softmax probabilities are:
$$\hat{p}_k(x) = \frac{\exp(z_k(x))}{\sum_{j=1}^K \exp(z_j(x))}$$
Under **Temperature Scaling**, a scalar temperature $T > 0$ is learned on the calibration partition $D_{\text{cal}}$ by minimizing the Negative Log-Likelihood (NLL) via L-BFGS:
$$\hat{p}_k^{(T)}(x) = \frac{\exp(z_k(x) / T)}{\sum_{j=1}^K \exp(z_j(x) / T)}, \quad T^* = \arg\min_T -\frac{1}{|D_{\text{cal}}|} \sum_{i \in D_{\text{cal}}} \log \hat{p}_{y_i}^{(T)}(x_i)$$

Metrics reported:
1. **Negative Log-Likelihood (NLL):** $\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \log \hat{p}_{y_i}(x_i)$
2. **Brier Score:** $\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (\hat{p}_k(x_i) - \mathbb{I}(y_i = k))^2$
3. **Expected Calibration Error (ECE):** Grouping predictions into $M=15$ confidence bins $B_m \subset (0, 1]$:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   Both uniform-width ($1/15$) and adaptive quantile-frequency binning are computed.

### 3.2 Selective Classification (Abstention & Geologist Referral)
A confidence score $g(x) \in \mathbb{R}$ is evaluated against an acceptance threshold $\tau$. The classifier predicts $\hat{y}(x) = \arg\max_k \hat{p}_k(x)$ if $g(x) \ge \tau$, and abstains (refers sample to a human geologist) if $g(x) < \tau$.

Confidence scoring functions evaluated:
1. **Maximum Softmax Probability (MSP):** $g_{\text{MSP}}(x) = \max_k \hat{p}_k(x)$
2. **Predictive Entropy (ENT):** $g_{\text{ENT}}(x) = \sum_{k=1}^K \hat{p}_k(x) \log \hat{p}_k(x)$ (abstaining on high entropy)
3. **Margin (MAR):** $g_{\text{MAR}}(x) = \hat{p}_{(1)}(x) - \hat{p}_{(2)}(x)$

Key selective classification estimands:
- **Coverage $\Phi(\tau)$:** Fraction of samples accepted:
  $$\Phi(\tau) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(g(x_i) \ge \tau)$$
- **Referral Rate $R(\tau)$:** Fraction of samples abstained / referred to human:
  $$R(\tau) = 1 - \Phi(\tau)$$
- **Selective Risk $\hat{r}(\tau)$:** Classification error rate among accepted samples:
  $$\hat{r}(\tau) = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \ne y_i) \mathbb{I}(g(x_i) \ge \tau)}{\sum_{i=1}^N \mathbb{I}(g(x_i) \ge \tau)}$$
- **Area Under Risk-Coverage Curve (AURC):** Integrated empirical risk across coverage levels $\Phi \in [0, 1]$:
  $$\text{AURC} = \int_0^1 \hat{r}(\Phi) d\Phi$$
- **Predeclared Operating Points:** Thresholds $\tau_{1\%}$ and $\tau_{5\%}$ are calibrated on $D_{\text{cal}}$ such that empirical risk $\le 1\%$ and $\le 5\%$, respectively. The realized selective risk and realized coverage are subsequently evaluated on $D_{\text{eval}}$.

### 3.3 Split Conformal Prediction
Given calibration partition $D_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^n$ and nonconformity score:
$$s(x, y) = 1 - \hat{p}_y(x)$$
For a target miscoverage level $\alpha \in \{0.01, 0.05, 0.10\}$, the conformal quantile is defined with finite-sample correction:
$$\hat{q}_{1-\alpha} = \text{Quantile}\left(\frac{\lceil (n+1)(1-\alpha) \rceil}{n}; \{s(x_i, y_i)\}_{i=1}^n\right)$$
The prediction set for a test instance $x$ is:
$$C_\alpha(x) = \{k \in \{1, \dots, K\} : 1 - \hat{p}_k(x) \le \hat{q}_{1-\alpha}\}$$

Conformal estimands reported:
1. **Empirical Marginal Coverage:** $\hat{C}_\alpha = \frac{1}{M} \sum_{j=1}^M \mathbb{I}(y_j \in C_\alpha(x_j))$
2. **Mean Set Size:** $\bar{S}_\alpha = \frac{1}{M} \sum_{j=1}^M |C_\alpha(x_j)|$
3. **Singleton Fraction:** Fraction of evaluation samples where $|C_\alpha(x)| = 1$ (unambiguous automatic calls).
4. **Singleton Error Rate:** Error rate among singleton predictions.
5. **Multi-Class Referral Rate:** Fraction of evaluation samples where $|C_\alpha(x)| > 1$ (ambiguous cores requiring geologist review).
6. **Empty Set Rate:** Fraction of samples where $|C_\alpha(x)| = 0$.

---

## 4. Preregistered Hypotheses & Statistical Tests

### 4.1 Primary Research Hypothesis ($H_1$)
> **Hypothesis $H_1$ (Calibration-Domain Mismatch):**  
> Under defective drill-core image acquisition ($D_{\text{eval,def}}$), a selective referral pipeline calibrated on clean images ($D_{\text{cal,clean}}$) fails to achieve its target selective risk, whereas a domain-matched pipeline calibrated on representative defective images ($D_{\text{cal,def}}$) successfully controls selective risk at nominal levels ($\le 1\%$ and $\le 5\%$) and yields significantly higher valid coverage.

**Primary Estimand:** Realized selective risk $\hat{r}(\tau^*)$ and realized coverage $\Phi(\tau^*)$ on $D_{\text{eval,def}}$ under Clean Calibration vs. Defective Calibration.

### 4.2 Secondary Conformal Hypothesis ($H_2$)
> **Hypothesis $H_2$ (Conformal Coverage Under Acquisition Shift):**  
> Split conformal prediction calibrated on clean images ($D_{\text{cal,clean}}$) will exhibit significant empirical undercoverage ($\hat{C}_{0.05} < 0.95$) when evaluated on the defective cohort ($D_{\text{eval,def}}$) due to covariate/acquisition shift, whereas domain-matched conformal prediction calibrated on $D_{\text{cal,def}}$ will satisfy $\hat{C}_{0.05} \ge 0.95$ within exchangeable bounds.

### 4.3 Statistical Testing & Uncertainty Quantification
1. **Bootstrap Confidence Intervals:** All summary metrics (Accuracy, ECE, NLL, Brier, AURC, Coverage at target risk, Conformal coverage, Mean set size) are reported with 95% class-stratified percentile bootstrap confidence intervals computed over $B = 2,000$ resamples.
2. **Paired Image-Level Tests:**
   - For binary error comparisons on accepted samples: McNemar's exact test.
   - For continuous prediction set size differences: Two-sided Wilcoxon signed-rank test.
3. **P-Value Reporting Standards:** Exact $p$-values are reported down to $p < 0.001$. No claims of $p = 0.000$ are permitted.

---

## 5. Execution Protocol & Verification Lock

1. Split manifests generated deterministically and SHA-256 hashed.
2. Scripts will extract logits on frozen checkpoints with no retraining.
3. Operating points will be determined exclusively on $D_{\text{cal}}$ before applying to $D_{\text{eval}}$.
4. Deduplicated sensitivity analysis will be computed excluding dHash $\le 2$ candidate pairs to evaluate potential near-duplicate bias.
5. All findings will be documented faithfully in subsequent reports regardless of whether hypotheses are confirmed or rejected.
