# Phase 6 v1.1 — Protocol Final Check & Pre-Experiment Verification

**Document ID:** `reports/phase6_snn_feasibility_v1_1/09_protocol_final_check.md`  
**Date:** October 2026  
**Status:** COMPLETE & FORMALLY VERIFIED  
**Objective:** Perform a rigorous pre-experiment reconciliation of all scientific definitions, formulas, cohort boundaries, statistical tests, and model-selection rules prior to authorizing the full comparative campaign.

---

## 1. Explicit Cohort Definitions

| Cohort | Data Source & Partition | Nature of Cohort | Permitted Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| **Clean Validation Cohort** | `DCID-512-7/train` (400 images/class, 2,800 total) | High-quality photographic core scans without surface mud, glare, or breakage | Baseline representation competence on standard geological core |
| **Defective Validation Cohort** | `noise-512-7/train` (160 images/class, 1,120 total) | Naturally defective core scans containing fractures, mud smears, uneven lighting | **Independent sample cohort** representing natural acquisition distribution shift |
| **Controlled Synthetic Perturbations** | Clean validation images perturbed with Gaussian blur ($\sigma=2.0$), noise ($\sigma=0.05$), and 10% border occlusion | **Identical physical rock scenes** subjected to algorithmic corruptions | **Paired causal corruption analysis** (Secondary analysis) |

### Strict Cohort Boundary
- Performance difference between the clean and natural-defective cohorts is an **inter-cohort distribution shift**, NOT a paired causal perturbation effect.
- Controlled synthetic corruptions applied directly to the clean validation set serve as the **sole valid measure of paired causal corruption resistance**.

---

## 2. Relative Degradation Formula & Success Criteria

### Relative Macro-F1 Degradation Formula
$$\Delta F_{1,\text{rel}} = \frac{F_{1,\text{clean}} - F_{1,\text{defective}}}{F_{1,\text{clean}}}$$
where $F_{1,\text{clean}}$ is the macro-averaged F1 score on the clean validation cohort, and $F_{1,\text{defective}}$ is the macro-averaged F1 score on the defective validation cohort.

### Primary Comparison Hypothesis ($H_1$)
$$\Delta F_{1,\text{rel}}(\text{ConvSNN4}) < \Delta F_{1,\text{rel}}(\text{ConvANN4}) - 0.025$$

### Formal Decision Thresholds

| Decision Outcome | Quantitative Condition | Required Action |
| :--- | :--- | :--- |
| **SUPPORTED (Success)** | 1. $F_{1,\text{clean}}(\text{SNN}) \ge 0.85 \times F_{1,\text{clean}}(\text{ANN})$<br>2. $\Delta F_{1,\text{rel}}(\text{SNN}) \le \Delta F_{1,\text{rel}}(\text{ANN}) - 0.025$<br>3. $\bar{S}_{\text{weighted}} \le 0.30$ | SNN hypothesis supported; proceed to paper preparation highlighting acquisition robustness. |
| **FALSIFIED (Failure)** | $\Delta F_{1,\text{rel}}(\text{SNN}) \ge \Delta F_{1,\text{rel}}(\text{ANN})$ OR $F_{1,\text{clean}}(\text{SNN}) < F_{1,\text{clean}}(\text{ANN}) - 0.15$ | SNN hypothesis refuted; report negative finding that temporal thresholding does not enhance static rock texture robustness. |
| **INCONCLUSIVE** | $|\Delta F_{1,\text{rel}}(\text{SNN}) - \Delta F_{1,\text{rel}}(\text{ANN})| < 0.025$ AND 95% bootstrap confidence intervals overlap | Difference is statistically indistinguishable from zero; declare inconclusive. |

---

## 3. Statistical Testing Architecture

1. **Multi-Seed Training Variation:**
   - 3 deterministic training runs (`seed = 42`, `seed = 1337`, `seed = 2026`).
   - All metrics reported as $\text{mean} \pm \text{standard deviation}$ across the 3 seeds.
2. **Within-Cohort Paired Per-Image Testing:**
   - Paired McNemar test on individual image predictions within the clean validation set and within the defective validation set ($p < 0.05$).
3. **Class-Stratified Bootstrap Confidence Intervals:**
   - 1,000 bootstrap resamples stratified by class on the validation cohort to compute non-parametric 95% confidence intervals for macro-F1 and $\Delta F_{1,\text{rel}}$.

---

## 4. Locked Model Configurations & Architecture Parity

Both models share topological identity:

| Hyperparameter / Setting | ConvSNN4 | Matched ConvANN4 |
| :--- | :--- | :--- |
| **Input Resolution** | $224 \times 224 \times 3$ (`InterpolationMode.BICUBIC`) | $224 \times 224 \times 3$ (`InterpolationMode.BICUBIC`) |
| **Channel Progression** | 3 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 | 3 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 |
| **Total Parameters** | **390,695** | **390,695** |
| **Neuron / Activation** | Multi-Step LIF ($\tau=2.0$, $V_{\text{th}}=1.0$, FastSigmoid) | Standard ReLU |
| **Readout Decoding** | Floating-point logit averaging across $T=4$ steps | Direct linear logits ($T=1$ step) |
| **Optimizer & Schedule** | AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), Cosine Annealing | AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), Cosine Annealing |
| **Batch Size & AMP** | Batch Size 16, AMP enabled | Batch Size 16, AMP enabled |

---

## 5. Model Selection & Test Isolation Rule

- **STRICT PROHIBITION:** The official test sets (`DCID-512-7/test` and `noise-512-7/test`) must **NEVER** be used to select architectures, tune hyperparameters, adjust thresholds, or choose optimal training epochs.
- **Model Selection Protocol:** The best model checkpoint is chosen exclusively based on the peak macro-F1 achieved on the **validation partition** during training.
- Official test evaluation is reserved strictly for final, locked benchmarking after all training and architecture decisions are frozen.

---

## 6. Protocol Verification Verdict

$$\mathbf{PROTOCOL\ FINAL\ CHECK:\ PASS}$$

All definitions, formulas, statistical tests, and experimental boundaries are locked without ambiguity.
