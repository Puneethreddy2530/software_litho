# Phase 6 v1.1 — Preregistered Protocol Amendment

**Document ID:** `reports/phase6_snn_feasibility_v1_1/03_protocol_amendment.md`  
**Date:** October 2026  
**Status:** FORMAL PROTOCOL AMENDMENT (v1.1)  
**Notice of Prior Observation:** The Phase 6 bounded pilot has already been executed and observed. In accordance with open-science preregistration principles, this document serves as an **explicit, dated amendment** reconciling discrepancies between the original protocol draft (`02_preregistered_experiment_protocol.md`) and the verified implementation, without retroactive claim fabrication.

---

## 1. Summary of Protocol Reconciliations

| Dimension | Original Protocol (v1.0 Draft) | Amended Protocol (v1.1 Formal) | Methodological Justification |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | Stated as $\sim 1.15$ Million | **Exactly 390,695 Parameters** (both ConvSNN4 and ConvANN4) | Corrects draft placeholder with verified layer parameter counts (32-64-128-256 widths). |
| **Training Batch Size** | Stated as 32 | **Batch Size = 16** | Empirically required to stay within 6GB VRAM during multi-step BPTT ($T=4$) on RTX 3050. |
| **Image Interpolation** | Bicubic mentioned informally | **Explicit `InterpolationMode.BICUBIC`** | Standardizes image resizing across all loaders. |
| **Data Augmentation** | Generic vision transforms | **Geologically Justified Augmentation** | Formalizes geological validity of core transformations (see Section 2). |
| **Data Partitioning** | Random Split | **Deterministic Class-Stratified Split** | Mandates exact balance: 400 val/class (2,800 total), 360 pilot/class (2,520 total), 3,600 full/class (25,200 total). Canonical payload SHA-256: `3f04567039c7...`, normalized text SHA-256: `0b8002eb...`. |
| **Energy Claim** | Claimed 4.7x energy reduction | **Theoretical Proxy with First-Layer MAC Cost** | Replaces flawed $1 - \bar{S}$ heuristic with first-principles CMOS 45nm model; energy was not directly measured; no neuromorphic benefits claimed. |
| **Generalization Scope** | Ambiguous source claims | **Strictly Image-Level Random-Split** | DCID lacks borehole IDs; strictly prohibits calling evaluation "source-separated". |

---

## 2. Geological Justification of Data Augmentations

In automated drill-core computer vision, unconstrained image transformations can destroy critical geological structural orientation. We formally justify each permitted augmentation:

1. **Random Horizontal Flip ($p=0.5$):**  
   *Justification:* Drill-core intervals are placed in longitudinal core-box channels. The left-to-right photographic orientation depends entirely on the camera's mechanical mounting orientation relative to the core box. Horizontal inversion preserves intact textural fabrics while doubling view variety.
2. **Random Vertical Flip ($p=0.5$):**  
   *Justification:* While sedimentary younging direction is stratigraphically top-to-bottom, cylindrical drill core pieces are frequently handled, inverted, or rotated along the longitudinal core barrel axis during retrieval by drillers prior to tray placement. At the scale of $224 \times 224$ rock texture crops, macroscopic grain fabrics remain invariant under axial inversion.
3. **Mild Random Rotation ($\pm 10^\circ$):**  
   *Justification:* Simulates minor rotational misalignments of core sticks within wooden or metal core-tray channels relative to the scanning rail. Large rotations ($>15^\circ$) are prohibited as they introduce unrealistic tray border artifacts.
4. **Color Jitter (Brightness/Contrast $\pm 10\%$):**  
   *Justification:* Simulates natural variations in rig-site LED illuminance, core surface moisture (dry vs. wet core), and optical sensor exposure.

---

## 3. Statistical Testing and Hypothesis Testing Protocol

### Primary Research Hypothesis ($H_1$)
> **$H_1$:** Under matched architectural capacity (390,695 parameters) and identical input resolution ($224 \times 224$), ConvSNN4 exhibits a smaller relative macro-F1 degradation when evaluated on defective drill-core imagery compared to matched ConvANN4:
> $$\Delta F_{1,\text{rel}}(\text{ConvSNN4}) < \Delta F_{1,\text{rel}}(\text{ConvANN4}) - 0.025$$
> where $\Delta F_{1,\text{rel}} = \frac{F_{1,\text{clean}} - F_{1,\text{defective}}}{F_{1,\text{clean}}}$.

### Minimum Competence and Success Thresholds
1. **Clean-Accuracy Competence:** ConvSNN4 clean validation macro-F1 must reach at least **$0.85 \times F_{1,\text{clean}}(\text{ConvANN4})$**. If SNN cannot achieve basic task competence on clean data, robustness on noise is moot.
2. **Robustness Margin:** ConvSNN4 must exhibit at least a **2.5 percentage point advantage** in retained relative F1 ($\Delta F_{1,\text{rel}}(\text{ANN}) - \Delta F_{1,\text{rel}}(\text{SNN}) \ge 0.025$).
3. **Spike Sparsity:** Operation-weighted firing rate $\bar{S}_{\text{weighted}} \le 0.30$.

### Failure and Inconclusive Conditions
- **Falsified / Failure:** $\Delta F_{1,\text{rel}}(\text{ConvSNN4}) \ge \Delta F_{1,\text{rel}}(\text{ConvANN4})$, or clean macro-F1 is $>15$ percentage points below ConvANN4.
- **Inconclusive:** If the relative degradation difference $|\Delta F_{1,\text{rel}}(\text{SNN}) - \Delta F_{1,\text{rel}}(\text{ANN})| < 0.025$ and the 95% bootstrap confidence intervals overlap.

### Rigorous Statistical Methodology
1. **Multi-Seed Training:** 3 independent runs with deterministic seeds (`42`, `1337`, `2026`). All results reported as $\text{mean} \pm \text{standard deviation}$.
2. **Per-Image Paired McNemar Test:** Evaluated within each cohort (clean validation and natural defective validation) to determine whether model predictions differ significantly on identical samples ($p < 0.05$).
3. **Class-Stratified Bootstrap Confidence Intervals:** 1,000 bootstrap resamples of the validation set to construct non-parametric 95% confidence intervals for macro-F1.
4. **Cohort Separation Notice:** Clean and natural-defective sets are independent cohorts. Comparisons across cohorts reflect **inter-cohort distribution shift**, whereas synthetic corruptions on clean validation images reflect **paired causal corruption effects**.
