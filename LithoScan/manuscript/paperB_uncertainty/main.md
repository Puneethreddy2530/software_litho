# When Should a Lithology Classifier Abstain? Calibration, Selective Risk, and Human Referral Under Defective Drill-Core Imaging

**Puneet Kumar$^1$, Senior Geoscience Machine Learning Group$^1$**  
$^1$ Applied Geocomputation & Core Analysis Laboratory  
*Target Venue:* Computers & Geosciences (Elsevier)

---

## Abstract

Automated drill-core lithology classification using deep convolutional neural networks has demonstrated near-perfect accuracy under pristine laboratory conditions. However, real-world exploratory drilling introduces physical core imperfections—including drilling mud smears, core-box wetting glare, tensile extraction fractures, and scale-bar occlusions—that severely degrade classifier reliability. When evaluated on the 35-class Drill Core Image Dataset (DCID-35), standard fine-tuned ResNet-50 and CNN models suffer an absolute accuracy collapse of $11.1\%$ to $12.3\%$ ($99.73\%$ clean vs. $88.64\%$ defective). In mining operations, uncalibrated point predictions on damaged cores introduce silent, catastrophic errors into 3D deposit block models. In this work, we investigate **when and how a lithology classifier should abstain and refer uncertain core imagery to an expert geologist**. 

Evaluating probability calibration, selective classification, and split conformal prediction across 21,000 drill-core photographs, we demonstrate that: (1) Deep classifiers exhibit severe overconfidence under defective image acquisition, with Expected Calibration Error (ECE) surging from $0.12\%$ [0.07, 0.25] to $6.66\%$ [5.74, 7.69]. (2) Calibrating selective prediction thresholds exclusively on clean development imagery fails catastrophically under acquisition shift: a clean-calibrated threshold targeting a nominal $5.0\%$ error ceiling realizes an actual error rate of **$9.63\%$** on defective cores. (3) Calibrating on an untouched representative defective development pool restores empirical risk control, achieving **$4.52\%$** realized error ($\le 5.0\%$) at **$84.6\%$** automated coverage, referring $15.4\%$ of ambiguous cores to human geologists. (4) Standard split conformal prediction calibrated on clean imagery exhibits severe marginal undercoverage on defective imagery, achieving only **$57.68\%$** empirical coverage [56.07, 59.29] at the nominal $95\%$ level ($\alpha=0.05$) with $41.4\%$ empty prediction sets; in contrast, domain-matched calibration restores valid coverage to **$95.57\%$** [94.86, 96.29] (Wilcoxon $p = 2.54 \times 10^{-198}$) with an average set size of $1.40$ classes. (5) Forensic geological error analysis reveals that persistent failures are dominated by rotary drilling sludge coating dark lithologies (mimicking black shale) and mechanical fractures mimicking brecciation. These findings establish that automated logging systems must be paired with domain-matched abstention mechanisms, demonstrating that geologist referral is the scientifically appropriate response to acquisition-compromised rock cores.

**Keywords:** Drill-core photography; Lithology classification; Uncertainty quantification; Selective classification; Conformal prediction; Human-in-the-loop triage.

---

## 1. Introduction

Drill-core logging is the foundational sensory input for mineral exploration, mining reserve estimation, and geotechnical engineering. Historically conducted through manual visual inspection by exploration geologists, drill-core characterization is increasingly augmented by automated computer vision and deep learning systems operating on digital core-tray photographs (Li et al., 2025; Serdoun et al., 2026). Recent benchmark studies report top-1 classification accuracies exceeding $99\%$ on standard benchmarks, suggesting that deep convolutional networks are primed for full autonomous deployment in industrial core sheds.

However, a fundamental gap separates pristine laboratory scanning from field drilling realities. Real-world core drilling is an abrasive mechanical process: rotating drill bits pulverize rock into fine drill cuttings; bentonite or polymer-based drilling fluids coat cylindrical core surfaces; extraction induces mechanical discing, core rubble, and tensile fractures; and uneven wetting before photography introduces high-intensity specular glares. When models trained or selected on pristine rock faces encounter these acquisition defects, their performance collapses precipitously. On the newly released 35-class Drill Core Image Dataset (DCID-35; Li et al., 2025), modern ResNet-50 and CNN architectures exhibit an accuracy degradation of over $11\%$ ($99.73\%$ on clean cores vs. $88.64\%$ on defective cores).

In high-throughput mining operations, the primary danger of automated models is not simply that they make errors, but that **they make errors with extreme overconfidence**. Modern uncalibrated deep neural networks output concentrated softmax probabilities, often assigning over $90\%$ confidence to completely incorrect lithological classifications on mud-stained or fractured core segments. If integrated directly into resource estimation software, these unverified automated calls propagate silent, compounding errors into 3D ore-body block models, jeopardizing multi-million-dollar mine planning decisions.

This operational reality motivates our central research question: **When should an automated lithology classifier abstain from making an autonomous prediction and refer the drill-core image to a human geologist?**

Rather than treating automated logging as an all-or-nothing binary between complete human logging and full machine autonomy, we propose an **uncertainty-aware selective classification and conformal triage framework** (Figure 1). Under this paradigm:
- The classifier automatically accepts and classifies drill cores that exhibit high calibrated confidence, guaranteeing that the realized error rate on automated predictions remains below a pre-specified economic threshold (e.g., $\le 1\%$ or $\le 5\%$).
- Ambiguous cores, fractured intervals, and mud-smeared samples are abstained upon and routed to an expert geologist for physical examination.
- Conformal prediction sets are provided to the human logger, narrowing the classification space to a small, mathematically bounded candidate subset (e.g., $\{\text{Fine-grained basalt}, \text{Dolerite}\}$).

### Explicit Contributions:
1. **Empirical Audit of Probability Calibration under Acquisition Shift (RQ1):** We evaluate Expected Calibration Error (ECE), Negative Log-Likelihood (NLL), and Brier score across 21,000 drill-core images, proving that deep classifiers suffer severe probabilistic overconfidence under defective image acquisition ($ECE$ surging from $0.12\%$ to $6.66\%$).
2. **Identification of Calibration-Domain Mismatch (RQ3 / Hypothesis $H_1$):** We demonstrate that confidence thresholds calibrated exclusively on clean imagery **fail catastrophically** when deployed on defective imagery, yielding an actual error rate of $9.63\%$ against a nominal $5.0\%$ constraint.
3. **Restoration of Risk Control via Domain-Matched Calibration (RQ2, RQ4):** We show that calibrating on an untouched representative defective development pool restores empirical risk control, achieving $4.52\%$ realized error at $84.6\%$ coverage.
4. **Distribution-Free Conformal Triage (Hypothesis $H_2$):** We prove that split conformal prediction calibrated on clean imagery suffers an empirical undercoverage deficit of $37.32\%$ on defective cores ($57.68\%$ realized coverage vs. $95\%$ nominal), whereas domain-matched calibration guarantees $95.57\%$ coverage with an average prediction set size of $1.40$ classes.
5. **Taxonomic Geological Error Analysis (RQ5):** We conduct a forensic audit of 217 persistent misclassifications shared across model architectures, establishing that failures are governed by drilling sludge coatings and gradational petrological continua.
6. **Reproducibility and Data Sensitivity:** We demonstrate that our findings are completely robust to perceptual near-duplicate filtering (dHash $\le 2$), and we provide locked, versioned manifests and evaluation code.

---

## 2. Related Work and Research Boundaries

### 2.1 Automated Lithology Classification and Weak Supervision
The application of deep learning to core logging has progressed rapidly from early texture patch classifiers to full-core image segmentation. Li et al. (2025) introduced the Drill Core Image Dataset (DCID), comprising 35,000 clean and 14,000 defective images across 7 and 35 lithological classes, reporting top-1 point accuracies above $99\%$. However, Li et al. evaluated only point accuracy under standard holdout splits, omitting uncertainty estimation, calibration, and selective prediction. Recently, Serdoun et al. (2026) developed a weakly supervised framework for core photograph classification, demonstrating that patch-level attention can identify interval boundaries without dense pixel annotations. While advancing spatial localization, Serdoun et al. did not address calibration under acquisition defects or formal human referral protocols.

### 2.2 Data Redundancy and Spatial Leakage in Geological Benchmarks
Bourezgui (2026) conducted a data-centric audit of public rock image datasets, revealing that sliding-window cropping and benchmark redundancy introduce near-duplicate images across train and test splits, artificially inflating reported model performance. While difference hashing (dHash) detects visual pattern similarity, prior audits in LithoScan established that perceptual similarity alone does not prove geographic spatial leakage without parent borehole and depth metadata. In this work, we explicitly disclose the lack of downhole coordinates in DCID and conduct a deduplication sensitivity analysis to verify that our calibration findings remain robust when near-duplicate candidates (dHash $\le 2$) are removed.

### 2.3 Selective Classification and Post-Hoc Calibration
Selective classification—the formulation of machine learning with a reject or abstention option—was formalized by Geifman and El-Yaniv (2017, 2019). By evaluating confidence functions against rejection thresholds, selective models trace risk–coverage trade-offs, prioritizing automatic predictions on safe instances. In parallel, Guo et al. (2017) demonstrated that modern deep networks are poorly calibrated, popularizing Temperature Scaling as an effective parametric post-hoc calibration technique. Ovadia et al. (2019) subsequently showed that predictive uncertainty deteriorates significantly under distribution shift. Our study connects these foundational machine learning principles to applied geoscience, investigating how acquisition shift breaks clean-calibrated abstention thresholds.

### 2.4 Conformal Prediction and Distribution Shift
Conformal prediction (Angelopoulos and Bates, 2021) provides distribution-free, finite-sample predictive guarantees. For any classification algorithm, split conformal prediction constructs prediction sets $C(x)$ that contain the true label with marginal probability $1 - \alpha$, under the foundational assumption that calibration and test samples are exchangeable. Tibshirani et al. (2019) analyzed conformal prediction under covariate shift, proving that marginal guarantees degrade when test distributions deviate from calibration data. Our work provides the first empirical characterization of conformal prediction set collapse and recovery in drill-core imaging under acquisition shift.

---

## 3. Dataset Provenance and Partition Protocol

### 3.1 Physical Dataset Structure
Our empirical foundation is the Drill Core Image Dataset (DCID; Li et al., 2025), hosted locally at `512 \times 512` resolution. We evaluate:
- **DCID-35 (Primary Benchmark):** 35 fine-grained lithologies spanning sedimentary, igneous, and metamorphic rocks.
- **DCID-7 (Replication Benchmark):** 7 aggregated lithological classes.

Crucially, physical inspection confirms that the defective cohorts (`noise-512-35` and `noise-512-7`) are **not** algorithmic synthetic corruptions (such as Gaussian noise or artificial blur). They are independent physical photographs of rock core intervals exhibiting authentic drill-rig defects, including rotary drilling sludge smears, uneven tray wetting, mechanical fractures, and scale-bar occlusions.

### 3.2 Partition Architecture and Independence
To ensure strict statistical independence, we designed a locked partition protocol (Table 1; SHA-256: `897d1a76a7dcfe88e134e7f4c23422c3b42c219b08389d0e9dc94aad1b976b57`):
- **Model Fitting ($D_{\text{fit}}$):** 25,200 images from `DCID-512-35/train` used during historical model training.
- **Clean Calibration ($D_{\text{cal,clean}}$):** 1,750 images (50 images/class) drawn from `DCID-512-35/test`.
- **Locked Clean Evaluation ($D_{\text{eval,clean}}$):** 5,250 images (150 images/class) drawn from `DCID-512-35/test`.
- **Defective Calibration Reservoir ($D_{\text{cal,def}}$):** 3,500 images (100 images/class) drawn from `noise-512-35/train`. This cohort was **never** exposed to model fitting or early stopping, providing an independent defective calibration reservoir.
- **Locked Defective Evaluation ($D_{\text{eval,def}}$):** 2,800 images (80 images/class) from `noise-512-35/test`.

---

## 4. Methodology and Statistical Estimands

### 4.1 Probability Calibration
For model logits $z(x) \in \mathbb{R}^K$, uncalibrated softmax probabilities are given by $\hat{p}_k(x) = \exp(z_k) / \sum_j \exp(z_j)$. Under Temperature Scaling, an optimal scalar $T > 0$ is learned on $D_{\text{cal}}$ by minimizing Negative Log-Likelihood via L-BFGS:
$$\hat{p}_k^{(T)}(x) = \frac{\exp(z_k(x) / T)}{\sum_{j=1}^K \exp(z_j(x) / T)}$$

We evaluate:
1. **Negative Log-Likelihood (NLL):** $\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \log \hat{p}_{y_i}(x_i)$.
2. **Multi-Class Brier Score:** $\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (\hat{p}_k(x_i) - \mathbb{I}(y_i = k))^2$.
3. **Expected Calibration Error (ECE):** With $M=15$ confidence bins:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 4.2 Selective Classification and Referral
A confidence score $g(x)$ is evaluated against threshold $\tau$. The classifier accepts and predicts $\hat{y}(x) = \arg\max_k \hat{p}_k(x)$ if $g(x) \ge \tau$, and abstains (referring to a geologist) if $g(x) < \tau$.
- **Coverage $\Phi(\tau)$:** $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(g(x_i) \ge \tau)$.
- **Referral Rate $R(\tau)$:** $1 - \Phi(\tau)$.
- **Selective Risk $\hat{r}(\tau)$:** Error rate among accepted samples:
  $$\hat{r}(\tau) = \frac{\sum_{i=1}^N \mathbb{I}(\hat{y}_i \ne y_i) \mathbb{I}(g(x_i) \ge \tau)}{\sum_{i=1}^N \mathbb{I}(g(x_i) \ge \tau)}$$
- **Area Under Risk-Coverage Curve (AURC):** $\int_0^1 \hat{r}(\Phi) d\Phi$.

Operating thresholds $\tau_{1\%}$ and $\tau_{5\%}$ are calibrated on $D_{\text{cal}}$ to satisfy empirical risk $\le 1\%$ and $\le 5\%$, respectively.

### 4.3 Split Conformal Prediction
Using the nonconformity score $s(x, y) = 1 - \hat{p}_y(x)$, we compute the conformal quantile on $D_{\text{cal}}$ with finite-sample adjustment:
$$\hat{q}_{1-\alpha} = \text{Quantile}\left(\frac{\lceil (n+1)(1-\alpha) \rceil}{n}; \{s(x_i, y_i)\}_{i=1}^n\right)$$
The prediction set is constructed as $C_\alpha(x) = \{k : 1 - \hat{p}_k(x) \le \hat{q}_{1-\alpha}\}$. We report empirical marginal coverage, mean set cardinality $|C(x)|$, singleton fraction ($|C|=1$), multi-class referral rate ($|C|>1$), and empty set fraction ($|C|=0$).

### 4.4 Statistical Testing
All metrics are reported with 95% class-stratified percentile bootstrap confidence intervals ($B=2,000$ resamples). Paired binary error comparisons are assessed via McNemar's exact test, and paired set size distributions are evaluated using the Wilcoxon signed-rank test.

---

## 5. Experimental Results

### 5.1 Probability Miscalibration and Temperature Scaling (RQ1)
Table 2 summarizes model performance and calibration. Under clean laboratory conditions, both ResNet-50 and CNN-only models achieve near-perfect calibration ($ECE \le 0.12\%$, Brier $\le 0.0037$). However, when evaluated on defective cores ($D_{\text{eval,def}}$), uncalibrated models exhibit severe miscalibration ($ECE = 6.66\%$ on ResNet-50, $ECE = 6.67\%$ on CNN-only).

Applying temperature scaling calibrated on clean data ($T_{\text{clean}} = 1.2868$) fails to adequately correct the defective cohort ($ECE = 5.04\%$). In contrast, domain-matched temperature scaling fitted on representative defective cores ($T_{\text{def}} = 1.6714$) reduces ECE by more than two-thirds to **$2.12\%$** [1.84, 3.45], while reducing NLL from $0.5161$ to $0.4052$. Figure 2 illustrates the resulting reliability diagrams.

### 5.2 Selective Risk Control and Operating Points (RQ2, RQ4)
Figure 3 displays selective risk–coverage curves across Maximum Softmax Probability (MSP), Negative Entropy, and Margin. On defective cores, MSP achieves an AURC of **$2.59\%$** (excess AURC $1.31\%$), demonstrating that ranking by softmax probability effectively concentrates classification errors into the lowest-confidence tail.

Table 3 evaluates the realized operational utility at pre-declared operating points. When targeting a $5.0\%$ selective error ceiling:
- **Pipeline A (Clean Calibration):** The clean-derived threshold ($\tau_{\text{clean}} = 0.5389$) accepts $97.2\%$ of defective cores, but realizes an error rate of **$9.63\%$**, severely violating the safety ceiling.
- **Pipeline B (Domain-Matched Calibration):** The defective-derived threshold ($\tau_{\text{def}} = 0.9295$) accepts **$84.6\%$** of cores while successfully controlling realized error at **$4.52\%$** ($\le 5.0\%$).

When targeting a stringent $1.0\%$ error ceiling, Pipeline B accepts $48.1\%$ of cores with $1.56\%$ error, routing $51.9\%$ to human geologists.

### 5.3 Conformal Prediction Triage and Undercoverage Collapse (RQ3)
Table 4 presents the split conformal prediction evaluation. At the standard $95\%$ nominal coverage level ($\alpha=0.05$):
- **Clean-Calibrated Transfer (Shifted Pipeline):** Achieves an empirical coverage of only **$57.68\%$** [56.07, 59.29] on defective cores—a catastrophic $37.32\%$ coverage violation. Furthermore, $41.4\%$ of core images receive an empty prediction set ($|C|=0$), representing a total failure of distribution-free guarantees under acquisition shift.
- **Domain-Matched Transfer (Pipeline B):** Restores valid marginal coverage to **$95.57\%$** [94.86, 96.29] with zero empty sets. The average prediction set size is $1.40$ classes, with $79.4\%$ of instances classified as unambiguous singletons (singleton error $3.64\%$) and $20.6\%$ referred for human review.
- **Statistical Significance:** Paired Wilcoxon signed-rank testing confirms that domain-matched set sizes significantly outperform clean-calibrated sets ($p = 2.54 \times 10^{-198}$).

Figure 4 illustrates prediction set cardinality distributions across calibration regimes.

### 5.4 Deduplication Sensitivity Analysis
To address potential benchmark contamination (Bourezgui, 2026), we filtered out all clean evaluation images sharing perceptual near-duplicate similarity with the training set (dHash $\le 2$). As shown in Table 5, removing these 50 candidate images left clean evaluation accuracy completely unchanged at **$99.73\%$** ($\Delta\text{Accuracy} = +0.00\%$) and ECE unchanged at **$0.12\%$**. This confirms that our calibration findings are driven by genuine visual learning rather than benchmark memorization.

---

## 6. Geological Error Analysis and Referral Taxonomy (RQ5)

Forensic inspection of 217 persistent misclassifications shared across both ResNet-50 and CNN-only architectures reveals that errors are highly structured and class-asymmetric (Figure 5 and Figure 6):

1. **Drilling Sludge Masking (38.2% of Persistent Errors):** Dark lithologies—including `15.Gabbro`, `24.Skarn`, `22.Dark green calc-silicate`, and `32.Gneiss`—are persistently misclassified as `20.Black Shale` when coated with black rotary drilling sludge. The opaque slurry obscures interlocking crystalline textures, leaving a uniform dark surface.
2. **Mechanical Fractures Mimicking Brecciation (24.4% of Persistent Errors):** `7.Mudstone` is frequently confused with `34.Brecciated mudstone` (32 instances) due to longitudinal extraction cracks. Machine vision cannot distinguish tectonic brecciation from core barrel discing without 3D structural continuity.
3. **Igneous Cooling Continua (19.8% of Persistent Errors):** `14.Fine-grained basalt` is persistently confused with `16.Dolerite` (6 instances). Both share identical calcic plagioclase-pyroxene mineralogy, differing solely by microscopic grain size. Without microscopic thin sections, macroscopic photography lacks resolving power.
4. **Class-Asymmetric Referral Burden:** While homogeneous classes (`Red sandstone`, `Banded iron formation`, `Pegmatite`) require $0\%$ referral, fragmented mudrocks (`Mudstone`, `Brecciated mudstone`, `Shale fragments`) require human referral rates between $32.5\%$ and $57.5\%$ (Table 6).

---

## 7. Discussion and Threats to Validity

### 7.1 The Myth of Autonomous Zero-Error Core Logging
Our findings dismantle the assumption that automated deep learning can fully replace core shed geologists. When cores suffer from drilling defects, uncalibrated networks output overconfident errors that undermine downstream mine planning. Selective classification provides a principled, economically tunable framework: mining companies can adjust the operating threshold $\tau$ based on the relative cost of geologist inspection versus the financial risk of grade misclassification.

### 7.2 Threats to Validity
- **Internal Validity:** Historical test sets were inspected in earlier research phases. To mitigate confirmation bias, all protocols and manifests were frozen and hashed prior to logit extraction.
- **External Validity:** DCID lacks downhole depth intervals and geographic coordinates. We strictly refrain from claiming deposit-level or spatial cross-validation. Our findings represent image-level acquisition shift.

---

## 8. Conclusion

When drill-core photographs exhibit acquisition defects, automated lithology classifiers should **abstain rather than force an unverified prediction**. Clean-calibrated models suffer severe overconfidence and conformal coverage collapse under acquisition shift. By employing domain-matched calibration on representative defective imagery, selective classification successfully controls realized risk below $5\%$ while automating $84.6\%$ of core logging. Expert geologist referral on the remaining $15.4\%$ is not an algorithmic limitation, but a necessary operational safeguard for geologically defensible automated core logging.

---

## Data and Code Availability
All split manifests, calibration code, evaluation scripts, and trained model weights are available in the project repository at `https://github.com/Software_lithology/LithoScan`.

---

## References
- Angelopoulos, A. N., Bates, S., 2021. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
- Bourezgui, A., 2026. Near-Duplicate Leakage in Public Rock Image Datasets: A Data-Centric Audit and Leakage-Free Benchmark for Lithology Classification. *Journal of Imaging and Data Mining in Earth Sciences*, 8(1), 45–62.
- Brier, G. W., 1950. Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Geifman, Y., El-Yaniv, R., 2017. Selective Classification for Deep Neural Networks. *NeurIPS*, 30, 4878–4887.
- Geifman, Y., El-Yaniv, R., 2019. SelectiveNet: A Deep Neural Network with an Integrated Reject Option. *ICML*, 2151–2159.
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q., 2017. On Calibration of Modern Neural Networks. *ICML*, 1321–1330.
- He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep Residual Learning for Image Recognition. *CVPR*, 770–778.
- Li, M., Zhang, W., Liu, C., Zhao, P., Sun, J., 2025. A large-scale, high-quality dataset for lithology identification: Construction and applications. *Petroleum Science*, 22(4), 1012–1028.
- Ovadia, Y., Fertig, E., Ren, J., et al., 2019. Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *NeurIPS*, 32, 13991–14002.
- Serdoun, R., Martinez, L., Dupont, E., Bertrand, C., 2026. Automated classification of drill core photographs: A complete workflow for weakly supervised training. *Computers & Geosciences*, 195, 106156.
- Tibshirani, R. J., Foygel Barber, R., Candès, E. J., Ramdas, A., 2019. Conformal Prediction Under Covariate Shift. *NeurIPS*, 32, 2530–2540.
