# 05: LITERATURE & NOVELTY INVESTIGATION MATRIX — LITHOSCAN

**Audit Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Computer Vision Reviewer  
**Methodology:** Direct verification of peer-reviewed articles, published DOIs, Crossref records, and conference proceedings via active academic web search.

---

## 1. The Five Closest Related Studies (Deep Comparative Analysis)

### 1. Li et al. (2025) — The DCID Benchmark Paper
- **Full Citation:** Jia-Yu Li, Ji-Zhou Tang, Xian-Zheng Zhao, Bo Fan, Wen-Ya Jiang, Shun-Yao Song, Jian-Bing Li, Kai-Da Chen, Zheng-Guang Zhao (2025). *A large-scale, high-quality dataset for lithology identification: Construction and applications.* **Petroleum Science**, ISSN: 1995-8226. DOI: [10.1016/j.petsci.2025.04.013](https://doi.org/10.1016/j.petsci.2025.04.013).
- **Dataset Used:** DCID (98,000 images, 512×512, 7 and 35 classes, clean and RWDA noise variants).
- **Method:** Evaluated standard CNNs (VGG, ResNet, DenseNet, MobileNet), Vision Transformers (ViT), and MLP-Mixer. Introduced Real-World Data Augmentation (RWDA).
- **Main Finding:** Deep CNNs and ViTs achieve near-perfect classification on clean DCID (>98–99%), but experience noticeable performance drops when subjected to defective field noise and lighting variations.
- **Direct Implications for LithoScan:** **This is the paper that created the dataset LithoScan uses.** Because Li et al. *already* benchmarked ResNet and Vision Transformers on DCID with RWDA noise, simply running a ResNet-50 or a standard CNN-Transformer hybrid on DCID provides **zero delta of contribution** over Li et al. LithoScan cannot claim architectural novelty on this benchmark.

### 2. Baraboshkin et al. (2020) — Automated Core Rock Typing
- **Full Citation:** Evgeny E. Baraboshkin, Leyla S. Ismailova, Dmitry M. Orlov, et al. (2020). *Deep convolutions for in-depth automated rock typing.* **Computers & Geosciences**, Vol. 134, Article 104330. DOI: [10.1016/j.cageo.2019.104330](https://doi.org/10.1016/j.cageo.2019.104330).
- **Dataset Used:** High-resolution core photography spanning 50 meters of continuous core.
- **Method:** Benchmarked AlexNet, VGG, GoogLeNet, and ResNet for automated continuous core description.
- **Main Finding:** CNNs achieve up to 95% validation precision and can process 50 meters of core in ~1 minute, validating deep learning as an operational assistant for sedimentologists.
- **Direct Implications for LithoScan:** Establishes that CNN-based core logging is an industry-standard baseline. Reviewers in *Computers & Geosciences* will expect comparisons to continuous borehole logging or operational speed, not just cropped patch classification.

### 3. Pires de Lima et al. (2019) — CNNs in Core Lithofacies Classification
- **Full Citation:** Rafael Pires de Lima, Fnu Suriamin, Kurt J. Marfurt, Matthew J. Pranter (2019). *Convolutional neural networks as aid in core lithofacies classification.* **Interpretation**, 7(3), SF27–SF40. DOI: [10.1190/INT-2018-0245.1](https://doi.org/10.1190/INT-2018-0245.1).
- **Dataset Used:** Whole-core slab photographs from multiple oil and gas boreholes.
- **Method:** Sliding-window cropping of core images into standardized patches, fine-tuning ImageNet-pretrained CNNs.
- **Main Finding:** Pretrained CNNs successfully categorize core textures and lithofacies, standardizing human logging descriptions across boreholes.
- **Direct Implications for LithoScan:** Pires de Lima established the patch-based classification methodology in 2019. Any new paper must address the well-known limitation of this approach: spatial autocorrelation between adjacent cropped patches.

### 4. Ploton et al. (2020) — Spatial Validation vs. Inflated Performance
- **Full Citation:** Pierre Ploton, Maxime Réjou-Méchain, Carsten F. Dormann, et al. (2020). *Spatial validation reveals poor predictive performance of large-scale ecological mapping models.* **Nature Communications**, 11, Article 4580. DOI: [10.1038/s41467-020-18321-y](https://doi.org/10.1038/s41467-020-18321-y).
- **Dataset Used:** Large-scale spatial mapping datasets (11.8 million records).
- **Method:** Comparative evaluation of random k-fold cross-validation versus spatially blocked cross-validation.
- **Main Finding:** Conventional random cross-validation yields artificially inflated performance ($R^2 > 0.5$) due to spatial autocorrelation; when spatial validation is strictly enforced, predictive power drops dramatically (approaching zero).
- **Direct Implications for LithoScan:** Provides the exact theoretical and empirical framework explaining why LithoScan's random split achieves >99.7% accuracy: spatial autocorrelation between adjacent core patches severely inflates standard cross-validation metrics.

### 5. Geirhos et al. (2020) — Shortcut Learning in Deep Neural Networks
- **Full Citation:** Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, Felix A. Wichmann (2020). *Shortcut learning in deep neural networks.* **Nature Machine Intelligence**, 2(11), 665–673. DOI: [10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z).
- **Dataset Used:** Multiple visual benchmarks under domain shift and controlled cue conflicts.
- **Method:** Conceptual and empirical taxonomy of shortcut learning across deep computer vision models.
- **Main Finding:** Deep neural networks systematically exploit unintended, spurious correlations (backgrounds, high-frequency textures, border artifacts) that satisfy the loss function on i.i.d. benchmarks but fail catastrophically under domain shift.
- **Direct Implications for LithoScan:** Directly informs LithoScan's Phase 3 and Phase 4 border/noise interventions. Proves that high clean accuracy is often a symptom of shortcut exploitation rather than deep geological understanding.

---

## 2. Additional Verified References (11 Further Studies)

6. **Roberts et al. (2017)** — *Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure.* **Ecography**, 40(8), 913–929. DOI: [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).  
   *Contribution:* Definitive review demonstrating that block cross-validation is mandatory when spatial/group dependencies exist.
7. **Hendrycks & Dietterich (2019)** — *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* **ICLR 2019**, arXiv:1903.12261.  
   *Contribution:* Standardized corruption benchmarking (ImageNet-C), foundational for LithoScan's clean-vs-noise degradation quantification.
8. **Geifman & El-Yaniv (2017)** — *Selective Classification for Deep Neural Networks.* **NeurIPS 2017**, 4877–4887.  
   *Contribution:* Risk-coverage curves and selective classification (reject option), allowing automated core loggers to triage uncertain samples to human geologists.
9. **Angelopoulos & Bates (2021)** — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* **Foundations and Trends in Machine Learning**, arXiv:2107.07511.  
   *Contribution:* Distribution-free prediction sets with rigorous finite-sample coverage guarantees.
10. **Tibshirani et al. (2019)** — *Conformal Prediction Under Covariate Shift.* **NeurIPS 2019**, 2530–2540.  
    *Contribution:* Extends conformal prediction guarantees to situations where test data distributions drift from calibration data (e.g., from clean core to fractured/muddy core).
11. **Pires de Lima et al. (2020)** — *Petrographic microfacies classification with deep convolutional neural networks.* **Computers & Geosciences**, 138, 104449. DOI: [10.1016/j.cageo.2020.104449](https://doi.org/10.1016/j.cageo.2020.104449).  
    *Contribution:* Thin-section microfacies automation, reinforcing the primacy of standard transfer learning in geological imagery.
12. **Pires de Lima, Galvis-Portilla, et al. (2022)** — *Generating a labeled data set to train machine learning algorithms for lithologic classification of drill cuttings.* **Interpretation**, 10(4), SD67–SD76. DOI: [10.1190/INT-2022-0019.1](https://doi.org/10.1190/INT-2022-0019.1).  
    *Contribution:* Open-source drill cutting SEM dataset creation; highlights data curation bottlenecks in subsurface ML.
13. **Xiao et al. (2024)** — *Rock image classification based on multi-scale feature fusion convolutional neural network.* **Earth Science Informatics**, 17, 1245–1258. DOI: [10.1007/s12145-024-01288-w](https://doi.org/10.1007/s12145-024-01288-w).  
    *Contribution:* Evaluated multi-scale feature fusion specifically for macroscopic rock textures; illustrates existing literature coverage of feature fusion.
14. **Ran et al. (2019)** — *Rock thin section identification based on deep convolutional neural networks.* **Petroleum Science**, 16(4), 756–766. DOI: [10.1007/s12182-019-0354-7](https://doi.org/10.1007/s12182-019-0354-7).  
    *Contribution:* Demonstrated >97% accuracy on microscopic rock thin sections using standard fine-tuned CNNs.
15. **Karimpouli & Tahmasebi (2019)** — *Image-based rock classification using convolutional neural networks.* **Computers & Geosciences**, 126, 142–150. DOI: [10.1016/j.cageo.2019.02.003](https://doi.org/10.1016/j.cageo.2019.02.003).  
    *Contribution:* Early benchmark comparing CNNs with traditional hand-crafted geostatistical features (variograms, gray-level co-occurrence).
16. **Alzubaidi et al. (2021)** — *Review of deep learning: concepts, CNN architectures, challenges, applications, and future directions.* **Journal of Big Data**, 8, Article 53. DOI: [10.1186/s40537-021-00444-8](https://doi.org/10.1186/s40537-021-00444-8).  
    *Contribution:* Authoritative review on the saturation of standard vision architectures on public benchmark tasks.

---

## 3. Novelty and Contribution Matrix

| Prior Work | Problem Addressed | Dataset | Evaluation Protocol | Method | Main Finding | Remaining Gap | LithoScan Implications |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Li et al. (2025)** | Lack of standardized large-scale core image benchmark | DCID (98,000 images, 35 classes) | Official random train/test split (80/20) | ResNet, VGG, DenseNet, ViT, MLP-Mixer + RWDA | Near-perfect clean accuracy; RWDA degradation exposed | No uncertainty quantification; no selective prediction; no spatial leakage audit | LithoScan cannot claim novelty on DCID benchmarking; must focus on selective triage and leakage |
| **Baraboshkin et al. (2020)** | Slow manual core logging | 50m of whole-core continuous images | Stratigraphic depth hold-out | AlexNet, GoogLeNet, ResNet | 95% precision; logged 50m in 1 minute | Evaluated single continuous hole; no fine-grained 35-class evaluation | LithoScan has more classes (35), but lacks continuous depth coordinates |
| **Pires de Lima et al. (2019)** | Inconsistent human core descriptions | Multiple borehole core slabs | Random patch split | Pretrained CNNs (ResNet, Inception) | Validated deep transfer learning for core logging | Suffered from patch-level spatial autocorrelation | LithoScan exhibits the exact same autocorrelation problem across splits |
| **Ploton et al. (2020)** | Overoptimistic machine learning performance in spatial data | 11.8M forest survey records | Random k-fold vs. Spatially blocked CV | Random Forest spatial mapping | Random split accuracy collapsed under spatial holdout | Not evaluated on drill-core computer vision | LithoScan can adopt this exact critique to challenge DCID's 99.7% benchmark |
| **Geirhos et al. (2020)** | Models learning spurious correlations | ImageNet-C, cue-conflict datasets | OOD corruption and style-swapping | Behavioral psychophysics and feature intervention | Deep nets latch onto shortcuts rather than high-level semantic shapes | Focused on natural images, not geological drill-cores | LithoScan's border occlusion and RWDA noise tests directly align with this paradigm |

---

## 4. Candid Assessment of LithoScan Novelty Claims

| Claimed Feature | Literature Reality | LithoScan Novelty Verdict |
| :--- | :--- | :--- |
| *"Novel CNN-Transformer fusion for drill cores"* | Hybrid CNN-Transformers (ViT/BoTNet/TransUNet) have been standard since 2020–2021. Li et al. (2025) already tested ViT on DCID. Furthermore, LithoScan's transformer underperforms ResNet-50 on DCID-7 and matches it on DCID-35. | **ZERO NOVELTY. DO NOT CLAIM.** |
| *"Texture-aware core architecture"* | No texture-specific modules exist in the code. Standard ResNet-50 layers. | **UNSUPPORTED / INVALID.** |
| *"High-accuracy (>99.7%) lithology classifier"* | Frozen ImageNet features achieve 99.28% with a simple linear probe. High accuracy is a property of the dataset's separability and spatial leakage, not model superiority. | **COMMODITY RESULT / SENSITIVITY FLAW.** |
| *"Empirical study on benchmark fragility, spatial leakage, and selective prediction"* | While Li et al. introduced DCID, no paper has critically investigated its perceptual train-test leakage, conducted rigorous selective triage (conformal prediction), or mapped risk-coverage trade-offs under real-world defects. | **GENUINE, DEFENSIBLE NOVELTY POTENTIAL.** |
