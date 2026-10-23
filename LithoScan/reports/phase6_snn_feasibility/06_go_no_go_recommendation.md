# Phase 6 — GO / NO-GO Recommendation and Strategic Decision

**Document ID:** `reports/phase6_snn_feasibility/06_go_no_go_recommendation.md`  
**Date:** October 2026  
**Status:** COMPLETE & FORMALLY VERIFIED  
**Final Verdict:** **CONDITIONAL GO** (Proceed to a controlled SNN-vs-ANN robustness campaign ONLY under four mandatory scientific boundaries)

---

## 1. Direct Answers to the 10 Scientific Decision Questions

### 1. Is an SNN study a defensible potential research contribution?
**YES, conditionally.**  
If framed strictly around the **accuracy–latency–resource trade-off under natural drilling acquisition defects** (evaluating whether temporal membrane integration and discrete event thresholds provide inherent noise robustness compared to matched conventional CNNs), it addresses an open, well-defined scientific question.  
However, if framed as a "novel SNN architecture for drill core" or claiming real-world energy efficiency on current GPU hardware, it is **indefensible and would be immediately rejected by peer reviewers**.

### 2. What is the closest existing work?
- **In Drill-Core ML:** Li et al. (2025) introduced the DCID benchmark (*Petroleum Science*), evaluating standard CNNs (ResNet, MobileNet) and ViTs. Alzubaidi et al. (2021) established CNN transfer learning on GSSA core trays (*JPSE*). Günther et al. (2025) comprehensively surveyed drill-core ML (*Ore Geology Reviews*), identifying edge power and acquisition noise as key bottlenecks.
- **In SNN Geoscience / Remote Sensing:** Wang et al. (2023) developed BDSNN (*Remote Sensing*, MDPI) for hyperspectral classification, demonstrating 65–78% SOP savings over 3D-CNNs. Li et al. (2024) developed Hyper-S3NN (*Infrared Physics & Technology*).
- **In SNN Robustness:** Kundu et al. (2021) introduced HIRE-SNN (*ICCV*), demonstrating that low-latency directly trained SNNs exhibit inherent resilience to input noise compared to matched ANNs.

### 3. What is actually different from CoreLog Vision?
- CoreLog Vision (Santibáñez-Leal, July/Sept 2026) is a browser-based, client-side interactive tool using conventional floating-point MobileNetV3 backbones for automated lithology segmentation on Chilean porphyry suites.
- CoreLog Vision does **not** evaluate Spiking Neural Networks, neuromorphic event dynamics, surrogate gradients, or spike-based energy proxies.
- LithoScan Phase 6 differs fundamentally by evaluating **bio-inspired temporal spike integration, membrane threshold dynamics, and theoretical synaptic operation savings under acquisition defects**.

### 4. What evidence supports a gap specifically in RGB drill-core imagery?
- Systematic multi-database searches across IEEE Xplore, ScienceDirect, MDPI, ACM, arXiv, and Google Scholar confirmed that:  
  > *"No directly matching study was identified in the searches conducted"* regarding the application of SNNs to static photographic RGB drill-core imagery.
- While SNNs have been applied to hyperspectral remote sensing cubes and seismic event streams, drill-core photographic classification remains exclusively reliant on conventional CNNs and ViTs.

### 5. What is the minimum convincing full experiment?
1. **Architectural Parity:** Matched pair **ConvSNN-4** ($T \in \{1, 2, 4\}$) vs. **ConvANN-4** vs. edge-standard **MobileNetV3-Small** at identical input resolution ($224 \times 224$).
2. **Data Evaluation:** DCID-7 evaluated across 3 random seeds on:
   - Official clean validation partition.
   - Official natural defective validation partition.
   - Controlled paired synthetic corruptions (Gaussian blur, noise, border occlusion) applied to clean validation images.
3. **Efficiency Benchmarking:**
   - Single-sample inference latency (BS=1) on GPU and CPU.
   - Layer-wise spike firing rates and theoretical Synaptic Operation Proxy (SOP) energy calculations.

### 6. Which source-separated dataset is genuinely usable?
- **Current Finding:** Neither DCID nor the public Mendeley GSSA dataset contains borehole or depth identifiers in their released filenames.
- The only dataset capable of supporting true Leave-One-Well-Out (LOWO) source separation is the raw **AuScope National Virtual Core Library (NVCL)**, which requires extensive continuous tray segmentation engineering.
- **MANDATORY PROTOCOL:** The SNN study must be reported strictly as **image-level evaluation under natural acquisition distribution shift**, with no unsubstantiated claims of source-separated borehole generalization.

### 7. What energy claims can actually be supported on available hardware?
- **Empirical Reality:** The workstation possesses an NVIDIA RTX 3050 Laptop GPU and x86 CPU. **No physical neuromorphic silicon (e.g., Intel Loihi 2) is present.**
- On current hardware, SNN inference is **~10.4x slower on GPU (9.54 ms vs. 0.92 ms)** and consumes more total electrical power than conventional CNNs.
- **PERMITTED CLAIM:** Energy claims are strictly restricted to **Theoretical Synaptic Operation Proxies (SOP)**:
  $$E_{\text{SNN,theory}} = \text{SOP} \times 0.9\text{ pJ} \quad \text{vs.} \quad E_{\text{ANN}} = \text{FLOPs} \times 4.6\text{ pJ}$$
  Every report and paper draft must explicitly state that this is a theoretical neuromorphic hardware proxy.

### 8. What would falsify the hypothesis?
The hypothesis is falsified if:
- ConvSNN exhibits equal or greater relative macro-F1 degradation than ConvANN on defective data ($\Delta F_{1,\text{rel}}(\text{SNN}) \ge \Delta F_{1,\text{rel}}(\text{ANN})$).
- Clean validation macro-F1 of ConvSNN is $>15$ percentage points below ConvANN.
- Average spike firing rate $\bar{S} > 0.40$, eliminating any theoretical SOP energy advantage.

### 9. What is the expected compute cost?
- In our bounded pilot, 1 epoch of ConvSNN4 on 2,520 images took ~89 seconds.
- Full DCID-7 training (25,200 images, 30 epochs) will require ~15 minutes per epoch, or **~7.5 hours per run**.
- A full 3-seed campaign for ConvSNN4 + ConvANN4 requires **~35–45 hours of GPU compute**, which is fully feasible on the available RTX 3050 GPU without multi-week training.

### 10. Is the SNN path more credible than the original conformal-only route?
- **YES.** The original conformal-only route hit a fundamental theoretical barrier: standard split conformal prediction requires exchangeability. Calibrating on clean data and testing on defective data without verified borehole metadata breaks exchangeability, rendering theoretical error-rate bounds invalid without complex covariate-shift reweighting.
- The SNN path asks an empirical, testable, and falsifiable question about acquisition robustness and computational trade-offs that can be rigorously answered using the existing DCID assets.

---

## 2. Verdict: CONDITIONAL GO

The recommendation is **CONDITIONAL GO**. The project should proceed to the full controlled SNN-vs-ANN campaign **ONLY IF** the following four non-negotiable boundaries are formally maintained:

### The 4 Non-Negotiable Scientific Boundaries
1. **No Silicon Energy Claims:** Energy discussions must be explicitly labeled as theoretical CMOS 45nm synaptic operation proxies (SOPs). No claims of empirical power reduction on GPU/CPU.
2. **No Source-Separation Overstatement:** Evaluation must be honestly designated as *image-level evaluation under natural acquisition distribution shift*.
3. **Apples-to-Apples Architecture Parity:** ConvSNN-4 must be evaluated exclusively against topologically matched ConvANN-4 and MobileNetV3-Small at $224 \times 224$ resolution. Comparisons against historical $512 \times 512$ models must be explicitly marked as non-equivalent.
4. **Preregistration and Falsifiability:** The preregistered hypothesis must be adhered to; if the SNN fails to demonstrate superior noise robustness, the negative result must be reported as a valid scientific finding.

---

## 3. Recommended Next Single Action

**Do not begin training immediately.**  
The next single action is to review the Phase 6 documentation and approve the preregistered experimental protocol before initiating full training.
