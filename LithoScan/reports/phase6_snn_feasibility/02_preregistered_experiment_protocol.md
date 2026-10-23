# Phase 6 — Preregistered Experiment Protocol

**Document ID:** `reports/phase6_snn_feasibility/02_preregistered_experiment_protocol.md`  
**Date:** October 2026  
**Status:** PREREGISTERED & LOCKED  
**Objective:** Formally define the primary falsifiable hypothesis, secondary research questions, experimental design, baseline matching rules, statistical tests, and explicit success/failure criteria before executing any comparative campaign.

---

## 1. Candidate Hypothesis and Research Questions

### Primary Research Question (RQ1 — Falsifiable Hypothesis)
> **Primary Hypothesis ($H_1$):** Under matched architectural capacity (parameter count within $\pm 20\%$) and identical input resolution ($224 \times 224$), a directly trained Leaky Integrate-and-Fire Spiking Neural Network (ConvSNN, $T=4$) demonstrates greater robustness to natural drilling acquisition defects than an equivalent conventional Convolutional Neural Network (ConvANN), exhibiting a statistically significantly smaller relative macro-F1 degradation when evaluated on defective drill-core imagery:
> $$\Delta F_{1,\text{rel}}(\text{SNN}) < \Delta F_{1,\text{rel}}(\text{ANN})$$
> where $\Delta F_{1,\text{rel}} = \frac{F_{1,\text{clean}} - F_{1,\text{noisy}}}{F_{1,\text{clean}}}$.

> **Null Hypothesis ($H_0$):** ConvSNN exhibits equal or greater relative macro-F1 degradation compared to ConvANN ($\Delta F_{1,\text{rel}}(\text{SNN}) \ge \Delta F_{1,\text{rel}}(\text{ANN})$), or ConvSNN fails to achieve at least 80% of ConvANN's absolute clean macro-F1 ($F_{1,\text{clean}}(\text{SNN}) < 0.80 \times F_{1,\text{clean}}(\text{ANN})$).

### Secondary Research Questions
- **RQ2 (Temporal Budget Trade-off):** How does the simulation time-step budget $T \in \{1, 2, 4, 8\}$ affect the Pareto frontier between classification accuracy (clean and noisy) and inference latency / synaptic operation count (SOP)?
- **RQ3 (Resource & Sparsity Feasibility):** Does the average spike firing rate across network layers remain sufficiently sparse ($<25\%$ average activation) to yield a theoretical synaptic operation energy advantage ($E_{\text{SNN}} < E_{\text{ANN}}$) under standard neuromorphic proxy calculations?

---

## 2. Experimental Scope and Datasets

1. **Dataset Scope:**
   - Primary evaluation on **DCID-7** (7 major lithological classes: Red sandstone, Gray-black mudstone, Light gray fine sandstone, Conglomerate, Gray sandstone, Sandstone, Gray mudstone).
   - Clean partition: `DCID-512-7` (28,000 train images, 7,000 test images).
   - Defective/noisy partition: `noise-512-7` (11,200 train images, 2,800 test images).
2. **Deterministic Partitions:**
   - In accordance with our safety rules, the official test sets (`DCID-512-7/test` and `noise-512-7/test`) are strictly isolated and **never accessed during model development, pilot verification, or hyperparameter tuning**.
   - Model selection is conducted strictly on a deterministic 10% validation split carved out of the official training set using fixed random seed `42`.
3. **Input Resolution:**
   - Both SNN and ANN models will process images resized to **$224 \times 224$ pixels** with bicubic interpolation.
   - *Rationale:* Simulating SNN dynamics across $T=4$ time-steps at $512 \times 512$ exceeds 6GB VRAM during backpropagation through time (BPTT). $224 \times 224$ is the universal computer-vision benchmark resolution and allows strictly matched apples-to-apples comparisons.

---

## 3. Matched Architecture Families

To prevent unfair comparisons between mismatched models, we pair each SNN with a topologically identical ANN:

| Specification | Spiking Model: **ConvSNN-4** | Matched Baseline: **ConvANN-4** |
| :--- | :--- | :--- |
| **Backbone Topology** | 4-stage Conv-BN-Neuron + Global Pool + Linear | 4-stage Conv-BN-ReLU + Global Pool + Linear |
| **Channel Progression** | 3 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 | 3 $\to$ 32 $\to$ 64 $\to$ 128 $\to$ 256 |
| **Kernel Size / Stride** | $3 \times 3$, stride 1 (pool stride 2) | $3 \times 3$, stride 1 (pool stride 2) |
| **Activation / Neuron** | Leaky Integrate-and-Fire (LIF, $\tau=2.0$, $V_{\text{th}}=1.0$) | Standard ReLU |
| **Surrogate Gradient** | FastSigmoid ($k=25.0$) | Exact analytical ReLU gradient |
| **Temporal Steps ($T$)** | $T=4$ (static repeat encoding) | $T=1$ (single forward pass) |
| **Output Decoding** | Mean membrane / spike rate over $T$ steps | Direct softmax logits |
| **Parameter Count** | $\sim 1.15 \times 10^6$ parameters | $\sim 1.15 \times 10^6$ parameters |
| **Weight Equality** | Same layer dimensions and weights | Same layer dimensions and weights |

In addition, an edge-standard baseline (**MobileNetV3-Small**, $\sim 1.5 \times 10^6$ parameters, trained from scratch under identical hyperparameter conditions) will serve as an external reference.

---

## 4. Evaluation Metrics

### Primary Metric
- **Macro-Averaged F1 Score ($F_{1,\text{macro}}$):** Evaluated independently on the clean validation set ($F_{1,\text{clean}}$) and noisy validation set ($F_{1,\text{noisy}}$).

### Secondary Metrics
1. **Top-1 Classification Accuracy ($A_{\text{top1}}$)**
2. **Balanced Accuracy** (macro-averaged recall across the 7 classes)
3. **Relative Robustness Degradation Index ($\Delta F_{1,\text{rel}}$):**
   $$\Delta F_{1,\text{rel}} = \frac{F_{1,\text{clean}} - F_{1,\text{noisy}}}{F_{1,\text{clean}}}$$
4. **Latency at Batch Size 1 ($L_{\text{BS1}}$, ms):** Measured on GPU and CPU after 50 warm-up runs across 200 iterations.
5. **Peak CUDA Allocation (MB):** Profiled via `torch.cuda.max_memory_allocated()`.
6. **Average Spike Firing Rate ($\bar{S}$):** Fraction of neurons firing spikes per time step across all convolutional layers:
   $$\bar{S} = \frac{1}{T \cdot N} \sum_{t=1}^T \sum_{i=1}^N s_{i,t} \in [0, 1]$$
7. **Theoretical Synaptic Operations Proxy (SOP):**
   $$\text{SOP}_{\text{layer}} = \bar{S} \times \text{FLOPs}_{\text{layer}}$$

---

## 5. Training, Optimization, and Seed Protocol

1. **Optimizer:** AdamW with weight decay $10^{-4}$.
2. **Learning Rate Schedule:** Initial learning rate $\eta_0 = 10^{-3}$, cosine annealing decay down to $10^{-5}$ over 30 epochs.
3. **Batch Size:** 32.
4. **Loss Function:** Multi-class Cross-Entropy Loss applied to time-averaged logits.
5. **Data Augmentation:** Identical for both models:
   - Random horizontal flip ($p=0.5$)
   - Random vertical flip ($p=0.5$)
   - Random affine rotation ($\pm 15^\circ$)
   - Color jitter (brightness=0.1, contrast=0.1)
   - Normalization (ImageNet mean and std)
6. **Multi-Seed Replication Policy:**
   - 3 deterministic random seeds (`42`, `1337`, `2026`) for all comparative runs.
   - All results reported as $\text{mean} \pm \text{standard deviation}$.
7. **Statistical Testing:**
   - Two-tailed paired Student's $t$-test or Wilcoxon signed-rank test across seeds and per-class F1 differentials with significance threshold $\alpha = 0.05$.

---

## 6. Explicit Decision Criteria: Success vs. Failure Conditions

### Success Conditions (Proceed to Full Campaign)
The primary hypothesis is considered supported if **ALL** of the following hold:
1. **Basic Task Competence:** ConvSNN achieves clean validation macro-F1 of at least $0.85 \times F_{1,\text{clean}}(\text{ConvANN})$.
2. **Demonstrated Robustness:** ConvSNN exhibits a smaller relative performance degradation on noisy validation data: $\Delta F_{1,\text{rel}}(\text{ConvSNN}) < \Delta F_{1,\text{rel}}(\text{ConvANN})$ by at least 2.5 percentage points.
3. **Spike Sparsity:** Average firing rate $\bar{S} \le 0.30$, confirming that computation is predominantly sparse.

### Failure Conditions (Declare Scientific Negative Result / Stop SNN Track)
The SNN track will be declared a **scientific negative result** and stopped if **ANY** of the following occur:
1. **Severe Accuracy Gap:** ConvSNN clean macro-F1 is more than 15 percentage points below ConvANN ($F_{1,\text{clean}}(\text{ConvSNN}) < F_{1,\text{clean}}(\text{ConvANN}) - 0.15$).
2. **No Robustness Advantage:** ConvSNN degrades equally or more severely than ConvANN under defective data ($\Delta F_{1,\text{rel}}(\text{ConvSNN}) \ge \Delta F_{1,\text{rel}}(\text{ConvANN})$).
3. **Dense Firing / High Latency:** ConvSNN requires dense firing ($\bar{S} > 0.50$) and GPU inference latency is $> 4\times$ ConvANN latency without theoretical SOP savings.

> **SCIENTIFIC INTEGRITY NOTE:** A failure to confirm the primary hypothesis is an entirely acceptable, highly informative scientific outcome. It directly demonstrates that biological temporal integration in SNNs does not automatically translate to acquisition robustness for high-resolution static geological textures, ruling out a major open question.
