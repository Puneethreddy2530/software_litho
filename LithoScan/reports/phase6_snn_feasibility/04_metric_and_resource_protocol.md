# Phase 6 — Metric, Resource, and Energy Protocol

**Document ID:** `reports/phase6_snn_feasibility/04_metric_and_resource_protocol.md`  
**Date:** October 2026  
**Status:** COMPLETE & METHODOLOGICALLY RIGOROUS  
**Objective:** Define the measurement standards for classification performance, hardware resource profiling, and energy claims, establishing strict boundaries to prevent misleading or unverified neuromorphic claims.

---

## 1. Non-Equivalence Notice for Historical Baselines

### Existing Baselines (Phases 1–5A)
- **Phase 2B ResNet-50:** Trained at $512 \times 512$ resolution on DCID-35 (50 epochs, 6,089 s).
- **Phase 5A CNN-only (Custom 4-block):** Trained at $512 \times 512$ on DCID-35 (50 epochs, 12,215 s, Macro-F1 96.6% clean / 84.7% noisy).
- **Phase 5A CNN-Transformer Hybrid:** Trained at $512 \times 512$ on DCID-35 (50 epochs, 23,874 s, Macro-F1 96.7% clean / 85.3% noisy).

### Strict Rule of Non-Equivalence
- The Phase 6 SNN pilot and candidate experiments use **$224 \times 224$ input resolution** to remain within a 6GB GPU memory envelope during multi-step backpropagation through time (BPTT).
- **MANDATORY SCIENTIFIC CONSTRAINT:** Results obtained at $224 \times 224$ resolution **CANNOT be compared directly against Phase 5A $512 \times 512$ results**.
- Any comparison must be conducted against an **identically matched $224 \times 224$ ConvANN baseline** trained on the exact same data partitions, random seeds, and optimization schedule.

---

## 2. Fair Baseline Design: ConvSNN-4 vs. ConvANN-4

To ensure genuine scientific comparability, the architectures are topologically matched:

| Property | ConvSNN-4 (Spiking) | ConvANN-4 (Standard) | MobileNetV3-Small (Edge ANN Reference) |
| :--- | :--- | :--- | :--- |
| **Input Resolution** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ |
| **Layer Structure** | 4 Conv-BN-LIF blocks | 4 Conv-BN-ReLU blocks | Depthwise separable inverted bottlenecks |
| **Channel Progression** | 32 $\to$ 64 $\to$ 128 $\to$ 256 | 32 $\to$ 64 $\to$ 128 $\to$ 256 | MobileNetV3 standard widths |
| **Parameters** | $\sim 1.15$ Million | $\sim 1.15$ Million | $\sim 1.52$ Million |
| **Checkpoint Size** | $\sim 4.6$ MB | $\sim 4.6$ MB | $\sim 6.1$ MB |
| **Time Steps ($T$)** | $T=4$ | $T=1$ | $T=1$ |
| **Input Encoding** | Direct constant repetition | Single static tensor | Single static tensor |
| **Loss Function** | Cross-Entropy on mean rate | Cross-Entropy on logits | Cross-Entropy on logits |

---

## 3. Classification Metrics Protocol

All evaluations report:
1. **Top-1 Accuracy:** $\frac{\text{Correct Predictions}}{\text{Total Samples}}$
2. **Macro-Averaged F1 Score ($F_1$):**
   $$F_{1,\text{macro}} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
   where $P_c$ and $R_c$ are precision and recall for class $c$.
3. **Balanced Accuracy:** Macro-averaged recall across all $C$ classes.
4. **Per-Class Precision, Recall, and F1:** Full breakdown across all 7 lithological classes.
5. **Confusion Matrix:** Normalized by true class row to display class confusion patterns.

### Cohort Evaluation Rules
- **Clean Cohort:** Evaluated on clean validation set (`DCID-512-7/val`).
- **Defective Cohort:** Evaluated on natural defective validation set (`noise-512-7/val`).
- **Controlled Perturbations (Secondary Intervention):**
  - Gaussian blur ($\sigma = 2.0$), additive Gaussian noise ($\sigma = 0.05$), and border masking ($10\%$ border occlusion) applied to the *same clean validation images* to measure paired causal corruption resistance.
  - Defective cohort vs. clean cohort is reported as **inter-cohort distribution shift**, while synthetic perturbations on the clean cohort are reported as **paired causal corruptions**.

---

## 4. Hardware Resource and Latency Protocol

### Latency Measurement Protocol
- **Batch Size:** Evaluated at **Batch Size 1** (simulating single core-patch edge logging) and **Batch Size 16** (throughput profiling).
- **Execution Environments:**
  1. GPU: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA).
  2. CPU: Intel Core i7 / AMD host processor (single-thread and multi-thread).
- **Timing Standards:**
  - Mandatory warm-up: 50 iterations prior to timing.
  - Measurement: 200 consecutive iterations timed via `torch.cuda.Event(enable_timing=True)` on GPU and `time.perf_counter()` on CPU.
  - Reported values: Median latency and 95th percentile latency (ms).

### Memory Profiling Protocol
- Peak allocated GPU memory recorded via `torch.cuda.max_memory_allocated() / (1024 ** 2)` in MB.
- Evaluated during both forward inference (BS=1) and training forward+backward (BS=16).

---

## 5. Neuromorphic Energy and Synaptic Operation Claims

### A. Non-Availability of Neuromorphic Silicon
- **FACT:** The current research workstation has an NVIDIA GPU and an x86 CPU. **No dedicated neuromorphic hardware (e.g., Intel Loihi 2, SynSense Speck, SpiNNaker) is physically attached.**
- **MANDATORY RULE:** We **CANNOT** claim real-world energy consumption measurements for neuromorphic hardware.
- Any energy discussion must be explicitly designated as a **Theoretical Proxy Calculation**.

### B. Standard Theoretical Synaptic Operation Proxy (SOP)
Following standard literature (Wang et al., 2023; Kundu et al., 2021; Eshraghian et al., 2023):
1. **ANN Energy Model (Floating-Point multiply-accumulate, MAC):**
   $$E_{\text{ANN}} = \text{FLOPs} \times E_{\text{MAC}}$$
   where in 45nm CMOS technology, $E_{\text{MAC}} \approx 4.6\text{ pJ}$ (32-bit floating point multiply-accumulate).
2. **SNN Energy Model (Spike-based accumulation, AC):**
   Because spiking neurons output binary pulses $s \in \{0, 1\}$, neuromorphic hardware replaces expensive multiply-accumulate operations with sparse additions:
   $$\text{SOP} = \sum_{l=1}^L \bar{S}_l \times \text{FLOPs}_l$$
   $$E_{\text{SNN}} = \text{SOP} \times E_{\text{AC}}$$
   where in 45nm CMOS, an accumulation operation consumes $E_{\text{AC}} \approx 0.9\text{ pJ}$ (32-bit integer addition).
3. **First-Layer Encoding Cost:**
   The first convolutional layer processes continuous RGB pixel values and performs standard floating-point operations over $T$ steps:
   $$E_{\text{enc}} = T \times \text{FLOPs}_{\text{conv1}} \times E_{\text{MAC}}$$
   $$E_{\text{SNN,total}} = E_{\text{enc}} + \sum_{l=2}^L \bar{S}_l \times \text{FLOPs}_l \times E_{\text{AC}}$$

### C. Falsification Rule for Theoretical Energy Advantage
An SNN provides a theoretical energy advantage **only if**:
$$E_{\text{SNN,total}} < E_{\text{ANN}}$$
If the average firing rate $\bar{S}$ is high (e.g., $\bar{S} > 0.40$) or time steps $T \ge 8$, the repeated execution of the first encoding layer ($T \times \text{FLOPs}_{\text{conv1}}$) combined with dense accumulations can cause $E_{\text{SNN,total}} \ge E_{\text{ANN}}$. In that scenario, the SNN is officially deemed **computationally inefficient**.
