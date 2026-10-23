# Phase 6 — SNN Bounded Pilot Results

**Document ID:** `reports/phase6_snn_feasibility/05_pilot_results.md`  
**Date:** October 2026  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  
**Objective:** Document the execution, training dynamics, convergence, spike sparsity, and single-sample inference latency of the bounded Phase 6 Spiking Neural Network pilot on DCID-7.

---

## 1. Bounded Pilot Protocol and Safety Verification

The pilot strictly adhered to all preregistered safety constraints:
1. **Dataset Scope:** Restricted exclusively to **DCID-7** (7 major classes).
2. **Data Partitioning:**
   - Training subset: Exactly **10% of the training partition (2,520 images)** sampled deterministically from `DCID-512-7/train`.
   - Validation partition: Deterministic 10% stratified validation split (**2,800 images**) from `DCID-512-7/train`.
   - **SAFETY RULE SATISFACTION:** The official test partitions (`DCID-512-7/test` and `noise-512-7/test`) were **NEVER loaded, evaluated, or accessed**.
3. **Training Budget:** Exactly **3 epochs** (no hyperparameter tuning or architecture sweeps).
4. **Hardware Environment:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124, AMP mixed precision).

---

## 2. Model Specification & Structural Parity

| Specification | ConvSNN4 (Phase 6 Pilot) | ConvANN4 (Matched Baseline) |
| :--- | :--- | :--- |
| **Model Type** | Spiking Convolutional Neural Network | Standard Convolutional Neural Network |
| **Neuron Dynamics** | Multi-Step Leaky Integrate-and-Fire (LIF) | Rectified Linear Unit (ReLU) |
| **Membrane Parameters** | $\tau = 2.0$ ($\beta = 0.6065$), $V_{\text{th}} = 1.0$ | N/A |
| **Reset Mechanism** | Reset-by-subtraction ($V \leftarrow V - V_{\text{th}}$) | Memoryless |
| **Surrogate Gradient** | FastSigmoid ($k = 25.0$) | Standard analytical gradient |
| **Simulation Time Steps ($T$)** | $T = 4$ steps | $T = 1$ step |
| **Trainable Parameters** | **390,695 parameters** | **390,695 parameters** |
| **Checkpoint Size** | **1.5 MB** (`best_model.pt`) / **4.7 MB** (full state) | **1.5 MB** / **4.7 MB** |

---

## 3. Training Progression and Learning Dynamics

The model was trained using AdamW ($\text{lr}_0 = 10^{-3}$, cosine decay to $10^{-5}$, weight decay $10^{-4}$) and AMP mixed precision with batch size 16:

| Epoch | Train Loss | Val Loss | Val Top-1 Accuracy | Val Macro-F1 | Mean Spike Firing Rate ($\bar{S}$) | Peak CUDA Memory | Wall-Clock Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.6211 | 0.4586 | 82.54% | 0.8208 | 24.4% | 2,521.4 MB | 96.7 s |
| **2** | 0.3282 | 0.1650 | 94.86% | **0.9474** | 23.5% | 2,521.4 MB | 88.5 s |
| **3** | 0.2413 | 0.1399 | 94.86% | 0.9467 | 23.6% | 2,521.4 MB | 89.4 s |

### Key Training Observations
1. **Definite Convergence:**
   - Training loss dropped smoothly from 0.6211 to 0.2413.
   - Validation loss decreased from 0.4586 to 0.1399.
   - Validation Macro-F1 reached **0.9474 (94.74%)** by Epoch 2 using only 2,520 training examples.
   - **VERDICT:** The PyTorch LIF neuron implementation with FastSigmoid surrogate gradients learns representations reliably without numerical divergence or gradient stagnation.
2. **Spike Sparsity:**
   - Across all four convolutional stages, the average firing rate remained tightly bounded between **23.5% and 24.4%**.
   - This satisfies the preregistered sparsity criterion ($\bar{S} \le 0.30$), indicating that >75% of operations in a neuromorphic accumulator would be idle/sparse.
3. **Memory Footprint:**
   - Peak CUDA memory was **2,521.4 MB**, operating safely within the 6,144 MB hardware ceiling.

---

## 4. Latency and Hardware Profiling (Batch Size 1)

Inference latency was measured at single-patch input ($1 \times 3 \times 224 \times 224$), simulating real-time rig-site logging:

| Device | Metric | ConvSNN4 ($T=4$ steps) | Matched ConvANN4 ($T=1$ step) | Latency Overhead Ratio (SNN / ANN) |
| :--- | :--- | :--- | :--- | :--- |
| **GPU (RTX 3050 Laptop)** | **Median Latency** | **9.54 ms** | **0.92 ms** | **10.37x slower** |
| | **95th Percentile** | 10.34 ms | 1.67 ms | 6.19x slower |
| | **Mean $\pm$ Std** | $9.49 \pm 0.50$ ms | $1.02 \pm 0.26$ ms | 9.30x slower |
| **CPU (Host Single-Core)**| **Median Latency** | **45.29 ms** | **9.25 ms** | **4.90x slower** |
| | **95th Percentile** | 46.99 ms | 9.53 ms | 4.93x slower |
| | **Mean $\pm$ Std** | $45.59 \pm 0.99$ ms | $9.23 \pm 0.24$ ms | 4.94x slower |

---

## 5. Critical Hardware Disconnect: Simulation vs. Silicon

1. **GPU Simulation Reality:**
   - On synchronous SIMD hardware (GPUs and CPUs), simulating an SNN across $T=4$ time-steps requires sequential loop execution and repeated memory fetches of intermediate membrane potentials.
   - Consequently, **simulating ConvSNN4 on the GPU is approximately 10.4x slower than running the topologically matched ConvANN4 (9.54 ms vs. 0.92 ms)**.
2. **Theoretical Neuromorphic Energy Advantage:**
   - With an average spike firing rate $\bar{S} \approx 0.236$, the theoretical Synaptic Operation (SOP) calculation yields:
     $$E_{\text{SNN,theory}} \approx 0.236 \times \text{FLOPs} \times 0.9\text{ pJ} \approx 0.21 \times E_{\text{ANN,FLOPs}}$$
   - **Theoretical SOP ratio: ~4.7x energy reduction** on dedicated event-driven neuromorphic silicon (e.g., Intel Loihi 2).
3. **The Empirical Truth:**
   - Because no neuromorphic silicon is physically connected, **any claim that LithoScan currently runs faster or with less energy on GPU/CPU is FALSE**.
   - On current hardware, SNN inference is significantly slower and consumes more total energy (due to extended GPU active time).

---

## 6. Pilot Verdict

- **Functional Feasibility:** **CONFIRMED.** The custom PyTorch SNN architecture trains reliably, converges quickly, and achieves 94.74% validation macro-F1 at 23.6% spike sparsity.
- **Hardware Profile:** **BOUNDED.** While viable for research evaluation, GPU simulation incurs a 10.4x latency penalty at $T=4$.
