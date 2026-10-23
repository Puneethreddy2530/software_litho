# Phase 6 v1.1 — End-to-End Smoke Test Engineering Results

**Document ID:** `reports/phase6_snn_feasibility_v1_1/06_smoke_test_results.md`  
**Date:** October 2026  
**Status:** COMPLETE & EMPIRICALLY VERIFIED  
**Engineering Notice:** All numerical metrics reported in this document are derived from an isolated, deterministic 28-sample smoke test executed exclusively for software and hardware pipeline verification. **They are engineering diagnostics, NOT scientific lithology classification results.**

---

## 1. Smoke Test Execution Setup

- **Compute Environment:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124) and host x86 CPU.
- **Deterministic Sample Selection:** Exactly 4 samples per class across all 7 classes drawn deterministically from the locked manifest:
  - Training subset: 28 images (drawn from `pilot_train` split).
  - Validation subset: 28 images (drawn from `validation` split).
  - Image size: $224 \times 224$ pixels, explicit `InterpolationMode.BICUBIC`.
- **Minibatch Configuration:** Batch size = 8, yielding 4 minibatches per epoch to thoroughly exercise multi-batch aggregation and gradient step mechanics.

---

## 2. Model Forward, Backward & Parameter Verification

| Property | ConvSNN4 (Spiking Prototype) | ConvANN4 (Matched Conventional Baseline) | Verification Result |
| :--- | :--- | :--- | :---: |
| **Model Class** | `src.models.phase6_snn_v1_1.ConvSNN4` | `src.models.phase6_snn_v1_1.ConvANN4` | **MATCHED** |
| **Parameter Count** | **390,695** (computed via `count_parameters()`) | **390,695** (computed via `count_parameters()`) | **EXACT MATCH** |
| **Simulation Time Steps** | $T = 4$ steps | $T = 1$ step | **VERIFIED** |
| **Single-Epoch Train Loss** | 1.8096 (Finite) | 1.9169 (Finite) | **FINITE** |
| **Gradient Check** | Gradients present and finite on all conv/BN/linear weights | Gradients present and finite on all conv/BN/linear weights | **PASS** |
| **Single-Epoch Val Loss** | 1.9467 (Finite) | 1.8224 (Finite) | **FINITE** |
| **Single-Epoch Val Accuracy**| 14.29% (Finite) | 42.86% (Finite) | **FINITE** |
| **Single-Epoch Val Macro-F1**| 0.0357 (Finite) | 0.3176 (Finite) | **FINITE** |

Both architectures executed forward propagation, loss calculation, backward gradient computation, optimizer stepping, and evaluation without warnings, NaN values, or runtime exceptions.

---

## 3. Dataset-Wide Multi-Batch Spike Activity Accumulation

In contrast to v1.0 (which recorded only the final minibatch), `SpikeActivityTracker` accumulated spike activations across all 4 validation minibatches:

| Layer / Stage | Total Spikes Recorded | Total Neuron-Time Positions | Layer Firing Rate ($\bar{S}_l$) | Denominator Validity |
| :--- | :---: | :---: | :---: | :---: |
| **Block 1 LIF** | 8,947,265 | 179,830,784 | **0.049754 (4.98%)** | Valid ($T \times B_{\text{total}} \times 32 \times 112^2$) |
| **Block 2 LIF** | 556,405 | 89,915,392 | **0.006188 (0.62%)** | Valid ($T \times B_{\text{total}} \times 64 \times 56^2$) |
| **Block 3 LIF** | 27,071 | 44,957,696 | **0.000602 (0.06%)** | Valid ($T \times B_{\text{total}} \times 128 \times 28^2$) |
| **Block 4 LIF** | 22 | 22,478,848 | **0.000001 (0.00%)** | Valid ($T \times B_{\text{total}} \times 256 \times 28^2$) |
| **Mean Firing Rate** | — | — | **0.014136 (1.41%)** | Unweighted mean across 4 layers |
| **Weighted Firing Rate** | — | — | **0.014136 (1.41%)** | Weighted by layer FLOP count |
**Diagnostic Confirmation:** Firing rates are strictly bounded within $[0.0, 1.0]$, positions accumulated across all batches without truncation, and denominators reflect the exact tensor shapes.

---

## 4. Standardized Latency Benchmarking (Task 5)

### Benchmark Provenance & Setup
The reported latency figures were obtained under the following standardized, deterministic experimental protocol:
- **Device Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, Driver 572.16) and host x86 CPU (Intel Core i5-12450H).
- **Input Resolution & Precision:** Single synthetic RGB image tensor of shape $(1, 3, 224, 224)$ in 32-bit floating point (`torch.float32`), matching the standard inference input specification.
- **Batch Size:** Exactly $1$ (single-sample sequential inference).
- **Iteration Counts:**
  - GPU: 50 warmup iterations, followed by 200 timed iterations.
  - CPU: 20 warmup iterations, followed by 50 timed iterations.
- **Synchronization Procedure:**
  - **GPU:** Prior to timing, `torch.cuda.synchronize()` was invoked. Timing was recorded strictly via CUDA events: `starter = torch.cuda.Event(enable_timing=True)` and `ender = torch.cuda.Event(enable_timing=True)`. Each timed iteration recorded `starter.record()`, executed `model(dummy_input)`, recorded `ender.record()`, and immediately invoked `torch.cuda.synchronize()` before reading `starter.elapsed_time(ender)`. This eliminates asynchronous kernel queueing and pipelining artifacts across iterations.
  - **CPU:** Timing was recorded using Python's high-resolution monotonic timer `time.perf_counter()`. Explicit single-thread execution was enforced via `torch.set_num_threads(1)` to eliminate multi-core thread contention, scheduling noise, and frequency scaling artifacts.
- **Model Checkpoint / Weight States:**
  - Both architectures were evaluated strictly in evaluation mode (`model.eval()`) with autograd disabled (`torch.no_grad()`).
  - **ConvSNN4 ($T=4$):** Evaluated using the weights resulting from 1 epoch of smoke-test training (`snn_model`, 390,695 parameters).
  - **ConvANN4 ($T=1$):** Evaluated using randomly initialized weights matching the topologically identical Conv4 architecture (390,695 parameters), specifically evaluated for architectural latency comparison.
- **Measurement Scope & Boundary:**
  - **Explicit Non-Universal Qualification:** These latency figures are measurements strictly from the specified benchmark setup on standard synchronous digital hardware under PyTorch. They reflect the software execution overhead of sequentially simulating $T=4$ discrete time steps on SIMD/SIMT architectures versus a single-pass ANN, and **must NOT be cited as universal SNN-versus-ANN performance**.

### Benchmark Latency Results

| Platform / Device | Model | Timed Iterations | Warmup Runs | Median Latency (Unrounded / Displayed) | 95th Percentile Latency | Mean $\pm$ Std | Latency Ratio (SNN / ANN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPU (RTX 3050 Laptop)** | **ConvSNN4 ($T=4$)** | 200 | 50 | 8.528 ms / **8.53 ms** | 9.16 ms | $8.61 \pm 0.34$ ms | **9.33x** (unrounded: 9.3315x)<br>*[$\approx 9.37\text{x}$ from displayed]* |
| | **ConvANN4 ($T=1$)** | 200 | 50 | 0.914 ms / **0.91 ms** | 1.60 ms | $0.99 \pm 0.24$ ms | — |
| **CPU (Single-Thread)** | **ConvSNN4 ($T=4$)** | 50 | 20 | 136.154 ms / **136.15 ms** | 138.99 ms | $136.54 \pm 3.11$ ms | **4.39x** (unrounded: 4.3866x)<br>*[$\approx 4.39\text{x}$ from displayed]* |
| | **ConvANN4 ($T=1$)** | 50 | 20 | 31.039 ms / **31.04 ms** | 32.35 ms | $31.13 \pm 0.66$ ms | — |

> **Mathematical Disclosure on Latency Ratios:**  
> The exact median latency observations recorded in `smoke_test_summary.json` are $8.527679920196533\text{ ms}$ for ConvSNN4 and $0.9138559997081757\text{ ms}$ for ConvANN4 on GPU. Dividing these unrounded floating-point observations yields a ratio of **$9.3315\times \approx 9.33\times$**. If calculated directly from the two-decimal rounded displayed table values ($8.53\text{ ms} / 0.91\text{ ms}$), the ratio evaluates to **approximately $9.37\times$** ($9.3736\times$). Both figures are disclosed to avoid ambiguity. On single-thread CPU, the unrounded ratio is $136.15405 / 31.03850 = 4.3866\times \approx 4.39\times$, matching the ratio from displayed values ($136.15 / 31.04 = 4.3863\times \approx 4.39\times$).

### Benchmark Limitations & Physical Distinction
- **Architectural Latency Diagnostic Only:** ConvSNN4 was evaluated using checkpoint weights obtained after one epoch of smoke training, while ConvANN4 was evaluated with random initialization matching identical topology. This benchmark is strictly an architectural timing diagnostic; it **must NOT be presented as a trained-model task-accuracy comparison or as evidence of robustness superiority**.
- **Separation of Latency from Energy:** Latency is the measured execution duration on synchronous digital hardware; it does not measure electrical energy. Simulating $T=4$ time steps sequentially on GPUs/CPUs incurs a substantial runtime penalty (**8.53 ms vs. 0.91 ms on GPU, 136.15 ms vs. 31.04 ms on CPU**).
- **Explicit Measurement Disclaimer:** **Energy was not directly measured.** No physical power or electrical energy consumption was recorded via hardware power meters, NVML, RAPL, or external monitoring equipment during this benchmark.
- **Correction of Unsupported Claims:** Any assertion that GPU simulation consumes substantially more electrical energy based solely on latency is unsupported and is formally qualified: latency differences cannot be directly converted to physical energy without empirical power profiling under load.

---

## 5. Theoretical CMOS 45nm Energy Proxy

### Mathematical Proxy Formulation
To assess potential arithmetic efficiency on hypothetical neuromorphic hardware, we compute a first-principles theoretical energy proxy based on standard 45nm CMOS synaptic operation models (Horowitz, 2014; literature proxy parameters: $E_{\text{MAC}} = 4.6\text{ pJ}$, $E_{\text{AC}} = 0.9\text{ pJ}$, membrane update = $1.8\text{ pJ}$):

| Energy Component | Formula / Term | Energy ($\mu$J) | Percentage of SNN Energy |
| :--- | :--- | :---: | :---: |
| **Layer 1 Continuous MACs** | $T \times \text{FLOPs}_{\text{conv1}} \times 4.6\text{ pJ}$ | $797.68\ \mu\text{J}$ | **93.71%** |
| **Internal Synaptic ACs** | $\sum_{l=2}^4 \bar{S}_{l-1} \times T \times \text{FLOPs}_l \times 0.9\text{ pJ}$ | $47.07\ \mu\text{J}$ | **5.53%** |
| **Membrane Potential Updates** | $\text{Neurons}_{\text{total}} \times T \times 1.8\text{ pJ}$ | $6.49\ \mu\text{J}$ | **0.76%** |
| **Total Theoretical SNN Energy** | $E_{\text{layer1}} + E_{\text{internal}} + E_{\text{membrane}}$ | **$851.25\ \mu\text{J}$** | **100.00%** |
| **Total Baseline ANN Energy** | $\sum_{l=1}^4 \text{FLOPs}_l \times 4.6\text{ pJ}$ | **$3,390.13\ \mu\text{J}$** | — |
| **Theoretical Energy Ratio ($E_{\text{SNN}} / E_{\text{ANN}}$)** | — | **0.2511 (~3.98x theoretical reduction)** | — |

### Assumptions, Literature Citations, and Limitations of the Theoretical Proxy

1. **Theoretical Operation-Energy Proxy, Not Measured Savings:**
   - The reported $0.2511$ SNN/ANN ratio ($851.25\ \mu\text{J} / 3390.13\ \mu\text{J}$) is purely a theoretical operation-energy proxy under the stated arithmetic assumptions. It represents hypothetical synaptic-operation energy on an idealized digital neuromorphic ASIC.
   - It is **NOT measured energy savings** and **NOT evidence of a demonstrated neuromorphic hardware advantage**.
   - **Reconciliation with Scientific Claims:** While the arithmetic model indicates a theoretical $\sim 3.98\times$ operation-energy reduction under idealized ASIC conditions, **no energy-superiority claim is made**. Physical energy was not directly measured, and standard workstation inference exhibited higher latency without demonstrated energy benefits.

2. **Citations for Operation-Energy Assumptions:**
   - Operation energies are derived from Mark Horowitz, *"1.1 Computing's energy problem (and what we can do about it),"* IEEE International Solid-State Circuits Conference (ISSCC) Digest of Technical Papers, 2014, pp. 10–14, doi: 10.1109/ISSCC.2014.6757323. Under 45nm CMOS: 32-bit floating-point Multiply-Accumulate (MAC) is estimated at $4.6\text{ pJ}$ ($3.7\text{ pJ}$ mult + $0.9\text{ pJ}$ add); 32-bit Accumulate (AC) is estimated at $0.9\text{ pJ}$; and membrane update is modeled as $2 \times 0.9\text{ pJ} = 1.8\text{ pJ}$.

3. **Continuous First-Layer Bottleneck ($93.71\%$):**
   - Because the input is a continuous photographic RGB image, the first convolutional layer performs standard floating-point MAC operations repeatedly across all $T=4$ timesteps. This single layer accounts for **93.71% of total theoretical SNN operation energy** ($797.68\ \mu\text{J}$ of $851.25\ \mu\text{J}$). Even if downstream spike sparsity were 100% (zero spikes), the SNN could never exceed a theoretical $\sim 4.25\times$ energy reduction over the ANN due to this continuous encoding front-end.

4. **Dominance of Omitted Memory Access Costs:**
   - As documented by Horowitz (2014, Table 1.1.2) and Han et al. (*"EIE: Efficient Inference Engine on Compressed Deep Neural Network,"* ACM/IEEE International Symposium on Computer Architecture [ISCA], 2016), memory access dominates arithmetic energy in physical silicon:
     - 32-bit SRAM cache read: $\sim 5\text{ pJ}$ (8KB) to $\sim 50\text{ pJ}$ (32KB), comparable to or exceeding a 32-bit FP MAC ($4.6\text{ pJ}$).
     - 32-bit DRAM external access: $\sim 1,300\text{ pJ}$ to $2,600\text{ pJ}$ ($1.3\text{ nJ}$–$2.6\text{ nJ}$), which is **$\sim 280\times$ to $560\times$ more energy than an FP MAC** and **$\sim 1,400\times$ to $2,800\times$ more than an AC**.
   - Because SNNs must maintain and access membrane potentials across $T=4$ timesteps, memory traffic can easily eliminate theoretical arithmetic savings unless mapped to specialized on-chip SRAM crossbars. The proxy completely omits memory access, cache line refills, and weight buffering costs.

5. **Omitted Hardware and System-Level Costs:**
   - The proxy omits static leakage power, clock tree distribution, peripheral I/O circuitry, voltage regulator inefficiencies, and PCIe bus communication overhead.

6. **Absence of Neuromorphic Silicon:**
   - No physical neuromorphic hardware (e.g., Intel Loihi, SynSense Speck, SpiNNaker) was utilized or evaluated. Therefore, **neuromorphic hardware benefits were NOT demonstrated**.

---

## 6. Smoke Test Verdict & Training Status

$$\mathbf{SMOKE\ TEST\ EXECUTION:\ PASS}$$

- All numerical operations, gradient flows, multi-batch trackers, and latency harnesses executed without defect.
- **Clarification of Training Status:** Smoke-test training was executed for engineering verification; the full three-seed comparative campaign has not been launched.
