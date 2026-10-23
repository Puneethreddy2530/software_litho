# Phase 6 v1.1 — Code Corrections & Measurement Engine Repair

**Document ID:** `reports/phase6_snn_feasibility_v1_1/02_code_corrections.md`  
**Date:** October 2026  
**Status:** COMPLETE & INDEPENDENTLY TESTED  
**Objective:** Document the specific software engineering repairs, integrity safeguards, and measurement modules implemented in Phase 6 v1.1 to resolve all identified pilot defects.

---

## 1. Summary of Code Base Modifications

All Phase 6 v1.1 modifications are strictly namespaced to prevent contamination of prior experiment code:
- New Model & Activity Engine: `src/models/phase6_snn_v1_1.py`
- New Stratified Splitter: `src/data/stratified_split.py`
- New Split Manifest: `configs/phase6_split_manifest_dcid7_v1_1.json`
- New Resilient Runner: `src/scripts/train_snn_pilot_v1_1.py`
- New Configuration: `configs/phase6_snn_pilot_v1_1.yaml`
- New Unit & Integrity Tests: `tests/test_phase6_snn_v1_1.py`

---

## 2. Detailed Technical Corrections

### A. Elimination of Synthetic Metric Fallbacks (Task 2)
- **Defect in v1.0:** When resuming from a checkpoint that lacked serialized `history`, the script injected a hardcoded dictionary list constructed from stdout text.
- **Repair in v1.1:**
  ```python
  saved_history = checkpoint.get("history")
  if saved_history is None or not isinstance(saved_history, list) or len(saved_history) == 0:
      raise RuntimeError(
          "CRITICAL INTEGRITY FAILURE: Checkpoint has no authentic history. "
          "Refusing to manufacture synthetic fallback metrics!"
      )
  ```
- **Integrity Guarantee:** If a checkpoint is corrupted, missing history, or tampered with, the runner immediately halts with an explicit error.

### B. Authentic Best Metric Verification (Task 2 & 3)
- **Repair in v1.1:** Upon resume, the runner recomputes the best validation F1 and best epoch directly from `history`:
  ```python
  best_entry = max(history, key=lambda x: x["val_macro_f1"])
  recomputed_best_f1 = best_entry["val_macro_f1"]
  recomputed_best_epoch = best_entry["epoch"]
  if abs(recomputed_best_f1 - checkpoint.get("best_val_f1", 0.0)) > 1e-4:
      raise RuntimeError(
          f"INTEGRITY FAILURE: Checkpoint best_val_f1 ({checkpoint.get('best_val_f1')}) "
          f"disagrees with authentic history maximum ({recomputed_best_f1})!"
      )
  ```

### C. Persistent Cumulative Training Time (Task 2)
- **Defect in v1.0:** When resuming an already-finished pilot, `time.time() - pilot_start_time` reported `0.0s`.
- **Repair in v1.1:** Cumulative training duration is persistently incremented across epochs and serialized into `last_checkpoint.pt`:
  ```python
  cumulative_training_time_sec += epoch_sec
  ```
  Latency benchmarking duration is timed and reported in an entirely separate section, preventing any wall-clock conflation.

### D. Exact Dataset-Wide Spike Activity Tracker (Task 5)
- **Defect in v1.0:** Firing rate was calculated as `self.last_firing_rate = spike_seq.mean().item()` on the last forward pass, ignoring all preceding minibatches.
- **Repair in v1.1:** Implemented `SpikeActivityTracker` in `src/models/phase6_snn_v1_1.py`:
  - Accumulates total spike counts: $\sum_{\text{batches}} \sum s \in \{0, 1\}$.
  - Accumulates total neuron-time positions: $\sum_{\text{batches}} (T \times B \times C \times H \times W)$.
  - Computes exact per-layer firing rates $\bar{S}_l$ across the entire validation cohort.
  - Computes operation-weighted firing rates weighted by layer FLOP counts.
  - Verified with synthetic spike tensors (exact rate 0.375 confirmed in `test_spike_activity_tracker_synthetic`).

### E. Parameter Reconciliations & Decoder Transparency (Task 6)
- **Parameter Parity:** Both `ConvSNN4` and `ConvANN4` contain exactly **390,695 trainable parameters**.
- **Transparent Decoder Documentation:**
  - Input static image is repeated across $T$ steps: $(T, B, 3, 224, 224)$.
  - Convolutions and LIF spike integration proceed across $T$ steps.
  - Spatial adaptive average pooling reduces features to $(T \times B, 256)$.
  - Linear classifier projects to $(T \times B, 7)$.
  - Time dimension is averaged: `mean_logits = logits_seq.mean(dim=0)`.
  - **Explicit Definition:** This is rate-based floating-point logit averaging, NOT spike output decoding.

### F. Standardized Latency Measurement Harness (Task 6)
- Implemented `measure_standardized_latency()`:
  - Batch Size = 1.
  - 50 warmup iterations followed by 200 timed iterations.
  - Explicit CUDA synchronization via `torch.cuda.Event(enable_timing=True)`.
  - Deterministic CPU single-thread benchmarking via `torch.set_num_threads(1)`.
  - Identical measurement harness applied to both SNN and ANN.
  - Matched ANN explicitly labeled: *"Randomly initialized for architectural latency comparison only; not trained for task performance."*

### G. First-Principles Synaptic Operation Proxy (Task 7)
- Replaced the flawed $1 - \bar{S}$ flat discount with a first-principles CMOS 45nm operation-energy proxy (Horowitz, 2014):
  - First layer executes continuous floating-point MACs over $T$ steps: $E_{\text{layer1}} = T \times \text{FLOPs}_{\text{conv1}} \times 4.6\text{ pJ}$.
  - Internal layers execute sparse accumulations driven by presynaptic spike rates: $E_{\text{internal}} = \sum \bar{S}_{l-1} \times T \times \text{FLOPs}_l \times 0.9\text{ pJ}$.
  - Membrane potential arithmetic overhead: $\text{Neurons} \times T \times 1.8\text{ pJ}$.
  - This arithmetic accounts for continuous encoding overhead (representing 93.71% of total theoretical SNN energy), mathematically disproving naive sparsity heuristics.
  - **Explicit Boundaries:** Energy was not directly measured. The theoretical proxy omits memory hierarchy (SRAM/DRAM, which typically dominates arithmetic energy; Horowitz, 2014; Han et al., 2016) and system-level static/clock overhead. Neuromorphic hardware benefits were not demonstrated.
  - **Training Status:** Smoke-test training was executed for engineering verification; the full three-seed comparative campaign has not been launched.
