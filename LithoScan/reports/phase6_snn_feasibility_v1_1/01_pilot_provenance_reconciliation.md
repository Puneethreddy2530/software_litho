# Phase 6 v1.1 — Pilot Provenance & Metric Reconciliation

**Document ID:** `reports/phase6_snn_feasibility_v1_1/01_pilot_provenance_reconciliation.md`  
**Date:** October 2026  
**Status:** COMPLETE & FORENSICALLY RECONCILED  
**Objective:** Perform an exhaustive forensic audit of the Phase 6 SNN pilot execution, reconciling runtime logs, physical checkpoint contents, summary files, and code flaws without manufacturing unverified metrics.

---

## 1. Forensic Evidence Trail

We inspected five primary artifacts:
1. `C:\Users\punee\.gemini\antigravity-ide\brain\d72fa4fc-6368-4c43-b91b-06352007d392\.system_generated\tasks\task-376.log`
2. `checkpoints/phase6_snn_pilot/last_checkpoint.pt`
3. `checkpoints/phase6_snn_pilot/best_model.pt`
4. `reports/phase6_snn_feasibility/pilot_summary.json`
5. `reports/phase6_snn_feasibility/05_pilot_results.md`

---

## 2. Checkpoint State Audit

Direct programmatic inspection of the physical checkpoint files yielded:

### `checkpoints/phase6_snn_pilot/last_checkpoint.pt`
- **File Size:** 4,728,724 bytes (~4.7 MB)
- **Top-level Keys:** `['epoch', 'best_val_f1', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'rng_torch', 'rng_cuda', 'rng_numpy', 'rng_python', 'config']`
- **`epoch`:** `2` (indicates completion of 0-indexed epochs 0, 1, 2 = 3 epochs completed)
- **`best_val_f1`:** `0.9474283841891407`
- **`history`:** **KEY MISSING (`None`).** In the original runner executed in `task-376`, `save_checkpoint` did not serialize the `history` list to disk.
- **RNG States:** Fully preserved (`rng_torch`, `rng_cuda`, `rng_numpy`, `rng_python`).

### `checkpoints/phase6_snn_pilot/best_model.pt`
- **File Size:** 1,576,482 bytes (~1.5 MB)
- **Top-level Keys:** `['epoch', 'val_macro_f1', 'val_acc', 'model_state_dict', 'class_report']`
- **`epoch`:** `2` (saved when Epoch 2 achieved the peak validation score)
- **`val_macro_f1`:** `0.9474283841891407`
- **`val_acc`:** `0.9485714285714286`
- **`class_report`:** Contains authentic dictionary containing per-class precision, recall, and F1 across classes 0–6.

---

## 3. Discrepancy & Provenance Reconciliation Matrix

| Metric / Field | Value in `pilot_summary.json` | Authentic Source Evidence | Provenance Classification | Root Cause / Forensic Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Epoch 1 Train/Val Metrics** | Train Loss: 0.6211, Val Loss: 0.4586, Val Acc: 82.54%, Val F1: 0.8208 | `task-376.log` stdout | **PARTIALLY VERIFIED** | Recorded in `task-376.log` stdout stream; absent from `last_checkpoint.pt` binary payload. |
| **Epoch 2 Train/Val Metrics** | Train Loss: 0.3282, Val Loss: 0.1650, Val Acc: 94.86%, Val F1: 0.9474 | `best_model.pt` & `task-376.log` | **VERIFIED FACT** | Explicitly preserved in `best_model.pt` (`val_macro_f1 = 0.947428...`, `val_acc = 0.948571...`). |
| **Epoch 3 Train/Val Metrics** | Train Loss: 0.2413, Val Loss: 0.1399, Val Acc: 94.86%, Val F1: 0.9467 | `task-376.log` stdout | **PARTIALLY VERIFIED** | Recorded in `task-376.log` stdout stream; absent from `last_checkpoint.pt` binary payload. |
| **Total Wall-Clock Training Time** | `0.0` seconds | Sum of epochs = 274.6 s | **UNVERIFIED / DEFECTIVE** | Code bug: When resuming at Epoch 3 (training already complete), `pilot_start_time` was initialized at script startup and measured `time.time() - pilot_start_time` across 0 epochs, overwriting the actual 274.6s training time with `0.0s`. |
| **Mean Spike Firing Rates (Epochs 1–3)** | 24.4%, 23.5%, 23.6% | `task-376.log` stdout | **PARTIALLY VERIFIED (Batch-level only)** | Critical architectural bug: `model.get_firing_rates()` in `phase6_snn.py` recorded only the final minibatch of the epoch, NOT a dataset-wide or validation-wide average. |
| **Inference Latency (GPU BS=1)** | Median: 9.54 ms, p95: 10.34 ms | `pilot_summary.json` & script output | **VERIFIED FACT** | Directly measured on RTX 3050 across 100 timed iterations after warmup during resume session. |
| **Inference Latency (CPU BS=1)** | Median: 45.29 ms, p95: 46.99 ms | `pilot_summary.json` & script output | **VERIFIED FACT** | Directly measured on host CPU across 30 timed iterations during resume session. |
| **Matched ANN Latency** | GPU: 0.92 ms, CPU: 9.25 ms | `pilot_summary.json` & script output | **VERIFIED FACT** | Measured on randomly initialized `ConvANN4` baseline (not trained for classification). |

---

## 4. Methodological Defects Identified

1. **Defect 1: Hardcoded Metric Fallback on Resume**
   - In `train_snn_pilot.py`, when resuming an execution where `history` was missing from `last_checkpoint.pt`, the script injected a hardcoded list of dictionaries copied from earlier console outputs.
   - **Correction:** This practice violates scientific reproducibility standards. In v1.1, if a checkpoint lacks authentic history or disagrees with `best_model.pt`, the runner must halt immediately with an explicit error.
2. **Defect 2: Wall-Clock Training vs. Latency Benchmark Conflation**
   - The runner computed total training time without accumulating prior session runtimes, reporting `0.0s` in the JSON summary.
   - **Correction:** v1.1 implements persistent tracking of `cumulative_training_time_sec` across resume boundaries and isolates benchmark timing into a dedicated section.
3. **Defect 3: Non-Accumulating Firing-Rate Measurement**
   - `MultiStepLIFNode` tracked `self.last_firing_rate` on whatever forward pass ran last (the last training minibatch), ignoring the rest of the dataset.
   - **Correction:** v1.1 introduces an exact `SpikeActivityTracker` that accumulates all spikes and all neuron-time positions across the entire validation cohort.
4. **Defect 4: Random Split vs. Stratified Split**
   - The original pilot used `torch.utils.data.random_split`, which produces stochastic class allocations without guaranteed per-class balance.
   - **Correction:** v1.1 implements a deterministic class-stratified splitter with verified manifests (360 train, 400 val per class).

---

## 5. Formal Scientific Verdict on the Original Pilot

- The Phase 6 pilot successfully proved **pipeline functionality and gradient backpropagation** (loss decreased from 0.6211 to 0.1399; validation accuracy reached 94.86%).
- However, due to Defects 1–4, **the numerical metrics from the original pilot are marked PARTIALLY VERIFIED and CANNOT be cited as definitive experimental data**.
- All controlled evaluations must proceed under the repaired v1.1 protocol.
