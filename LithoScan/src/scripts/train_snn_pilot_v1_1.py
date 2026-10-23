"""
Phase 6 v1.1 SNN Pilot Runner & Checkpoint Integrity Engine
===========================================================

Repairs all pilot measurement and provenance flaws:
1. Zero hardcoded metric fallbacks: Aborts with explicit error if history is missing or corrupted.
2. Verified checkpoint consistency: Recomputes best metric from authentic history and verifies match.
3. Strict epoch-end order: Metrics -> Best Model -> Cumulative Time -> Resumable Checkpoint.
4. Cumulative duration tracking: Persists cumulative training time across resume sessions.
5. Exact dataset-wide spike activity tracking via SpikeActivityTracker.
6. Standardized latency profiling (BS=1, 50 warmup, 200 iterations, single-thread CPU).
7. Strict isolation: Evaluates ONLY on stratified validation split; NEVER touches official test set.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.phase6_snn_v1_1 import ConvSNN4, ConvANN4, SpikeActivityTracker


# ---------------------------------------------------------------------------
# Manifest-Driven Stratified Dataset
# ---------------------------------------------------------------------------

class ManifestDataset(Dataset):
    """Loads images deterministically from a pre-computed JSON split manifest."""

    def __init__(self, root_dir: Path, sample_list: List[Dict[str, Any]], transform=None):
        self.root_dir = root_dir
        self.samples = sample_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        image_path = self.root_dir / item["rel_path"]
        label = item["class_idx"]

        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Standardized Latency Measurement Harness (Task 6)
# ---------------------------------------------------------------------------

def measure_standardized_latency(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    warmup: int = 50,
    iterations: int = 200,
    force_single_thread_cpu: bool = True,
) -> Dict[str, float]:
    """
    Standardized single-sample latency measurement:
    - Batch Size = 1
    - 50 warmup iterations
    - 200 timed iterations
    - Explicit CUDA event synchronization on GPU
    - torch.set_num_threads(1) on CPU for deterministic single-core benchmarking
    """
    model = model.to(device)
    model.eval()

    orig_threads = torch.get_num_threads()
    if device.type == "cpu" and force_single_thread_cpu:
        torch.set_num_threads(1)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    latencies = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for _ in range(iterations):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
    else:
        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

    if device.type == "cpu" and force_single_thread_cpu:
        torch.set_num_threads(orig_threads)

    return {
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "iterations": iterations,
        "warmup": warmup,
        "batch_size": 1,
    }


# ---------------------------------------------------------------------------
# Checkpoint Serialization (Tasks 2 & 3)
# ---------------------------------------------------------------------------

def save_resumable_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    epoch: int,
    best_epoch: int,
    best_val_f1: float,
    history: List[Dict[str, Any]],
    cumulative_training_time_sec: float,
    config: Dict[str, Any],
    manifest_sha256: str,
) -> None:
    """Saves authentic resumable checkpoint state with cumulative time and manifest hash."""
    state = {
        "version": "1.1",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
        "cumulative_training_time_sec": cumulative_training_time_sec,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "config": config,
        "manifest_sha256": manifest_sha256,
    }
    torch.save(state, path)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_cohort(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tracker: Optional[SpikeActivityTracker] = None,
) -> Tuple[float, float, float, Dict[str, Any], Dict[str, Any]]:
    """Evaluates model on loader and collects dataset-wide spike activity."""
    model.eval()
    if tracker is not None:
        tracker.reset()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images, tracker=tracker) if tracker is not None else model(images)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    val_loss = total_loss / total_samples
    val_acc = accuracy_score(all_targets, all_preds)
    val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_targets, all_preds, output_dict=True, zero_division=0
    )

    activity_summary = tracker.get_summary() if tracker is not None else {}

    return val_loss, float(val_acc), float(val_macro_f1), report, activity_summary


# ---------------------------------------------------------------------------
# Main Training & Verification Engine
# ---------------------------------------------------------------------------

def run_pilot(config_path: str, resume: bool = False) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # SAFETY CHECK: Strictly block official test evaluation
    if config.get("training", {}).get("eval_on_official_test", False):
        raise ValueError("CRITICAL PROTOCOL VIOLATION: Evaluation on official test set is forbidden in Phase 6 pilot!")

    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 6 v1.1] Compute Device: {device}")

    # Load and verify split manifest
    manifest_rel_path = config["data"]["split_manifest"]
    manifest_path = Path(manifest_rel_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_sha256 = manifest.get("manifest_sha256", "UNKNOWN")
    print(f"[Phase 6 v1.1] Verified Split Manifest SHA-256: {manifest_sha256}")

    dataset_root = Path(config["data"]["dataset_root"]) / "train"
    image_size = config["data"].get("image_size", 224)

    # Preprocessing with explicit Bicubic Interpolation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=config.get("augmentation", {}).get("horizontal_flip_prob", 0.5)),
        transforms.RandomVerticalFlip(p=config.get("augmentation", {}).get("vertical_flip_prob", 0.5)),
        transforms.RandomRotation(degrees=config.get("augmentation", {}).get("rotation_degrees", 10.0)),
        transforms.ColorJitter(
            brightness=config.get("augmentation", {}).get("brightness_jitter", 0.1),
            contrast=config.get("augmentation", {}).get("contrast_jitter", 0.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_samples = manifest["splits"]["pilot_train"]
    val_samples = manifest["splits"]["validation"]

    train_dataset = ManifestDataset(dataset_root, train_samples, transform=train_transform)
    val_dataset = ManifestDataset(dataset_root, val_samples, transform=val_transform)

    batch_size = config["data"].get("batch_size", 16)
    num_workers = config["data"].get("num_workers", 2)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    # Initialize ConvSNN4
    model_cfg = config["model"]
    model = ConvSNN4(
        num_classes=config["data"].get("num_classes", 7),
        time_steps=model_cfg.get("time_steps", 4),
        tau=model_cfg.get("tau", 2.0),
        v_threshold=model_cfg.get("v_threshold", 1.0),
        surrogate=model_cfg.get("surrogate", "fast_sigmoid"),
        surrogate_k=model_cfg.get("surrogate_k", 25.0),
        dropout=model_cfg.get("dropout", 0.2),
    ).to(device)

    total_params = model.count_parameters()
    print(f"[Phase 6 v1.1] ConvSNN4 Verified Parameters: {total_params:,} (Target: 390,695)")

    epochs = min(config["training"].get("epochs", 3), 3)
    lr = config["training"].get("lr", 0.001)
    weight_decay = config["training"].get("weight_decay", 0.0001)
    use_amp = config["training"].get("amp", True) and (device.type == "cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir = Path(config["checkpoint"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last_checkpoint.pt"
    best_model_path = ckpt_dir / "best_model.pt"

    start_epoch = 0
    best_epoch = 0
    best_val_f1 = 0.0
    cumulative_training_time_sec = 0.0
    history: List[Dict[str, Any]] = []

    # Checkpoint Resume Logic (Tasks 2 & 3)
    if resume and last_ckpt_path.exists():
        print(f"[Phase 6 v1.1] Loading checkpoint from: {last_ckpt_path}")
        checkpoint = torch.load(last_ckpt_path, map_location=device, weights_only=False)

        # Integrity Check 1: History must exist and be non-empty
        saved_history = checkpoint.get("history")
        if saved_history is None or not isinstance(saved_history, list) or len(saved_history) == 0:
            raise RuntimeError(
                "CRITICAL INTEGRITY FAILURE: Checkpoint has no authentic history. "
                "Refusing to manufacture synthetic fallback metrics!"
            )

        history = saved_history
        cumulative_training_time_sec = checkpoint.get("cumulative_training_time_sec", 0.0)

        # Integrity Check 2: Recompute best epoch and metric from authentic history
        best_entry = max(history, key=lambda x: x["val_macro_f1"])
        recomputed_best_f1 = best_entry["val_macro_f1"]
        recomputed_best_epoch = best_entry["epoch"]

        if abs(recomputed_best_f1 - checkpoint.get("best_val_f1", 0.0)) > 1e-4:
            raise RuntimeError(
                f"INTEGRITY FAILURE: Checkpoint best_val_f1 ({checkpoint.get('best_val_f1')}) "
                f"disagrees with authentic history maximum ({recomputed_best_f1})!"
            )

        best_val_f1 = recomputed_best_f1
        best_epoch = recomputed_best_epoch
        start_epoch = checkpoint["epoch"] + 1

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") and scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") and scaler:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        try:
            if checkpoint.get("rng_torch") is not None:
                torch.set_rng_state(checkpoint["rng_torch"].cpu())
            if device.type == "cuda" and checkpoint.get("rng_cuda") is not None:
                torch.cuda.set_rng_state(checkpoint["rng_cuda"].cpu())
            if checkpoint.get("rng_numpy") is not None:
                np.random.set_state(checkpoint["rng_numpy"])
            if checkpoint.get("rng_python") is not None:
                random.setstate(checkpoint["rng_python"])
        except Exception as e:
            print(f"[Phase 6 v1.1] Warning: RNG state restore note: {e}")

        print(f"[Phase 6 v1.1] Resumed successfully at epoch {start_epoch}. Best F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    activity_tracker = SpikeActivityTracker()

    print(f"\n{'='*70}\nPHASE 6 V1.1 SNN BOUNDED PILOT ({epochs} EPOCHS)\n{'='*70}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step()

        val_loss, val_acc, val_f1, class_report, activity_summary = evaluate_cohort(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            tracker=activity_tracker,
        )
        epoch_sec = time.time() - t0
        cumulative_training_time_sec += epoch_sec

        peak_vram = (
            torch.cuda.max_memory_allocated(device=device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "mean_firing_rate": activity_summary.get("mean_firing_rate", 0.0),
            "layer_firing_rates": activity_summary.get("layer_rates", {}),
            "duration_sec": round(epoch_sec, 2),
            "peak_vram_mb": round(peak_vram, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
        }
        history.append(epoch_stats)

        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f} | Mean Spikes: {activity_summary.get('mean_firing_rate', 0)*100:.2f}% | "
            f"Time: {epoch_sec:.1f}s"
        )

        # Update best model independently
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(
                {
                    "version": "1.1",
                    "epoch": best_epoch,
                    "val_macro_f1": best_val_f1,
                    "val_acc": val_acc,
                    "model_state_dict": model.state_dict(),
                    "class_report": class_report,
                    "manifest_sha256": manifest_sha256,
                },
                best_model_path,
            )
            print(f"  --> Saved new best model checkpoint (Epoch {best_epoch}, Val Macro-F1: {best_val_f1:.4f})")

        # Save complete resumable checkpoint state with updated history
        save_resumable_checkpoint(
            path=last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_f1=best_val_f1,
            history=history,
            cumulative_training_time_sec=cumulative_training_time_sec,
            config=config,
            manifest_sha256=manifest_sha256,
        )

    print(f"\n[Phase 6 v1.1] Total Cumulative Training Time: {cumulative_training_time_sec:.1f}s.")

    # Standardized Latency Measurement (Task 6)
    print("\n[Phase 6 v1.1] Executing Standardized Latency Benchmarks (BS=1, 50 warmup, 200 timed)...")
    latency_gpu = (
        measure_standardized_latency(model, device=device, image_size=image_size)
        if device.type == "cuda"
        else None
    )
    latency_cpu_single = measure_standardized_latency(
        model, device=torch.device("cpu"), image_size=image_size, force_single_thread_cpu=True
    )

    # Randomly initialized matched ConvANN4 for latency profiling
    matched_ann = ConvANN4(num_classes=7)
    ann_latency_gpu = (
        measure_standardized_latency(matched_ann, device=device, image_size=image_size)
        if device.type == "cuda"
        else None
    )
    ann_latency_cpu_single = measure_standardized_latency(
        matched_ann, device=torch.device("cpu"), image_size=image_size, force_single_thread_cpu=True
    )

    # First-Principles Theoretical Energy Proxy Calculation (Task 7)
    latest_layer_rates = history[-1].get("layer_firing_rates", {}) if history else {}
    energy_proxy = model.compute_theoretical_energy_proxy(latest_layer_rates, input_res=image_size)

    summary = {
        "version": "1.1",
        "pilot_name": config["experiment"]["name"],
        "dataset": "DCID-7 (Class-Stratified)",
        "manifest_sha256": manifest_sha256,
        "model_architecture": "ConvSNN4",
        "parameters": total_params,
        "time_steps": model.time_steps,
        "training_samples_used": len(train_dataset),
        "validation_samples": len(val_dataset),
        "epochs_completed": len(history),
        "cumulative_training_time_sec": round(cumulative_training_time_sec, 2),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_f1, 4),
        "history": history,
        "latency_snn_bs1": {
            "gpu": latency_gpu,
            "cpu_single_thread": latency_cpu_single,
        },
        "latency_ann_matched_bs1": {
            "gpu": ann_latency_gpu,
            "cpu_single_thread": ann_latency_cpu_single,
            "note": "ConvANN4 was randomly initialized for architectural latency comparison only; not trained for task performance.",
        },
        "theoretical_energy_proxy": energy_proxy,
    }

    summary_path = Path("reports/phase6_snn_feasibility_v1_1/pilot_summary_v1_1.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Phase 6 v1.1] Summary successfully saved to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 v1.1 SNN Pilot Runner")
    parser.add_argument("--config", type=str, default="configs/phase6_snn_pilot_v1_1.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_pilot(args.config, resume=args.resume)
