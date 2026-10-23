"""
run_paperB_experiments.py — End-to-end execution of Paper B experiments.
Extracts logits from frozen checkpoints, performs temperature scaling, selective classification,
split conformal prediction, and stratified bootstrap statistical inference across DCID-35 and DCID-7.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

sys.path.insert(0, ".")
from src.models.server_fusion import LithoScanServerFusion
from src.calibration import paperB_methods as pbm

# ---------------------------------------------------------------------------
# Constants and Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DCID_ROOT = Path(os.environ.get("DCID_ROOT", PROJECT_ROOT.parent / "DCID" / "DCID"))
RESULTS_DIR = PROJECT_ROOT / "results" / "paperB_uncertainty"
LOGITS_DIR = RESULTS_DIR / "logits"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

LOGITS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 2026

# Standard eval transforms
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---------------------------------------------------------------------------
# Dataset for Arbitrary Manifest File Lists
# ---------------------------------------------------------------------------
class ManifestDataset(Dataset):
    def __init__(self, file_rel_paths: List[str], class_to_idx: Dict[str, int], transform=EVAL_TRANSFORMS):
        self.file_rel_paths = file_rel_paths
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples = []
        for rel_path in file_rel_paths:
            # Format: 'DCID-512-35/test/1.Red sandstone/1.Red sandstone_1.jpg'
            parts = rel_path.replace("\\", "/").split("/")
            class_name = parts[-2]
            label = self.class_to_idx[class_name]
            abs_path = DCID_ROOT / rel_path
            self.samples.append((str(abs_path), label, rel_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        abs_path, label, rel_path = self.samples[idx]
        with Image.open(abs_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)
        return tensor, label, rel_path


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------
def load_frozen_model(benchmark: str, model_type: str, num_classes: int) -> nn.Module:
    print(f"Loading frozen {model_type} for {benchmark} ({num_classes} classes)...")
    if benchmark == "DCID-35":
        if model_type == "resnet50":
            ckpt_path = PROJECT_ROOT / "checkpoints/exp_B/best_model.pt"
            model = models.resnet50(num_classes=num_classes)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        elif model_type == "cnn_only":
            ckpt_path = PROJECT_ROOT / "checkpoints/phase5/dcid35_cnn_only/best_model.pt"
            model = LithoScanServerFusion(num_classes=num_classes, architecture_mode="cnn_only", embed_dim=512, pretrained=False)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    elif benchmark == "DCID-7":
        if model_type == "resnet50":
            ckpt_path = PROJECT_ROOT / "checkpoints/exp_A/best_model.pt"
            model = models.resnet50(num_classes=num_classes)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        elif model_type == "cnn_only":
            ckpt_path = PROJECT_ROOT / "checkpoints/phase4/dcid7_cnn_only/best_model.pt"
            model = LithoScanServerFusion(num_classes=num_classes, architecture_mode="cnn_only", embed_dim=512, pretrained=False)
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    model.to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ---------------------------------------------------------------------------
# Logit Extraction and Caching
# ---------------------------------------------------------------------------
def extract_or_load_logits(
    model: nn.Module,
    dataset: ManifestDataset,
    cache_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if cache_path.exists():
        print(f"Loading cached logits from {cache_path.name}...")
        data = np.load(cache_path, allow_pickle=True)
        return data["logits"], data["probs"], data["labels"], data["paths"].tolist()

    print(f"Extracting forward logits to {cache_path.name} ({len(dataset)} samples)...")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    all_logits = []
    all_labels = []
    all_paths = []

    t0 = time.time()
    with torch.inference_mode():
        for batch_x, batch_y, batch_paths in loader:
            batch_x = batch_x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch_x)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(batch_y.numpy())
            all_paths.extend(batch_paths)

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute softmax probabilities
    exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
    all_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    elapsed = time.time() - t0
    print(f"  Extracted {len(all_labels)} samples in {elapsed:.2f}s ({len(all_labels)/elapsed:.1f} img/s)")

    np.savez_compressed(
        cache_path,
        logits=all_logits,
        probs=all_probs,
        labels=all_labels,
        paths=np.array(all_paths)
    )
    return all_logits, all_probs, all_labels, all_paths


# ---------------------------------------------------------------------------
# Benchmark Evaluator
# ---------------------------------------------------------------------------
def run_benchmark_evaluation(benchmark: str, model_name: str, config: Dict[str, Any]):
    print(f"\n=======================================================")
    print(f"RUNNING EXPERIMENT: {benchmark} | MODEL: {model_name}")
    print(f"=======================================================")

    manifest_file = PROJECT_ROOT / f"configs/paperB_uncertainty/paperB_split_manifest_{benchmark.lower().replace('-', '')}.json"
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    # Discover classes naturally ordered by numeric prefix
    clean_test_dir = DCID_ROOT / ("DCID-512-35/test" if benchmark == "DCID-35" else "DCID-512-7/test")
    raw_classes = os.listdir(clean_test_dir)
    classes = sorted(raw_classes, key=lambda x: int(x.split(".")[0]))
    class_to_idx = {c: int(c.split(".")[0]) - 1 for c in classes}
    num_classes = len(classes)

    # Load frozen model
    model = load_frozen_model(benchmark, model_name, num_classes)

    # Prepare datasets
    partitions = manifest["partitions"]
    ds_cal_clean = ManifestDataset(partitions["cal_clean"]["files"], class_to_idx)
    ds_eval_clean = ManifestDataset(partitions["eval_clean"]["files"], class_to_idx)
    ds_cal_def = ManifestDataset(partitions["cal_def"]["files"], class_to_idx)
    ds_eval_def = ManifestDataset(partitions["eval_def"]["files"], class_to_idx)

    # Extract or load logits
    prefix = f"{benchmark.lower().replace('-', '')}_{model_name}"
    cal_clean_logits, cal_clean_probs, cal_clean_y, cal_clean_paths = extract_or_load_logits(
        model, ds_cal_clean, LOGITS_DIR / f"{prefix}_cal_clean.npz"
    )
    eval_clean_logits, eval_clean_probs, eval_clean_y, eval_clean_paths = extract_or_load_logits(
        model, ds_eval_clean, LOGITS_DIR / f"{prefix}_eval_clean.npz"
    )
    cal_def_logits, cal_def_probs, cal_def_y, cal_def_paths = extract_or_load_logits(
        model, ds_cal_def, LOGITS_DIR / f"{prefix}_cal_def.npz"
    )
    eval_def_logits, eval_def_probs, eval_def_y, eval_def_paths = extract_or_load_logits(
        model, ds_eval_def, LOGITS_DIR / f"{prefix}_eval_def.npz"
    )

    # -----------------------------------------------------------------------
    # Step 1: Temperature Scaling Fitting
    # -----------------------------------------------------------------------
    print("\n--- Fitting Temperature Scaling ---")
    ts_clean = pbm.TemperatureScaling()
    t_clean = ts_clean.fit(torch.tensor(cal_clean_logits), torch.tensor(cal_clean_y))
    print(f"Optimal Temperature (Clean Calibration): T_clean = {t_clean:.4f}")

    ts_def = pbm.TemperatureScaling()
    t_def = ts_def.fit(torch.tensor(cal_def_logits), torch.tensor(cal_def_y))
    print(f"Optimal Temperature (Defective Calibration): T_def = {t_def:.4f}")

    # Scaled probabilities
    def scale_probs(logits, temp):
        scaled = logits / temp
        exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    eval_clean_probs_tclean = scale_probs(eval_clean_logits, t_clean)
    eval_def_probs_tclean = scale_probs(eval_def_logits, t_clean)
    eval_def_probs_tdef = scale_probs(eval_def_logits, t_def)

    # -----------------------------------------------------------------------
    # Step 2: Probability Calibration Metrics & Bootstrap 95% CIs
    # -----------------------------------------------------------------------
    print("\n--- Computing Probability Calibration Metrics (ECE, NLL, Brier) ---")
    calibration_results = {}

    eval_cases = [
        ("clean_uncal", eval_clean_probs, eval_clean_y, "Clean Evaluation (Uncalibrated)"),
        ("clean_tclean", eval_clean_probs_tclean, eval_clean_y, "Clean Evaluation (T_clean Scaled)"),
        ("def_uncal", eval_def_probs, eval_def_y, "Defective Evaluation (Uncalibrated)"),
        ("def_tclean", eval_def_probs_tclean, eval_def_y, "Defective Evaluation (T_clean Scaled, Shifted)"),
        ("def_tdef", eval_def_probs_tdef, eval_def_y, "Defective Evaluation (T_def Scaled, Matched)")
    ]

    for key, probs, labels, desc in eval_cases:
        base_metrics = pbm.compute_calibration_metrics(probs, labels)
        # Bootstrap CIs for Accuracy, NLL, Brier, ECE
        data_d = {"probs": probs, "labels": labels}

        _, acc_l, acc_u = pbm.stratified_bootstrap_ci(data_d, lambda d: np.mean(np.argmax(d["probs"], axis=1) == d["labels"]))
        _, nll_l, nll_u = pbm.stratified_bootstrap_ci(data_d, lambda d: pbm.compute_calibration_metrics(d["probs"], d["labels"])["nll"])
        _, bs_l, bs_u = pbm.stratified_bootstrap_ci(data_d, lambda d: pbm.compute_calibration_metrics(d["probs"], d["labels"])["brier"])
        _, ece_l, ece_u = pbm.stratified_bootstrap_ci(data_d, lambda d: pbm.compute_calibration_metrics(d["probs"], d["labels"])["ece"])

        calibration_results[key] = {
            "description": desc,
            "accuracy": {"est": base_metrics["accuracy"], "ci": [acc_l, acc_u]},
            "nll": {"est": base_metrics["nll"], "ci": [nll_l, nll_u]},
            "brier": {"est": base_metrics["brier"], "ci": [bs_l, bs_u]},
            "ece": {"est": base_metrics["ece"], "ci": [ece_l, ece_u]},
            "ece_adaptive": base_metrics["ece_adaptive"],
            "bin_confs": base_metrics["bin_confs"],
            "bin_accs": base_metrics["bin_accs"],
            "bin_counts": base_metrics["bin_counts"]
        }
        print(f"  [{key}] Acc: {base_metrics['accuracy']*100:.2f}% [{acc_l*100:.2f}, {acc_u*100:.2f}] | ECE: {base_metrics['ece']*100:.2f}% [{ece_l*100:.2f}, {ece_u*100:.2f}] | NLL: {base_metrics['nll']:.4f} [{nll_l:.4f}, {nll_u:.4f}] | Brier: {base_metrics['brier']:.4f}")

    # -----------------------------------------------------------------------
    # Step 3: Selective Classification & Risk-Coverage Analysis
    # -----------------------------------------------------------------------
    print("\n--- Evaluating Selective Classification & Operating Points ---")
    selective_results = {}
    scoring_methods = ["msp", "entropy", "margin"]

    for sm in scoring_methods:
        selective_results[sm] = {}
        # Calibration scores
        cal_clean_scores = pbm.compute_confidence_scores(cal_clean_probs, method=sm)
        cal_clean_preds = np.argmax(cal_clean_probs, axis=1)
        cal_def_scores = pbm.compute_confidence_scores(cal_def_probs, method=sm)
        cal_def_preds = np.argmax(cal_def_probs, axis=1)

        # Evaluation scores
        eval_clean_scores = pbm.compute_confidence_scores(eval_clean_probs, method=sm)
        eval_clean_preds = np.argmax(eval_clean_probs, axis=1)
        eval_def_scores = pbm.compute_confidence_scores(eval_def_probs, method=sm)
        eval_def_preds = np.argmax(eval_def_probs, axis=1)

        # Full curves & AURC
        clean_curve = pbm.compute_risk_coverage_curve(eval_clean_scores, eval_clean_preds, eval_clean_y)
        def_curve = pbm.compute_risk_coverage_curve(eval_def_scores, eval_def_preds, eval_def_y)

        selective_results[sm]["curves"] = {
            "eval_clean_aurc": clean_curve["aurc"],
            "eval_clean_e_aurc": clean_curve["e_aurc"],
            "eval_def_aurc": def_curve["aurc"],
            "eval_def_e_aurc": def_curve["e_aurc"],
            "eval_clean_curve": {"cov": clean_curve["coverages"], "risk": clean_curve["risks"]},
            "eval_def_curve": {"cov": def_curve["coverages"], "risk": def_curve["risks"]}
        }
        print(f"  Scoring '{sm}': Clean AURC = {clean_curve['aurc']*100:.3f}% | Defective AURC = {def_curve['aurc']*100:.3f}%")

        # Operating points (Target risks: 1%, 5%, 10%)
        selective_results[sm]["operating_points"] = {}
        for r_star in [0.01, 0.05, 0.10]:
            # Pipeline A: Clean Calibrated Threshold applied to Clean Eval & Defective Eval
            tau_clean = pbm.find_threshold_for_target_risk(cal_clean_scores, cal_clean_preds, cal_clean_y, target_risk=r_star)
            res_clean_on_clean = pbm.evaluate_at_threshold(eval_clean_scores, eval_clean_preds, eval_clean_y, tau_clean)
            res_clean_on_def = pbm.evaluate_at_threshold(eval_def_scores, eval_def_preds, eval_def_y, tau_clean)

            # Pipeline B: Defective Calibrated Threshold applied to Defective Eval
            tau_def = pbm.find_threshold_for_target_risk(cal_def_scores, cal_def_preds, cal_def_y, target_risk=r_star)
            res_def_on_def = pbm.evaluate_at_threshold(eval_def_scores, eval_def_preds, eval_def_y, tau_def)

            selective_results[sm]["operating_points"][f"target_risk_{int(r_star*100)}pct"] = {
                "target_risk": r_star,
                "tau_clean": tau_clean,
                "tau_def": tau_def,
                "pipeline_A_clean_eval": res_clean_on_clean,
                "pipeline_A_def_eval": res_clean_on_def,
                "pipeline_B_def_eval": res_def_on_def
            }
            print(f"    Target Risk {int(r_star*100)}%: tau_clean={tau_clean:.4f} -> Def Cov={res_clean_on_def['coverage']*100:.1f}%, Risk={res_clean_on_def['selective_risk']*100:.2f}% | tau_def={tau_def:.4f} -> Def Cov={res_def_on_def['coverage']*100:.1f}%, Risk={res_def_on_def['selective_risk']*100:.2f}%")

    # -----------------------------------------------------------------------
    # Step 4: Split Conformal Prediction
    # -----------------------------------------------------------------------
    print("\n--- Split Conformal Prediction (Coverage, Set Size, Referral) ---")
    conformal_results = {}
    alphas = [0.01, 0.05, 0.10]

    for alpha in alphas:
        # 1. Clean-calibrated conformal predictor
        cp_clean = pbm.SplitConformalClassifier(alpha=alpha)
        q_clean = cp_clean.fit(cal_clean_probs, cal_clean_y)
        res_cp_clean_on_clean = cp_clean.evaluate(eval_clean_probs, eval_clean_y)
        res_cp_clean_on_def = cp_clean.evaluate(eval_def_probs, eval_def_y)

        # 2. Defective-calibrated conformal predictor
        cp_def = pbm.SplitConformalClassifier(alpha=alpha)
        q_def = cp_def.fit(cal_def_probs, cal_def_y)
        res_cp_def_on_def = cp_def.evaluate(eval_def_probs, eval_def_y)

        # Bootstrap CIs for marginal coverage and set size
        # Clean on Def
        d_cov_clean_def = {"covered": res_cp_clean_on_def["covered_array"], "labels": eval_def_y}
        _, cov_c_l, cov_c_u = pbm.stratified_bootstrap_ci(d_cov_clean_def, lambda d: np.mean(d["covered"]))
        # Def on Def
        d_cov_def_def = {"covered": res_cp_def_on_def["covered_array"], "labels": eval_def_y}
        _, cov_d_l, cov_d_u = pbm.stratified_bootstrap_ci(d_cov_def_def, lambda d: np.mean(d["covered"]))

        # Paired Wilcoxon test on set sizes
        w_test = pbm.paired_wilcoxon_test(res_cp_clean_on_def["set_sizes_array"], res_cp_def_on_def["set_sizes_array"])

        conformal_results[f"alpha_{int(alpha*100)}"] = {
            "alpha": alpha,
            "nominal_coverage": 1.0 - alpha,
            "q_clean": q_clean,
            "q_def": q_def,
            "clean_on_clean": {
                "coverage": res_cp_clean_on_clean["marginal_coverage"],
                "mean_set_size": res_cp_clean_on_clean["mean_set_size"],
                "singleton_frac": res_cp_clean_on_clean["singleton_fraction"],
                "singleton_error": res_cp_clean_on_clean["singleton_error"],
                "referral_rate": res_cp_clean_on_clean["multi_class_referral_rate"],
                "empty_frac": res_cp_clean_on_clean["empty_fraction"],
                "size_dist": res_cp_clean_on_clean["set_size_distribution"]
            },
            "clean_on_def": {
                "coverage": res_cp_clean_on_def["marginal_coverage"],
                "coverage_ci": [cov_c_l, cov_c_u],
                "mean_set_size": res_cp_clean_on_def["mean_set_size"],
                "singleton_frac": res_cp_clean_on_def["singleton_fraction"],
                "singleton_error": res_cp_clean_on_def["singleton_error"],
                "referral_rate": res_cp_clean_on_def["multi_class_referral_rate"],
                "empty_frac": res_cp_clean_on_def["empty_fraction"],
                "size_dist": res_cp_clean_on_def["set_size_distribution"]
            },
            "def_on_def": {
                "coverage": res_cp_def_on_def["marginal_coverage"],
                "coverage_ci": [cov_d_l, cov_d_u],
                "mean_set_size": res_cp_def_on_def["mean_set_size"],
                "singleton_frac": res_cp_def_on_def["singleton_fraction"],
                "singleton_error": res_cp_def_on_def["singleton_error"],
                "referral_rate": res_cp_def_on_def["multi_class_referral_rate"],
                "empty_frac": res_cp_def_on_def["empty_fraction"],
                "size_dist": res_cp_def_on_def["set_size_distribution"]
            },
            "wilcoxon_test": w_test
        }
        print(f"  alpha={alpha:.2f} (Target Cov={100*(1-alpha):.0f}%):")
        print(f"    Clean-Cal on Def: Coverage={res_cp_clean_on_def['marginal_coverage']*100:.2f}% [{cov_c_l*100:.2f}, {cov_c_u*100:.2f}] | Set Size={res_cp_clean_on_def['mean_set_size']:.2f} | Singletons={res_cp_clean_on_def['singleton_fraction']*100:.1f}%")
        print(f"    Def-Cal on Def  : Coverage={res_cp_def_on_def['marginal_coverage']*100:.2f}% [{cov_d_l*100:.2f}, {cov_d_u*100:.2f}] | Set Size={res_cp_def_on_def['mean_set_size']:.2f} | Singletons={res_cp_def_on_def['singleton_fraction']*100:.1f}%")
        print(f"    Wilcoxon p-value: {w_test['p_value']:.4e} (mean diff: {w_test['mean_diff']:.2f})")

    # -----------------------------------------------------------------------
    # Step 5: Per-Class Fine-Grained Failure & Referral Taxonomy
    # -----------------------------------------------------------------------
    print("\n--- Per-Class Performance and Referral Breakdown ---")
    per_class_summary = []
    eval_def_preds = np.argmax(eval_def_probs, axis=1)
    eval_def_msp = np.max(eval_def_probs, axis=1)
    # Using alpha=0.05 conformal sets
    cp_def_05 = pbm.SplitConformalClassifier(alpha=0.05)
    cp_def_05.fit(cal_def_probs, cal_def_y)
    eval_def_sets = cp_def_05.predict(eval_def_probs)

    # 5% selective risk threshold on MSP
    tau_def_5pct = selective_results["msp"]["operating_points"]["target_risk_5pct"]["tau_def"]
    eval_def_accepted = eval_def_msp >= tau_def_5pct

    for c_idx, c_name in enumerate(classes):
        c_mask = (eval_def_y == c_idx)
        c_n = int(np.sum(c_mask))
        c_acc = float(np.mean(eval_def_preds[c_mask] == eval_def_y[c_mask]))
        c_msp_mean = float(np.mean(eval_def_msp[c_mask]))
        c_referred = float(1.0 - np.mean(eval_def_accepted[c_mask]))

        # Conformal metrics for this class
        c_cov = float(np.mean([1.0 if c_idx in eval_def_sets[i] else 0.0 for i in np.where(c_mask)[0]]))
        c_size = float(np.mean([len(eval_def_sets[i]) for i in np.where(c_mask)[0]]))

        per_class_summary.append({
            "class_idx": c_idx,
            "class_name": c_name,
            "sample_count": c_n,
            "defective_accuracy": c_acc,
            "mean_msp_confidence": c_msp_mean,
            "selective_referral_rate_5pct": c_referred,
            "conformal_coverage_05": c_cov,
            "mean_set_size_05": c_size
        })

    # -----------------------------------------------------------------------
    # Step 6: Deduplication Sensitivity Analysis (dHash <= 2)
    # -----------------------------------------------------------------------
    print("\n--- Deduplication Sensitivity Analysis ---")
    # Identify duplicate paths from Phase 3 audit
    dup_csv = PROJECT_ROOT / "reports/phase3/near_duplicate_pairs.csv"
    if not dup_csv.exists():
        dup_csv = PROJECT_ROOT / "reports/phase3/near_duplicate_pairs.csv.gz"
    dedup_metrics = {}
    if dup_csv.exists():
        df_dup = pd.read_csv(dup_csv)
        ds_name = "DCID-512-35" if benchmark == "DCID-35" else "DCID-512-7"
        df_sub = df_dup[(df_dup["dataset"] == ds_name) & (df_dup["distance"] <= 2)]
        dup_test_paths = set()
        for _, row in df_sub.iterrows():
            if "test" in str(row["path_2"]).lower():
                # Extract filename
                p_base = os.path.basename(str(row["path_2"]))
                dup_test_paths.add(p_base)

        print(f"Identified {len(dup_test_paths)} test images with training near-duplicate neighbors (dHash <= 2)")
        # Filter eval_clean
        clean_filenames = [os.path.basename(p) for p in eval_clean_paths]
        non_dup_mask = np.array([f not in dup_test_paths for f in clean_filenames])
        n_retained = int(np.sum(non_dup_mask))
        n_excluded = len(non_dup_mask) - n_retained
        print(f"Retained {n_retained} / {len(non_dup_mask)} clean evaluation samples ({n_retained/len(non_dup_mask)*100:.1f}%)")

        if n_retained > 0:
            dedup_clean_probs = eval_clean_probs[non_dup_mask]
            dedup_clean_y = eval_clean_y[non_dup_mask]
            dedup_cal_metrics = pbm.compute_calibration_metrics(dedup_clean_probs, dedup_clean_y)
            dedup_metrics = {
                "n_excluded": n_excluded,
                "n_retained": n_retained,
                "accuracy": dedup_cal_metrics["accuracy"],
                "ece": dedup_cal_metrics["ece"],
                "nll": dedup_cal_metrics["nll"],
                "brier": dedup_cal_metrics["brier"]
            }
            print(f"  Deduplicated Clean Eval: Acc={dedup_cal_metrics['accuracy']*100:.2f}% | ECE={dedup_cal_metrics['ece']*100:.2f}% | NLL={dedup_cal_metrics['nll']:.4f}")

    # -----------------------------------------------------------------------
    # Step 7: Packaging and Serializing Outputs
    # -----------------------------------------------------------------------
    experiment_results = {
        "benchmark": benchmark,
        "model_name": model_name,
        "temperature_clean": t_clean,
        "temperature_def": t_def,
        "calibration": calibration_results,
        "selective_classification": selective_results,
        "conformal_prediction": conformal_results,
        "per_class_summary": per_class_summary,
        "deduplication_sensitivity": dedup_metrics
    }

    def to_serializable(val):
        if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
            return int(val)
        if isinstance(val, (np.float64, np.float32, np.float16)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {str(k): to_serializable(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [to_serializable(x) for x in val]
        return val

    serializable_results = to_serializable(experiment_results)
    out_file = METRICS_DIR / f"{prefix}_full_results.json"
    with open(out_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nExperiment results successfully serialized to {out_file}")
    return serializable_results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print(f"Starting LithoScan Paper B Experiments on {DEVICE}...")
    t_start = time.time()

    config_path = PROJECT_ROOT / "configs/paperB_uncertainty/experiment_config.yaml"
    with open(config_path, "r") as f:
        # Simple YAML-free loader or yaml parse
        import yaml
        config = yaml.safe_load(f)

    results = {}
    # 1. Primary Benchmark: DCID-35 ResNet-50
    results["dcid35_resnet50"] = run_benchmark_evaluation("DCID-35", "resnet50", config)

    # 2. Primary Benchmark: DCID-35 CNN-only
    results["dcid35_cnn_only"] = run_benchmark_evaluation("DCID-35", "cnn_only", config)

    # 3. Replication Benchmark: DCID-7 ResNet-50
    results["dcid7_resnet50"] = run_benchmark_evaluation("DCID-7", "resnet50", config)

    # 4. Replication Benchmark: DCID-7 CNN-only
    results["dcid7_cnn_only"] = run_benchmark_evaluation("DCID-7", "cnn_only", config)

    # Summary manifest
    summary_path = METRICS_DIR / "all_experiments_summary.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(DEVICE),
        "benchmarks_completed": list(results.keys()),
        "total_runtime_seconds": time.time() - t_start
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nAll Paper B experiments completed successfully in {time.time() - t_start:.2f} seconds!")


if __name__ == "__main__":
    main()
