"""
inspect_results.py — Print structured summary of Paper B experiment results.
"""

import json
from pathlib import Path

METRICS_DIR = Path("results/paperB_uncertainty/metrics")

def inspect_model(benchmark, model_name):
    p = METRICS_DIR / f"{benchmark}_{model_name}_full_results.json"
    if not p.exists():
        print(f"File {p} does not exist yet.")
        return False

    with open(p, "r") as f:
        d = json.load(f)

    print(f"\n=================================================================")
    print(f"BENCHMARK: {d['benchmark']} | ARCHITECTURE: {d['model_name']}")
    print(f"Optimal Temperatures: T_clean = {d['temperature_clean']:.4f}, T_def = {d['temperature_def']:.4f}")
    print(f"=================================================================")

    print("\n--- 1. PROBABILITY CALIBRATION (ACCURACY, ECE, NLL, BRIER) ---")
    for k, v in d["calibration"].items():
        acc = v["accuracy"]["est"] * 100
        acc_ci = [x * 100 for x in v["accuracy"]["ci"]]
        ece = v["ece"]["est"] * 100
        ece_ci = [x * 100 for x in v["ece"]["ci"]]
        nll = v["nll"]["est"]
        nll_ci = v["nll"]["ci"]
        brier = v["brier"]["est"]
        print(f"  {k:14s} | Acc: {acc:5.2f}% [{acc_ci[0]:5.2f}, {acc_ci[1]:5.2f}] | ECE: {ece:4.2f}% [{ece_ci[0]:4.2f}, {ece_ci[1]:4.2f}] | NLL: {nll:6.4f} [{nll_ci[0]:6.4f}, {nll_ci[1]:6.4f}] | Brier: {brier:6.4f}")

    print("\n--- 2. SELECTIVE CLASSIFICATION RISK-COVERAGE & OPERATING POINTS ---")
    for sm in ["msp", "entropy", "margin"]:
        aurc_clean = d["selective_classification"][sm]["curves"]["eval_clean_aurc"] * 100
        aurc_def = d["selective_classification"][sm]["curves"]["eval_def_aurc"] * 100
        print(f"  Scoring '{sm}': Clean AURC = {aurc_clean:.4f}% | Defective AURC = {aurc_def:.4f}%")
        ops = d["selective_classification"][sm]["operating_points"]
        for op_k, op_v in ops.items():
            t_risk = op_v["target_risk"] * 100
            pA = op_v["pipeline_A_def_eval"]
            pB = op_v["pipeline_B_def_eval"]
            print(f"    Target {t_risk:2.0f}% Risk: Clean-Cal (tau={op_v['tau_clean']:.4f}) -> Cov={pA['coverage']*100:5.1f}%, Risk={pA['selective_risk']*100:5.2f}% | Def-Cal (tau={op_v['tau_def']:.4f}) -> Cov={pB['coverage']*100:5.1f}%, Risk={pB['selective_risk']*100:5.2f}%")

    print("\n--- 3. SPLIT CONFORMAL PREDICTION (ALPHA = 0.05, TARGET COV = 95%) ---")
    cp5 = d["conformal_prediction"]["alpha_5"]
    print(f"  Clean-Calibrated on Defective Eval:")
    print(f"    Marginal Coverage: {cp5['clean_on_def']['coverage']*100:.2f}% [{cp5['clean_on_def']['coverage_ci'][0]*100:.2f}, {cp5['clean_on_def']['coverage_ci'][1]*100:.2f}]")
    print(f"    Mean Set Size:     {cp5['clean_on_def']['mean_set_size']:.2f}")
    print(f"    Singleton Frac:    {cp5['clean_on_def']['singleton_frac']*100:.1f}% (Singleton Error: {cp5['clean_on_def']['singleton_error']*100:.2f}%)")
    print(f"    Referral Rate:     {cp5['clean_on_def']['referral_rate']*100:.1f}%")
    print(f"    Empty Set Frac:    {cp5['clean_on_def']['empty_frac']*100:.1f}%")

    print(f"  Defective-Calibrated (Domain-Matched) on Defective Eval:")
    print(f"    Marginal Coverage: {cp5['def_on_def']['coverage']*100:.2f}% [{cp5['def_on_def']['coverage_ci'][0]*100:.2f}, {cp5['def_on_def']['coverage_ci'][1]*100:.2f}]")
    print(f"    Mean Set Size:     {cp5['def_on_def']['mean_set_size']:.2f}")
    print(f"    Singleton Frac:    {cp5['def_on_def']['singleton_frac']*100:.1f}% (Singleton Error: {cp5['def_on_def']['singleton_error']*100:.2f}%)")
    print(f"    Referral Rate:     {cp5['def_on_def']['referral_rate']*100:.1f}%")
    print(f"    Empty Set Frac:    {cp5['def_on_def']['empty_frac']*100:.1f}%")
    print(f"  Wilcoxon Paired Signed-Rank Test: p = {cp5['wilcoxon_test']['p_value']:.4e} (mean diff = {cp5['wilcoxon_test']['mean_diff']:.2f})")

    print("\n--- 4. DEDUPLICATION SENSITIVITY ---")
    dedup = d["deduplication_sensitivity"]
    if dedup:
        print(f"  Excluded: {dedup['n_excluded']} | Retained: {dedup['n_retained']}")
        print(f"  Deduplicated Clean Eval Acc: {dedup['accuracy']*100:.2f}% | ECE: {dedup['ece']*100:.2f}% | NLL: {dedup['nll']:.4f}")
    return True

if __name__ == "__main__":
    inspect_model("dcid35", "resnet50")
    inspect_model("dcid35", "cnn_only")
    inspect_model("dcid7", "resnet50")
    inspect_model("dcid7", "cnn_only")
