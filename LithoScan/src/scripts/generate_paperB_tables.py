"""
generate_paperB_tables.py — Generate LaTeX and CSV summary tables for Paper B manuscript.
Reads directly from authentic JSON results in results/paperB_uncertainty/metrics/.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "results" / "paperB_uncertainty" / "metrics"
TABLES_DIR = PROJECT_ROOT / "manuscript" / "paperB_uncertainty" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def load_metrics(benchmark, model_name):
    p = METRICS_DIR / f"{benchmark}_{model_name}_full_results.json"
    with open(p, "r") as f:
        return json.load(f)

def generate_table1_dataset_partitions():
    rows = [
        {"Benchmark": "DCID-35", "Partition": "Model Fitting ($D_{\\text{fit}}$)", "Source Directory": "DCID-512-35/train [:25200]", "Total Images": "25,200", "Images/Class": "720", "Role": "Historical Weight Fitting"},
        {"Benchmark": "DCID-35", "Partition": "Model Selection ($D_{\\text{sel}}$)", "Source Directory": "DCID-512-35/train [25200:]", "Total Images": "2,799", "Images/Class": "~80", "Role": "Historical Model Selection"},
        {"Benchmark": "DCID-35", "Partition": "Clean Calibration ($D_{\\text{cal,clean}}$)", "Source Directory": "DCID-512-35/test [:50]", "Total Images": "1,750", "Images/Class": "50", "Role": "Clean Probability & Conformal Calibration"},
        {"Benchmark": "DCID-35", "Partition": "Clean Evaluation ($D_{\\text{eval,clean}}$)", "Source Directory": "DCID-512-35/test [50:]", "Total Images": "5,250", "Images/Class": "150", "Role": "Locked Clean Benchmark Evaluation"},
        {"Benchmark": "DCID-35", "Partition": "Defective Calibration ($D_{\\text{cal,def}}$)", "Source Directory": "noise-512-35/train [:100]", "Total Images": "3,500", "Images/Class": "100", "Role": "Domain-Matched Calibration (Untouched Pool)"},
        {"Benchmark": "DCID-35", "Partition": "Defective Evaluation ($D_{\\text{eval,def}}$)", "Source Directory": "noise-512-35/test [full]", "Total Images": "2,800", "Images/Class": "80", "Role": "Locked Defective Benchmark Evaluation"},
        {"Benchmark": "DCID-7", "Partition": "Clean Calibration ($D_{\\text{cal,clean}}$)", "Source Directory": "DCID-512-7/test [:250]", "Total Images": "1,750", "Images/Class": "250", "Role": "Clean Calibration (Replication)"},
        {"Benchmark": "DCID-7", "Partition": "Clean Evaluation ($D_{\\text{eval,clean}}$)", "Source Directory": "DCID-512-7/test [250:]", "Total Images": "5,250", "Images/Class": "750", "Role": "Clean Evaluation (Replication)"},
        {"Benchmark": "DCID-7", "Partition": "Defective Calibration ($D_{\\text{cal,def}}$)", "Source Directory": "noise-512-7/train [:400]", "Total Images": "2,800", "Images/Class": "400", "Role": "Defective Calibration (Replication)"},
        {"Benchmark": "DCID-7", "Partition": "Defective Evaluation ($D_{\\text{eval,def}}$)", "Source Directory": "noise-512-7/test [full]", "Total Images": "2,800", "Images/Class": "400", "Role": "Defective Evaluation (Replication)"}
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table1_dataset_partitions.csv", index=False)
    print("Table 1 saved.")

def generate_table2_calibration():
    models_to_eval = [
        ("DCID-35", "ResNet-50", "dcid35", "resnet50"),
        ("DCID-35", "CNN-Only", "dcid35", "cnn_only"),
        ("DCID-7", "ResNet-50", "dcid7", "resnet50"),
        ("DCID-7", "CNN-Only", "dcid7", "cnn_only")
    ]
    rows = []
    for bench_name, m_label, b_code, m_code in models_to_eval:
        d = load_metrics(b_code, m_code)
        cal = d["calibration"]
        # 1. Clean Uncalibrated
        c_u = cal["clean_uncal"]
        rows.append({
            "Benchmark": bench_name, "Architecture": m_label, "Cohort & Strategy": "Clean (Uncalibrated)",
            "Accuracy (%) [95% CI]": f"{c_u['accuracy']['est']*100:.2f} [{c_u['accuracy']['ci'][0]*100:.2f}, {c_u['accuracy']['ci'][1]*100:.2f}]",
            "ECE (%) [95% CI]": f"{c_u['ece']['est']*100:.2f} [{c_u['ece']['ci'][0]*100:.2f}, {c_u['ece']['ci'][1]*100:.2f}]",
            "NLL [95% CI]": f"{c_u['nll']['est']:.4f} [{c_u['nll']['ci'][0]:.4f}, {c_u['nll']['ci'][1]:.4f}]",
            "Brier Score": f"{c_u['brier']['est']:.4f}"
        })
        # 2. Defective Uncalibrated
        d_u = cal["def_uncal"]
        rows.append({
            "Benchmark": bench_name, "Architecture": m_label, "Cohort & Strategy": "Defective (Uncalibrated)",
            "Accuracy (%) [95% CI]": f"{d_u['accuracy']['est']*100:.2f} [{d_u['accuracy']['ci'][0]*100:.2f}, {d_u['accuracy']['ci'][1]*100:.2f}]",
            "ECE (%) [95% CI]": f"{d_u['ece']['est']*100:.2f} [{d_u['ece']['ci'][0]*100:.2f}, {d_u['ece']['ci'][1]*100:.2f}]",
            "NLL [95% CI]": f"{d_u['nll']['est']:.4f} [{d_u['nll']['ci'][0]:.4f}, {d_u['nll']['ci'][1]:.4f}]",
            "Brier Score": f"{d_u['brier']['est']:.4f}"
        })
        # 3. Defective Clean-Calibrated (T_clean)
        d_tc = cal["def_tclean"]
        rows.append({
            "Benchmark": bench_name, "Architecture": m_label, "Cohort & Strategy": "Defective ($T_{\\text{clean}}$ Scaled, Shifted)",
            "Accuracy (%) [95% CI]": f"{d_tc['accuracy']['est']*100:.2f} [{d_tc['accuracy']['ci'][0]*100:.2f}, {d_tc['accuracy']['ci'][1]*100:.2f}]",
            "ECE (%) [95% CI]": f"{d_tc['ece']['est']*100:.2f} [{d_tc['ece']['ci'][0]*100:.2f}, {d_tc['ece']['ci'][1]*100:.2f}]",
            "NLL [95% CI]": f"{d_tc['nll']['est']:.4f} [{d_tc['nll']['ci'][0]:.4f}, {d_tc['nll']['ci'][1]:.4f}]",
            "Brier Score": f"{d_tc['brier']['est']:.4f}"
        })
        # 4. Defective Matched-Calibrated (T_def)
        d_td = cal["def_tdef"]
        rows.append({
            "Benchmark": bench_name, "Architecture": m_label, "Cohort & Strategy": "Defective ($T_{\\text{def}}$ Scaled, Matched)",
            "Accuracy (%) [95% CI]": f"{d_td['accuracy']['est']*100:.2f} [{d_td['accuracy']['ci'][0]*100:.2f}, {d_td['accuracy']['ci'][1]*100:.2f}]",
            "ECE (%) [95% CI]": f"{d_td['ece']['est']*100:.2f} [{d_td['ece']['ci'][0]*100:.2f}, {d_td['ece']['ci'][1]*100:.2f}]",
            "NLL [95% CI]": f"{d_td['nll']['est']:.4f} [{d_td['nll']['ci'][0]:.4f}, {d_td['nll']['ci'][1]:.4f}]",
            "Brier Score": f"{d_td['brier']['est']:.4f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table2_calibration_metrics.csv", index=False)
    print("Table 2 saved.")

def generate_table3_selective():
    models_to_eval = [
        ("DCID-35", "ResNet-50", "dcid35", "resnet50"),
        ("DCID-35", "CNN-Only", "dcid35", "cnn_only"),
        ("DCID-7", "ResNet-50", "dcid7", "resnet50"),
        ("DCID-7", "CNN-Only", "dcid7", "cnn_only")
    ]
    rows = []
    for bench_name, m_label, b_code, m_code in models_to_eval:
        d = load_metrics(b_code, m_code)
        msp_ops = d["selective_classification"]["msp"]["operating_points"]
        for r_code, r_val in [("target_risk_1pct", 1.0), ("target_risk_5pct", 5.0), ("target_risk_10pct", 10.0)]:
            op = msp_ops[r_code]
            pA = op["pipeline_A_def_eval"]
            pB = op["pipeline_B_def_eval"]
            rows.append({
                "Benchmark": bench_name, "Architecture": m_label, "Nominal Error Target": f"{r_val:.0f}%",
                "Pipeline A (Clean-Cal) Threshold": f"{op['tau_clean']:.4f}",
                "Pipeline A Realized Cov (%)": f"{pA['coverage']*100:.1f}%",
                "Pipeline A Referral Rate (%)": f"{pA['referral_rate']*100:.1f}%",
                "Pipeline A Realized Error (%)": f"{pA['selective_risk']*100:.2f}%",
                "Pipeline B (Def-Cal) Threshold": f"{op['tau_def']:.4f}",
                "Pipeline B Realized Cov (%)": f"{pB['coverage']*100:.1f}%",
                "Pipeline B Referral Rate (%)": f"{pB['referral_rate']*100:.1f}%",
                "Pipeline B Realized Error (%)": f"{pB['selective_risk']*100:.2f}%"
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table3_selective_risk_operating_points.csv", index=False)
    print("Table 3 saved.")

def generate_table4_conformal():
    models_to_eval = [
        ("DCID-35", "ResNet-50", "dcid35", "resnet50"),
        ("DCID-35", "CNN-Only", "dcid35", "cnn_only"),
        ("DCID-7", "ResNet-50", "dcid7", "resnet50"),
        ("DCID-7", "CNN-Only", "dcid7", "cnn_only")
    ]
    rows = []
    for bench_name, m_label, b_code, m_code in models_to_eval:
        d = load_metrics(b_code, m_code)
        for a_code, a_val, cov_target in [("alpha_1", 0.01, 99.0), ("alpha_5", 0.05, 95.0), ("alpha_10", 0.10, 90.0)]:
            cp = d["conformal_prediction"][a_code]
            clean_def = cp["clean_on_def"]
            def_def = cp["def_on_def"]
            w_test = cp["wilcoxon_test"]
            rows.append({
                "Benchmark": bench_name, "Architecture": m_label, "Nominal Coverage": f"{cov_target:.0f}% ($\\alpha={a_val}$)",
                "Shifted Pipeline Coverage (%) [95% CI]": f"{clean_def['coverage']*100:.2f}% [{clean_def['coverage_ci'][0]*100:.2f}, {clean_def['coverage_ci'][1]*100:.2f}]",
                "Shifted Mean Set Size": f"{clean_def['mean_set_size']:.2f}",
                "Shifted Singleton Frac (%)": f"{clean_def['singleton_frac']*100:.1f}%",
                "Shifted Empty Frac (%)": f"{clean_def['empty_frac']*100:.1f}%",
                "Domain-Matched Coverage (%) [95% CI]": f"{def_def['coverage']*100:.2f}% [{def_def['coverage_ci'][0]*100:.2f}, {def_def['coverage_ci'][1]*100:.2f}]",
                "Domain-Matched Mean Set Size": f"{def_def['mean_set_size']:.2f}",
                "Domain-Matched Singleton Frac (%)": f"{def_def['singleton_frac']*100:.1f}%",
                "Domain-Matched Referral Rate (%)": f"{def_def['referral_rate']*100:.1f}%",
                "Wilcoxon p-value": f"{w_test['p_value']:.4e}" if w_test['p_value'] >= 0.001 else "<0.001"
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table4_conformal_prediction_triage.csv", index=False)
    print("Table 4 saved.")

def generate_table5_deduplication():
    models_to_eval = [
        ("DCID-35", "ResNet-50", "dcid35", "resnet50"),
        ("DCID-35", "CNN-Only", "dcid35", "cnn_only"),
        ("DCID-7", "ResNet-50", "dcid7", "resnet50"),
        ("DCID-7", "CNN-Only", "dcid7", "cnn_only")
    ]
    rows = []
    for bench_name, m_label, b_code, m_code in models_to_eval:
        d = load_metrics(b_code, m_code)
        c_u = d["calibration"]["clean_uncal"]
        dedup = d["deduplication_sensitivity"]
        rows.append({
            "Benchmark": bench_name, "Architecture": m_label,
            "Total Clean Eval Images": "5,250",
            "Excluded Near-Duplicates (dHash $\\le 2$)": str(dedup["n_excluded"]),
            "Retained Pristine Images": f"{dedup['n_retained']} ({dedup['n_retained']/5250*100:.1f}%)",
            "Original Full Eval Accuracy (%)": f"{c_u['accuracy']['est']*100:.2f}%",
            "Deduplicated Subset Accuracy (%)": f"{dedup['accuracy']*100:.2f}%",
            "Accuracy Shift (\\Delta Acc)": f"{(dedup['accuracy'] - c_u['accuracy']['est'])*100:+.2f}%",
            "Original ECE (%)": f"{c_u['ece']['est']*100:.2f}%",
            "Deduplicated ECE (%)": f"{dedup['ece']*100:.2f}%"
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table5_deduplication_sensitivity.csv", index=False)
    print("Table 5 saved.")

def generate_table6_per_class():
    d = load_metrics("dcid35", "resnet50")
    classes = d["per_class_summary"]
    sorted_classes = sorted(classes, key=lambda x: x["selective_referral_rate_5pct"], reverse=True)
    rows = []
    for c in sorted_classes:
        rows.append({
            "Class ID": c["class_idx"] + 1,
            "Lithology Name": c["class_name"],
            "Defective Accuracy (%)": f"{c['defective_accuracy']*100:.1f}%",
            "Mean Confidence (MSP) (%)": f"{c['mean_msp_confidence']*100:.1f}%",
            "Selective Referral Rate at 5% Error (%)": f"{c['selective_referral_rate_5pct']*100:.1f}%",
            "Conformal Coverage (\\alpha=0.05) (%)": f"{c['conformal_coverage_05']*100:.1f}%",
            "Mean Set Size (\\alpha=0.05)": f"{c['mean_set_size_05']:.2f}"
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table6_per_class_referral_taxonomy.csv", index=False)
    print("Table 6 saved.")

if __name__ == "__main__":
    generate_table1_dataset_partitions()
    generate_table2_calibration()
    generate_table3_selective()
    generate_table4_conformal()
    generate_table5_deduplication()
    generate_table6_per_class()
    print("All Paper B tables generated successfully!")
