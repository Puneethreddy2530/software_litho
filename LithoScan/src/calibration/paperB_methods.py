"""
paperB_methods.py — Core statistical methods for LithoScan Paper B.
Implements:
  1. Temperature Scaling & Probability Calibration (NLL, Brier, ECE fixed/adaptive)
  2. Selective Classification (MSP, Entropy, Margin, AURC, threshold selection)
  3. Split Conformal Prediction (Coverage, Set Size, Singleton Triage)
  4. Statistical Inference (Stratified Bootstrap 95% CIs, McNemar, Wilcoxon)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any

# ---------------------------------------------------------------------------
# 1. Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """
    Learns a scalar temperature T > 0 by optimizing NLL on a calibration cohort.
    Uses L-BFGS optimizer following Guo et al. (2017).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature.clamp(min=0.01)
        return logits / temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100, lr: float = 0.01) -> float:
        """
        Fit temperature on calibration logits and labels.
        Returns the optimized temperature value.
        """
        self.temperature.data.fill_(1.5)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_step():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())


# ---------------------------------------------------------------------------
# 2. Calibration Metrics: NLL, Brier, ECE (Fixed & Adaptive)
# ---------------------------------------------------------------------------

def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    adaptive_bins: int = 15
) -> Dict[str, Any]:
    """
    Compute probabilistic calibration metrics:
      - NLL (Negative Log-Likelihood)
      - Brier Score
      - ECE (Uniform 15-bin Expected Calibration Error)
      - ECE_adaptive (Equal-frequency quantile binning)
      - Reliability diagram bin points
    """
    N, K = probs.shape
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    correct = (preds == labels).astype(float)

    # 1. NLL (clipped for numerical stability)
    eps = 1e-15
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    true_class_probs = clipped_probs[np.arange(N), labels]
    nll = float(-np.mean(np.log(true_class_probs)))

    # 2. Multi-class Brier Score
    one_hot = np.zeros((N, K), dtype=float)
    one_hot[np.arange(N), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    # 3. Uniform ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_uniform = 0.0
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        count = int(np.sum(in_bin))
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(correct[in_bin]))
            conf = float(np.mean(confs[in_bin]))
            ece_uniform += (count / N) * abs(acc - conf)
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    # 4. Adaptive (Equal-Frequency) ECE
    # Sort samples by confidence and divide into equal-sized bins
    sort_idx = np.argsort(confs)
    sorted_confs = confs[sort_idx]
    sorted_correct = correct[sort_idx]
    bin_size = N // adaptive_bins
    ece_adaptive = 0.0

    for i in range(adaptive_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < adaptive_bins - 1 else N
        bin_c = sorted_confs[start_idx:end_idx]
        bin_corr = sorted_correct[start_idx:end_idx]
        if len(bin_c) > 0:
            acc = float(np.mean(bin_corr))
            conf = float(np.mean(bin_c))
            ece_adaptive += (len(bin_c) / N) * abs(acc - conf)

    return {
        "accuracy": float(np.mean(correct)),
        "nll": nll,
        "brier": brier,
        "ece": float(ece_uniform),
        "ece_adaptive": float(ece_adaptive),
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
        "bin_edges": bin_boundaries.tolist()
    }


# ---------------------------------------------------------------------------
# 3. Selective Classification (Abstention, Risk-Coverage, Operating Points)
# ---------------------------------------------------------------------------

def compute_confidence_scores(probs: np.ndarray, method: str = "msp") -> np.ndarray:
    """
    Compute sample-level confidence scores:
      - 'msp': Maximum Softmax Probability: max_k p_k
      - 'entropy': Negative Shannon Entropy: sum(p_k * log p_k)
      - 'margin': Difference between top-1 and top-2 probabilities
    """
    if method == "msp":
        return np.max(probs, axis=1)
    elif method == "entropy":
        eps = 1e-15
        p = np.clip(probs, eps, 1.0)
        # Note: -sum(p * log p) is positive entropy.
        # We negate it so higher score = lower entropy = higher confidence
        return np.sum(p * np.log(p), axis=1)
    elif method == "margin":
        sorted_p = np.sort(probs, axis=1)
        return sorted_p[:, -1] - sorted_p[:, -2]
    else:
        raise ValueError(f"Unknown confidence scoring method: {method}")


def compute_risk_coverage_curve(
    scores: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 1000
) -> Dict[str, Any]:
    """
    Compute empirical risk-coverage curve and AURC across coverage levels.
    """
    N = len(labels)
    correct = (preds == labels).astype(float)
    errors = 1.0 - correct

    # Sort samples by confidence score ascending (lowest confidence first)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_errors = errors[order]

    # Evaluate at multiple threshold percentiles
    percentiles = np.linspace(0, 100, n_thresholds)
    thresholds = np.percentile(sorted_scores, percentiles)
    thresholds = np.unique(thresholds)

    coverages = []
    risks = []
    referral_rates = []

    for tau in thresholds:
        accepted = scores >= tau
        n_accepted = np.sum(accepted)
        if n_accepted == 0:
            continue
        cov = n_accepted / N
        risk = np.sum(errors[accepted]) / n_accepted
        coverages.append(float(cov))
        risks.append(float(risk))
        referral_rates.append(float(1.0 - cov))

    # Add endpoints (coverage 1.0: full test error; coverage 0.0: extrapolate)
    coverages = np.array(coverages)
    risks = np.array(risks)
    referral_rates = np.array(referral_rates)

    # Sort by coverage ascending for integration
    sort_cov = np.argsort(coverages)
    cov_sorted = coverages[sort_cov]
    risk_sorted = risks[sort_cov]

    # Compute AURC via trapezoidal rule
    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risk_sorted, cov_sorted))
    else:
        aurc = float(np.trapz(risk_sorted, cov_sorted))

    # Excess AURC relative to optimal selective classifier
    base_error = float(np.mean(errors))
    # Optimal AURC: accepts all correct first, then errors
    if base_error > 0:
        optimal_aurc = base_error + (1.0 - base_error) * np.log(1.0 - base_error) if (1.0 - base_error) > 0 else 0.0
    else:
        optimal_aurc = 0.0
    e_aurc = float(max(0.0, aurc - optimal_aurc))

    return {
        "aurc": aurc,
        "e_aurc": e_aurc,
        "coverages": coverages.tolist(),
        "risks": risks.tolist(),
        "referral_rates": referral_rates.tolist(),
        "thresholds": thresholds.tolist()
    }


def find_threshold_for_target_risk(
    cal_scores: np.ndarray,
    cal_preds: np.ndarray,
    cal_labels: np.ndarray,
    target_risk: float
) -> float:
    """
    Determine confidence threshold tau* on calibration data such that
    selective risk on calibration data is <= target_risk.
    Uses conservative threshold selection.
    """
    cal_errors = (cal_preds != cal_labels).astype(float)
    # Sort candidate thresholds
    cand_thresholds = np.sort(np.unique(cal_scores))

    valid_tau = None
    for tau in cand_thresholds:
        accepted = cal_scores >= tau
        if np.sum(accepted) == 0:
            continue
        r = np.sum(cal_errors[accepted]) / np.sum(accepted)
        if r <= target_risk:
            valid_tau = float(tau)
            break

    if valid_tau is None:
        # If even the strictest threshold cannot achieve target risk, return max score
        return float(np.max(cal_scores))
    return valid_tau


def evaluate_at_threshold(
    eval_scores: np.ndarray,
    eval_preds: np.ndarray,
    eval_labels: np.ndarray,
    threshold: float
) -> Dict[str, float]:
    """
    Evaluate realized coverage, referral rate, and selective risk at locked threshold.
    """
    N = len(eval_labels)
    accepted = eval_scores >= threshold
    n_accepted = int(np.sum(accepted))

    if n_accepted == 0:
        return {
            "coverage": 0.0,
            "referral_rate": 1.0,
            "selective_risk": 0.0,
            "selective_accuracy": 1.0,
            "accepted_count": 0,
            "referred_count": N
        }

    eval_errors = (eval_preds != eval_labels).astype(float)
    risk = float(np.sum(eval_errors[accepted]) / n_accepted)
    cov = float(n_accepted / N)

    return {
        "coverage": cov,
        "referral_rate": float(1.0 - cov),
        "selective_risk": risk,
        "selective_accuracy": float(1.0 - risk),
        "accepted_count": n_accepted,
        "referred_count": int(N - n_accepted)
    }


# ---------------------------------------------------------------------------
# 4. Split Conformal Prediction
# ---------------------------------------------------------------------------

class SplitConformalClassifier:
    """
    Split Conformal Prediction for Multi-Class Classification.
    Nonconformity score: s(x, y) = 1 - p_y(x) (Hinge / Softmax loss proxy).
    Guarantees 1 - alpha marginal coverage under exchangeability.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q_hat = None

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray) -> float:
        """
        Compute conformal threshold q_hat from calibration cohort.
        Uses finite-sample correction: quantile level = ceil((n+1)(1-alpha)) / n
        """
        n = len(cal_labels)
        # Compute nonconformity score on true labels: s_i = 1 - p_{i, y_i}
        true_probs = cal_probs[np.arange(n), cal_labels]
        scores = 1.0 - true_probs

        # Finite-sample adjusted quantile level
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self.q_hat

    def predict(self, eval_probs: np.ndarray) -> List[List[int]]:
        """
        Construct prediction sets for evaluation instances:
        C(x) = {k : 1 - p_k(x) <= q_hat}
        """
        if self.q_hat is None:
            raise RuntimeError("Conformal classifier must be fit before predict.")
        sets = []
        for p in eval_probs:
            # Class k is included if 1 - p_k <= q_hat
            included = np.where((1.0 - p) <= self.q_hat)[0].tolist()
            sets.append(included)
        return sets

    def evaluate(self, eval_probs: np.ndarray, eval_labels: np.ndarray) -> Dict[str, Any]:
        """
        Compute empirical conformal metrics:
          - marginal coverage
          - mean set size
          - singleton fraction
          - singleton error rate
          - multi-class referral rate
          - empty set fraction
        """
        pred_sets = self.predict(eval_probs)
        M = len(eval_labels)
        covered = []
        set_sizes = []
        singleton_indices = []
        multiclass_indices = []
        empty_indices = []

        for i in range(M):
            s = pred_sets[i]
            y = eval_labels[i]
            covered.append(1.0 if y in s else 0.0)
            set_sizes.append(len(s))
            if len(s) == 1:
                singleton_indices.append(i)
            elif len(s) > 1:
                multiclass_indices.append(i)
            else:
                empty_indices.append(i)

        covered = np.array(covered)
        set_sizes = np.array(set_sizes)

        # Singleton error rate
        if len(singleton_indices) > 0:
            singleton_preds = np.array([pred_sets[i][0] for i in singleton_indices])
            singleton_labels = eval_labels[singleton_indices]
            singleton_error = float(np.mean(singleton_preds != singleton_labels))
        else:
            singleton_error = 0.0

        return {
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "marginal_coverage": float(np.mean(covered)),
            "mean_set_size": float(np.mean(set_sizes)),
            "median_set_size": float(np.median(set_sizes)),
            "singleton_fraction": float(len(singleton_indices) / M),
            "singleton_error": singleton_error,
            "multi_class_referral_rate": float(len(multiclass_indices) / M),
            "empty_fraction": float(len(empty_indices) / M),
            "set_size_distribution": {
                str(int(size)): int(np.sum(set_sizes == size))
                for size in sorted(np.unique(set_sizes))
            },
            "covered_array": covered,
            "set_sizes_array": set_sizes
        }


# ---------------------------------------------------------------------------
# 5. Statistical Inference (Stratified Bootstrap, McNemar, Wilcoxon)
# ---------------------------------------------------------------------------

def stratified_bootstrap_ci(
    data_dict: Dict[str, np.ndarray],
    metric_fn: Any,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 2026
) -> Tuple[float, float, float]:
    """
    Computes class-stratified bootstrap confidence intervals.
    Returns (estimate, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    labels = data_dict["labels"]
    unique_classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}

    # Point estimate
    point_est = float(metric_fn(data_dict))

    boot_estimates = []
    for _ in range(n_resamples):
        sampled_indices = []
        for c in unique_classes:
            idx = class_indices[c]
            resampled = rng.choice(idx, size=len(idx), replace=True)
            sampled_indices.extend(resampled)
        sampled_indices = np.array(sampled_indices)

        boot_data = {k: v[sampled_indices] for k, v in data_dict.items()}
        boot_est = metric_fn(boot_data)
        boot_estimates.append(boot_est)

    alpha_ci = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_estimates, 100 * alpha_ci))
    ci_upper = float(np.percentile(boot_estimates, 100 * (1.0 - alpha_ci)))

    return point_est, ci_lower, ci_upper


def paired_mcnemar_test(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict[str, Any]:
    """
    Paired McNemar test on binary disagreement between two methods on matched samples.
    Table:
              Method B Correct | Method B Error
    Method A Correct    n00           n01
    Method A Error      n10           n11
    """
    b0 = (errors_a == 0) & (errors_b == 1)  # A correct, B error
    b1 = (errors_a == 1) & (errors_b == 0)  # A error, B correct

    n01 = int(np.sum(b0))
    n10 = int(np.sum(b1))

    # Exact binomial test on discordant pairs
    total_discordant = n01 + n10
    if total_discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        # McNemar with continuity correction
        statistic = float(((abs(n01 - n10) - 1.0) ** 2) / total_discordant)
        # Exact two-sided binomial p-value
        p_value = float(stats.binomtest(min(n01, n10), total_discordant, 0.5).pvalue)

    return {
        "n_discordant_A_better": n10,
        "n_discordant_B_better": n01,
        "total_discordant": total_discordant,
        "statistic": statistic,
        "p_value": p_value
    }


def paired_wilcoxon_test(sizes_a: np.ndarray, sizes_b: np.ndarray) -> Dict[str, Any]:
    """
    Two-sided Wilcoxon signed-rank test for paired set sizes.
    """
    diff = sizes_a - sizes_b
    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_ties": len(diff)}

    res = stats.wilcoxon(sizes_a, sizes_b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean_diff": float(np.mean(diff)),
        "n_samples": len(diff),
        "n_ties": int(np.sum(diff == 0))
    }
