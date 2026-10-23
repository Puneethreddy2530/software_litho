"""
visualize_dataset.py — Visualization suite for DCID.

Generates:
  1.  Class distribution bar chart — DCID-7
  2.  Class distribution bar chart — DCID-35
  3.  Per-class sample image grid — DCID-7
  4.  Per-class sample image grid — DCID-35
  5.  Clean vs noise side-by-side comparison panel

All figures are saved to reports/figures/.

Usage (from project root):
  python src/data/visualize_dataset.py

Environment variable overrides:
  DCID_ROOT=<path>             LITHO_PROJECT_ROOT=<path>
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configurable paths
# ---------------------------------------------------------------------------
_DEFAULT_DCID_ROOT = Path(
    os.environ.get(
        "DCID_ROOT",
        r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID",
    )
)
_DEFAULT_PROJECT_ROOT = Path(
    os.environ.get(
        "LITHO_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
)

VERSIONS_7  = ["DCID-512-7",  "noise-512-7"]
VERSIONS_35 = ["DCID-512-35", "noise-512-35"]

PALETTE = sns.color_palette("mako", as_cmap=False, n_colors=35)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sort_key(name: str) -> int:
    try:
        return int(name.split(".")[0])
    except ValueError:
        return 9999


def _collect_class_counts(
    dcid_root: Path,
    version: str,
    splits: List[str] = ("train", "test"),
) -> Dict[str, Dict[str, int]]:
    """
    Returns {class_name: {"train": N, "test": M}} for a given version.
    """
    counts: Dict[str, Dict[str, int]] = {}
    for split in splits:
        split_dir = dcid_root / version / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir(), key=lambda p: _sort_key(p.name)):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            n = sum(1 for f in cls_dir.iterdir() if f.is_file())
            if cls not in counts:
                counts[cls] = {}
            counts[cls][split] = n
    return counts


def _sample_images(
    dcid_root: Path,
    version: str,
    split: str,
    n_per_class: int = 4,
    seed: int = 42,
) -> Dict[str, List[Path]]:
    """Return up to n_per_class image paths per class for a given version/split."""
    rng = random.Random(seed)
    split_dir = dcid_root / version / split
    result: Dict[str, List[Path]] = {}
    if not split_dir.is_dir():
        return result
    for cls_dir in sorted(split_dir.iterdir(), key=lambda p: _sort_key(p.name)):
        if not cls_dir.is_dir():
            continue
        imgs = [f for f in cls_dir.iterdir() if f.is_file()]
        result[cls_dir.name] = rng.sample(imgs, min(n_per_class, len(imgs)))
    return result


# ---------------------------------------------------------------------------
# Plot 1 & 2 — Class distribution bar charts
# ---------------------------------------------------------------------------

def plot_class_distribution(
    dcid_root: Path,
    version: str,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    counts = _collect_class_counts(dcid_root, version)
    if not counts:
        logger.warning("No data found for version %s", version)
        return

    classes = sorted(counts.keys(), key=_sort_key)
    # Shorten names for readability in the 35-class chart
    short_names = [c.split(".", 1)[-1].strip() if "." in c else c for c in classes]
    train_vals = [counts[c].get("train", 0) for c in classes]
    test_vals  = [counts[c].get("test",  0) for c in classes]

    n = len(classes)
    fig_width = max(12, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = range(n)
    bar_w = 0.4
    bars_tr = ax.bar([i - bar_w / 2 for i in x], train_vals, bar_w,
                     label="Train", color="#2E86AB", alpha=0.85, edgecolor="white")
    bars_te = ax.bar([i + bar_w / 2 for i in x], test_vals,  bar_w,
                     label="Test",  color="#E84855", alpha=0.85, edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Image count", fontsize=11)
    ax.set_title(
        f"Class Distribution — {version}{' ' + title_suffix if title_suffix else ''}",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate totals on top of each grouped bar
    for bar in bars_tr:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=6, color="#2E86AB")
    for bar in bars_te:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                    f"{int(h):,}", ha="center", va="bottom", fontsize=6, color="#E84855")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 3 & 4 — Per-class sample grids
# ---------------------------------------------------------------------------

def plot_sample_grid(
    dcid_root: Path,
    version: str,
    out_path: Path,
    split: str = "train",
    n_per_class: int = 4,
    seed: int = 42,
    thumb_size: int = 128,
) -> None:
    samples = _sample_images(dcid_root, version, split, n_per_class, seed)
    if not samples:
        logger.warning("No samples found for %s/%s", version, split)
        return

    classes = sorted(samples.keys(), key=_sort_key)
    n_classes = len(classes)
    n_cols = n_per_class

    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(n_cols * 1.6, n_classes * 1.6),
    )

    # Ensure 2D axes array
    if n_classes == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, cls in enumerate(classes):
        imgs = samples[cls]
        short = cls.split(".", 1)[-1].strip() if "." in cls else cls
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx < len(imgs):
                try:
                    img = Image.open(imgs[col_idx]).convert("RGB")
                    img.thumbnail((thumb_size, thumb_size))
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, "ERR", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6)
            if col_idx == 0:
                ax.set_ylabel(short, rotation=0, labelpad=80,
                              fontsize=7, va="center")

    fig.suptitle(
        f"Sample Images — {version} | split={split}",
        fontsize=13, fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Plot 5 — Clean vs Noise comparison
# ---------------------------------------------------------------------------

def plot_clean_vs_noise(
    dcid_root: Path,
    out_path: Path,
    n_classes: int = 7,
    n_per_class: int = 3,
    seed: int = 42,
) -> None:
    """
    Side-by-side panel: DCID (clean) vs noise variant for DCID-7.
    """
    clean_version = "DCID-512-7"
    noise_version = "noise-512-7"
    split = "train"

    clean_samples = _sample_images(dcid_root, clean_version, split, n_per_class, seed)
    noise_samples = _sample_images(dcid_root, noise_version, split, n_per_class, seed)

    classes = sorted(clean_samples.keys(), key=_sort_key)[:n_classes]
    if not classes:
        logger.warning("No samples found for clean/noise comparison.")
        return

    n_cols = n_per_class * 2 + 1   # clean cols | divider | noise cols
    n_rows = len(classes)

    fig = plt.figure(figsize=(n_per_class * 3.2 * 2 + 1.2, n_rows * 1.8))
    fig.suptitle(
        "Clean (DCID-512-7)  vs  Noisy / Defective (noise-512-7)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    outer_gs = gridspec.GridSpec(n_rows, 1, hspace=0.05)

    for row_idx, cls in enumerate(classes):
        short = cls.split(".", 1)[-1].strip() if "." in cls else cls
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer_gs[row_idx], wspace=0.05
        )
        # Clean columns
        for col in range(n_per_class):
            ax = fig.add_subplot(inner_gs[0, col])
            ax.axis("off")
            imgs = clean_samples.get(cls, [])
            if col < len(imgs):
                try:
                    img = Image.open(imgs[col]).convert("RGB")
                    img.thumbnail((160, 160))
                    ax.imshow(img)
                except Exception:
                    pass
            if col == 0:
                ax.set_ylabel(short, rotation=0, labelpad=70, fontsize=7, va="center")
            if row_idx == 0:
                ax.set_title("Clean" if col == n_per_class // 2 else "", fontsize=9,
                             color="#2E86AB", fontweight="bold")

        # Divider column (blank)
        ax_div = fig.add_subplot(inner_gs[0, n_per_class])
        ax_div.axis("off")
        if row_idx == 0:
            ax_div.text(0.5, 0.5, "│", ha="center", va="center",
                        transform=ax_div.transAxes, fontsize=14, color="#666")

        # Noise columns
        for col in range(n_per_class):
            ax = fig.add_subplot(inner_gs[0, n_per_class + 1 + col])
            ax.axis("off")
            imgs = noise_samples.get(cls, [])
            if col < len(imgs):
                try:
                    img = Image.open(imgs[col]).convert("RGB")
                    img.thumbnail((160, 160))
                    ax.imshow(img)
                except Exception:
                    pass
            if row_idx == 0:
                ax.set_title("Noisy" if col == n_per_class // 2 else "", fontsize=9,
                             color="#E84855", fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_visualizations(
    dcid_root: Path = _DEFAULT_DCID_ROOT,
    project_root: Path = _DEFAULT_PROJECT_ROOT,
    seed: int = 42,
) -> None:
    figures_dir = project_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations into: %s", figures_dir)

    # 1. Class distribution — DCID-7
    plot_class_distribution(
        dcid_root, "DCID-512-7",
        figures_dir / "class_dist_dcid7.png",
    )

    # 2. Class distribution — DCID-35
    plot_class_distribution(
        dcid_root, "DCID-512-35",
        figures_dir / "class_dist_dcid35.png",
    )

    # 3. Sample grid — DCID-7
    plot_sample_grid(
        dcid_root, "DCID-512-7",
        figures_dir / "samples_dcid7.png",
        n_per_class=4,
        seed=seed,
    )

    # 4. Sample grid — DCID-35
    plot_sample_grid(
        dcid_root, "DCID-512-35",
        figures_dir / "samples_dcid35.png",
        n_per_class=4,
        seed=seed,
    )

    # 5. Clean vs noise comparison
    plot_clean_vs_noise(
        dcid_root,
        figures_dir / "clean_vs_noise.png",
        seed=seed,
    )

    logger.info("All visualizations complete. Check: %s", figures_dir)
    print(f"\n[OK] Figures saved to: {figures_dir}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LithoScan — DCID Visualization Suite"
    )
    parser.add_argument("--dcid-root", type=Path, default=_DEFAULT_DCID_ROOT)
    parser.add_argument("--project-root", type=Path, default=_DEFAULT_PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_visualizations(args.dcid_root, args.project_root, args.seed)
