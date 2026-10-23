"""
split_utils.py — Safe official-split access for DCID.

CRITICAL RULE:
  Never randomly re-split the DCID dataset.
  Only consume the official train/ and test/ sub-directories.
  Any attempt to override the split will raise SplitViolationError.

Also enforces:
  - assert_class_alignment: clean and noise variants must have identical
    class names in identical order.  Fails loudly if they differ.
  - validate_noise_alignment: convenience wrapper that checks both the
    7-class and 35-class clean↔noise pairs.

Usage:
  from src.data.split_utils import (
      get_official_split_dirs, build_split_loaders,
      assert_class_alignment, validate_noise_alignment,
  )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_SPLITS: Tuple[str, ...] = ("train", "test")
VALID_VERSIONS: Tuple[str, ...] = ("DCID-512-7", "DCID-512-35", "noise-512-7", "noise-512-35")


# ---------------------------------------------------------------------------
# Custom error
# ---------------------------------------------------------------------------
class SplitViolationError(RuntimeError):
    """Raised when caller attempts to bypass the official DCID split."""


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def get_official_split_dirs(
    dcid_root: str | Path,
    version: str,
    split: str,
) -> Path:
    """
    Return the path to the official split directory for a given DCID version.

    Parameters
    ----------
    dcid_root : str | Path
        Root directory that contains DCID-512-7, DCID-512-35, noise-512-7, noise-512-35.
    version : str
        One of VALID_VERSIONS.
    split : str
        Either ``"train"`` or ``"test"``.

    Returns
    -------
    Path
        Absolute path to ``<dcid_root>/<version>/<split>/``.

    Raises
    ------
    SplitViolationError
        If *split* is not one of the official values.
    FileNotFoundError
        If the resolved directory does not exist on disk.
    """
    if split not in VALID_SPLITS:
        raise SplitViolationError(
            f"'{split}' is not an official DCID split. "
            f"Allowed values: {VALID_SPLITS}. "
            "Do NOT randomly re-split the dataset."
        )

    if version not in VALID_VERSIONS:
        raise ValueError(
            f"Unknown dataset version '{version}'. "
            f"Expected one of: {VALID_VERSIONS}"
        )

    split_dir = Path(dcid_root) / version / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Official split directory not found: {split_dir}\n"
            "Make sure the DCID archive is extracted correctly."
        )

    logger.debug("Resolved split dir: %s", split_dir)
    return split_dir


def assert_no_leakage(train_paths: list[str], test_paths: list[str]) -> None:
    """
    Assert that no file path appears in both train and test sets.

    Parameters
    ----------
    train_paths : list[str]
    test_paths : list[str]

    Raises
    ------
    SplitViolationError
        If any overlap is detected.
    """
    train_set = set(os.path.abspath(p) for p in train_paths)
    test_set = set(os.path.abspath(p) for p in test_paths)
    overlap = train_set & test_set
    if overlap:
        sample = list(overlap)[:5]
        raise SplitViolationError(
            f"Data leakage detected! {len(overlap)} file(s) appear in both "
            f"train and test sets. Sample: {sample}"
        )
    logger.info("No train/test leakage detected (%d train, %d test).", len(train_set), len(test_set))


def build_split_loaders(
    train_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from official split datasets.

    Parameters
    ----------
    train_dataset : Dataset
        Dataset loaded from the official ``train/`` split.
    test_dataset : Dataset
        Dataset loaded from the official ``test/`` split.
    batch_size : int
        Batch size for both loaders.
    num_workers : int
        Number of worker processes.
    pin_memory : bool
        Pin memory for faster GPU transfer.
    seed : int
        Random seed for the train loader's generator.

    Returns
    -------
    (train_loader, test_loader)
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # Never shuffle test set
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        "DataLoaders ready — train batches: %d | test batches: %d",
        len(train_loader),
        len(test_loader),
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Class alignment validation (Correction 4)
# ---------------------------------------------------------------------------

def _get_class_names(dcid_root: Path, version: str, split: str = "train") -> List[str]:
    """
    Return sorted class-folder names for a given version/split.
    Uses the same numeric-prefix sort key as DCIDDataset._find_classes.
    """
    split_dir = dcid_root / version / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find split directory for alignment check: {split_dir}"
        )

    def _sort_key(p: Path) -> int:
        try:
            return int(p.name.split(".")[0])
        except ValueError:
            return 9999

    entries = sorted(
        [e for e in split_dir.iterdir() if e.is_dir()],
        key=_sort_key,
    )
    return [e.name for e in entries]


def assert_class_alignment(
    dcid_root: str | Path,
    clean_version: str,
    noise_version: str,
    split: str = "train",
) -> None:
    """
    Assert that a clean/noise DCID variant pair has **identical class names
    in identical order**.

    Raises
    ------
    ValueError
        If the class lists differ in length, content, or order.

    Examples
    --------
    assert_class_alignment(dcid_root, "DCID-512-7",  "noise-512-7")
    assert_class_alignment(dcid_root, "DCID-512-35", "noise-512-35")
    """
    root = Path(dcid_root)
    clean_classes = _get_class_names(root, clean_version, split)
    noise_classes = _get_class_names(root, noise_version, split)

    if clean_classes == noise_classes:
        logger.info(
            "Class alignment OK: %s ↔ %s (%d classes)",
            clean_version, noise_version, len(clean_classes),
        )
        return

    # Build a detailed diff for the error message.
    clean_set = set(clean_classes)
    noise_set = set(noise_classes)
    only_in_clean = sorted(clean_set - noise_set)
    only_in_noise = sorted(noise_set - clean_set)

    order_mismatches = [
        (i, c, n)
        for i, (c, n) in enumerate(zip(clean_classes, noise_classes))
        if c != n
    ]

    msg_parts = [
        f"Class alignment FAILED: {clean_version} ↔ {noise_version}",
        f"  clean classes : {len(clean_classes)}",
        f"  noise classes : {len(noise_classes)}",
    ]
    if only_in_clean:
        msg_parts.append(f"  Only in clean : {only_in_clean}")
    if only_in_noise:
        msg_parts.append(f"  Only in noise : {only_in_noise}")
    if order_mismatches:
        msg_parts.append(f"  Order mismatches (index, clean, noise):")
        for idx, c, n in order_mismatches[:10]:
            msg_parts.append(f"    [{idx}] '{c}' vs '{n}'")

    raise ValueError("\n".join(msg_parts))


def validate_noise_alignment(dcid_root: str | Path) -> None:
    """
    Validate class alignment for both DCID clean↔noise pairs.

    Checks:
      DCID-512-7  ↔  noise-512-7
      DCID-512-35 ↔  noise-512-35

    Raises
    ------
    ValueError
        If either pair has mismatched class names or ordering.
    """
    root = Path(dcid_root)
    logger.info("Validating clean↔noise class alignment for DCID root: %s", root)

    assert_class_alignment(root, "DCID-512-7",  "noise-512-7")
    assert_class_alignment(root, "DCID-512-35", "noise-512-35")

    logger.info("All clean↔noise class alignments verified ✅")
