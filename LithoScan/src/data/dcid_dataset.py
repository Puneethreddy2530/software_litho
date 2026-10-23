"""
dcid_dataset.py — Reusable PyTorch Dataset and DataLoader factory for DCID.

Supports:
  - DCID-7  (7 lithology classes, clean)
  - DCID-35 (35 lithology classes, clean)
  - noise-7 / noise-35  (real-world defective variants)
  - Configurable image size
  - Configurable augmentation pipeline
  - Reproducible random seed

Usage:
  from src.data.dcid_dataset import DCIDDataset, build_dcid_loaders
  from src.data.split_utils import build_split_loaders
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.split_utils import (
    VALID_SPLITS,
    VALID_VERSIONS,
    SplitViolationError,
    assert_no_leakage,
    build_split_loaders,
    get_official_split_dirs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default image transforms
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 512) -> transforms.Compose:
    """
    Standard augmentation pipeline for training.

    No random crop beyond the 512→image_size resize — avoids introducing
    out-of-distribution patches.  Augmentation strength is conservative
    to prevent leaking texture information.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(image_size: int = 512) -> transforms.Compose:
    """Deterministic pipeline for validation / test — no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_minimal_transforms(image_size: int = 512) -> transforms.Compose:
    """Resize + ToTensor only — useful for exploration / debugging."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DCIDDataset(Dataset):
    """
    PyTorch Dataset for the Drill Core Image Dataset (DCID).

    Parameters
    ----------
    dcid_root : str | Path
        Root directory containing the four DCID sub-directories.
    version : str
        Dataset variant — one of VALID_VERSIONS
        (``"DCID-512-7"``, ``"DCID-512-35"``, ``"noise-512-7"``, ``"noise-512-35"``).
    split : str
        ``"train"`` or ``"test"`` — only official splits are accepted.
    transform : Callable | None
        torchvision transform applied to each PIL image.
        If None, only ToTensor is applied.
    image_size : int
        Target square image size (used when *transform* is None to build a
        default Resize + ToTensor pipeline).
    skip_corrupt : bool
        If True, corrupt / unreadable images are **removed from the sample
        list during dataset construction** and a warning is logged with the
        count.  The dataset will be smaller than the raw file count but every
        remaining sample is guaranteed loadable.  Setting this to False
        (default) will raise RuntimeError at ``__getitem__`` time if a file
        is unreadable.

    Attributes
    ----------
    classes : List[str]
        Sorted list of class names discovered from the directory tree.
    class_to_idx : Dict[str, int]
        Mapping from class name to integer label.
    samples : List[Tuple[str, int]]
        (absolute_filepath, class_index) for every **valid** image.
    n_corrupt : int
        Number of files removed during construction because they were
        unreadable (only non-zero when skip_corrupt=True).
    """

    def __init__(
        self,
        dcid_root: str | Path,
        version: str,
        split: str,
        transform: Optional[Callable] = None,
        image_size: int = 512,
        skip_corrupt: bool = False,
    ) -> None:
        self.dcid_root = Path(dcid_root)
        self.version = version
        self.split = split
        self.image_size = image_size
        self.skip_corrupt = skip_corrupt
        self.n_corrupt: int = 0   # filled by _make_dataset when skip_corrupt=True

        # Validate and resolve the official split directory
        self.split_dir: Path = get_official_split_dirs(dcid_root, version, split)

        # Discover classes (sorted so that class_id is deterministic)
        self.classes, self.class_to_idx = self._find_classes()

        # Gather all valid image samples (corrupt files pruned here if skip_corrupt)
        self.samples: List[Tuple[str, int]] = self._make_dataset()

        # Attach transform (default: appropriate train/eval pipeline)
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_eval_transforms(image_size)

        logger.info(
            "[DCIDDataset] %s | split=%s | classes=%d | images=%d%s",
            version, split, len(self.classes), len(self.samples),
            f" | skipped_corrupt={self.n_corrupt}" if self.n_corrupt else "",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Scan split_dir for class sub-directories.

        Returns sorted class list and class→id mapping.
        The folder names follow the pattern ``"<id>.<ClassName>"``,
        e.g. ``"1.Red sandstone"``.  We sort numerically by the leading
        integer so IDs are stable across platforms.
        """
        entries = [e for e in self.split_dir.iterdir() if e.is_dir()]
        if not entries:
            raise RuntimeError(f"No class sub-directories found in {self.split_dir}")

        def _sort_key(p: Path) -> int:
            try:
                return int(p.name.split(".")[0])
            except ValueError:
                return 0

        sorted_entries = sorted(entries, key=_sort_key)
        classes = [e.name for e in sorted_entries]
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """
        Walk the split directory and collect (filepath, label) pairs.

        If ``skip_corrupt=True``, each file is probed with PIL at construction
        time.  Unreadable files are excluded from the sample list and counted
        in ``self.n_corrupt``.  This ensures every sample returned by
        ``__getitem__`` is guaranteed to be loadable — no placeholder images
        are ever used.
        """
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        samples: List[Tuple[str, int]] = []
        n_corrupt = 0

        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.split_dir / cls_name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                filepath = str(img_path.resolve())

                if self.skip_corrupt:
                    # Probe readability during construction; exclude bad files.
                    try:
                        with Image.open(filepath) as im:
                            im.verify()
                        # verify() closes the file; re-open briefly to confirm
                        # the image can be decoded (not just the header).
                        with Image.open(filepath) as im:
                            im.load()
                    except Exception as exc:
                        logger.warning(
                            "Corrupt image excluded from dataset: %s — %s",
                            filepath, exc,
                        )
                        n_corrupt += 1
                        continue

                samples.append((filepath, cls_idx))

        self.n_corrupt = n_corrupt
        if n_corrupt:
            logger.warning(
                "[DCIDDataset] %s/%s: excluded %d corrupt file(s) from sample list.",
                self.version, self.split, n_corrupt,
            )

        if not samples:
            raise RuntimeError(
                f"No valid images found under {self.split_dir}. "
                f"({n_corrupt} corrupt files were excluded.)"
            )
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[index]
        try:
            img = Image.open(filepath).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            # If skip_corrupt=True, this file passed construction-time probing
            # but may have become unreadable since (e.g. disk error).  Raise
            # explicitly — never silently substitute a black placeholder image.
            raise RuntimeError(
                f"Cannot open image (may have become corrupt since construction): "
                f"{filepath}"
            ) from exc

        tensor = self.transform(img)
        return tensor, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_class_counts(self) -> Dict[str, int]:
        """Return {class_name: image_count} for this split."""
        counts: Dict[str, int] = {cls: 0 for cls in self.classes}
        idx_to_cls = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            counts[idx_to_cls[label]] += 1
        return counts

    def get_filepaths(self) -> List[str]:
        """Return all image file paths for this split."""
        return [fp for fp, _ in self.samples]


# ---------------------------------------------------------------------------
# Factory — build matched train/test DataLoader pair
# ---------------------------------------------------------------------------

def build_dcid_loaders(
    dcid_root: str | Path,
    version: str,
    image_size: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
    skip_corrupt: bool = False,
    check_leakage: bool = True,
):
    """
    Build matched train + test DataLoaders for a DCID variant.

    Parameters
    ----------
    dcid_root : str | Path
        Root directory of the DCID archive.
    version : str
        One of VALID_VERSIONS.
    image_size : int
        Square image size to resize to.
    batch_size : int
        Batch size for both loaders.
    num_workers : int
        Worker processes for DataLoader.
    pin_memory : bool
        Pin memory for GPU.
    seed : int
        Reproducibility seed.
    train_transform : Callable | None
        Custom transform for training. Defaults to get_train_transforms().
    eval_transform : Callable | None
        Custom transform for evaluation. Defaults to get_eval_transforms().
    skip_corrupt : bool
        Skip unreadable images silently.
    check_leakage : bool
        Run leakage assertion before returning loaders.

    Returns
    -------
    (train_loader, test_loader, train_dataset, test_dataset)
    """
    train_ds = DCIDDataset(
        dcid_root=dcid_root,
        version=version,
        split="train",
        transform=train_transform,
        image_size=image_size,
        skip_corrupt=skip_corrupt,
    )
    test_ds = DCIDDataset(
        dcid_root=dcid_root,
        version=version,
        split="test",
        transform=eval_transform,
        image_size=image_size,
        skip_corrupt=skip_corrupt,
    )

    if check_leakage:
        assert_no_leakage(train_ds.get_filepaths(), test_ds.get_filepaths())

    train_loader, test_loader = build_split_loaders(
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
    )

    return train_loader, test_loader, train_ds, test_ds
