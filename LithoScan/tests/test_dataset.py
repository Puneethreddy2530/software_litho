"""
test_dataset.py — Unit tests for DCID dataset infrastructure.

Tests:
  1. Dataset discovery — DCID-7 and DCID-35
  2. Label consistency — class names map to stable integer IDs
  3. Image loading — valid tensor shape and value range
  4. No train/test leakage — filepath prefix verification
  5. Class count correctness
  6. Split safety — SplitViolationError on invalid split names
  7. Class alignment — clean↔noise variants have identical class lists
  8. skip_corrupt behaviour — corrupt files removed at construction, no placeholders

Usage (from project root):
  python -m pytest tests/test_dataset.py -v
  python -m pytest tests/test_dataset.py -v --tb=short
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Make project root importable when running from tests/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dcid_dataset import DCIDDataset, get_eval_transforms, get_train_transforms
from src.data.split_utils import (
    SplitViolationError,
    assert_class_alignment,
    assert_no_leakage,
    get_official_split_dirs,
    validate_noise_alignment,
)

# ---------------------------------------------------------------------------
# Configuration — must match the real extracted archive
# ---------------------------------------------------------------------------
DCID_ROOT = Path(
    os.environ.get(
        "DCID_ROOT",
        r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID",
    )
)

# Expected class names (sorted by leading integer)
DCID_7_CLASSES = [
    "1.Red sandstone",
    "2.Light sandstone",
    "3.Gray siltstone",
    "4.Mudstone",
    "5.Granite",
    "6.Basalt",
    "7.Marble",
]

# Just the first 5 of 35 for a quick membership check
DCID_35_CLASSES_SAMPLE = [
    "1.Red sandstone",
    "2.Light sandstone",
    "3.Gray-green sandstone",
    "4.Gray siltstone",
    "5.Brown siltstone",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dcid7_train() -> DCIDDataset:
    """DCID-7 training dataset (session-scoped for speed)."""
    return DCIDDataset(
        dcid_root=DCID_ROOT,
        version="DCID-512-7",
        split="train",
        transform=get_eval_transforms(image_size=64),  # small size for fast tests
        image_size=64,
    )


@pytest.fixture(scope="session")
def dcid7_test() -> DCIDDataset:
    return DCIDDataset(
        dcid_root=DCID_ROOT,
        version="DCID-512-7",
        split="test",
        transform=get_eval_transforms(image_size=64),
        image_size=64,
    )


@pytest.fixture(scope="session")
def dcid35_train() -> DCIDDataset:
    return DCIDDataset(
        dcid_root=DCID_ROOT,
        version="DCID-512-35",
        split="train",
        transform=get_eval_transforms(image_size=64),
        image_size=64,
    )


@pytest.fixture(scope="session")
def dcid35_test() -> DCIDDataset:
    return DCIDDataset(
        dcid_root=DCID_ROOT,
        version="DCID-512-35",
        split="test",
        transform=get_eval_transforms(image_size=64),
        image_size=64,
    )


# ---------------------------------------------------------------------------
# 1. Dataset Discovery
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_dcid_root_exists(self):
        assert DCID_ROOT.is_dir(), (
            f"DCID root not found: {DCID_ROOT}\n"
            "Set the DCID_ROOT environment variable."
        )

    @pytest.mark.parametrize("version", [
        "DCID-512-7", "DCID-512-35", "noise-512-7", "noise-512-35"
    ])
    def test_all_versions_exist(self, version):
        assert (DCID_ROOT / version).is_dir(), (
            f"Version directory missing: {DCID_ROOT / version}"
        )

    @pytest.mark.parametrize("version,split", [
        ("DCID-512-7",  "train"), ("DCID-512-7",  "test"),
        ("DCID-512-35", "train"), ("DCID-512-35", "test"),
        ("noise-512-7", "train"), ("noise-512-7", "test"),
        ("noise-512-35","train"), ("noise-512-35","test"),
    ])
    def test_split_dirs_exist(self, version, split):
        split_path = DCID_ROOT / version / split
        assert split_path.is_dir(), f"Split directory missing: {split_path}"

    def test_dcid7_has_correct_class_count(self, dcid7_train):
        assert len(dcid7_train.classes) == 7, (
            f"Expected 7 classes, got {len(dcid7_train.classes)}: {dcid7_train.classes}"
        )

    def test_dcid35_has_correct_class_count(self, dcid35_train):
        assert len(dcid35_train.classes) == 35, (
            f"Expected 35 classes, got {len(dcid35_train.classes)}: {dcid35_train.classes}"
        )


# ---------------------------------------------------------------------------
# 2. Label Consistency
# ---------------------------------------------------------------------------

class TestLabels:
    def test_dcid7_class_names_match(self, dcid7_train):
        assert dcid7_train.classes == DCID_7_CLASSES, (
            f"Class name mismatch.\n"
            f"Expected: {DCID_7_CLASSES}\n"
            f"Got:      {dcid7_train.classes}"
        )

    def test_dcid7_class_to_idx_is_injective(self, dcid7_train):
        """class_to_idx must have unique IDs for each class."""
        idx_values = list(dcid7_train.class_to_idx.values())
        assert len(idx_values) == len(set(idx_values)), "Duplicate class IDs found!"

    def test_dcid7_class_to_idx_range(self, dcid7_train):
        idx_values = sorted(dcid7_train.class_to_idx.values())
        assert idx_values == list(range(7)), (
            f"class_to_idx should map to 0..6, got: {idx_values}"
        )

    def test_dcid35_class_to_idx_range(self, dcid35_train):
        idx_values = sorted(dcid35_train.class_to_idx.values())
        assert idx_values == list(range(35)), (
            f"class_to_idx should map to 0..34, got: {idx_values}"
        )

    def test_dcid35_first_classes_match(self, dcid35_train):
        for cls in DCID_35_CLASSES_SAMPLE:
            assert cls in dcid35_train.class_to_idx, (
                f"Expected class '{cls}' not found in DCID-35."
            )

    def test_labels_are_integers_in_range(self, dcid7_train):
        _, label = dcid7_train[0]
        assert isinstance(label, int), f"Label should be int, got {type(label)}"
        assert 0 <= label < 7, f"Label {label} out of range [0, 7)"

    def test_train_test_share_same_classes(self, dcid7_train, dcid7_test):
        assert dcid7_train.classes == dcid7_test.classes, (
            "Train and test datasets have different class lists!"
        )

    def test_train_test_share_same_class_to_idx(self, dcid7_train, dcid7_test):
        assert dcid7_train.class_to_idx == dcid7_test.class_to_idx


# ---------------------------------------------------------------------------
# 3. Image Loading
# ---------------------------------------------------------------------------

class TestImageLoading:
    def test_first_item_returns_tensor(self, dcid7_train):
        img_tensor, label = dcid7_train[0]
        assert isinstance(img_tensor, torch.Tensor), "Expected a torch.Tensor"

    def test_tensor_shape_is_correct(self, dcid7_train):
        img_tensor, _ = dcid7_train[0]
        # [C, H, W] = [3, 64, 64] since we loaded at image_size=64
        assert img_tensor.shape == (3, 64, 64), (
            f"Expected shape (3, 64, 64), got {img_tensor.shape}"
        )

    def test_tensor_dtype(self, dcid7_train):
        img_tensor, _ = dcid7_train[0]
        assert img_tensor.dtype == torch.float32

    def test_tensor_values_normalized(self, dcid7_train):
        """After ImageNet normalization the values should not be in [0, 1]."""
        img_tensor, _ = dcid7_train[0]
        # At least some negative values expected after subtracting mean
        assert img_tensor.min() < 0.5, (
            "Tensor does not appear to be normalized (all values >= 0.5)"
        )

    def test_multiple_items_load_without_error(self, dcid7_train):
        for i in range(min(20, len(dcid7_train))):
            img, label = dcid7_train[i]
            assert img.shape == (3, 64, 64)
            assert 0 <= label < 7

    def test_all_classes_represented_in_train(self, dcid7_train):
        labels_seen = set()
        for _, label in dcid7_train.samples:
            labels_seen.add(label)
        assert labels_seen == set(range(7)), (
            f"Not all 7 classes represented in train set. Missing: {set(range(7)) - labels_seen}"
        )

    def test_noise_dataset_loads(self):
        noise_ds = DCIDDataset(
            dcid_root=DCID_ROOT,
            version="noise-512-7",
            split="train",
            transform=get_eval_transforms(image_size=64),
            image_size=64,
        )
        assert len(noise_ds) > 0
        img, label = noise_ds[0]
        assert img.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# 4. No Train/Test Leakage
# ---------------------------------------------------------------------------

class TestNoLeakage:
    def test_train_and_test_paths_are_disjoint(self, dcid7_train, dcid7_test):
        """File paths must never appear in both splits."""
        assert_no_leakage(
            dcid7_train.get_filepaths(),
            dcid7_test.get_filepaths(),
        )

    def test_train_paths_contain_train_prefix(self, dcid7_train):
        for fp in dcid7_train.get_filepaths():
            assert "\\train\\" in fp or "/train/" in fp, (
                f"Train filepath does not contain 'train' folder: {fp}"
            )

    def test_test_paths_contain_test_prefix(self, dcid7_test):
        for fp in dcid7_test.get_filepaths():
            assert "\\test\\" in fp or "/test/" in fp, (
                f"Test filepath does not contain 'test' folder: {fp}"
            )

    def test_dcid35_no_leakage(self, dcid35_train, dcid35_test):
        assert_no_leakage(
            dcid35_train.get_filepaths(),
            dcid35_test.get_filepaths(),
        )


# ---------------------------------------------------------------------------
# 5. Class Count Correctness
# ---------------------------------------------------------------------------

class TestClassCounts:
    def test_dcid7_train_has_expected_size(self, dcid7_train):
        """DCID-7: 7 classes × 4000 train = 28,000 train images."""
        assert len(dcid7_train) == 28_000, (
            f"Expected 28,000 train images, got {len(dcid7_train)}"
        )

    def test_dcid7_test_has_expected_size(self, dcid7_test):
        """DCID-7: 7 classes × 1000 test = 7,000 test images."""
        assert len(dcid7_test) == 7_000, (
            f"Expected 7,000 test images, got {len(dcid7_test)}"
        )

    def test_dcid7_class_counts_balanced(self, dcid7_train):
        counts = dcid7_train.get_class_counts()
        for cls, count in counts.items():
            assert count == 4_000, (
                f"Class '{cls}' has {count} images, expected 4,000"
            )

    def test_dcid35_train_has_expected_size(self, dcid35_train):
        """DCID-35: 35 classes × expected train count each."""
        assert len(dcid35_train) > 0, "DCID-35 train dataset is empty!"

    def test_dcid35_test_has_expected_size(self, dcid35_test):
        assert len(dcid35_test) > 0, "DCID-35 test dataset is empty!"


# ---------------------------------------------------------------------------
# 6. Split Safety
# ---------------------------------------------------------------------------

class TestSplitSafety:
    def test_invalid_split_name_raises(self):
        with pytest.raises(SplitViolationError):
            get_official_split_dirs(DCID_ROOT, "DCID-512-7", "val")

    def test_random_split_name_raises(self):
        with pytest.raises(SplitViolationError):
            get_official_split_dirs(DCID_ROOT, "DCID-512-7", "random")

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError):
            get_official_split_dirs(DCID_ROOT, "DCID-512-99", "train")

    def test_assert_no_leakage_clean(self):
        """Should not raise for disjoint lists."""
        assert_no_leakage(["a.jpg", "b.jpg"], ["c.jpg", "d.jpg"])

    def test_assert_no_leakage_detects_overlap(self):
        """Should raise when a path appears in both splits."""
        with pytest.raises(SplitViolationError):
            assert_no_leakage(["a.jpg", "shared.jpg"], ["shared.jpg", "b.jpg"])


# ---------------------------------------------------------------------------
# 7. Class Alignment -- clean vs noise variants (Correction 4)
# ---------------------------------------------------------------------------

class TestClassAlignment:
    # Clean and noise variants must have identical class names in identical
    # order. These tests fail loudly if they diverge.

    def test_dcid7_noise_class_names_match(self):
        # DCID-512-7 and noise-512-7 must have exactly the same classes.
        assert_class_alignment(DCID_ROOT, "DCID-512-7", "noise-512-7")

    def test_dcid35_noise_class_names_match(self):
        # DCID-512-35 and noise-512-35 must have exactly the same classes.
        assert_class_alignment(DCID_ROOT, "DCID-512-35", "noise-512-35")

    def test_validate_noise_alignment_passes(self):
        # Convenience wrapper must succeed for both pairs simultaneously.
        validate_noise_alignment(DCID_ROOT)

    def test_assert_class_alignment_detects_name_mismatch(self, tmp_path):
        # Differing class names must raise ValueError.
        (tmp_path / "fake_clean" / "train" / "1.ClassA").mkdir(parents=True)
        (tmp_path / "fake_clean" / "train" / "2.ClassB").mkdir(parents=True)
        (tmp_path / "fake_noise" / "train" / "1.ClassA").mkdir(parents=True)
        (tmp_path / "fake_noise" / "train" / "2.ClassC").mkdir(parents=True)
        import pytest
        with pytest.raises(ValueError, match="Class alignment FAILED"):
            assert_class_alignment(tmp_path, "fake_clean", "fake_noise")

    def test_assert_class_alignment_detects_length_mismatch(self, tmp_path):
        # Different number of classes must raise ValueError.
        (tmp_path / "a" / "train" / "1.ClassA").mkdir(parents=True)
        (tmp_path / "a" / "train" / "2.ClassB").mkdir(parents=True)
        (tmp_path / "b" / "train" / "1.ClassA").mkdir(parents=True)
        import pytest
        with pytest.raises(ValueError, match="Class alignment FAILED"):
            assert_class_alignment(tmp_path, "a", "b")

    def test_clean_and_noise_dcid7_share_class_to_idx(self):
        # DCIDDataset for clean and noise DCID-7 must produce identical class_to_idx.
        clean_ds = DCIDDataset(
            dcid_root=DCID_ROOT, version="DCID-512-7", split="train",
            transform=get_eval_transforms(image_size=32), image_size=32,
        )
        noise_ds = DCIDDataset(
            dcid_root=DCID_ROOT, version="noise-512-7", split="train",
            transform=get_eval_transforms(image_size=32), image_size=32,
        )
        assert clean_ds.classes == noise_ds.classes, (
            "Clean/noise DCID-7 class lists differ: "
            + str(clean_ds.classes) + " vs " + str(noise_ds.classes)
        )
        assert clean_ds.class_to_idx == noise_ds.class_to_idx


# ---------------------------------------------------------------------------
# 8. skip_corrupt Behaviour (Correction 3)
# ---------------------------------------------------------------------------

class TestSkipCorrupt:
    # skip_corrupt=True must remove corrupt files from ds.samples at
    # construction time (tracked in ds.n_corrupt). No black placeholder ever.

    def test_skip_corrupt_true_excludes_from_sample_list(self, tmp_path):
        # Corrupt files must be absent from ds.samples; ds.n_corrupt reflects count.
        import shutil
        src_cls = Path(DCID_ROOT) / "DCID-512-7" / "train" / "1.Red sandstone"
        fake_root = tmp_path / "fake_dcid"
        fake_split = fake_root / "DCID-512-7" / "train" / "1.Red sandstone"
        fake_split.mkdir(parents=True)

        for img in sorted(src_cls.iterdir())[:4]:
            shutil.copy(img, fake_split / img.name)

        (fake_split / "bad1.jpg").write_bytes(b"NOTANIMAGE_1")
        (fake_split / "bad2.jpg").write_bytes(b"NOTANIMAGE_2")

        ds = DCIDDataset(
            dcid_root=fake_root, version="DCID-512-7", split="train",
            transform=get_eval_transforms(image_size=32), image_size=32,
            skip_corrupt=True,
        )
        assert ds.n_corrupt == 2, "Expected n_corrupt=2, got " + str(ds.n_corrupt)
        assert len(ds.samples) == 4, "Expected 4 valid samples, got " + str(len(ds.samples))
        for fp, _ in ds.samples:
            assert "bad1" not in fp and "bad2" not in fp, (
                "Corrupt file still in sample list: " + fp
            )

    def test_skip_corrupt_false_raises_not_placeholder(self, tmp_path):
        # With skip_corrupt=False a corrupt image raises RuntimeError, not a black tensor.
        import shutil
        src_cls = Path(DCID_ROOT) / "DCID-512-7" / "train" / "1.Red sandstone"
        fake_root = tmp_path / "fake_dcid2"
        fake_split = fake_root / "DCID-512-7" / "train" / "1.Red sandstone"
        fake_split.mkdir(parents=True)

        for img in sorted(src_cls.iterdir())[:2]:
            shutil.copy(img, fake_split / img.name)
        # Plain ASCII text - PIL cannot identify this as any image format.
        (fake_split / "corrupt.jpg").write_bytes(b"THIS IS NOT AN IMAGE FILE AT ALL")

        ds = DCIDDataset(
            dcid_root=fake_root, version="DCID-512-7", split="train",
            transform=get_eval_transforms(image_size=32), image_size=32,
            skip_corrupt=False,
        )
        # Use Path(fp).name to match only the filename, NOT the full path.
        # (pytest tmp_path dirs contain the test name which includes "corrupt".)
        corrupt_idx = next(
            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in Path(fp).name
        )
        import pytest
        with pytest.raises(RuntimeError):
            _ = ds[corrupt_idx]

    def test_default_dataset_samples_are_loadable(self, dcid7_train):
        # Spot-check 10 items -- all must return valid tensors, no errors.
        for i in range(min(10, len(dcid7_train))):
            img, label = dcid7_train[i]
            assert img.shape == (3, 64, 64)
            assert isinstance(label, int)
