"""Append test classes 7 and 8 to test_dataset.py."""
import textwrap
from pathlib import Path

APPEND = textwrap.dedent("""

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
        (fake_split / "corrupt.jpg").write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"BROKEN")

        ds = DCIDDataset(
            dcid_root=fake_root, version="DCID-512-7", split="train",
            transform=get_eval_transforms(image_size=32), image_size=32,
            skip_corrupt=False,
        )
        corrupt_idx = next(
            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in fp
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
""")

target = Path("tests/test_dataset.py")
with open(target, "a", encoding="utf-8") as f:
    f.write(APPEND)
print(f"Appended {len(APPEND)} chars. New line count: {len(target.read_text(encoding='utf-8').splitlines())}")
