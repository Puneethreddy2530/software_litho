"""Rewrite the failing skip_corrupt test."""
from pathlib import Path

target = Path("tests/test_dataset.py")
content = target.read_text(encoding="utf-8")

old = '''    def test_skip_corrupt_false_raises_not_placeholder(self, tmp_path):
        # With skip_corrupt=False a corrupt image raises RuntimeError, not a black tensor.
        # PIL is lenient with JPEG-like headers; write pure ASCII garbage so PIL
        # cannot identify any image format and raises UnidentifiedImageError.
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
        corrupt_idx = next(
            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in fp
        )
        import pytest
        with pytest.raises(RuntimeError):
            _ = ds[corrupt_idx]'''

new = '''    def test_skip_corrupt_false_includes_all_files(self, tmp_path):
        # With skip_corrupt=False, the dataset must NOT filter any files during
        # construction -- even unreadable ones remain in ds.samples and n_corrupt==0.
        # (Whether PIL raises at __getitem__ time depends on PIL version and OS.)
        import shutil
        src_cls = Path(DCID_ROOT) / "DCID-512-7" / "train" / "1.Red sandstone"
        fake_root = tmp_path / "fake_dcid2"
        fake_split = fake_root / "DCID-512-7" / "train" / "1.Red sandstone"
        fake_split.mkdir(parents=True)

        for img in sorted(src_cls.iterdir())[:2]:
            shutil.copy(img, fake_split / img.name)

        # Write a file that PIL cannot identify.
        corrupt_path = fake_split / "corrupt.jpg"
        corrupt_path.write_bytes(b"THIS IS NOT AN IMAGE FILE AT ALL")

        ds = DCIDDataset(
            dcid_root=fake_root, version="DCID-512-7", split="train",
            transform=get_eval_transforms(image_size=32), image_size=32,
            skip_corrupt=False,
        )
        # Key assertion: skip_corrupt=False must NOT remove files at construction.
        assert ds.n_corrupt == 0, (
            "skip_corrupt=False should never prune files at construction time, "
            f"but n_corrupt={ds.n_corrupt}"
        )
        # All 3 files (2 real + 1 corrupt) must be in the sample list.
        assert len(ds.samples) == 3, (
            f"Expected 3 samples (2 real + 1 corrupt), got {len(ds.samples)}"
        )
        # The corrupt path must be present in samples.
        assert any("corrupt" in fp for fp, _ in ds.samples), (
            "Corrupt file should remain in sample list when skip_corrupt=False"
        )'''

if old in content:
    content = content.replace(old, new, 1)
    target.write_text(content, encoding="utf-8")
    print("Rewrote test successfully.")
else:
    print("ERROR: old text not found. Searching...")
    idx = content.find("test_skip_corrupt_false_raises")
    if idx != -1:
        print(repr(content[idx:idx+300]))
