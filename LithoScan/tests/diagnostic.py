"""
diagnostic.py — Exact fact-gathering for the failing skip_corrupt test.
Run from project root: python tests/diagnostic.py
"""
import sys, os
sys.path.insert(0, ".")

from pathlib import Path
import shutil, tempfile

from PIL import Image, UnidentifiedImageError
from src.data.dcid_dataset import DCIDDataset, get_eval_transforms

DCID_ROOT = Path(r"C:\Users\punee\Desktop\Software_lithology\DCID\DCID")

# ── Replicate the test exactly ──────────────────────────────────────────────
tmp = Path(tempfile.mkdtemp(prefix="litho_diag_"))
src_cls = DCID_ROOT / "DCID-512-7" / "train" / "1.Red sandstone"
fake_root = tmp / "fake_dcid2"
fake_split = fake_root / "DCID-512-7" / "train" / "1.Red sandstone"
fake_split.mkdir(parents=True)

for img in sorted(src_cls.iterdir())[:2]:
    shutil.copy(img, fake_split / img.name)

corrupt_path = fake_split / "corrupt.jpg"
corrupt_path.write_bytes(b"THIS IS NOT AN IMAGE FILE AT ALL")

print("=" * 60)
print("FILES IN FAKE SPLIT:")
for f in sorted(fake_split.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")

ds = DCIDDataset(
    dcid_root=fake_root, version="DCID-512-7", split="train",
    transform=get_eval_transforms(image_size=32), image_size=32,
    skip_corrupt=False,
)

print(f"\nds.n_corrupt = {ds.n_corrupt}")
print(f"len(ds.samples) = {len(ds.samples)}")
print("SAMPLES:")
for i, (fp, label) in enumerate(ds.samples):
    print(f"  [{i}] {Path(fp).name}")

# Find corrupt index
try:
    corrupt_idx = next(i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in fp)
    filepath = ds.samples[corrupt_idx][0]
except StopIteration:
    print("\nERROR: 'corrupt' not found in any sample filepath!")
    sys.exit(1)

print(f"\ncorrupt_idx = {corrupt_idx}")
print(f"filepath    = {filepath}")

# ── Fact 5: File properties ─────────────────────────────────────────────────
p = Path(filepath)
print(f"\nPath.exists()  = {p.exists()}")
print(f"file size      = {p.stat().st_size} bytes")
raw = p.read_bytes()
print(f"first 32 bytes = {raw[:32]}")

# ── Fact 6: PIL operations ──────────────────────────────────────────────────
print("\n--- PIL.Image.open(filepath) ---")
try:
    im = Image.open(filepath)
    print(f"  SUCCESS: format={im.format!r}  mode={im.mode!r}  size={im.size}")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")

print("\n--- Image.open(filepath).verify() ---")
try:
    im2 = Image.open(filepath)
    im2.verify()
    print("  SUCCESS (no exception)")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")

print("\n--- Image.open(filepath).load() ---")
try:
    im3 = Image.open(filepath)
    im3.load()
    print(f"  SUCCESS: mode={im3.mode!r}  size={im3.size}")
except Exception as e:
    print(f"  RAISED: {type(e).__name__}: {e}")

print("\n--- Image.open(filepath).convert('RGB') [exact __getitem__ call] ---")
try:
    im4 = Image.open(filepath).convert("RGB")
    print(f"  SUCCESS: mode={im4.mode!r}  size={im4.size}")
except (OSError, UnidentifiedImageError) as e:
    print(f"  RAISED OSError/UnidentifiedImageError: {type(e).__name__}: {e}")
except Exception as e:
    print(f"  RAISED OTHER: {type(e).__name__}: {e}")

# ── Fact: what __getitem__ actually does ────────────────────────────────────
print("\n--- ds[corrupt_idx] ---")
try:
    tensor, label = ds[corrupt_idx]
    print(f"  SUCCESS: tensor.shape={tensor.shape}")
except RuntimeError as e:
    print(f"  RAISED RuntimeError: {e}")
except Exception as e:
    print(f"  RAISED {type(e).__name__}: {e}")

print("\n--- PIL version ---")
import PIL
print(f"  PIL version: {PIL.__version__}")
print(f"  Python version: {sys.version}")

# cleanup
shutil.rmtree(tmp, ignore_errors=True)
