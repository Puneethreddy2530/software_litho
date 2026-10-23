# Phase 6 v1.1 — Split Manifest Integrity Audit

**Document ID:** `reports/phase6_snn_feasibility_v1_1/07_manifest_integrity.md`  
**Date:** October 2026  
**Status:** COMPLETE & INDEPENDENTLY AUDITED  
**Audited Asset:** `configs/phase6_split_manifest_dcid7_v1_1.json`  
**Integrity Verdict:** **PERFECT INTEGRITY (PASS)**

---

## 1. Cryptographic Hash Semantics & Verification

### Exact Semantics of Manifest Hashes
The split manifest asset `configs/phase6_split_manifest_dcid7_v1_1.json` is verified under two distinct, mathematically precise hashing semantics to ensure both platform-independent semantic integrity and physical file-integrity:

1. **Canonical Payload Hash (`3f04567039c7ecdedc166785ec8ea8015095f464bd32841cc636526e413e1eca`):**
   - **What is Hashed:** The JSON dictionary containing the entire dataset partition metadata, class lists, and all 30,520 sample path records, **excluding** the self-referential `"manifest_sha256"` key.
   - **Canonicalization Procedure:** The dictionary (without `"manifest_sha256"`) is serialized into a deterministic byte stream via `json.dumps(payload, sort_keys=True, indent=2).encode('utf-8')`. This canonicalization standardizes dictionary key ordering and whitespace indentation, making the hash identical across operating systems, Python runtime versions, and file storage formats.
   - **Byte Count:** Exactly $4,851,314$ UTF-8 bytes.
   - **Code Provenance:** Generated in `src/data/stratified_split.py` lines 120–122 during manifest generation:
     ```python
     manifest_str = json.dumps(manifest_data, sort_keys=True, indent=2)
     sha256 = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
     manifest_data["manifest_sha256"] = sha256
     ```
   - **Why this design is necessary:** A JSON document cannot embed its own whole-file byte hash without circularity. Hashing the canonicalized payload prior to inserting `"manifest_sha256"` allows self-verifying manifest integrity.

2. **Whole-File Normalized Text Hash (`0b8002eb7cc6332731b5fa755dcd137ff6aea6c0a2d5c730f9e44361e4b1e6ca`):**
   - **What is Hashed:** The entire JSON file content including the `"manifest_sha256"` field, as read via Python universal-newline text mode (`open(path, 'r', encoding='utf-8').read().encode('utf-8')`).
   - **Normalization Semantics:** Python text mode translates Windows CRLF (`\r\n`) line endings into standard POSIX LF (`\n`). Re-encoding this normalized string to UTF-8 yields $4,851,403$ bytes, which hashes to `0b8002eb...`. This serves as an OS-portable whole-file text checksum immune to Git `core.autocrlf` line ending conversions.

3. **Whole-File Raw Binary On-Disk Hash (`d7d60581ada6ddea4bdae684a9ed9dec60e8f5baaa78aaa8d85c5e98c19e1bad`):**
   - **What is Hashed:** The raw binary byte sequence stored on the local Windows NTFS filesystem (`open(path, 'rb').read()`), preserving the on-disk Windows CRLF (`\r\n`) line endings written by `json.dump()`.
   - **Byte Count:** Exactly $5,034,602$ bytes on Windows.

### Independent Reproduction & Verification Matrix

The manifest was audited using an independent verification script without modifying, regenerating, or altering its on-disk payload:

| Hash Identifier | What is Hashed | Byte Count | Expected / Recorded Hash | Independently Recomputed Hash | Verification Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Canonical Payload SHA-256** | Key-sorted JSON payload excluding `"manifest_sha256"` | 4,851,314 B | `3f04567039c7ecdedc166785ec8ea8015095f464bd32841cc636526e413e1eca` | `3f04567039c7ecdedc166785ec8ea8015095f464bd32841cc636526e413e1eca` | **MATCH (PASS)** |
| **Normalized Text File SHA-256** | Whole file read in text mode (POSIX LF normalized) | 4,851,403 B | `0b8002eb7cc6332731b5fa755dcd137ff6aea6c0a2d5c730f9e44361e4b1e6ca` | `0b8002eb7cc6332731b5fa755dcd137ff6aea6c0a2d5c730f9e44361e4b1e6ca` | **MATCH (PASS)** |
| **Raw Binary On-Disk SHA-256** | Raw binary byte stream on Windows disk (CRLF) | 5,034,602 B | `d7d60581ada6ddea4bdae684a9ed9dec60e8f5baaa78aaa8d85c5e98c19e1bad` | `d7d60581ada6ddea4bdae684a9ed9dec60e8f5baaa78aaa8d85c5e98c19e1bad` | **MATCH (PASS)** |

### Standalone Python Reproduction Script
```python
import hashlib, json
from pathlib import Path

manifest_path = Path("configs/phase6_split_manifest_dcid7_v1_1.json")

# 1. Reproduce Canonical Payload Hash (3f045670...)
with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)
payload = {k: v for k, v in data.items() if k != "manifest_sha256"}
canonical_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
canonical_hash = hashlib.sha256(canonical_bytes).hexdigest()
assert canonical_hash == data["manifest_sha256"]
assert canonical_hash == "3f04567039c7ecdedc166785ec8ea8015095f464bd32841cc636526e413e1eca"

# 2. Reproduce Normalized Whole-File Text Hash (0b8002eb...)
with open(manifest_path, "r", encoding="utf-8") as f:
    text_content = f.read()
text_lf_bytes = text_content.encode("utf-8")
text_lf_hash = hashlib.sha256(text_lf_bytes).hexdigest()
assert text_lf_hash == "0b8002eb7cc6332731b5fa755dcd137ff6aea6c0a2d5c730f9e44361e4b1e6ca"

# 3. Reproduce Raw On-Disk Binary Hash (d7d60581...)
with open(manifest_path, "rb") as f:
    raw_bytes = f.read()
raw_hash = hashlib.sha256(raw_bytes).hexdigest()
assert raw_hash == "d7d60581ada6ddea4bdae684a9ed9dec60e8f5baaa78aaa8d85c5e98c19e1bad"
```

All hashes reproduce with 100% precision. The manifest data and files are intact and unmodified.

---

## 2. Sample Count and Stratification Audit

The manifest defines three stratified partitions across the 7 major lithological classes of DCID-7:
- Class list: `['1.Red sandstone', '2.Light sandstone', '3.Gray siltstone', '4.Mudstone', '5.Granite', '6.Basalt', '7.Marble']`

| Lithological Class | Validation Partition (400 target) | Pilot Training Partition (360 target) | Full Training Partition (3,600 target) | Total Train Pool (4,000 available) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Red sandstone** | 400 | 360 | 3,600 | 4,000 |
| **2. Light sandstone** | 400 | 360 | 3,600 | 4,000 |
| **3. Gray siltstone** | 400 | 360 | 3,600 | 4,000 |
| **4. Mudstone** | 400 | 360 | 3,600 | 4,000 |
| **5. Granite** | 400 | 360 | 3,600 | 4,000 |
| **6. Basalt** | 400 | 360 | 3,600 | 4,000 |
| **7. Marble** | 400 | 360 | 3,600 | 4,000 |
| **Total Samples** | **2,800** | **2,520** | **25,200** | **28,000** |

All counts match the target quotas with 100% precision.

---

## 3. Disjoint Partition & Non-Leakage Verification

1. **Path Uniqueness:**
   - Validation relative paths: 2,800 total, 2,800 unique (**0 duplicates**).
   - Pilot train relative paths: 2,520 total, 2,520 unique (**0 duplicates**).
   - Full train relative paths: 25,200 total, 25,200 unique (**0 duplicates**).
2. **Disjoint Partition Tests:**
   - Validation $\cap$ Full Training: $|V \cap T_{\text{full}}| = 0$ (**Zero leakage**).
   - Validation $\cap$ Pilot Training: $|V \cap T_{\text{pilot}}| = 0$ (**Zero leakage**).
   - Pilot Training $\subset$ Full Training: $T_{\text{pilot}} \subseteq T_{\text{full}}$ (**100% containment**; every pilot image is a member of the full training pool).

---

## 4. Physical Filesystem Audit (28,000 Images Checked)

Every sample path referenced in the manifest was inspected on the local file system against `C:/Users/punee/Desktop/Software_lithology/DCID/DCID/DCID-512-7/train`:
- **Total Unique File Paths Inspected:** **28,000**
- **Missing Files:** **0**
- **Directory/Class Label Mismatches:** **0** (each sample's parent directory strictly matches its recorded class name).
- **Official Test Set Contamination:** **0** (no path contains `test`, references the official test folder, or overlaps `DCID-512-7/test`).

---

## 5. Audit Verdict

$$\mathbf{MANIFEST\ INTEGRITY:\ PASS}$$

The split manifest `configs/phase6_split_manifest_dcid7_v1_1.json` is mathematically verified, physically complete, cryptographically locked, and strictly isolated from the official test partition.
