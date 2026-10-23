# LithoScan — DCID Dataset Reconnaissance Report

Generated: 2026-09-13 22:55:17  |  Elapsed: 71.8s

> **License**: DCID is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  Academic / non-commercial use only.

---

## 1  Structure Verification

✅  All four expected DCID directories found.

| Directory | train | test | Total |
|-----------|------:|-----:|------:|
| `DCID-512-7` | 28,000 | 7,000 | 35,000 |
| `DCID-512-35` | 27,999 | 7,000 | 34,999 |
| `noise-512-7` | 11,200 | 2,800 | 14,000 |
| `noise-512-35` | 11,200 | 2,800 | 14,000 |

---

## 2  Class Distributions

### DCID-512-7  (7 classes)

| Class | Train | Test | Total |
|-------|------:|-----:|------:|
| 1.Red sandstone | 4,000 | 1,000 | 5,000 |
| 2.Light sandstone | 4,000 | 1,000 | 5,000 |
| 3.Gray siltstone | 4,000 | 1,000 | 5,000 |
| 4.Mudstone | 4,000 | 1,000 | 5,000 |
| 5.Granite | 4,000 | 1,000 | 5,000 |
| 6.Basalt | 4,000 | 1,000 | 5,000 |
| 7.Marble | 4,000 | 1,000 | 5,000 |

### DCID-512-35  (35 classes)

| Class | Train | Test | Total |
|-------|------:|-----:|------:|
| 1.Red sandstone | 800 | 200 | 1,000 |
| 2.Light sandstone | 800 | 200 | 1,000 |
| 3.Gray-green sandstone | 800 | 200 | 1,000 |
| 4.Gray siltstone | 800 | 200 | 1,000 |
| 5.Brown siltstone | 800 | 200 | 1,000 |
| 6.Carbonaceous siltstone | 800 | 200 | 1,000 |
| 7.Mudstone | 800 | 200 | 1,000 |
| 8.Granite | 800 | 200 | 1,000 |
| 9.Megacrystic granite | 800 | 200 | 1,000 |
| 10.Monzogranite | 800 | 200 | 1,000 |
| 11.Fine-grained granite | 800 | 200 | 1,000 |
| 12.Pegmatite | 800 | 200 | 1,000 |
| 13.Basalt | 800 | 200 | 1,000 |
| 14.Fine-grained basalt | 800 | 200 | 1,000 |
| 15.Gabbro | 800 | 200 | 1,000 |
| 16.Dolerite | 800 | 200 | 1,000 |
| 17.White marble | 800 | 200 | 1,000 |
| 18.Blue marble | 800 | 200 | 1,000 |
| 19.Dolomite | 800 | 200 | 1,000 |
| 20.Black Shale | 800 | 200 | 1,000 |
| 21.Red shale | 800 | 200 | 1,000 |
| 22.Dark green calc-silicate | 800 | 200 | 1,000 |
| 23.Brown calc-silicate | 800 | 200 | 1,000 |
| 24.Skarn | 800 | 200 | 1,000 |
| 25.Red-brown dacite | 800 | 200 | 1,000 |
| 26.Black dacite | 800 | 200 | 1,000 |
| 27.Rhyolite | 800 | 200 | 1,000 |
| 28.Banded iron formation | 799 | 200 | 999 |
| 29.Hematite breccia | 800 | 200 | 1,000 |
| 30.Altered breccia | 800 | 200 | 1,000 |
| 31.Schist | 800 | 200 | 1,000 |
| 32.Gneiss | 800 | 200 | 1,000 |
| 33.Porphyry | 800 | 200 | 1,000 |
| 34.Brecciated mudstone | 800 | 200 | 1,000 |
| 35.Shale fragments | 800 | 200 | 1,000 |

### noise-512-7  (7 classes)

| Class | Train | Test | Total |
|-------|------:|-----:|------:|
| 1.Red sandstone | 1,600 | 400 | 2,000 |
| 2.Light sandstone | 1,600 | 400 | 2,000 |
| 3.Gray siltstone | 1,600 | 400 | 2,000 |
| 4.Mudstone | 1,600 | 400 | 2,000 |
| 5.Granite | 1,600 | 400 | 2,000 |
| 6.Basalt | 1,600 | 400 | 2,000 |
| 7.Marble | 1,600 | 400 | 2,000 |

### noise-512-35  (35 classes)

| Class | Train | Test | Total |
|-------|------:|-----:|------:|
| 1.Red sandstone | 320 | 80 | 400 |
| 2.Light sandstone | 320 | 80 | 400 |
| 3.Gray-green sandstone | 320 | 80 | 400 |
| 4.Gray siltstone | 320 | 80 | 400 |
| 5.Brown siltstone | 320 | 80 | 400 |
| 6.Carbonaceous siltstone | 320 | 80 | 400 |
| 7.Mudstone | 320 | 80 | 400 |
| 8.Granite | 320 | 80 | 400 |
| 9.Megacrystic granite | 320 | 80 | 400 |
| 10.Monzogranite | 320 | 80 | 400 |
| 11.Fine-grained granite | 320 | 80 | 400 |
| 12.Pegmatite | 320 | 80 | 400 |
| 13.Basalt | 320 | 80 | 400 |
| 14.Fine-grained basalt | 320 | 80 | 400 |
| 15.Gabbro | 320 | 80 | 400 |
| 16.Dolerite | 320 | 80 | 400 |
| 17.White marble | 320 | 80 | 400 |
| 18.Blue marble | 320 | 80 | 400 |
| 19.Dolomite | 320 | 80 | 400 |
| 20.Black Shale | 320 | 80 | 400 |
| 21.Red shale | 320 | 80 | 400 |
| 22.Dark green calc-silicate | 320 | 80 | 400 |
| 23.Brown calc-silicate | 320 | 80 | 400 |
| 24.Skarn | 320 | 80 | 400 |
| 25.Red-brown dacite | 320 | 80 | 400 |
| 26.Black dacite | 320 | 80 | 400 |
| 27.Rhyolite | 320 | 80 | 400 |
| 28.Banded iron formation | 320 | 80 | 400 |
| 29.Hematite breccia | 320 | 80 | 400 |
| 30.Altered breccia | 320 | 80 | 400 |
| 31.Schist | 320 | 80 | 400 |
| 32.Gneiss | 320 | 80 | 400 |
| 33.Porphyry | 320 | 80 | 400 |
| 34.Brecciated mudstone | 320 | 80 | 400 |
| 35.Shale fragments | 320 | 80 | 400 |

---

## 3  Image Dimensions

| Width | Height | Channels |
|------:|-------:|---------:|
| 512 | 512 | 3 |

---

## 4  Data Quality

- **Corrupt / unreadable images**: 0

- **Filename duplicates** (within splits): 0
- **Cross-split hash duplicates** (train ↔ test leakage): 0
  ✅  No leakage detected.

---

## 5  Summary

| Metric | Value |
|--------|-------|
| Total images scanned | 97,999 |
| Corrupt images | 0 |
| Unique dimension combinations | 1 |
| Cross-split hash duplicates | 0 |
| Scan time | 71.8s |

---

## 6  Citation

```
Jia-Yu Li et al. A large-scale, high-quality dataset for lithology identification:
Construction and applications. Petroleum Science, 2025.
https://doi.org/10.1016/j.petsci.2025.04.013
```
