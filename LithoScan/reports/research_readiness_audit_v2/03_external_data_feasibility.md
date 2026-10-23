# 03: EXTERNAL DATASET FEASIBILITY AUDIT — LITHOSCAN

**Audit Gate:** Source Provenance & Claim Correction  
**Date:** October 22, 2026  
**Auditor:** Skeptic Senior Research Scientist & Methodology Auditor  
**Objective:** Identify and evaluate candidate open-access drill-core image datasets to support an independent, source-separated out-of-domain evaluation.

---

## 1. Evaluation Criteria for External Datasets

To resolve the provenance deadlock in DCID without compromising academic standards, candidate external datasets must satisfy five non-negotiable criteria:
1. **Credible Geological Ground Truth:** Labels must derive from formal geological surveys, certified core logs, or thin-section/assay validation—not crowdsourced guesswork.
2. **Traceable Provenance & Licensing:** Permissive academic license (e.g., CC BY, Open Government Licence) with clear institutional authorship.
3. **Source Group Metadata:** Explicit borehole/well identifiers, downhole depth intervals, or geographic coordinates allowing a strict **Leave-One-Well-Out (LOWO)** or **Leave-One-Site-Out (LOSO)** evaluation split.
4. **Distinctness from DCID:** Different core photographic equipment, lighting conditions, rock formations, and drilling facilities to ensure a genuine test of domain shift.
5. **No Label Fabrication:** Evaluated strictly on legitimate petrological intersections or representation transfer; no artificial label remapping.

---

## 2. Comprehensive Candidate Assessment

### Candidate 1: GSSA HyLogger Lithology Dataset (Alzubaidi et al., Mendeley Data)

- **Source / Citation:** Fatimah Alzubaidi et al. (2022). *Lithology dataset.* **Mendeley Data**, V1. DOI: [10.17632/b6rj29mrss.1](https://doi.org/10.17632/b6rj29mrss.1). Accompanying paper: *Automated lithology classification from drill core images using convolutional neural networks.* **Journal of Petroleum Science and Engineering**, 2021.
- **Data Host / Archive:** Mendeley Data open repository.
- **License:** **CC BY 4.0** (Open academic reuse with attribution).
- **Core Origin & Provenance:** High-resolution digital core photography acquired by the **Geological Survey of South Australia (GSSA)** using the HyLogger-3 core scanning system across South Australian petroleum and mineral exploration boreholes.
- **Image Specifications:** Standardized cropped $2\times 2\text{ cm}$ core patches ($224\times 224$ pixels), formatted similarly to patch-based classification workflows.
- **Geological Classes:**
  1. Sandstone
  2. Limestone
  3. Shale
  4. Non-core / Tray artifacts ("Garbage" class)
- **Source-Separation Feasibility: HIGH.**  
  Because source images trace to specific South Australian boreholes cataloged in GSSA SARIG database, patches can be grouped by original well identifier.
- **Distinctness from DCID: VERY HIGH.**  
  Acquired using a standardized linescan camera system with calibrated LED illumination in an Australian government core library, contrasting sharply with DCID's Chinese geological core archive.
- **Technical Bottleneck & Limitation:**  
  Taxonomic granularity is coarse (3 geological classes + 1 artifact class). It overlaps directly with DCID classes (*Sandstone*, *Shale*, *Marble/Limestone*), but cannot evaluate DCID's 35 fine-grained sub-classes without zero-shot feature probing or head retraining.

---

### Candidate 2: National Virtual Core Library (NVCL) / AuScope / Geoscience Australia

- **Source / Citation:** AuScope / National Virtual Core Library (NVCL). Hosted collaboratively by Geoscience Australia and Australian State Geological Surveys (GSWA, GSSA, GSQ, GSNSW).
- **Data Host / Archive:** AuScope Discovery Portal (`portal.auscope.org.au`) and state survey open data services (e.g., GSWA WAMEX / GeoVIEW).
- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** / Australian Government Open Access and Licensing Framework (AusGOAL).
- **Core Origin & Provenance:** Thousands of kilometers of continuous mineral and energy exploration drill cores spanning major mining terranes (Pilbara, Yilgarn, Mount Isa, Gawler Craton).
- **Metadata Traceability: EXCEPTIONAL.**  
  Every core tray image is tied to:
  - Unique National Borehole Identifier (ENO / WAMEX ID)
  - Exact Geographic Coordinates (WGS84 Lat/Long)
  - Continuous Downhole Depth Interval (Start Depth / End Depth in meters)
  - Stratigraphic Unit and Petrographic Core Logs
  - Multi-sensor spectral data (TIR/VNIR/SWIR mineralogy from HyLogger)
- **Source-Separation Feasibility: PERFECT.**  
  Supports true Leave-One-Well-Out, Leave-One-Basin-Out, and spatial block cross-validation.
- **Distinctness from DCID: MAXIMUM.**  
  Completely distinct geological basins, metamorphic grades, and operational scanning rigs.
- **Technical Bottleneck & Limitation:**  
  The data is archived as full continuous core-tray photographs (often $4000\times 3000$ pixels) rather than pre-cropped $512\times 512$ image patches. Ingestion requires building a core-run detection and patch cropping pipeline (e.g., using sliding windows or Mask R-CNN), requiring non-trivial engineering effort.

---

### Candidate 3: International Ocean Discovery Program (IODP / ODP) Visual Core Descriptions

- **Source / Citation:** International Ocean Discovery Program (IODP), Texas A&M University Core Repository.
- **Data Host / Archive:** IODP JANUS and LIMS Scientific Ocean Drilling Databases (`web.iodp.tamu.edu`).
- **License:** **Public Domain / Open Data Policy** (U.S. National Science Foundation).
- **Core Origin & Provenance:** Scientific ocean drilling cores retrieved globally from oceanic crust and marine basins across hundreds of international expeditions.
- **Metadata Traceability: RIGOROUS.**  
  Full hierarchical sample identifier: `Expedition-Site-Hole-Core-Type-Section-Interval_cm`. Top/bottom depths recorded to millimeter accuracy. Detailed shipboard Visual Core Descriptions (VCDs) authored by international geological science parties.
- **Source-Separation Feasibility: PERFECT.**  
  Evaluation can hold out entire drilling Sites (e.g., Site U1500 vs. Site U1501).
- **Technical Bottleneck & Limitation:**  
  Marine ocean-drilling sedimentological and volcanic facies (e.g., pelagic nannofossil ooze, basaltic pillow lavas, volcanic ash) have limited direct taxonomic overlap with continental petroleum and mining exploration lithologies.

---

## 3. Comparative Summary Matrix

| Dataset | Access & License | Provenance Traceability | Source Separation Unit | Geological Quality | Taxonomic Compatibility with DCID | Engineering Effort | Overall Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSSA HyLogger (Alzubaidi)** | Open (CC BY 4.0) | Verified (GSSA) | Well / Borehole ID | High (State Survey) | Moderate (Coarse 3-class overlap: Sandstone, Shale, Limestone) | **Very Low** (Pre-cropped patches, 224px) | **HIGH (Best Quick Validation)** |
| **AuScope NVCL** | Open (CC BY 4.0) | Complete (Geoscience Australia) | Well ID, Basin, Lat/Long | Gold Standard (Multi-sensor) | High (All 35 rock types represented across Australian terranes) | **High** (Requires tray-to-patch pipeline) | **VERY HIGH (Best for Final Paper)** |
| **IODP Ocean Drilling** | Public Domain | Complete (IODP LIMS) | Expedition / Site / Hole | Research Grade (Shipboard VCD) | Low (Primarily pelagic marine sediments & crustal volcanics) | **High** (Custom image extraction required) | **MODERATE (Domain Shift Only)** |

---

## 4. Feasibility Verdict

1. **Quickest Turnaround External Proof:**  
   The **GSSA HyLogger Lithology Dataset (Alzubaidi et al., Mendeley Data)** is the most immediately viable external evaluation asset. It is pre-cropped, openly licensed (CC BY 4.0), and provides a clean test of whether DCID-trained representations retain any discriminative power on an entirely different government core library.
2. **Gold-Standard Publication Benchmark:**  
   The **AuScope NVCL** core archive provides the ideal long-term foundation for a definitive journal paper. If LithoScan ingests 3–5 representative NVCL boreholes with known coordinates, it can execute the first true **Leave-One-Well-Out spatial evaluation** directly addressing the spatial leakage concerns raised in this audit.
3. **Strict Constraint:** Under no circumstances should labels be arbitrarily merged or forced to fit. Evaluation on external datasets must be framed as a representation transfer test or evaluated on the subset of unambiguously shared lithologies.
