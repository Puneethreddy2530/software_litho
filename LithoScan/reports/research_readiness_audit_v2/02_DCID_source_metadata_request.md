# 02: DCID SOURCE METADATA REQUEST (DRAFT AUTHOR INQUIRY)

**Document Type:** Formal Data Request (Draft Only — Do Not Send)  
**Target Recipients:** Dr. Jia-Yu Li, Prof. Ji-Zhou Tang, and Prof. Xian-Zheng Zhao (Corresponding Authors)  
**Affiliation:** State Key Laboratory of Marine Geology / School of Ocean and Earth Science, Tongji University, Shanghai, China  
**Reference Publication:** *A large-scale, high-quality dataset for lithology identification: Construction and applications.* **Petroleum Science**, 2025, DOI: [10.1016/j.petsci.2025.04.013](https://doi.org/10.1016/j.petsci.2025.04.013).

---

> **IMPORTANT ADMINISTRATIVE NOTICE:**  
> This document is an unexecuted internal research draft. Under strict audit guidelines, no electronic communication, email, or external contact has been initiated.

---

### Draft Letter to Dataset Authors

**Subject:** Inquiry regarding source borehole metadata and split procedures for the Drill Core Image Dataset (DCID)

Dear Dr. Li, Prof. Tang, and Prof. Zhao,

We are writing to express our sincere appreciation for your groundbreaking contribution to geological machine learning through the release of the **Drill Core Image Dataset (DCID)** published in *Petroleum Science* (2025). Your comprehensive benchmark of 98,000 high-resolution images across 35 lithologies has provided the subsurface community with an invaluable open asset.

Our research group is currently conducting an empirical investigation into automated core logging reliability, focusing specifically on spatial data partitioning, out-of-distribution acquisition robustness, and calibrated selective prediction. As part of our commitment to rigorous methodology and reproducibility, we wish to evaluate how models generalize across independent geological units.

To support a methodologically sound, leakage-resistant evaluation that adheres to modern spatial cross-validation standards (e.g., Leave-One-Well-Out or block cross-validation), we would be deeply grateful if you could share additional provenance metadata regarding the original core acquisition, or clarify whether such records can be made available for academic research reuse:

1. **Crop-to-Parent-Image Mapping:**  
   Is there a mapping table or manifest linking each cropped $512\times 512$ image patch (e.g., `1.Red sandstone_1.jpg`) to its original continuous core-box photo, core tray identifier, or bounding box pixel coordinates?
2. **Borehole and Well Identifiers:**  
   Are there anonymized well, borehole, or drill-hole identifiers (e.g., Well-A, Well-B) associated with the physical core samples from which the classes were extracted?
3. **Stratigraphic Depth Intervals:**  
   Are downhole depth markers (top/bottom meterage or downhole order) available for the sampled intervals?
4. **Clean-to-Noise Cohort Correspondence:**  
   In the Real-World Data Augmentation (RWDA) subsets (`noise-512-7` and `noise-512-35`), do the defective images represent independently photographed core sections displaying physical damage, or do any samples directly correspond to corrupted versions of specific clean images in `DCID-512`?
5. **Official Split Generation Procedure:**  
   Could you confirm whether the official 80/20 train/test split was generated via uniform random sampling across all cropped patches, or if entire core boxes or drill runs were held out as contiguous blocks?
6. **Academic Reuse and Attribution:**  
   We fully respect that DCID is licensed under Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0). Any metadata shared would be used strictly for non-commercial academic research, cited prominently, and archived in accordance with your specified licensing and attribution requirements.

If full coordinates or well names remain commercially sensitive, even coarse grouping indicators (such as an anonymized column `core_run_id` or `well_group_id` mapping each crop index to a cluster) would fully suffice to allow our study to evaluate spatial generalization without compromising proprietary information.

Thank you very much for your time, consideration, and continued leadership in open geoscience.

Sincerely,  

*[Research Team Name / Affiliation]*  
*[Contact Email / Department]*  
*[Institutional Webpage]*
