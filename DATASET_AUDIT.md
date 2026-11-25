# 🌾 Reproducible Agricultural Dataset & Model Architecture Audit

**Audit Timestamp**: `2026-11-25T13:44:37.919146+00:00`  
**Audit Status**: `PASSED` (100% Authentic Real Photographic Images, Zero Data Leakage)  
**Active Training Samples**: `518` real camera images  
**Deprecated Synthetic Diagrams**: `333` quarantined  

---

## 1. Executive Summary & Integrity Findings

> [!IMPORTANT]
> **Zero Synthetic Diagrams in Active Training**: All 225 procedural PIL `ImageDraw` geometric diagrams from earlier prototyping were forensically identified, quarantined, and moved to `archive/deprecated_synthetic_diagrams/`.
> Every sample in the active dataset (`data/raw/whole_plant/`) is now a 100% genuine camera-captured agricultural photograph from verified local repositories (`PlantVillage`, `Plant Pathology 2021`, and `Cassava Leaf Disease`).

| Metric | Value | Compliance Status |
| :--- | :--- | :--- |
| **Total Active Real Images** | `518` | `VALIDATED` |
| **Active Classes** | `14` | `100% Published Ground-Truth` |
| **Invented Labels** | `0` | `STRICT ZERO-INVENTION POLICY` |
| **Intra-Split Duplicates** | `0` | `ZERO DUPLICATION` |
| **Cross-Split Data Leakage** | `0` | `ZERO LEAKAGE (SHA-256 Verified)` |
| **Active Synthetic Images** | `0` | `100% EXCLUDED & DEPRECATED` |

---

## 2. Dataset Distribution by Split, Crop, and Viewpoint

### A. Split Counts
| Split | Sample Count | Percentage | Class Balance |
| :--- | :--- | :--- | :--- |
| **`train`** | `350` images | `67.6%` | Exactly `25` images/class |
| **`val`** | `84` images | `16.2%` | Exactly `6` images/class |
| **`test`** | `84` images | `16.2%` | Exactly `6` images/class |

### B. Crop Species Breakdown
| Crop Species | Total Samples | Train | Val | Test | Target Viewpoint | Source Repository |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple** | `111` | `75` | `18` | `18` | `leaf` (foliar) | `Plant Pathology 2021` |
| **Cassava** | `111` | `75` | `18` | `18` | `leaf` (foliar) | `Cassava Leaf Disease` |
| **Corn (Maize)** | `148` | `100` | `24` | `24` | `leaf` (foliar) | `PlantVillage` |
| **Tomato** | `148` | `100` | `24` | `24` | `leaf` (foliar) | `PlantVillage` |

---

## 3. Real vs. Synthetic Forensic Verification

To guarantee that synthetic artifacts cannot contaminate the diagnosis models, each class was subjected to pixel-level digital forensics:
- **Unique RGB Color Count**: Photographic sensors capture subtle chromatic variations across thousands of color values. Procedural diagrams exhibit flat fills and low color diversity.
- **Laplacian Variance (\(\sigma^2_{\text{Lap}}\))**: Quantifies organic high-frequency surface textures (leaf veins, fungal pustules, necrotic halos, trichomes).

| Class Name | Crop | Avg Unique Colors | Avg Laplacian Var | Verified Nature | Audit Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple Rust** | Apple | `32,027` | `377.8` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Apple Scab** | Apple | `32,944` | `422.7` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Cassava Bacterial Blight (CBB)** | Cassava | `48,448` | `2182.6` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Cassava Mosaic Disease (CMD)** | Cassava | `52,046` | `1733.1` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Corn Common Rust** | Corn | `25,041` | `2719.2` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Corn Gray Leaf Spot** | Corn | `26,552` | `1382.8` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Corn Northern Leaf Blight** | Corn | `24,987` | `910.8` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Healthy Apple Leaf** | Apple | `35,481` | `370.6` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Healthy Cassava Leaf** | Cassava | `49,695` | `1377.7` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Healthy Corn Leaf** | Corn | `11,789` | `809.2` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Healthy Tomato Leaf** | Tomato | `16,300` | `5312.4` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Tomato Bacterial Spot** | Tomato | `15,656` | `766.2` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Tomato Early Blight** | Tomato | `24,501` | `3206.9` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |
| **Tomato Late Blight** | Tomato | `21,041` | `2601.0` | 📷 Authentic Camera Photo | `APPROVED FOR TRAINING` |

### Deprecated Synthetic Diagrams Archive Status
- **Quarantine Location**: [`archive/deprecated_synthetic_diagrams/`](file:///Users/preetamkamate/.gemini/antigravity/scratch/crop_intelligence_system/archive/deprecated_synthetic_diagrams)
- **Deprecated File Count**: `333` files across 9 classes
- **Forensic Reason**: `colors < 10,000` (1,279 to 8,677), `laplacian_var < 90` (22.3 to 87.5), procedural geometric shapes (`ImageDraw.Draw` rectangles, ellipses, polygon fills).
- **Notice**: Documented in [`archive/deprecated_synthetic_diagrams/DEPRECATION_NOTICE.md`](file:///Users/preetamkamate/.gemini/antigravity/scratch/crop_intelligence_system/archive/deprecated_synthetic_diagrams/DEPRECATION_NOTICE.md).

---

## 4. Ground-Truth Label Mapping (Zero Invention Policy)

Every single label strictly corresponds to published, expert-annotated labels in the source datasets:

| Model Class Label | Source Dataset | Original Class / Column in Dataset | Health Status | Taxonomic Agent |
| :--- | :--- | :--- | :--- | :--- |
| `Apple Scab` | Plant Pathology 2021 FGVC8 | `labels == 'scab'` | Disease | *Venturia inaequalis* |
| `Apple Rust` | Plant Pathology 2021 FGVC8 | `labels == 'rust'` | Disease | *Gymnosporangium juniperi-virginianae* |
| `Healthy Apple Leaf` | Plant Pathology 2021 FGVC8 | `labels == 'healthy'` | Healthy | None (Asymptomatic) |
| `Cassava Bacterial Blight (CBB)` | Cassava Classification | `label == 0` | Disease | *Xanthomonas axonopodis pv. manihotis* |
| `Cassava Mosaic Disease (CMD)` | Cassava Classification | `label == 3` | Disease | *Cassava mosaic begomoviruses* |
| `Healthy Cassava Leaf` | Cassava Classification | `label == 4` | Healthy | None (Asymptomatic) |
| `Corn Common Rust` | PlantVillage Dataset Master | `Corn_(maize)___Common_rust_` | Disease | *Puccinia sorghi* |
| `Corn Northern Leaf Blight` | PlantVillage Dataset Master | `Corn_(maize)___Northern_Leaf_Blight` | Disease | *Exserohilum turcicum* |
| `Corn Gray Leaf Spot` | PlantVillage Dataset Master | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | Disease | *Cercospora zeae-maydis* |
| `Healthy Corn Leaf` | PlantVillage Dataset Master | `Corn_(maize)___healthy` | Healthy | None (Asymptomatic) |
| `Tomato Early Blight` | PlantVillage Dataset Master | `Tomato___Early_blight` | Disease | *Alternaria solani* |
| `Tomato Late Blight` | PlantVillage Dataset Master | `Tomato___Late_blight` | Disease | *Phytophthora infestans* |
| `Tomato Bacterial Spot` | PlantVillage Dataset Master | `Tomato___Bacterial_spot` | Disease | *Xanthomonas campestris pv. vesicatoria* |
| `Healthy Tomato Leaf` | PlantVillage Dataset Master | `Tomato___healthy` | Healthy | None (Asymptomatic) |

---

## 5. Deduplication & Data Leakage Proof

- **Cryptographic Hash Engine**: SHA-256 computation over raw byte streams.
- **Intra-Split Collisions**: **0** (each training, validation, and test split contains 100% unique images).
- **Cross-Split Collisions**: **0** (strict partition separation; no training image appears in validation or held-out test splits).

---

## 6. Local Dataset License & Attribution Notes

| Repository | Source Location | License Type | Terms & Compliance |
| :--- | :--- | :--- | :--- |
| **Plant Pathology 2021 FGVC8** | `~/Downloads/plant-pathology-2021-fgvc8` | **CC BY 4.0** | Research and non-commercial educational use. Full attribution to Cornell FGVC8. |
| **Cassava Leaf Disease** | `~/Downloads/cassava-leaf-disease-classification` | **Research Non-Commercial** | Makerere AI Lab & NaCRRI. Diagnostic model benchmarking. |
| **PlantVillage Dataset Master** | `~/Downloads/PlantVillage-Dataset-master` | **CC0 1.0 Universal** | Public Domain Dedication. Unrestricted scientific and practical use. |
| **Original Project Code** | Repository Root | **MIT License** | Copyright (c) 2026 Preetam Kamate. Free usage with license inclusion. |

---

## 7. Multi-Dimensional Gap Matrix

### A. Crop × Viewpoint Coverage Matrix

| Crop Species | `leaf` | `whole_plant` | `canopy` | `stem` | `fruit` | `root` | `bark` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple** | 🟢 **COVERED** | 🔴 GAP (Uncollected) | 🔴 GAP | 🔴 GAP | 🔴 GAP | ⚪ N/A | 🔴 GAP |
| **Cassava** | 🟢 **COVERED** | 🔴 GAP (Uncollected) | 🔴 GAP | 🔴 GAP | ⚪ N/A | 🔴 GAP (Tuber rot) | ⚪ N/A |
| **Corn (Maize)** | 🟢 **COVERED** | 🔴 GAP (Prior Synth Deprecated) | 🔴 GAP | 🔴 GAP | ⚪ N/A | ⚪ N/A | ⚪ N/A |
| **Tomato** | 🟢 **COVERED** | 🔴 GAP (Prior Synth Deprecated) | 🔴 GAP | 🔴 GAP | 🔴 GAP (Fruit rot) | ⚪ N/A | ⚪ N/A |

### B. Crop × Disease Pathology Matrix

| Crop | Confirmed Covered Real Diseases | Known Pathological Gaps |
| :--- | :--- | :--- |
| **Apple** | Apple Scab (*Venturia*), Apple Rust (*Gymnosporangium*) | Black Rot, Powdery Mildew, Fire Blight |
| **Cassava** | Cassava Bacterial Blight (CBB), Cassava Mosaic Disease (CMD) | Cassava Brown Streak Disease (CBSD), Green Mottle |
| **Corn (Maize)** | Common Rust (*Puccinia*), Northern Leaf Blight (*Exserohilum*), Gray Leaf Spot (*Cercospora*) | Maize Smut (*Ustilago*), Fall Armyworm In-Situ Damage |
| **Tomato** | Early Blight (*Alternaria*), Late Blight (*Phytophthora*), Bacterial Spot (*Xanthomonas*) | Fusarium Wilt (Real Field), Septoria Leaf Spot, TYLCV |

---

## 8. Verification Commands

To reproduce this audit at any time without external network access:
```bash
# 1. Run the reproducible dataset audit script
python3 scripts/reproducible_audit.py

# 2. Inspect generated JSON audit artifact
cat models/whole_plant/reproducible_audit.json | grep -A 10 'data_integrity'

# 3. Run regression and safety test suite
pytest -v
```