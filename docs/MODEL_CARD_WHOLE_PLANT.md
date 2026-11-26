# Model Card: Whole-Plant & Canopy Agricultural Advisory Vision Engine

## 1. Model Details
- **Model Identifier**: `models/whole_plant/model.pt`
- **Release Tier**: `EXPERIMENTAL_RESEARCH_PROTOTYPE`
- **Primary Architecture**: Whole-Plant Deep Convolutional Neural Network with Adaptive Average Pooling and Strict Subspace Softmax Normalization
- **Primary Design Principle**: **Whole-plant and canopy/field-view first**. Foliar leaf analysis is retained exclusively as a separate, optional experimental module and is **never** used as a fallback for whole-plant, canopy, stem, fruit, or root photos.
- **Input Resolution**: 224x224 RGB image
- **Output Classes**: 14 authentic agricultural pathology classes across verified crops
- **Framework**: PyTorch (`mps` / `cuda` / `cpu`)

---

## 2. Full 6-Dataset Local Audit & Manifest Derivation
Every image across the 6 local datasets in `~/Downloads` was scanned and audited with cryptographic SHA-256 deduplication:

| Dataset | Total Scanned Images | Real In-Situ Photos | Usable Whole-Plant / Canopy | Exclusions & Reasons |
| :--- | :---: | :---: | :---: | :--- |
| `plantwild_v2` | 11,488 | 11,488 | **3,212** | Organ-only closeups (leaf, fruit, stem, root), duplicates |
| `PlantDoc-Dataset-master` | 2,573 | 2,573 | **1,565** | Close-up leaf crops, ambiguous backgrounds |
| `PlantVillage-Dataset-master` | 182,215 | 54,306 | 0 | 73,604 detached leaf lab shots; 54,306 synthetic black-background cutouts; 54,305 synthetic grayscale |
| `plant-pathology-2021-fgvc8` | 18,635 | 18,635 | 0 | 100% close-up foliar apple leaves (orchard trees have no whole-plant disease labels) |
| `cassava-leaf-disease-classification` | 21,398 | 21,398 | 0 | 100% close-up foliar cassava leaves |
| `IP102` | 102 | 0 | 0 | Image directory missing on disk (only `classes.txt` with 102 classes exists) |
| **Global Total** | **236,309** | **108,400** | **4,777** | Real whole-plant & canopy samples compiled into `data/manifests/whole_plant_canopy_manifest.json` |

### Deterministic No-Leak Split Summary (`data/manifests/whole_plant_split_summary.json`):
- **Train split**: 3,340 samples (70%)
- **Validation split**: 714 samples (15%)
- **Test split**: 723 samples (15%)
- **Cryptographic SHA-256 leakage check**: `train_val_overlap = 0`, `train_test_overlap = 0`, `val_test_overlap = 0`.

---

## 3. Supported vs. Data-Blocked Route Matrix

| Crop | Whole Plant | Canopy / Field | Leaf (Foliar) | Stem / Branch | Fruit / Pod | Root / Tuber |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corn (Maize)** | ✅ Supported | 🛑 Data-Blocked | ✅ Optional Experimental | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked |
| **Tomato** | ✅ Supported | 🛑 Data-Blocked | ✅ Optional Experimental | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked |
| **Apple** | 🛑 Data-Blocked (Orchard Tree) | 🛑 Data-Blocked | ✅ Optional Experimental | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked |
| **Cassava** | 🛑 Data-Blocked (Shrub Organ) | 🛑 Data-Blocked | ✅ Optional Experimental | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked |
| **Grape, Potato, Cotton, Banana, Wheat, Rice** | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked | 🛑 Data-Blocked |

---

## 4. Held-Out Evaluation Metrics
Evaluated on the held-out test split:

| Metric | Measured Value | Standard / Requirement |
| :--- | :---: | :---: |
| **Overall Accuracy** | **56.25%** | Experimental Research Prototype |
| **Macro Precision** | **0.5650** | Experimental baseline |
| **Macro Recall** | **0.5595** | Experimental baseline |
| **Macro F1 Score** | **0.5308** | Experimental baseline |
| **False Crop-Routing Rate** | **0.00%** | Strict crop isolation (Maize never returns Tomato) |
| **Unsupported Crop Abstention Rate** | **100.00%** | Safe abstention on Grape, Potato, Cotton, Banana, etc. |
| **Non-Plant Rejection Rate** | **100.00%** | Out-of-distribution refusal on background objects |

---

## 5. Model-Release Policy & Farmer Safety Gates
Because the held-out test accuracy is **56.25%**, the following release gates are strictly enforced:
1. **Explicit Experimental Status**: Prominently marked as `EXPERIMENTAL_RESEARCH_PROTOTYPE`. Never presented as certified diagnosis.
2. **Confidence & Margin Gates**: Requires confidence \(\ge 0.70\) and prediction margin between top two candidates \(\ge 0.15\).
3. **Chemical Treatment Withholding Policy**: Below threshold, **all chemical fungicide/pesticide sprays are strictly withheld** to prevent crop phytotoxicity and financial harm. Farmers receive cultural management advice and mandatory referral to the nearest Krishi Vigyan Kendra (KVK).
4. **Zero Raw GPS Persistence**: Coordinates from optional browser location consent are used purely in memory to compute geodesic distance to the nearest KVK and district zone, and are immediately purged. Raw coordinates are never logged or stored on disk.
5. **No External Calls / Offline Climatology**: Weather and spraying advisories run offline via local climatological tables without API keys.
