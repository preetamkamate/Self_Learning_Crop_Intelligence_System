# Model Card: Foliar & Whole-Plant Crop Disease Classifier

## 1. Model Details
- **Model Identifier**: `models/whole_plant/model.pt`
- **Release Tier**: `EXPERIMENTAL_RESEARCH_PROTOTYPE`
- **Architecture**: Deep Convolutional Neural Network with Adaptive Average Pooling and Conditional Subspace Softmax Normalization
- **Input Resolution**: 224x224 RGB image
- **Output Classes**: 14 authentic agricultural pathology classes (Apple, Cassava, Corn, Tomato)
- **Framework**: PyTorch (`mps` / `cuda` / `cpu`)

---

## 2. Intended Use & Safety Envelope
- **Intended Use**: Educational and research demonstration of computer vision for foliar disease symptom identification in Indian agriculture.
- **Supported Scope**:
  - Crops: **Apple**, **Cassava**, **Corn (Maize)**, **Tomato**
  - Viewpoint: **Foliar (`leaf`) close-ups** only
- **Out-of-Scope & Prohibited Use**:
  - Never use as a standalone, certified, or legally dependable laboratory diagnosis.
  - Never apply to non-leaf organs (`stem_branch`, `fruit`, `canopy`, `whole_plant`, `root`).
  - Never apply to untrained crop species (Grape, Potato, Citrus, Rice, Wheat, etc.).

---

## 3. Training & Validation Data
- **Training Paradigm**: Supervised learning on derived, verified real-camera manifests (`data/manifests/disease_classification_manifest.json`).
- **Zero-Synthetic Guarantee**: All deprecated synthetic diagrams have been permanently quarantined.
- **Deterministic No-Leak Split**: SHA-256 fingerprint verification ensuring 0 cross-split duplicates across:
  - **Train**: 350 samples
  - **Val**: 84 samples
  - **Test (Held-Out)**: 84 samples (balanced 6 samples per class)
- **Primary Datasets**:
  1. `PlantVillage-Dataset-master` (Mohanty et al., 2016)
  2. `cassava-leaf-disease-classification` (Kaggle/FGVC)
  3. `plant-pathology-2021-fgvc8` (FGVC8)

---

## 4. Held-Out Evaluation Metrics
Evaluated strictly on the held-out test split (84 real-camera images):

| Metric | Measured Value | Standard / Target |
| :--- | :---: | :---: |
| **Overall Accuracy** | **58.33%** | Experimental Tier (< 70%) |
| **Macro Precision** | **0.6673** | - |
| **Macro Recall** | **0.6107** | - |
| **Macro F1 Score** | **0.5881** | Baseline |
| **False Crop-Routing Rate** | **0.0%** | Zero cross-crop leakage |
| **Unsupported-Route Abstention** | **100.0%** | Safe abstention |
| **Non-Plant Rejection Rate** | **100.0%** | Out-of-distribution refusal |

### Per-Dataset Performance Breakdown:
- **`PlantVillage-Dataset-master`**: 48 test samples, Accuracy: 66.67%, **Macro F1: 0.6428**
- **`cassava-leaf-disease-classification`**: 18 test samples, Accuracy: 55.56%, **Macro F1: 0.5556**
- **`plant-pathology-2021-fgvc8`**: 18 test samples, Accuracy: 44.44%, **Macro F1: 0.4750**

### Per-Crop Performance Breakdown:
- **Corn (Maize)** (4 classes, 24 samples): Precision: 0.8125, Recall: 0.7417, **F1: 0.7204**
- **Tomato** (4 classes, 24 samples): Precision: 0.6375, Recall: 0.6167, **F1: 0.5983**
- **Cassava** (3 classes, 18 samples): Precision: 0.6806, Recall: 0.5833, **F1: 0.5556**
- **Apple** (3 classes, 18 samples): Precision: 0.5000, Recall: 0.5000, **F1: 0.4750**

---

## 5. Model-Release Policy & Validation Gates
Because the held-out test accuracy is **58.33%**, the system enforces strict release gates:
1. **Explicit Experimental Status**: Always labeled as `EXPERIMENTAL_RESEARCH_PROTOTYPE`.
2. **High Confidence Threshold Gate**: Requires probability \(\ge 0.70\) and margin between top-1 and top-2 candidates \(\ge 0.15\).
3. **Chemical Treatment Withholding Policy**:
   - If confidence \(< 0.70\) or margin \(< 0.15\): **Chemical fungicide and bactericide sprays are strictly withheld**.
   - The user is provided solely with safe cultural sanitation, organic bioprotectants (e.g. cold-pressed neem oil), and mandatory referral to the nearest Krishi Vigyan Kendra (KVK).
4. **Data-Blocked Routes**: Non-leaf viewpoints (`stem_branch`, `fruit`, `canopy`, `whole_plant`, `root`) and unverified crops safely return `disease=None` with honest photo guidance.
