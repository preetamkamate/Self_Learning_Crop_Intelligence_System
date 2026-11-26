# Raitha Mitra: Crop Intelligence System 🌾
### Production-Safe, Whole-Plant & Canopy Agricultural Advisory Engine

A production-safe, privacy-preserving local farmer advisory application designed for responsible agricultural decision support. Built with a **Whole-Plant and Canopy/Field-View First** architecture, verified local dataset auditing, zero cross-crop contamination, and strict chemical spray withholding gates.

---

## ⚠️ Model-Release Policy & Safety Envelope

- **Experimental Research Prototype**: The active whole-plant pathology model (`models/whole_plant/model.pt`) achieves a held-out test accuracy of **56.25%** (Macro F1: **0.5308**). It is designated strictly as an **Experimental Research Prototype** and is **never presented as a dependable or certified laboratory diagnosis**.
- **Confidence & Margin Threshold Gates**:
  - Requires a minimum confidence threshold (\(\ge 0.70\)) and a decision margin (\(\ge 0.15\)) between the top two candidate classes.
  - **Chemical Treatment Withholding Policy**: Below these thresholds, synthetic chemical fungicides, bactericides, and insecticides are **strictly withheld**. The application returns a tentative observation with safe cultural sanitation and mandatory referral to the nearest Krishi Vigyan Kendra (KVK).
- **Whole-Plant First & Isolated Leaf Feature**: Whole-plant and field canopy diagnosis is the primary active system. Foliar leaf analysis is retained exclusively as a separate, optional experimental feature and is **never** used as a fallback for whole-plant, canopy, stem, fruit, or root photos.
- **Zero-Guessing on Unverified Routes**: Non-leaf anatomical organs (`stem_branch`, `fruit`, `root`) and unsupported crop species safely abstain with honest bilingual photo-capture guidance.

---

## 📊 Full 6-Dataset Local Audit Summary

Scanned and audited across all 6 local dataset directories in `~/Downloads` (cryptographic SHA-256 deduplicated, synthetic diagrams quarantined):

| Dataset | Scanned Images | Usable Whole-Plant / Canopy | Exclusions & Quarantine Reasons |
| :--- | :---: | :---: | :--- |
| **`plantwild_v2`** | 11,488 | **3,212** | Organ closeups (leaf, fruit, stem, root), duplicates |
| **`PlantDoc-Dataset-master`** | 2,573 | **1,565** | Close-up leaf crops, ambiguous backgrounds |
| **`PlantVillage-Dataset-master`** | 182,215 | 0 | 73,604 detached leaves; 54,306 synthetic black cutouts; 54,305 synthetic grayscale |
| **`plant-pathology-2021-fgvc8`** | 18,635 | 0 | 100% close-up foliar apple leaves (orchard trees have no whole-plant disease labels) |
| **`cassava-leaf-disease-classification`** | 21,398 | 0 | 100% close-up foliar cassava leaves |
| **`IP102`** | 102 | 0 | Missing image directory on disk (only `classes.txt` with 102 classes exists) |
| **Global Total** | **236,309** | **4,777** | Verified real-camera samples compiled in `data/manifests/whole_plant_canopy_manifest.json` |

---

## 📋 Operational Route Status

| Route Category | Target Organ / Viewpoint | Supported Crop Scope | Status | Operational Behavior |
| :--- | :--- | :--- | :---: | :--- |
| **Whole-Plant Field Diagnosis** | `whole_plant` | Corn (Maize), Tomato | 🧪 **Active Primary** | Active whole-plant neural inference; high confidence (\(\ge 0.70\)) & margin (\(\ge 0.15\)) required for chemical guidance. |
| **Canopy Field Overviews** | `canopy` | All crops | 🛑 **Data-Blocked** | Safely withheld; provides field photography distance guidance and KVK referral. |
| **Foliar Leaf Screening** | `leaf` | Apple, Cassava, Corn (Maize), Tomato | 🧪 **Optional Leaf Feature** | Separate optional feature; never a fallback for whole plant, canopy, stem, fruit, or root. |
| **Stem & Branch Cankers** | `stem_branch` | All crops | 🛑 **Data-Blocked** | Safely withheld; advises vascular inspection and state agricultural university clinic referral. |
| **Fruit Rots / Boreholes** | `fruit` | All crops | 🛑 **Data-Blocked** | Safely withheld; provides calyx close-up and borehole inspection guidance. |
| **Root & Tuber Pathology** | `root` | All crops | 🛑 **Data-Blocked** | Safely withheld; advises root gall wash and certified diagnostic clinic testing. |
| **Entomology (Spider Mites)** | `leaf` (underside) | Tomato | 🧪 **Pilot** | Dedicated binary CNN (`models/pest_detector/model.pt`); ICAR-IIHR organic/low-toxicity protocols. |
| **Entomology (Armyworm/Borers)** | `whorl`, `stem` | All crops | 🛑 **Data-Blocked** | IP102 image files missing on disk; provides Integrated Pest Management (IPM) scouting guidelines. |
| **Unsupported Crops** | Any viewpoint | Grape, Potato, Cotton, Banana, Wheat, Rice | 🛑 **Data-Blocked** | Safely returns exact message: *"Valid whole plant detected, but this crop is not trained for diagnosis yet."* |
| **Non-Plant Objects** | Non-agricultural photos | N/A | 🛡️ **Rejected** | Refuses diagnosis with zero chemical prescription (`status=out_of_distribution`). |

---

## 🔒 Privacy & Location Architecture

- **Consent-Gated Geolocation**: Live location uses the browser HTML5 Geolocation API with explicit user opt-in.
- **Zero Raw GPS Persistence**: Latitude and longitude coordinates are used purely in-memory to compute Great-Circle Haversine distances to the nearest Karnataka district and KVK centroid. Coordinates are **never saved to disk, session state, databases, or log files**, and URL query parameters are cleared immediately upon resolution.
- **Offline Weather Abstraction**: Climatological monthly temperature, humidity, wind, and rain probability models operate 100% locally with zero external network requests or API keys.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or Python 3.13
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/preetamkamate/Self_Learning_Crop_Intelligence_System.git
cd Self_Learning_Crop_Intelligence_System

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Commands: Audit, Train, Evaluate & Launch

### 1. Run Reproducible Audit
```bash
python3 scripts/reproducible_audit.py
```

### 2. Create Whole-Plant & Canopy Manifests
```bash
python3 scripts/create_whole_plant_manifests.py
```

### 3. Train Whole-Plant Model
```bash
python3 src/whole_plant/train.py --epochs 8
```

### 4. Evaluate Held-Out Test Split
```bash
python3 src/whole_plant/evaluate.py
```

### 5. Run Full Verification Test Suite (Bounded Timeout, No Cache)
```bash
python3 -m pytest -p no:cacheprovider -o faulthandler_timeout=60 -o faulthandler_exit_on_timeout=true
```

### 6. Launch Streamlit Application Locally
```bash
streamlit run interface/app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Repository Structure

```text
├── interface/
│   └── app.py                     # Streamlit user interface (Home journey, coverage matrix, audit cards)
├── src/
│   ├── common/
│   │   └── startup_checks.py      # Startup artifact integrity checks & graceful degradation
│   ├── location/
│   │   ├── component.py           # Geolocation HTML5 component with instant query-parameter purge
│   │   ├── registry.py            # 20 Karnataka districts & 19 ICAR/SAU KVK centers
│   │   └── service.py             # Haversine distance, zero GPS persistence, offline weather
│   ├── modular_vision/
│   │   ├── crop_id.py             # Crop species normalizer & verified crop gate
│   │   ├── disease_classifier.py  # Foliar & field disease classification engine
│   │   ├── pest_detector.py       # Dedicated binary pest CNN & IPM scouting guidance
│   │   ├── photo_guidance.py      # Bilingual (English & Kannada) photo-capture checklists
│   │   ├── pipeline.py            # ModularCropVisionPipeline
│   │   ├── train_pest_detector.py # Training script for DedicatedPestCNN
│   │   └── viewpoint_router.py    # 6-viewpoint canonical router
│   ├── orchestrator/              # Autonomous intent router, state management, and LLM service
│   ├── rag/                       # ICAR / SAU guideline retriever
│   ├── tabular/                   # Crop recommendation model based on soil NPK
│   └── whole_plant/               # Whole-plant dataset, training, evaluation, and predictor
├── tests/                         # Comprehensive unit & regression test suite (69 tests)
├── docs/
│   ├── MODEL_CARD_WHOLE_PLANT.md  # Model card: Whole-Plant & Canopy Vision Engine
│   ├── MODEL_CARD_DISEASE.md      # Model card: Optional Leaf-Only Pathology Classifier
│   ├── MODEL_CARD_PEST.md         # Model card: Dedicated Pest Detection CNN
│   └── MODEL_RELEASE_POLICY.md    # Strict release policy, validation gates, and chemical withholding
├── data/
│   └── manifests/                 # Deterministic, SHA-256 verified manifests & audit reports
├── scripts/                       # Reproducible audit, manifest generation, and evaluation scripts
├── LICENSE                        # MIT License
└── THIRD_PARTY_NOTICES.md         # Full 6-dataset attribution and academic licenses
```

---

## 📚 Dataset Attribution & Licenses

This project relies on publicly available agricultural research datasets under their respective open terms:
- **`plantwild_v2`**: In-situ real-camera field crop collection (Academic Research Use).
- **`PlantDoc-Dataset-master`**: Singh et al. (2019), CoDS-COMAD (Academic Research Use).
- **`PlantVillage-Dataset-master`**: Hughes & Salathé (2015), Nature Scientific Data (CC BY-SA 4.0).
- **`plant-pathology-2021-fgvc8`**: FGVC8 Workshop, Kaggle Research License.
- **`cassava-leaf-disease-classification`**: Makerere AI Lab / FGVC Research (CC BY 4.0).
- **`IP102`**: Wu et al. (2019), IEEE CVPR (Academic Research Use; local images missing).
- **Institutional Protocols**: Indian Council of Agricultural Research (ICAR) & State Agricultural Universities (UAS Dharwad, UAS Bangalore).
