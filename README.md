# Raitha Mitra: Crop Intelligence System 🌾

A production-safe, privacy-preserving local farmer advisory application designed for responsible agricultural decision support.

---

## ⚠️ Model-Release Policy & Safety Envelope

- **Experimental Research Prototype**: The active foliar pathology model (`models/whole_plant/model.pt`) achieves a held-out test accuracy of **58.33%** (Macro F1: **0.5881**). It is designated strictly as an **Experimental Research Prototype** and is **never presented as a dependable or certified laboratory diagnosis**.
- **Confidence & Margin Threshold Gates**:
  - Requires a minimum confidence threshold (\(\ge 0.70\)) and a decision margin (\(\ge 0.15\)) between the top two candidate classes.
  - **Chemical Treatment Withholding Policy**: Below these thresholds, synthetic chemical fungicides, bactericides, and insecticides are **strictly withheld**. The application returns a tentative observation with safe cultural sanitation and mandatory referral to the nearest Krishi Vigyan Kendra (KVK).
- **Zero-Guessing on Unverified Routes**: Non-leaf anatomical viewpoints and unsupported crop species safely abstain with honest bilingual photo-capture guidance.

---

## 📋 Operational Route Status

| Route Category | Target Organ / Viewpoint | Crop Coverage | Status | Operational Behavior |
| :--- | :--- | :--- | :---: | :--- |
| **Foliar Pathology** | `leaf` | Apple, Cassava, Corn (Maize), Tomato | 🧪 **Experimental** | Active neural classification; high confidence (\(\ge 0.70\)) & margin (\(\ge 0.15\)) required for chemical guidance. |
| **Foliar Pathology** | `leaf` | Grape, Potato, Citrus, Banana, etc. | 🛑 **Data-Blocked** | Safely abstains; returns `disease=None`, `supported=False`; provides photo guidance. |
| **Stem / Branch Cankers** | `stem_branch` | All crops | 🛑 **Data-Blocked** | Safely withheld; advises vascular inspection and state agricultural university clinic referral. |
| **Fruit Rots / Boreholes** | `fruit` | All crops | 🛑 **Data-Blocked** | Safely withheld; provides calyx close-up and borehole inspection guidance. |
| **Canopy Field Overviews** | `canopy` | All crops | 🛑 **Data-Blocked** | Safely withheld; requests step-in foliar close-up photo. |
| **Whole Plant View** | `whole_plant` | All crops | 🛑 **Data-Blocked** | Safely withheld; full plant architecture guidance; asks for single affected leaf photo. |
| **Root / Tuber Rots** | `root` | All crops | 🛑 **Data-Blocked** | Safely withheld; advises root gall wash and certified diagnostic clinic testing. |
| **Entomology (Mites)** | `leaf` (underside) | Tomato | 🧪 **Pilot** | Dedicated binary CNN (`models/pest_detector/model.pt`); ICAR-IIHR organic/low-toxicity protocols. |
| **Entomology (Insects)** | `leaf`, `stem`, `whorl` | All crops (Armyworm, Borers, Aphids) | 🛑 **Data-Blocked** | Zero chemical prescription; provides Integrated Pest Management (IPM) scouting guidelines. |
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

## 🚀 Running the Application & Tests

### Run Full Verification Test Suite
```bash
python3 -m pytest -p no:cacheprovider -o faulthandler_timeout=60 -o faulthandler_exit_on_timeout=true
```

### Launch Streamlit Application Locally
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
│   └── whole_plant/               # Whole-plant inference, dataset manifests, and evaluation
├── tests/                         # Comprehensive unit & regression test suite (69 tests)
├── docs/
│   ├── MODEL_CARD_DISEASE.md      # Model card: Foliar Disease Classifier (58.33% accuracy)
│   ├── MODEL_CARD_PEST.md         # Model card: Dedicated Pest Detection CNN
│   └── MODEL_RELEASE_POLICY.md    # Strict release policy, validation gates, and chemical withholding
├── data/
│   └── manifests/                 # Deterministic, SHA-256 verified real-camera manifests (592 samples)
├── scripts/                       # Reproducible manifest and evaluation scripts
├── LICENSE                        # MIT License
└── THIRD_PARTY_NOTICES.md         # Third-party data and citation acknowledgments
```

---

## 📚 Dataset Attribution & Licenses

This project relies on publicly available agricultural research datasets under their respective open terms:
- **PlantVillage Dataset**: Hughes & Salathé (2015), Nature Scientific Data (CC-BY 4.0 / Open Access).
- **Cassava Leaf Disease Dataset**: Mwebaze et al. (2019), Makerere AI Lab / FGVC Research.
- **Plant Pathology 2021**: FGVC8 Competition Dataset, Research Access.
- **Open-Meteo & Historical Climatology**: Open Database License (ODbL) / CC-BY 4.0.
- **Institutional Protocols**: Indian Council of Agricultural Research (ICAR) & State Agricultural Universities (UAS Dharwad, UAS Bangalore).
