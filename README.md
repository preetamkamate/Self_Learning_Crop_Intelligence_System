# Self-Learning Crop Intelligence System 🌾
### NHCE 5th-Semester AI & ML Mini-Project

An end-to-end, multimodal, context-aware agricultural advisory platform built strictly with **100% real verifiable datasets, domain-trained checkpoints, and authentic ICAR/IIHR guidelines**.

---

## 🌟 Key Architectural Features

1. **Unified Conversational Interface**:
   - Zero fragmented tabs or disconnected pages.
   - Single continuous chat stream with persistent multi-turn session state.
   - Unified multimodal input area (natural language + plant leaf photo upload).
   - Dual bilingual advisory output (**English & ಕನ್ನಡ - Kannada**).

2. **Autonomous Internal Router**:
   - Dynamically inspects user text and attachments to activate the appropriate AI/ML subsystem without requiring user configuration.

3. **Subsystems & Models**:
   - **Tabular Crop Suitability ML (`src/tabular/`)**:
     - Trained on the real 2,200-row Indian Agriculture Crop Recommendation dataset.
     - Evaluated with stratified 80/20 train/test split.
     - Production Model: `RandomForestClassifier` (Test Accuracy: **99.55%** vs. GradientBoosting: 99.09%).
     - Recommends top-3 ranked crops with confidence percentages and Kannada names.
   - **Plant Disease Vision Engine + Whole-Plant Field Understanding (`src/vision/`)**:
     - **Dual-Model Vision Architecture**:
       - `facebook/dinov2-small` (~85 MB) vision transformer backbone: Extracts whole-plant global structural embeddings (`[CLS]` tokens) and executes an Out-of-Distribution (OOD) cosine similarity check against authentic agricultural references. Non-plant objects (furniture, animals, vehicles, walls) are immediately rejected (`similarity < 0.35`).
       - `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` (~14 MB): Fine-tuned pathology head.
     - **Multi-Scale Whole-Plant Tiling**:
       - Evaluates 5 views per upload: Full resized canopy view + 4 overlapping quadrant crops (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
     - **Spatial Agronomic Reasoning & Uncertainty Gate**:
       - Pinpoints `affected_regions`: Lower leaves & stem base (ground-adjacent splash blights), Upper canopy (viral leaf curl / young shoot stress), or Whole canopy (systemic wilting).
       - Computes Shannon predictive entropy $H(p) = -\sum p_i \log(p_i)$ across aggregated views to flag ambiguous or conflicted symptoms.
   - **Live Weather Intelligence (`src/orchestrator/weather.py`)**:
     - Direct integration with the free, keyless **Open-Meteo API**.
     - Real-time temperature, humidity, precipitation, and 3-day rainfall forecast for Karnataka agricultural zones.
     - Real-time spray safety gate (warns against pesticide drift during high winds or wash-off during rain).
   - **Authoritative RAG Knowledge Base (`src/rag/`)**:
     - Documented according to **ICAR**, **IIHR Bengaluru**, **UAS Bengaluru/Dharwad**, and **Karnataka Raitha Mitra** standards.
     - Covers **Tomato, Paddy (Rice), Ragi (Finger Millet), Maize, and Cotton**.
     - Chunked and indexed using `sentence-transformers/all-MiniLM-L6-v2` and `faiss-cpu` (Cosine Similarity IndexFlatIP).
   - **Responsible Batch Self-Learning (`src/feedback/`)**:
     - Interactions logged to `data/processed/feedback_queue.csv`.
     - Strict evaluation gate: Retrains candidate model ONLY on confirmed agronomist-verified labels and promotes candidate to production ONLY if validation accuracy strictly improves.

---

## 📂 Directory Layout

```
crop_intelligence_system/
├── config.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── Crop_recommendation.csv
│   │   ├── sample_apple_scab.jpg
│   │   ├── sample_tomato_early_blight.jpg
│   │   ├── sample_corn_common_rust.jpg
│   │   └── knowledge_docs/
│   │       ├── tomato_icar_iihr_guidelines.md
│   │       ├── paddy_rice_icar_karnataka_guidelines.md
│   │       ├── ragi_finger_millet_uas_guidelines.md
│   │       ├── maize_icar_iihr_guidelines.md
│   │       └── cotton_icar_uas_guidelines.md
│   └── processed/
│       └── feedback_queue.csv
├── models/
│   ├── tabular/
│   │   ├── crop_model.joblib
│   │   ├── scaler.joblib
│   │   ├── label_encoder.joblib
│   │   └── metrics.json
│   ├── vision/
│   └── rag/
│       ├── faiss_index.bin
│       └── chunks_metadata.json
├── src/
│   ├── __init__.py
│   ├── tabular/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── predict.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── router.py
│   │   ├── state.py
│   │   ├── weather.py
│   │   └── llm_service.py
│   └── feedback/
│       ├── __init__.py
│       ├── feedback_logger.py
│       └── batch_retrain.py
└── interface/
    └── app.py
```

---

## 🚀 Running the System

### 1. Launch the Streamlit Interface
```bash
cd /Users/preetamkamate/.gemini/antigravity/scratch/crop_intelligence_system
streamlit run interface/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Retraining & Pipeline Verification
```bash
# Tabular ML training
python3 src/tabular/train.py

# Vision model & OOD test
python3 src/vision/predict.py

# Rebuild FAISS index
python3 src/rag/indexer.py

# Test RAG semantic search
python3 src/rag/retriever.py

# Batch self-learning evaluation gate
python3 src/feedback/batch_retrain.py
```

---

## 📊 Computed Evaluation Metrics

### Tabular Crop Recommendation (`models/tabular/metrics.json`)
- **Dataset Size**: 2,200 rows across 22 Indian crop classes
- **Split**: 1,760 train samples / 440 test samples (Stratified)
- **RandomForest Accuracy**: **99.55%** (Selected Production Model)
- **GradientBoosting Accuracy**: 99.09%
- **Macro Average F1-Score**: 1.00
- **Weighted Average F1-Score**: 1.00

### Computer Vision & OOD Engine
- **Model**: MobileNetV2 (PlantVillage 38 classes)
- **Apple Scab Test Sample**: 99.79% confidence, Shannon Entropy $H = 0.018$ (Status: `confident`)
- **Tomato Early Blight Test Sample**: 83.94% confidence, Shannon Entropy $H = 0.735$ (Status: `confident`)
- **Synthetic Noise / Non-Crop Test Sample**: 33.47% confidence, Shannon Entropy $H = 2.478$ (Status: `uncertain_or_ood` -> Safe Refusal Triggered)

