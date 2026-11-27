# 🌾 Self-Learning Crop Intelligence System

An intelligent, multimodal agricultural decision support platform engineered to provide accessible, real-time crop disease diagnosis, pest scouting, and agronomic advisory. Powered by self-supervised vision transformers, retrieval-augmented intelligence, and interactive web technologies, the system empowers farmers with actionable, science-backed agricultural insights.

---

## 🌟 Key Highlights & Capabilities

- **Intelligent Visual Diagnosis**: Identifies crop diseases and foliar symptoms across diverse plant structures (whole plant, canopy, leaf, stem, fruit, and root) using foundation vision representations.
- **Multimodal Advisory Engine**: Combines visual features with conversational intelligence to generate structured, 5-point farmer advisories:
  1. **Diagnosis**: Precise identification of the condition or pathogen.
  2. **Reason**: Underpinning agronomic and environmental causes.
  3. **Chemical Solution**: Targeted synthetic treatments with appropriate application guidelines.
  4. **Organic Solution**: Eco-friendly, biological, and cultural management strategies.
  5. **What to Do Next**: Actionable monitoring steps, field hygiene, and preventive care.
- **Precision Soil & NPK Recommendations**: Analyzes soil nutrients (Nitrogen, Phosphorus, Potassium) and soil pH to suggest optimal crops and fertilization schedules.
- **Local Weather & Spray Window Guidance**: Computes real-time spray safety indicators based on temperature, relative humidity, wind speed, and precipitation probability.
- **Privacy-Preserving Geolocation**: Derives proximity to regional agricultural extension centers (Krishi Vigyan Kendras) in-memory without storing raw GPS coordinates.

---

## 🛠️ Tools & Technologies Used

| Technology / Tool | Category | Role in the System |
| :--- | :--- | :--- |
| **Python 3.11+** | Core Language | Powers the entire backend architecture, data pipelines, and orchestration services. |
| **Meta DINOv2 (`dinov2_vits14`)** | Foundation Vision Transformer | Extracts self-supervised visual embeddings to identify plant parts, crop types, and pathological symptoms without overfitting. |
| **PyTorch** | Deep Learning Framework | Handles neural probe training, tensor transformations, feature projection, and model inference. |
| **Streamlit** | Web Application Framework | Delivers an intuitive, mobile-friendly chat interface with inline photo uploads and real-time parameter controls. |
| **Google Gemini API** | Multimodal LLM | Orchestrates context-aware agronomic synthesis, translating visual predictions and soil data into clear, empathetic farmer advice. |
| **FAISS (Facebook AI Similarity Search)** | Vector Search Engine | Performs dense semantic retrieval across institutional agricultural extension documents and pest management guides. |
| **Scikit-learn** | Machine Learning | Implements soil-to-crop tabular classification algorithms and model evaluation metrics. |
| **NumPy & Pandas** | Data Processing | Manages tabular manifests, soil nutrient calculations, and vector operations. |
| **Pillow (PIL)** | Image Processing | Manages image loading, normalization, aspect-ratio preservation, and optical entropy analysis. |
| **Open-Meteo & Haversine Distance** | Weather & Geolocation | Provides local meteorological insights and computes geodesic distances to agricultural clinics. |
| **Pytest** | Quality Assurance | Comprehensive unit and integration test suite ensuring safety, correctness, and reliable routing. |

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit User Interface                 │
│         (Chat History, Inline Upload, Soil & NPK Menu)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     Autonomous Router                       │
│    (Intent Classifier, Viewpoint Routing & Gating Logic)    │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
               ▼                               ▼
┌──────────────────────────────┐ ┌────────────────────────────┐
│      Modular Vision Engine   │ │     Agronomic Services     │
│  - DINOv2 Feature Extractor  │ │  - Soil NPK Recommender    │
│  - Plant Part Classifier     │ │  - Offline Weather Engine  │
│  - Calibrated Disease Probes │ │  - District / KVK Registry │
└──────────────┬───────────────┘ └─────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Synthesis Service                    │
│    (Multimodal Gemini + FAISS Knowledge Retrieval)          │
│    -> Formats Structured 5-Point Farmer Advisory            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/preetamkamate/Self_Learning_Crop_Intelligence_System.git
cd Self_Learning_Crop_Intelligence_System
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Run the Test Suite
```bash
python3 -m pytest -p no:cacheprovider -o faulthandler_timeout=60 -o faulthandler_exit_on_timeout=true
```

### 6. Launch the Application
```bash
streamlit run interface/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to access the assistant.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
