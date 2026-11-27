# Crop Intelligence System

A zero-checkpoint, cloud-native multimodal agricultural diagnostic and advisory platform powered by **Google Gemini (`gemini-2.5-flash`)**, **ICAR & IIHR agronomic knowledge retrieval**, and a minimalist **Gemini-style Streamlit chat interface**.

## Architecture

```text
root/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── orchestrator/
│   │   └── router.py            # Direct Gemini API routing (no local weight bottlenecks)
│   ├── modular_vision/
│   │   └── pipeline.py          # Multimodal image encoder & 5-point response formatter
│   └── rag/
│       └── kb_retriever.py      # ICAR and IIHR knowledge base mapping for 5 datasets
├── interface/
│   └── app.py                   # Minimalist Gemini-style Streamlit UI
└── tests/
    └── test_system.py           # Pytest suite for zero-error verification
```

## Key Features

- **Direct Multimodal Vision (`gemini-2.5-flash`)**: Analyzes crop foliage, fruits, and stems directly via the Google GenAI SDK without local `.pt` or `.pth` model weight dependencies.
- **5 Grounded Taxonomies**: Maps visual symptoms and field queries across:
  1. PlantVillage (Corn)
  2. FGVC8 (Apple)
  3. Cassava Leaf Disease
  4. ICAR Research
  5. IIHR Advisory
- **Strict 5-Point Diagnostic Format**:
  1. Diagnosis
  2. Reason
  3. Chemical Solution
  4. Organic Solution
  5. What to do next
- **Minimalist Gemini-Style UI**:
  - Clean header with top-right `⋮` menu for Soil NPK (`N`, `P`, `K`) and `pH` settings.
  - Fixed bottom chat input bar with embedded `+` file/image attachment and drag-and-drop support.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # Add your GEMINI_API_KEY
python3 -m pytest -p no:cacheprovider
streamlit run interface/app.py
```
