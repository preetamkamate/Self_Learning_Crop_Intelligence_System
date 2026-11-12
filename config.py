"""
Configuration settings for the Self-Learning Crop Intelligence System.
"""
import os
from pathlib import Path
import torch

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KNOWLEDGE_DOCS_DIR = RAW_DATA_DIR / "knowledge_docs"
MODELS_DIR = BASE_DIR / "models"
TABULAR_MODEL_DIR = MODELS_DIR / "tabular"
VISION_MODEL_DIR = MODELS_DIR / "vision"
RAG_MODEL_DIR = MODELS_DIR / "rag"

# Ensure runtime directories exist
for p in [RAW_DATA_DIR, PROCESSED_DATA_DIR, KNOWLEDGE_DOCS_DIR,
          TABULAR_MODEL_DIR, VISION_MODEL_DIR, RAG_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Datasets
CROP_DATASET_URL = (
    "https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture"
    "/main/Crop%20Recommendation%20dataset.csv"
)
CROP_DATA_PATH = RAW_DATA_DIR / "Crop_recommendation.csv"
FEEDBACK_QUEUE_PATH = PROCESSED_DATA_DIR / "feedback_queue.csv"

# Hardware / Acceleration: macOS M1 / Apple Silicon MPS or CPU fallback
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Vision Model Configuration
# Whole-Plant DINOv2 Backbone + Fine-Tuned Plant Pathology Classifier
DINOV2_MODEL_ID = "facebook/dinov2-small"
DINOV2_PLANT_OOD_THRESHOLD = 0.35  # Cosine similarity below this flags non-plant/unrelated objects
VISION_MODEL_ID = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
OOD_ENTROPY_THRESHOLD = 1.8
MIN_CONFIDENCE_THRESHOLD = 0.60
VISION_TOP_K = 3

# Tabular ML Model Configuration
TABULAR_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TABULAR_TARGET = "label"
TABULAR_TEST_SIZE = 0.2
TABULAR_RANDOM_STATE = 42
MODEL_PATH = TABULAR_MODEL_DIR / "crop_model.joblib"
SCALER_PATH = TABULAR_MODEL_DIR / "scaler.joblib"
ENCODER_PATH = TABULAR_MODEL_DIR / "label_encoder.joblib"
METRICS_PATH = TABULAR_MODEL_DIR / "metrics.json"

# RAG Configuration
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = RAG_MODEL_DIR / "faiss_index.bin"
METADATA_PATH = RAG_MODEL_DIR / "chunks_metadata.json"
RAG_TOP_K = 3
CHUNK_SIZE_WORDS = 280
CHUNK_OVERLAP_WORDS = 50

# Weather API Configuration (Open-Meteo, free, no API key needed)
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Pre-configured Karnataka / South Indian Agricultural Districts coordinates
DISTRICT_COORDINATES = {
    "Bengaluru (Urban/Rural)": {"lat": 12.9716, "lon": 77.5946},
    "Mandya (Cauvery Basin)": {"lat": 12.5230, "lon": 76.8969},
    "Mysuru (Southern Dry Zone)": {"lat": 12.2958, "lon": 76.6394},
    "Dharwad (Northern Transition)": {"lat": 15.4589, "lon": 75.0078},
    "Belagavi (Northern Dry Zone)": {"lat": 15.8497, "lon": 74.4977},
    "Shivamogga (Central Zone)": {"lat": 13.9299, "lon": 75.5681},
    "Hassan (Hilly/Southern Zone)": {"lat": 13.0072, "lon": 76.0963},
    "Raichur (North Eastern Dry)": {"lat": 16.2120, "lon": 77.3439},
    "Tumakuru (Eastern Dry Zone)": {"lat": 13.3422, "lon": 77.1017},
    "Vijayapura (Northern Dry)": {"lat": 16.8302, "lon": 75.7100},
    "Kalaburagi (North Eastern Dry)": {"lat": 17.3297, "lon": 76.8343},
    "Udupi (Coastal Zone)": {"lat": 13.3409, "lon": 74.7421},
    "Ballari (Central Dry Zone)": {"lat": 15.1394, "lon": 76.9214},
    "Chikkamagaluru (Coffee Zone)": {"lat": 13.3161, "lon": 75.7720},
}

# Supported Languages
LANGUAGES = {
    "en": "English",
    "kn": "ಕನ್ನಡ (Kannada)"
}

# CHANGED: Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# CHANGED: Gemini LLM Configuration via Google GenAI SDK
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

