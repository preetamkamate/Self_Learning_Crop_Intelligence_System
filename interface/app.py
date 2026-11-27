"""
Crop Intelligence System
5th-Semester B.E. Project | Lead AI/ML Engineer: Preetam Kamate
Autonomous Conversational Agricultural Assistant.
Provides:
- Minimalist ChatGPT-style chat interface with message stream.
- 5-Point Structured Bot Response for image diagnoses.
- Whole-plant and multi-part DINOv2 vision across 8 anatomical viewpoints.
- Collapsible sidebar for District Weather, Location settings, Model cards, and Coverage matrix.
"""
import hashlib
import io
import json
import logging
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional
from PIL import Image, ImageOps
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.common.startup_checks import run_startup_artifact_checks
from src.feedback.feedback_logger import log_interaction
from src.location.component import render_location_widget
from src.location.registry import KARNATAKA_DISTRICTS, KVK_DIRECTORY
from src.location.service import AgroLocationService
from src.modular_vision.photo_guidance import get_photo_guidance
from src.orchestrator.llm_service import AdvisorySynthesisService
from src.orchestrator.router import AutonomousRouter
from src.orchestrator.state import FarmerProfile, SessionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Page Configuration: Clean title, collapsed sidebar by default
st.set_page_config(
    page_title="🌾 Crop Intelligence System | Preetam Kamate",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling for Minimalist ChatGPT-like Assistant Look
st.markdown(
    """
    <style>
    .main { background-color: #fafbfa; }
    .stChatInputContainer { padding-bottom: 15px; }
    .project-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .author-pill {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-confident { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
    .badge-ood { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_startup_and_services():
    """Run startup checks and initialize service singletons."""
    report = run_startup_artifact_checks()
    router = AutonomousRouter()
    synthesis = AdvisorySynthesisService()
    loc_service = AgroLocationService()
    return report, router, synthesis, loc_service


startup_report, router, synthesis_service, location_service = load_startup_and_services()

# Session State Initialization
if "farmer_state" not in st.session_state:
    st.session_state.farmer_state = SessionState()

if "last_processed_image_hash" not in st.session_state:
    st.session_state.last_processed_image_hash = None

if "last_diagnostic_result" not in st.session_state:
    st.session_state.last_diagnostic_result = None

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_en = (
        "👋 **Welcome to the Crop Intelligence System.**\n\n"
        "I am your local offline agricultural advisor designed for Karnataka farmers. "
        "You can ask me questions about crop management, spraying weather, and soil fertility, "
        "or attach a plant photograph below for full-plant and multi-part field pathology diagnosis."
    )
    welcome_kn = (
        "👋 **ಬೆಳೆ ಬುದ್ಧಿಮತ್ತೆ ವ್ಯವಸ್ಥೆಗೆ (Crop Intelligence System) ಸುಸ್ವಾಗತ.**\n\n"
        "ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಸಹಾಯಕ. ಬೆಳೆ ನಿರ್ವಹಣೆ, ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆ, ಮತ್ತು ಮಣ್ಣಿನ ಫಲವತ್ತತೆಯ ಬಗ್ಗೆ ಪ್ರಶ್ನಿಸಿ "
        "ಅಥವಾ ಗಿಡದ ಫೋಟೋವನ್ನು ಲಗತ್ತಿಸಿ ನಿಖರ ರೋಗನಿರ್ಣಯ ಪಡೆಯಿರಿ."
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content_en": welcome_en,
        "content_kn": welcome_kn,
        "telemetry": None,
        "turn_id": "welcome_turn",
    })


def compute_image_hash(data: bytes) -> str:
    """Compute MD5 hash for image deduplication."""
    return hashlib.md5(data).hexdigest()


DEFAULT_CONF_THRESHOLD = 0.70
DEFAULT_MARGIN_THRESHOLD = 0.15


# ==============================================================================
# 5-POINT STRUCTURED BOT RESPONSE FORMATTER
# ==============================================================================
def format_5_point_response(telemetry: Dict[str, Any], lang: str = "en") -> str:
    """Format diagnostic telemetry directly into the exact 5 points required."""
    vision = telemetry.get("vision_result") or {}
    crop = vision.get("crop") or vision.get("crop_species") or "Crop"
    viewpoint = vision.get("viewpoint") or vision.get("plant_part") or "whole_plant"
    vp_display = viewpoint.replace("_", " ").title()
    is_supp = vision.get("supported", False)
    status = vision.get("status", "unsupported")
    disease = vision.get("disease") or vision.get("top1_label") or "Unknown Condition"
    disease_kn = vision.get("top1_label_kannada", disease)

    conf = float(vision.get("confidence", vision.get("top1_confidence", 0.0)))
    if conf <= 1.0 and conf > 0.0:
        conf *= 100.0
    margin = float(vision.get("margin", 0.0))
    if margin <= 1.0 and margin > 0.0:
        margin *= 100.0

    is_healthy = vision.get("is_healthy", False) or "healthy" in disease.lower()
    chem_withheld = vision.get("chemical_treatment_withheld", False) or (conf < 70.0) or (margin < 15.0 and margin > 0.0)

    # 1. Non-Plant Rejection
    if not vision.get("plant_detected", True) or status == "out_of_distribution":
        if lang == "kn":
            return (
                "1. **ರೋಗನಿರ್ಣಯ (Diagnosis):** ಕೃಷಿಯೇತರ ವಸ್ತು / ಸಸ್ಯ ಪತ್ತೆಯಾಗಿಲ್ಲ (Non-Plant OOD)\n"
                "2. **ಕಾರಣ (Reason):** ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಫೋಟೋದಲ್ಲಿ ಯಾವುದೇ ಕೃಷಿ ಬೆಳೆ, ಎಲೆ, ಅಥವಾ ಕಾಂಡ ಕಂಡುಬಂದಿಲ್ಲ.\n"
                "3. **ರಾಸಾಯನಿಕ ಪರಿಹಾರ (Chemical Solution):** ಯಾವುದೂ ಇಲ್ಲ (ರೋಗನಿರ್ಣಯವನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ತಿರಸ್ಕರಿಸಲಾಗಿದೆ).\n"
                "4. **ಸಾವಯವ ಪರಿಹಾರ (Organic Solution):** ಯಾವುದೂ ಇಲ್ಲ.\n"
                "5. **ಮುಂದಿನ ಕ್ರಮ (What to do next):** ದಯವಿಟ್ಟು ನಿಜವಾದ ಕೃಷಿ ಬೆಳೆ, ಎಲೆ, ಅಥವಾ ಗಿಡದ ಸ್ಪಷ್ಟ ಫೋಟೋವನ್ನು ತೆಗೆದು ಕಳುಹಿಸಿ."
            )
        return (
            "1. **Diagnosis:** Non-Plant / Out-of-Distribution Image Detected.\n"
            "2. **Reason:** The uploaded photograph does not match recognizable agricultural crop foliage, stems, fruits, or field structures.\n"
            "3. **Chemical Solution:** None (Safely Refused — no chemical prescription for non-crop images).\n"
            "4. **Organic Solution:** None.\n"
            "5. **What to do next:** Please capture and upload a clear, well-focused photograph of an actual agricultural crop or plant foliage."
        )

    # 2. Untrained / Unsupported Route
    if not is_supp or status in ["unsupported", "unsupported_crop", "unsupported_viewpoint", "unsupported_crop_part", "needs_training_data", "pest_guidance_provided"]:
        if lang == "kn":
            return (
                f"1. **ರೋಗನಿರ್ಣಯ (Diagnosis):** ಬೆಳೆ: **{crop}** | ಭಾಗ: **{vp_display}** — ತರಬೇತಿ ಪಡೆಯದ ಕೋನ (Untrained Route)\n"
                f"2. **ಕಾರಣ (Reason):** ಈ ಬೆಳೆ ಮತ್ತು ಸಸ್ಯದ ಭಾಗಕ್ಕೆ ಪರಿಶೀಲಿಸಿದ ನೈಜ ತರಬೇತಿ ಮಾದರಿ ಲಭ್ಯವಿಲ್ಲ. ತಪ್ಪಾದ ಚಿಕಿತ್ಸೆ ತಪ್ಪಿಸಲು ಊಹೆಯನ್ನು ನಿಷೇಧಿಸಲಾಗಿದೆ.\n"
                f"3. **ರಾಸಾಯನಿಕ ಪರಿಹಾರ (Chemical Solution):** 🛑 **ಸುರಕ್ಷಿತವಾಗಿ ತಡೆಹಿಡಿಯಲಾಗಿದೆ**: ಅಧಿಕೃತ ದೃಢೀಕರಣವಿಲ್ಲದೆ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ.\n"
                f"4. **ಸಾವಯವ ಪರಿಹಾರ (Organic Solution):** ಗಿಡಗಳ ನಡುವೆ ಸರಿಯಾದ ಗಾಳಿ-ಬೆಳಕು ಇರುವಂತೆ ನಿರ್ವಹಿಸಿ ಮತ್ತು ತೋಟದ ನೈರ್ಮಲ್ಯ ಕಾಪಾಡಿ.\n"
                f"5. **ಮುಂದಿನ ಕ್ರಮ (What to do next):** ಎಲೆಯ ಹತ್ತಿರದ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆಯಿರಿ ಅಥವಾ ಮಾದರಿಯೊಂದಿಗೆ ಹತ್ತಿರದ ಕೆವಿಕೆ (KVK) ಕೇಂದ್ರಕ್ಕೆ ಭೇಟಿ ನೀಡಿ."
            )
        return (
            f"1. **Diagnosis:** Crop: **{crop}** | Plant Part: **{vp_display}** — Untrained Route (Safely Abstained).\n"
            f"2. **Reason:** Certified real-world training checkpoints are not yet deployed for this specific crop and anatomical part combination. The system abstains to avoid false chemical recommendations.\n"
            f"3. **Chemical Solution:** 🛑 **Safely Withheld**: No synthetic chemicals or pesticides are recommended without certified diagnostic evidence to prevent crop injury.\n"
            f"4. **Organic Solution:** Maintain field hygiene, prune visibly deteriorating foliage, and ensure adequate spacing for aeration.\n"
            f"5. **What to do next:** Capture a sharp close-up photo of an affected leaf in diffuse daylight, and consult your local Krishi Vigyan Kendra (KVK) for physical verification."
        )

    # 3. Healthy Crop Assessment
    if is_healthy:
        if lang == "kn":
            return (
                f"1. **ರೋಗನಿರ್ಣಯ (Diagnosis):** ಬೆಳೆ: **{crop}** | ಭಾಗ: **{vp_display}** — **ಆರೋಗ್ಯಕರ ಗಿಡ (Healthy Plant)** ({conf:.1f}% ಖಚಿತತೆ)\n"
                f"2. **ಕಾರಣ (Reason):** ಸಸ್ಯದ ಅಂಗಾಂಶಗಳಲ್ಲಿ ಯಾವುದೇ ಶಿಲೀಂಧ್ರ, ವೈರಸ್, ಅಥವಾ ಕೀಟಬಾಧೆಯ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದಿಲ್ಲ. ಹಸಿರು ಕ್ಲೋರೊಫಿಲ್ ಸಮೃದ್ಧವಾಗಿದೆ.\n"
                f"3. **ರಾಸಾಯನಿಕ ಪರಿಹಾರ (Chemical Solution):** ಯಾವುದೇ ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆಯ ಅಗತ್ಯವಿಲ್ಲ. ಅನಗತ್ಯ ಕೀಟನಾಶಕ ಬಳಕೆಯನ್ನು ತಪ್ಪಿಸಿ.\n"
                f"4. **ಸಾವಯವ ಪರಿಹಾರ (Organic Solution):** ರೋಗನಿರೋಧಕ ಶಕ್ತಿ ಹೆಚ್ಚಿಸಲು ತಿಂಗಳಿಗೊಮ್ಮೆ ಪಂಚಗವ್ಯ (30 ಮಿಲಿ/ಲೀ) ಅಥವಾ ಜೀವಾಮೃತ ಸಿಂಪಡಿಸಿ.\n"
                f"5. **ಮುಂದಿನ ಕ್ರಮ (What to do next):** ಸರಿಯಾದ ಸಮಯಕ್ಕೆ ನೀರು ಹಾಯಿಸಿ, ಕಳೆ ನಿಯಂತ್ರಣ ಮುಂದುವರಿಸಿ ಮತ್ತು ಸಾಪ್ತಾಹಿಕ ತಪಾಸಣೆ ನಡೆಸಿ."
            )
        return (
            f"1. **Diagnosis:** Crop: **{crop}** | Plant Part: **{vp_display}** — **Healthy Plant** ({conf:.1f}% Confidence)\n"
            f"2. **Reason:** Plant tissues display active photosynthetic chlorophyll integrity, intact cellular structure, and no visible pathogen sporulation or pest feeding lesions.\n"
            f"3. **Chemical Solution:** None required. Do not spray synthetic chemical fungicides or insecticides on healthy foliage.\n"
            f"4. **Organic Solution:** Apply preventive bio-stimulants such as 5% Neem Seed Kernel Extract (NSKE) or Panchagavya (30 ml/L) to maintain plant vigor.\n"
            f"5. **What to do next:** 1. Maintain regular irrigation schedules. 2. Continue weekly scouting under leaves. 3. Ensure balanced NPK fertilization."
        )

    # 4. Supported Disease Diagnosis
    dis_lower = disease.lower()
    if "early blight" in dis_lower:
        reason_en = "Caused by the fungal pathogen *Alternaria solani*, favored by warm temperatures (24–29°C) and prolonged foliar wetness from dew, rainfall splash, or overhead irrigation."
        reason_kn = "*ಆಲ್ಟರ್ನೇರಿಯಾ ಸೊಲಾನಿ* ಎಂಬ ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ. 24–29°C ಉಷ್ಣಾಂಶ ಮತ್ತು ಎಲೆಗಳ ಮೇಲೆ ನಿರಂತರ ತೇವಾಂಶವಿದ್ದಾಗ ಇದು ವೇಗವಾಗಿ ಹರಡುತ್ತದೆ."
        chem_rec_en = "Spray Chlorothalonil 75% WP @ 2 g/L or Mancozeb 75% WP @ 2.5 g/L."
        chem_rec_kn = "ಕ್ಲೋರೋಥಾಲೋನಿಲ್ 75% WP (2 ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP (2.5 ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        org_rec_en = "Spray 5% Neem Seed Kernel Extract (NSKE) or *Trichoderma viride* @ 5 g/L of water."
        org_rec_kn = "5% ಬೇವಿನ ಬೀಜದ ಕಷಾಯ (NSKE) ಅಥವಾ *ಟ್ರೈಕೋಡರ್ಮಾ ವಿರಿಡೆ* (5 ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        next_en = "1. Prune and destroy lower diseased leaves touching soil. 2. Avoid overhead sprinkler irrigation. 3. Monitor spraying weather before chemical application. 4. Consult nearest KVK if blight spreads."
        next_kn = "1. ನೆಲಕ್ಕೆ ತಾಗುತ್ತಿರುವ ಬಾಧಿತ ಎಲೆಗಳನ್ನು ಕಿತ್ತು ನಾಶಪಡಿಸಿ. 2. ಹನಿ ನೀರಾವರಿ ಬಳಸಿ. 3. ಹವಾಮಾನ ಪರಿಶೀಲಿಸಿ ಸಿಂಪಡಿಸಿ. 4. ರೋಗ ಮುಂದುವರಿದರೆ ಕೆವಿಕೆ ಸಂಪರ್ಕಿಸಿ."
    elif "late blight" in dis_lower:
        reason_en = "Caused by the oomycete water mold *Phytophthora infestans*, favored by cool, humid weather (15–20°C) with prolonged cloud cover and high relative humidity (>90%)."
        reason_kn = "*ಫೈಟೊಪ್ಥೊರಾ ಇನ್‌ಫೆಸ್ಟಾನ್ಸ್* ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ. ತಂಪಾದ ವಾತಾವರಣ (15–20°C) ಮತ್ತು ಹೆಚ್ಚಿನ ತೇವಾಂಶವಿದ್ದಾಗ ವೇಗವಾಗಿ ಹರಡುತ್ತದೆ."
        chem_rec_en = "Spray Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ) @ 2 g/L or Cymoxanil + Mancozeb @ 2 g/L."
        chem_rec_kn = "ಮೆಟಾಲಾಕ್ಸಿಲ್ + ಮ್ಯಾಂಕೋಜೆಬ್ (ರಿಡೋಮಿಲ್ ಎಂಝೆಡ್) @ 2 ಗ್ರಾಂ/ಲೀ ನೀರಿಗೆ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        org_rec_en = "Spray Bordeaux mixture 1% or Copper Oxychloride 50% WP @ 2.5 g/L. Apply *Pseudomonas fluorescens* @ 5 g/L."
        org_rec_kn = "1% ಬೋರ್ಡೋ ಮಿಶ್ರಣ ಅಥವಾ ತಾಮ್ರದ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ (2.5 ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        next_en = "1. Rogue out severely infected plants immediately. 2. Ensure good soil drainage. 3. Avoid night-time foliar wetting. 4. Take leaf sample to local KVK clinic."
        next_kn = "1. ತೀವ್ರವಾಗಿ ಸೋಂಕಿತ ಗಿಡಗಳನ್ನು ತಕ್ಷಣ ಕಿತ್ತು ಸುಟ್ಟುಹಾಕಿ. 2. ಮಣ್ಣಿನಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಬಸಿದುಹೋಗಲು ಚರಂಡಿ ಮಾಡಿ. 3. ಸ್ಥಳೀಯ ಕೆವಿಕೆ ಸಂಪರ್ಕಿಸಿ."
    elif "rust" in dis_lower:
        reason_en = "Caused by fungal rust pathogens (*Puccinia sorghi* / *Puccinia polysora*), producing powdery pustules on foliage, favored by moderate temperatures and humid canopy microclimates."
        reason_kn = "*ಪಕ್ಸಿನಿಯಾ ಸೋರ್ಗಿ* ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ. ಎಲೆಗಳ ಮೇಲೆ ಕೆಂಪು-ಕಂದು ಬಣ್ಣದ ರಂಧ್ರದಂತಹ ಗುಳ್ಳೆಗಳು ಉಂಟಾಗುತ್ತವೆ."
        chem_rec_en = "Spray Mancozeb 75% WP @ 2.5 g/L or Azoxystrobin 23% SC @ 1 ml/L at first appearance of pustules."
        chem_rec_kn = "ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP (2.5 ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ಅಜಾಕ್ಸಿಸ್ಟ್ರೋಬಿನ್ 23% SC (1 ಮಿಲಿ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        org_rec_en = "Spray wettable sulfur 80% WP @ 2 g/L or 3% cold-pressed neem oil with mild soap emulsifier."
        org_rec_kn = "ಕರಗುವ ಗಂಧಕ (Wettable Sulfur 80% WP) @ 2 ಗ್ರಾಂ/ಲೀ ಅಥವಾ 3% ಬೇವಿನ ಎಣ್ಣೆ ಸಿಂಪಡಿಸಿ."
        next_en = "1. Avoid excessive nitrogen top-dressing. 2. Inspect upper leaves protecting yield. 3. Practice crop rotation with broadleaf pulses."
        next_kn = "1. ಅತಿಯಾದ ಯೂರಿಯಾ ಗೊಬ್ಬರ ಬಳಕೆಯನ್ನು ನಿಯಂತ್ರಿಸಿ. 2. ದ್ವಿದಳ ಧಾನ್ಯಗಳೊಂದಿಗೆ ಬೆಳೆ ಪರಿವರ್ತನೆ ಮಾಡಿ."
    elif "scab" in dis_lower:
        reason_en = "Caused by *Venturia inaequalis*, surviving winter on fallen leaves and releasing ascospores during spring rains, causing velvety olive-brown lesions on leaves and fruit skin."
        reason_kn = "*ವೆಂಚುರಿಯಾ ಇನಾಕ್ವಾಲಿಸ್* ಶಿಲೀಂಧ್ರದಿಂದ ಉಂಟಾಗುತ್ತದೆ. ಎಲೆ ಮತ್ತು ಹಣ್ಣುಗಳ ಮೇಲೆ ಕಪ್ಪು ಕಲೆಗಳು ಉಂಟಾಗುತ್ತವೆ."
        chem_rec_en = "Spray Difenoconazole 25% EC (Score) @ 0.5 ml/L or Captan 50% WP @ 2.5 g/L during active leaf emergence."
        chem_rec_kn = "ಡೈಫೆನೊಕೊನಜೋಲ್ 25% EC @ 0.5 ಮಿಲಿ/ಲೀ ಅಥವಾ ಕ್ಯಾಪ್ಟಾನ್ 50% WP @ 2.5 ಗ್ರಾಂ/ಲೀ ಸಿಂಪಡಿಸಿ."
        org_rec_en = "Spray wettable sulfur 80% WP @ 3 g/L or copper octanoate (copper soap) @ 2 ml/L before anticipated rainfall."
        org_rec_kn = "ಕರಗುವ ಗಂಧಕ 80% WP @ 3 ಗ್ರಾಂ/ಲೀ ಅಥವಾ ತಾಮ್ರದ ಸೋಪ್ ಸಿಂಪಡಿಸಿ."
        next_en = "1. Rake and destroy fallen leaves to reduce overwintering fungal inocula. 2. Prune canopy branches to facilitate sunlight and rapid drying. 3. Verify fruit development with local horticulture extension."
        next_kn = "1. ಉದುರಿದ ಎಲೆಗಳನ್ನು ಹೆಕ್ಕಿ ಸುಟ್ಟುಹಾಕಿ. 2. ಮರದ ರೆಂಬೆಗಳನ್ನು ಕತ್ತರಿಸಿ ಸೂರ್ಯನ ಬೆಳಕು ಬೀಳುವಂತೆ ಮಾಡಿ."
    elif "spider mite" in dis_lower:
        reason_en = "Infestation by phytophagous mites (*Tetranychus urticae*), puncturing plant cells to feed on sap, favored by hot, dry weather and dusty field conditions."
        reason_kn = "*ಟೆಟ್ರಾನಿಕಸ್ ಉರ್ಟಿಕೆ* ಎಂಬ ಕೆಂಪು ನುಸಿ ಕೀಟಗಳು ಎಲೆಗಳ ರಸ ಹೀರುವುದರಿಂದ ಉಂಟಾಗುತ್ತದೆ. ಬಿಸಿ ಮತ್ತು ಒಣ ಹವೆ ಇದ್ದಾಗ ಇದು ತೀವ್ರವಾಗುತ್ತದೆ."
        chem_rec_en = "Spray Fenazaquin 10% EC @ 1.5 ml/L or Spiromesifen 22.9% SC @ 1 ml/L targeting the leaf undersides."
        chem_rec_kn = "ಫೆನಾಜಾಕ್ವಿನ್ 10% EC (1.5 ಮಿಲಿ/ಲೀ) ಅಥವಾ ಸ್ಪೈರೋಮೆಸಿಫೆನ್ (1 ಮಿಲಿ/ಲೀ) ಎಲೆಯ ಹಿಂಭಾಗಕ್ಕೆ ಸಿಂಪಡಿಸಿ."
        org_rec_en = "Spray 5% Neem Seed Kernel Extract (NSKE) or mineral oil @ 5 ml/L to smother mite populations and dissolve webbing."
        org_rec_kn = "5% ಬೇವಿನ ಬೀಜದ ಕಷಾಯ (NSKE) ಅಥವಾ ಬೇವಿನ ಎಣ್ಣೆ (5 ಮಿಲಿ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        next_en = "1. Hose down foliage with water sprays to dislodge webbing and mites. 2. Maintain soil moisture to alleviate plant drought stress. 3. Avoid broad-spectrum synthetic pyrethroids that kill predatory mites."
        next_kn = "1. ಎಲೆಯ ಹಿಂಭಾಗಕ್ಕೆ ಜೋರಾಗಿ ನೀರು ಸಿಂಪಡಿಸಿ ಜಾಲೆಯನ್ನು ತೊಳೆದುಹಾಕಿ. 2. ಮಣ್ಣಿನಲ್ಲಿ ಸಾಕಷ್ಟು ತೇವಾಂಶ ಕಾಪಾಡಿಕೊಳ್ಳಿ."
    else:
        reason_en = f"Visual foliar symptoms align with {disease}, typically triggered by localized pathogen proliferation, ambient temperature fluctuations, and moisture accumulation."
        reason_kn = f"ರೋಗಲಕ್ಷಣಗಳು {disease_kn} ಗೆ ಹೊಂದಿಕೆಯಾಗುತ್ತವೆ. ತೇವಾಂಶ ಮತ್ತು ವಾತಾವರಣದ ಬದಲಾವಣೆಯಿಂದ ಈ ಸಮಸ್ಯೆ ಉಂಟಾಗುತ್ತದೆ."
        chem_rec_en = "Spray broad-spectrum protective fungicide (Mancozeb 75% WP @ 2.5 g/L or Copper Oxychloride @ 2.5 g/L)."
        chem_rec_kn = "ಶಿಲೀಂಧ್ರನಾಶಕ ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP (2.5 ಗ್ರಾಂ/ಲೀ) ಅಥವಾ ತಾಮ್ರದ ಆಕ್ಸಿಕ್ಲೋರೈಡ್ (2.5 ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        org_rec_en = "Spray 5% Neem Seed Kernel Extract (NSKE) or *Trichoderma* / *Pseudomonas* formulation @ 5 g/L."
        org_rec_kn = "5% ಬೇವಿನ ಬೀಜದ ಕಷಾಯ (NSKE) ಅಥವಾ *ಟ್ರೈಕೋಡರ್ಮಾ* (5 ಗ್ರಾಂ/ಲೀ) ಸಿಂಪಡಿಸಿ."
        next_en = "1. Prune affected plant parts to restrict spread. 2. Maintain clean field borders. 3. Consult nearest KVK for certified physical diagnosis."
        next_kn = "1. ಬಾಧಿತ ಭಾಗಗಳನ್ನು ಕತ್ತರಿಸಿ ತೆಗೆಯಿರಿ. 2. ಹತ್ತಿರದ ಕೆವಿಕೆ ತಜ್ಞರನ್ನು ಸಂಪರ್ಕಿಸಿ."

    if chem_withheld:
        chem_sol_en = (
            f"🛑 **Chemical Spray Safely Withheld**: Confidence (`{conf:.1f}%`) or margin (`{margin:.1f}%`) "
            "is below the 70.0% release threshold. To prevent chemical injury and off-target pesticide damage, "
            "synthetic sprays are withheld. Practice organic sanitation and consult your local KVK."
        )
        chem_sol_kn = (
            f"🛑 **ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆ ಸುರಕ್ಷಿತವಾಗಿ ತಡೆಹಿಡಿಯಲಾಗಿದೆ**: ವಿಶ್ವಾಸಾರ್ಹತೆಯು (`{conf:.1f}%`) 70% ಮಿತಿಗಿಂತ ಕಡಿಮೆಯಿದೆ. "
            "ಬೆಳೆ ಹಾನಿ ತಪ್ಪಿಸಲು ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆಯನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ. ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ."
        )
    else:
        chem_sol_en = f"{chem_rec_en} *(Follow 7-day pre-harvest interval and wear PPE)*."
        chem_sol_kn = f"{chem_rec_kn} *(7 ದಿನಗಳ ಕಾಯುವ ಅವಧಿ ಪಾಲಿಸಿ ಮತ್ತು ಮುಖಗವಸು ಧರಿಸಿ)*."

    if lang == "kn":
        return (
            f"1. **ರೋಗನಿರ್ಣಯ (Diagnosis):** ಬೆಳೆ: **{crop}** | ಭಾಗ: **{vp_display}** — **{disease_kn}** ({conf:.1f}% ಖಚಿತತೆ)\n"
            f"2. **ಕಾರಣ (Reason):** {reason_kn}\n"
            f"3. **ರಾಸಾಯನಿಕ ಪರಿಹಾರ (Chemical Solution):** {chem_sol_kn}\n"
            f"4. **ಸಾವಯವ ಪರಿಹಾರ (Organic Solution):** {org_rec_kn}\n"
            f"5. **ಮುಂದಿನ ಕ್ರಮ (What to do next):** {next_kn}"
        )

    return (
        f"1. **Diagnosis:** Crop: **{crop}** | Plant Part: **{vp_display}** — **{disease}** ({conf:.1f}% Confidence, {margin:.1f}% Margin)\n"
        f"2. **Reason:** {reason_en}\n"
        f"3. **Chemical Solution:** {chem_sol_en}\n"
        f"4. **Organic Solution:** {org_rec_en}\n"
        f"5. **What to do next:** {next_en}"
    )


# ==============================================================================
# SIDEBAR: Secondary Tools, Weather, KVK, Model Cards & Settings
# ==============================================================================
with st.sidebar:
    st.markdown("### 🌾 Crop Intelligence System")
    st.caption("5th-Semester B.E. Project | Lead: **Preetam Kamate**")
    st.markdown("---")

    # 1. Language Selection
    st.markdown("#### ⚙️ Settings / ಸೆಟ್ಟಿಂಗ್ಸ್")
    lang_choice = st.radio(
        "Language / ಭಾಷೆ:",
        options=["English", "ಕನ್ನಡ (Kannada)"],
        index=0 if st.session_state.farmer_state.profile.language == "en" else 1,
    )
    active_lang = "kn" if "ಕನ್ನಡ" in lang_choice else "en"
    st.session_state.farmer_state.profile.language = active_lang

    # 2. Quick Prompts
    st.markdown("#### 💡 Quick Prompts")
    quick_query = None
    qp1, qp2 = st.columns(2)
    if qp1.button("🌱 Crops", use_container_width=True):
        quick_query = "What crops are most suitable for my soil and district?"
    if qp2.button("🌦️ Weather", use_container_width=True):
        quick_query = "Will it rain tomorrow for spraying pesticides?"
    qp3, qp4 = st.columns(2)
    if qp3.button("🍅 Blight Guide", use_container_width=True):
        quick_query = "What organic spray works for Tomato Early Blight?"
    if qp4.button("🌽 Maize Pest", use_container_width=True):
        quick_query = "What is the ICAR chemical dosage for Fall Armyworm in maize?"

    # 3. Benchmark Sample Photos
    st.markdown("#### 🧪 Test Benchmark Photos")
    sample_image_bytes = None
    sample_image_label = None
    sb1, sb2, sb3 = st.columns(3)
    if sb1.button("🌱 Tomato", use_container_width=True):
        p = config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Analyze this field tomato plant."
    if sb2.button("🌽 Maize", use_container_width=True):
        p = Path("data/raw/whole_plant/test/Corn_Common_Rust/img_000.jpg")
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Examine this maize plant for rust symptoms."
    if sb3.button("🍂 Apple", use_container_width=True):
        p = config.RAW_DATA_DIR / "sample_apple_scab.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Examine this apple leaf for scab lesions."

    sb4, sb5 = st.columns(2)
    if sb4.button("🕷️ Spider Mites", use_container_width=True):
        mite_dir = Path("data/raw/pest_detection/test/Tomato_Spider_Mites")
        mite_files = list(mite_dir.glob("*.jpg")) if mite_dir.exists() else []
        if mite_files:
            sample_image_bytes = open(mite_files[0], "rb").read()
            sample_image_label = "Check leaf for spider mites."
    if sb5.button("🐱 Non-Plant", use_container_width=True):
        p = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Evaluate this non-plant image."

    st.markdown("---")

    # 4. District Weather & Spray Safety
    st.markdown("#### 📍 District Weather & Spraying Safety")
    districts = list(KARNATAKA_DISTRICTS.keys())
    curr_d = st.session_state.farmer_state.profile.district or "Dharwad"
    d_idx = districts.index(curr_d) if curr_d in districts else 0
    sel_d = st.selectbox("Selected District / Zone:", districts, index=d_idx)
    st.session_state.farmer_state.profile.district = sel_d

    w_res = location_service.get_weather(sel_d)
    temp_c = w_res.get("temp_c", 26.5)
    rh = w_res.get("humidity", 65)
    wind = w_res.get("wind_speed_kmh", 12.0)
    rain_prob = w_res.get("rain_probability", 20)
    is_safe = w_res.get("spraying_safe", True)
    adv_msg = w_res.get("spraying_advisory", "Favorable conditions for spraying.")
    b_class = "badge-confident" if is_safe else "badge-ood"
    b_text = "🟢 SPRAY SAFE" if is_safe else "🔴 SPRAY UNSAFE"
    st.markdown(f'<span class="metric-badge {b_class}">{b_text}</span>', unsafe_allow_html=True)
    st.caption(f"Temp: {temp_c}°C | Humidity: {rh}% | Wind: {wind} km/h | Rain Prob: {rain_prob}%")
    st.caption(f"Advisory: {adv_msg}")

    # 5. Local KVK Directory
    st.markdown("#### 🏛️ Nearest KVK Directory")
    kvk_info = location_service.get_kvk_by_district(sel_d)
    if kvk_info:
        st.markdown(f"**{kvk_info.name}**")
        st.caption(f"📞 Phone: `{kvk_info.contact_phone}`")
        st.caption(f"📍 Location: {kvk_info.location}")

    st.markdown("---")

    # 6. Model Diagnostics & Telemetry
    with st.expander("🔍 Diagnostic & Model Reasoning", expanded=False):
        if st.session_state.last_diagnostic_result:
            t_data = st.session_state.last_diagnostic_result
            v_data = t_data.get("vision_result", {})
            st.markdown(f"• **Model Route**: `{v_data.get('selected_model', 'models/whole_plant/model.pt')}`")
            st.markdown(f"• **Checkpoint**: `{v_data.get('checkpoint_path', 'None')}`")
            preds = v_data.get("predictions", [])
            if preds:
                st.markdown("**Candidate Probabilities:**")
                for p in preds[:4]:
                    p_name = p.get("label", "Unknown")
                    p_val = float(p.get("prob", p.get("probability", 0.0)))
                    st.progress(p_val, text=f"{p_name}: {p_val*100:.1f}%")
        else:
            st.caption("No diagnostic inference run yet.")

    # 7. Operational Coverage Matrix
    with st.expander("📋 Supported Coverage Matrix", expanded=False):
        matrix_data = [
            {"Organ / Viewpoint": "Whole Plant", "Tomato": "✅ Active Primary", "Corn (Maize)": "✅ Active Primary", "Apple": "—", "Potato": "—", "Grape": "—", "Citrus": "—", "Bell Pepper": "—", "Coconut": "—", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Canopy / Field View", "Tomato": "✅ Active", "Corn (Maize)": "✅ Active", "Apple": "—", "Potato": "—", "Grape": "—", "Citrus": "—", "Bell Pepper": "—", "Coconut": "—", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Leaf (Foliar)", "Tomato": "✅ Active", "Corn (Maize)": "✅ Active", "Apple": "✅ Active", "Potato": "✅ Active", "Grape": "✅ Active", "Citrus": "—", "Bell Pepper": "✅ Active", "Coconut": "—", "Cassava": "✅ Active", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Stem / Branch", "Tomato": "✅ Active (F1 74%)", "Corn (Maize)": "✅ Active (F1 100%)", "Apple": "—", "Potato": "—", "Grape": "✅ Active (F1 100%)", "Citrus": "✅ Active (F1 100%)", "Bell Pepper": "—", "Coconut": "—", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Fruit / Pod / Ear", "Tomato": "✅ Active (F1 100%)", "Corn (Maize)": "—", "Apple": "✅ Active (F1 63%)", "Potato": "—", "Grape": "—", "Citrus": "✅ Active (F1 66%)", "Bell Pepper": "✅ Active (F1 100%)", "Coconut": "—", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Root / Tuber", "Tomato": "—", "Corn (Maize)": "—", "Apple": "—", "Potato": "✅ Active (F1 100%)", "Grape": "—", "Citrus": "—", "Bell Pepper": "—", "Coconut": "—", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Bark / Trunk", "Tomato": "—", "Corn (Maize)": "—", "Apple": "✅ Active (F1 100%)", "Potato": "—", "Grape": "—", "Citrus": "—", "Bell Pepper": "—", "Coconut": "✅ Active (F1 100%)", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
            {"Organ / Viewpoint": "Pests (Spider Mites)", "Tomato": "✅ Pilot (100%)", "Corn (Maize)": "—", "Apple": "—", "Potato": "—", "Grape": "—", "Citrus": "—", "Bell Pepper": "—", "Coconut": "—", "Cassava": "—", "Untrained Crops": "🛑 Abstain & Guide"},
        ]
        st.dataframe(matrix_data, use_container_width=True)
        st.caption("All 15 multi-part routes + foliar & pest routes are validated with zero cross-crop contamination (0.0%).")

    # 8. Model Cards & Audit
    with st.expander("📚 Model Cards & Dataset Info", expanded=False):
        st.caption("Whole-plant and multi-part DINOv2 neural probes trained exclusively on authentic camera images.")


# ==============================================================================
# MAIN PAGE: Minimalist ChatGPT-Style Chat Interface
# ==============================================================================

# Header & Branding with Top-Right Soil & NPK Parameters Menu (⋮)
head_col1, head_col2 = st.columns([0.76, 0.24])
with head_col1:
    st.markdown(
        """
        <div style="padding: 4px 0 10px 0;">
            <h2 style="margin: 0; padding: 0; font-weight: 700; color: #1e293b;">🌾 Crop Intelligence System</h2>
            <p style="margin: 2px 0 0 0; color: #64748b; font-size: 0.95rem;">
                Autonomous Agronomic & Plant Pathology Assistant &bull; DINOv2 Vision (8 Viewpoints)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with head_col2:
    with st.popover("⋮ Soil & NPK Parameters", use_container_width=True):
        st.markdown("#### 🌱 Soil & NPK Configuration")
        curr_profile = st.session_state.farmer_state.profile
        new_n = st.slider("Nitrogen (N) kg/ha", 0, 200, int(curr_profile.soil_n or 90), key="hdr_soil_n")
        new_p = st.slider("Phosphorus (P) kg/ha", 0, 150, int(curr_profile.soil_p or 42), key="hdr_soil_p")
        new_k = st.slider("Potassium (K) kg/ha", 0, 200, int(curr_profile.soil_k or 43), key="hdr_soil_k")
        new_ph = st.slider("Soil pH", 4.0, 9.5, float(getattr(curr_profile, "soil_ph", 6.5)), step=0.1, key="hdr_soil_ph")
        new_land = st.number_input("Land Holding (Acres)", 0.5, 50.0, float(getattr(curr_profile, "land_area_acres", 2.5)), step=0.5, key="hdr_land_acres")
        crop_opts = ["Tomato", "Corn (Maize)", "Apple", "Potato", "Grape", "Citrus", "Bell Pepper", "Cassava", "Coconut", "Cotton", "Sugarcane"]
        sel_crop_idx = crop_opts.index(curr_profile.active_crop) if curr_profile.active_crop in crop_opts else 0
        new_crop = st.selectbox("Active Crop", crop_opts, index=sel_crop_idx, key="hdr_active_crop")

        if st.button("💾 Update Soil Profile", type="primary", use_container_width=True):
            curr_profile.soil_n = float(new_n)
            curr_profile.soil_p = float(new_p)
            curr_profile.soil_k = float(new_k)
            curr_profile.soil_ph = float(new_ph)
            curr_profile.land_area_acres = float(new_land)
            curr_profile.active_crop = new_crop
            st.success("Soil parameters updated!")

# Render Chat Message Stream
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], caption="Uploaded Photograph", width=280)
        c_text = msg.get("content_kn") if active_lang == "kn" else msg.get("content_en")
        st.markdown(c_text or "")


# ==============================================================================
# BOTTOM BAR: Inline Photo Attachment + Chat Input Bar
# ==============================================================================

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
input_c1, input_c2 = st.columns([0.16, 0.84])

with input_c1:
    with st.popover("📎 Photo", use_container_width=True):
        st.markdown("**Attach Plant Photo**")
        uploaded_file = st.file_uploader(
            "Upload Photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="bottom_chat_photo_uploader",
            label_visibility="collapsed",
        )
        sel_crop = st.selectbox(
            "Crop (Optional):",
            ["Auto-detect", "Tomato", "Corn (Maize)", "Apple", "Cassava", "Potato", "Grape", "Citrus", "Bell Pepper", "Coconut", "Other"],
            index=0,
            key="bottom_chat_crop",
        )
        sel_viewpoint = st.selectbox(
            "Plant Part (Optional):",
            [
                "Auto-detect",
                "Whole Plant",
                "Canopy / Field View",
                "Leaf",
                "Stem / Branch",
                "Fruit / Pod / Ear",
                "Root / Tuber",
                "Bark / Trunk",
            ],
            index=0,
            key="bottom_chat_vp",
        )
        if uploaded_file:
            st.caption(f"✓ Ready: {uploaded_file.name}")

with input_c2:
    chat_input = st.chat_input("Ask a question (e.g. 'Tomato yellow leaf, what do I do?') or attach a photo via 📎...")

active_query = quick_query or chat_input

# Check if an image is provided (via upload or sample photo)
active_image_bytes = None
if uploaded_file is not None:
    active_image_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
elif sample_image_bytes:
    active_image_bytes = sample_image_bytes

# Execution: Either image diagnosis or text query
if active_image_bytes and (active_query or sample_image_bytes):
    query_text = active_query or sample_image_label or "Analyze this crop image for diseases or pests."
    img_h = compute_image_hash(active_image_bytes)

    with st.spinner("Analyzing plant photograph with DINOv2 vision & multimodal advisor..."):
        crop_hint_clean = sel_crop.split(" (")[0] if sel_crop != "Auto-detect" else None
        if "Whole Plant" in sel_viewpoint:
            viewpoint_hint_clean = "whole_plant"
        elif "Canopy" in sel_viewpoint:
            viewpoint_hint_clean = "canopy"
        elif "Leaf" in sel_viewpoint:
            viewpoint_hint_clean = "leaf"
        elif "Stem" in sel_viewpoint:
            viewpoint_hint_clean = "stem_branch"
        elif "Fruit" in sel_viewpoint:
            viewpoint_hint_clean = "fruit"
        elif "Root" in sel_viewpoint:
            viewpoint_hint_clean = "root"
        elif "Bark" in sel_viewpoint:
            viewpoint_hint_clean = "bark_trunk"
        else:
            viewpoint_hint_clean = None

        raw_pil = Image.open(io.BytesIO(active_image_bytes))
        pil_image = ImageOps.exif_transpose(raw_pil).convert("RGB")

        telemetry = router.route_and_execute(
            user_text=query_text,
            image_input=pil_image,
            is_new_image=True,
            image_hash=img_h,
            crop_hint=crop_hint_clean,
            part_hint=viewpoint_hint_clean,
            confidence_threshold=DEFAULT_CONF_THRESHOLD,
            state=st.session_state.farmer_state,
        )

        st.session_state.last_diagnostic_result = telemetry

        # Generate Structured 5-Point Advisory
        adv = synthesis_service.generate_advisory(telemetry, language=active_lang)

        # Append User & Assistant Turns to Chat History
        st.session_state.messages.append({
            "role": "user",
            "content_en": query_text,
            "content_kn": query_text,
            "image_bytes": active_image_bytes,
            "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content_en": adv.get("en", ""),
            "content_kn": adv.get("kn", ""),
            "telemetry": telemetry,
            "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
        })
        st.rerun()

elif active_query:
    st.session_state.messages.append({
        "role": "user",
        "content_en": active_query,
        "content_kn": active_query,
        "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
    })
    with st.spinner("Formulating agricultural advisory..."):
        telemetry = router.route_and_execute(
            user_text=active_query,
            image_input=None,
            is_new_image=False,
            state=st.session_state.farmer_state,
        )
        adv = synthesis_service.generate_advisory(telemetry, language=active_lang)
        st.session_state.messages.append({
            "role": "assistant",
            "content_en": adv.get("en", ""),
            "content_kn": adv.get("kn", ""),
            "telemetry": telemetry,
            "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
        })
        st.rerun()
