"""
Self-Learning Crop Intelligence System
Simple, Autonomous Conversational Agricultural Assistant.
Provides:
- Whole-plant and canopy primary field photo diagnosis (with safe abstention).
- Separate optional leaf-only experimental feature.
- Four quick-prompt inquiry buttons.
- Simple single chat input.
- Offline climatological weather and spray safety.
- Consent-based nearest KVK referral with zero GPS persistence.
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
from src.feedback.batch_retrain import run_batch_retrain_gate
from src.feedback.feedback_logger import (
    get_queue_statistics,
    log_interaction,
    update_feedback,
)
from src.location.component import render_location_widget
from src.location.registry import KARNATAKA_DISTRICTS, KVK_DIRECTORY
from src.location.service import AgroLocationService
from src.modular_vision.photo_guidance import PHOTO_GUIDANCE_CATALOG, get_photo_guidance
from src.orchestrator.llm_service import AdvisorySynthesisService
from src.orchestrator.router import AutonomousRouter
from src.orchestrator.state import FarmerProfile, SessionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Page Configuration: Simple clean title, collapsed sidebar by default
st.set_page_config(
    page_title="🌾 Self-Learning Crop Intelligence System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling for Simple, Clean Agricultural Assistant Look
st.markdown(
    """
    <style>
    .main { background-color: #fafbfa; }
    .stChatInputContainer { padding-bottom: 20px; }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-confident { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
    .badge-ood { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
    .badge-warn { background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
    .badge-module { background-color: #e0f2fe; color: #0369a1; border: 1px solid #38bdf8; }
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
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

if "pending_quick_prompt" not in st.session_state:
    st.session_state.pending_quick_prompt = None

if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_en = (
        "👋 **Welcome to the Self-Learning Crop Intelligence System.**\n\n"
        "I am your local offline agricultural assistant. Ask a question about crop management, soil NPK, or weather, "
        "or upload a plant photo below for whole-plant field diagnosis."
    )
    welcome_kn = (
        "👋 **ಸ್ವಯಂ-ಕಲಿಕೆಯ ಬೆಳೆ ಬುದ್ಧಿಮತ್ತೆ ವ್ಯವಸ್ಥೆಗೆ ಸುಸ್ವಾಗತ.**\n\n"
        "ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಸಲಹೆಗಾರ. ಬೆಳೆ ನಿರ್ವಹಣೆ, ಮಣ್ಣಿನ ಪೋಷಕಾಂಶ, ಹವಾಮಾನದ ಬಗ್ಗೆ ಪ್ರಶ್ನಿಸಿ ಅಥವಾ ಗಿಡದ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
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


# Release-Quality Safety Thresholds (Configurable via settings expander)
DEFAULT_CONF_THRESHOLD = 0.70
DEFAULT_MARGIN_THRESHOLD = 0.15

# ==============================================================================
# COMPACT SIDEBAR: Language Selector & Quick Settings (Clean & Minimal)
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings / ಸೆಟ್ಟಿಂಗ್ಸ್")
    lang_choice = st.radio(
        "Language / ಭಾಷೆ:",
        options=["English", "ಕನ್ನಡ (Kannada)"],
        index=0 if st.session_state.farmer_state.profile.language == "en" else 1,
    )
    active_lang = "kn" if "ಕನ್ನಡ" in lang_choice else "en"
    st.session_state.farmer_state.profile.language = active_lang

    st.markdown("---")
    st.caption("Self-Learning Crop Intelligence System")
    st.caption("Whole-Plant & Field Diagnostic Architecture")


# ==============================================================================
# MAIN PAGE: Clean Agricultural Assistant Layout
# ==============================================================================

# 1. Main-Page Title & Subtitle
st.title("🌾 Self-Learning Crop Intelligence System")
st.markdown("##### Autonomous Agricultural Assistant for Field Diagnosis, Weather & Crop Management")

# 2. Short Friendly Welcome Message
if active_lang == "kn":
    st.markdown(
        "👋 **ನಮಸ್ಕಾರ!** ಕೃಷಿ ನಿರ್ವಹಣೆ, ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ, ಅಥವಾ ಬೆಳೆ ರೋಗಗಳ ಬಗ್ಗೆ ಪ್ರಶ್ನಿಸಿ. "
        "ಸಂಪೂರ್ಣ ಗಿಡದ (Whole-Plant) ಅಥವಾ ಎಲೆಯ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ ನಿಖರ ಮಾರ್ಗದರ್ಶನ ಪಡೆಯಿರಿ."
    )
else:
    st.markdown(
        "👋 **Welcome!** Ask a question about crop management, spraying weather, or soil suitability, "
        "or upload a crop photograph below for whole-plant field diagnosis."
    )

st.markdown("---")

# 3. Four Quick-Prompt Buttons
st.markdown("**Quick Prompts / ತ್ವರಿತ ಪ್ರಶ್ನೆಗಳು:**")
qp1, qp2, qp3, qp4 = st.columns(4)

quick_query = None
if qp1.button("🌱 Recommend Crops", use_container_width=True):
    quick_query = "What crops are most suitable for my soil and district?"
if qp2.button("🌦️ Spraying Weather", use_container_width=True):
    quick_query = "Will it rain tomorrow for spraying pesticides?"
if qp3.button("🍅 Tomato Blight Guide", use_container_width=True):
    quick_query = "What organic spray works for Tomato Early Blight?"
if qp4.button("🌽 Maize Pest Management", use_container_width=True):
    quick_query = "What is the ICAR chemical dosage for Fall Armyworm in maize?"

# 4. Clear Plant-Photo Upload Section
st.markdown("### 📷 Plant Photo Diagnosis")
u_col1, u_col2 = st.columns([1.8, 1.2])

sample_image_bytes = None
sample_image_label = None

with u_col1:
    uploaded_files = st.file_uploader(
        "Upload plant photo(s) (1–3 crop photos across whole plant, canopy, leaf, stem, fruit, root, bark):",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="main_plant_photo_uploader",
    )

with u_col2:
    viewpoint_options = [
        "Whole Plant (Active Primary Diagnosis)",
        "Canopy / Field View (Active)",
        "Leaf (Active - Foliar Pathology)",
        "Stem / Branch (Active)",
        "Fruit / Pod / Ear (Active)",
        "Root / Tuber (Active)",
        "Bark / Trunk (Active)",
        "Auto-detect (Autonomous Multi-Part)",
    ]
    sel_viewpoint = st.selectbox(
        "Photo Type (Whole-Plant default):",
        viewpoint_options,
        index=0,
        help="Whole-plant diagnosis is the active primary route for Corn and Tomato. Multi-part routes cover stem, fruit, root, bark, and canopy across verified crops.",
    )
    sel_crop = st.selectbox(
        "Crop Species (Optional):",
        ["Auto-detect", "Tomato", "Corn (Maize)", "Apple", "Cassava", "Potato", "Grape", "Citrus", "Bell Pepper", "Coconut", "Other"],
        index=0,
    )

# Quick sample benchmark buttons
with st.expander("🧪 Test with Benchmark Photo Samples", expanded=False):
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns(5)
    if s_col1.button("🌱 Field Tomato", use_container_width=True):
        p = config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Analyze this field tomato plant."
    if s_col2.button("🌽 Maize Rust", use_container_width=True):
        p = Path("data/raw/whole_plant/test/Corn_Common_Rust/img_000.jpg")
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Examine this maize plant for rust symptoms."
    if s_col3.button("🍂 Apple Scab", use_container_width=True):
        p = config.RAW_DATA_DIR / "sample_apple_scab.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Examine this apple leaf for scab lesions."
    if s_col4.button("🕷️ Spider Mites", use_container_width=True):
        mite_dir = Path("data/raw/pest_detection/test/Tomato_Spider_Mites")
        mite_files = list(mite_dir.glob("*.jpg")) if mite_dir.exists() else []
        if mite_files:
            sample_image_bytes = open(mite_files[0], "rb").read()
            sample_image_label = "Check leaf for spider mites."
    if s_col5.button("🐱 Non-Plant Cat", use_container_width=True):
        p = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Evaluate this non-plant image."

# Determine active image bytes list (1 to 3 images)
active_images = []
if uploaded_files:
    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    for uf in files[:3]:
        b = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
        active_images.append(b)
elif sample_image_bytes:
    active_images.append(sample_image_bytes)

# Diagnostic Execution
if active_images:
    img_hashes = [compute_image_hash(b) for b in active_images]
    combined_hash = "_".join(img_hashes)

    st.markdown(f"**Uploaded Photographs ({len(active_images)} image{'s' if len(active_images) > 1 else ''})**")
    cols = st.columns(max(len(active_images), 1))
    for idx, (c, b) in enumerate(zip(cols, active_images)):
        with c:
            st.image(b, caption=f"Photo #{idx+1}", use_container_width=True)

    run_btn = st.button("🔍 Diagnose Crop Photo(s)", type="primary", use_container_width=True)

    if run_btn or (combined_hash != st.session_state.last_processed_image_hash):
        st.session_state.last_processed_image_hash = combined_hash

        with st.spinner("Analyzing plant photograph(s) through multi-part vision engine..."):
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

            query_text = sample_image_label or "Analyze this crop image for diseases or pests."

            # Execute vision analysis for each uploaded photo
            all_telemetries = []
            for idx, img_b in enumerate(active_images):
                raw_pil = Image.open(io.BytesIO(img_b))
                pil_image = ImageOps.exif_transpose(raw_pil).convert("RGB")
                h = img_hashes[idx]

                t = router.route_and_execute(
                    user_text=query_text,
                    image_input=pil_image,
                    is_new_image=True,
                    image_hash=h,
                    crop_hint=crop_hint_clean,
                    part_hint=viewpoint_hint_clean,
                    confidence_threshold=DEFAULT_CONF_THRESHOLD,
                    state=st.session_state.farmer_state,
                )
                all_telemetries.append(t)

            # Designate primary telemetry (prefer supported diagnosis)
            primary_telemetry = None
            for t in all_telemetries:
                v = t.get("vision_result", {})
                if v.get("supported") and v.get("status") in ["confident", "probable_diagnosis_needs_confirmation"]:
                    primary_telemetry = t
                    break
            if not primary_telemetry:
                primary_telemetry = all_telemetries[0]

            primary_telemetry["multi_image_telemetries"] = all_telemetries

            advisory = synthesis_service.generate_advisory(
                router_payload=primary_telemetry,
                language=active_lang,
            )
            primary_telemetry["advisory"] = advisory
            st.session_state.last_diagnostic_result = primary_telemetry

            # Add to conversation stream
            st.session_state.messages.append({
                "role": "user",
                "content_en": query_text,
                "content_kn": query_text,
                "image_bytes": active_images[0],
                "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content_en": advisory["en"],
                "content_kn": advisory["kn"],
                "telemetry": primary_telemetry,
                "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
            })

# 5. Clean Result Card Display
if st.session_state.last_diagnostic_result:
    res = st.session_state.last_diagnostic_result
    multi_t = res.get("multi_image_telemetries", [res])

    st.markdown("#### 🌿 Diagnostic Assessment")
    if len(multi_t) > 1:
        st.caption(f"Synthesized analysis across {len(multi_t)} uploaded crop photographs:")

    for idx, t_item in enumerate(multi_t):
        vision = t_item.get("vision_result")
        if not vision:
            continue

        prefix = f"**Photo #{idx+1}**: " if len(multi_t) > 1 else ""
        v_crop = vision.get("crop", "Unknown")
        v_part = vision.get("viewpoint", "whole_plant")
        v_status = vision.get("status", "unsupported")
        is_supp = vision.get("supported", False)
        disease = vision.get("disease")
        conf = vision.get("confidence", 0.0) * 100.0
        margin = vision.get("margin", 0.0) * 100.0
        chem_withheld = vision.get("chemical_treatment_withheld", False) or (conf < DEFAULT_CONF_THRESHOLD * 100.0)
        fallback = vision.get("fallback_reason", "")
        pg = vision.get("photo_guidance")

        # Case 1: Non-Plant Rejection
        if not vision.get("plant_detected", True) or v_status == "out_of_distribution":
            st.error(f"🚫 {prefix}**Non-Plant Image Detected**")
            st.write("The uploaded photograph does not match recognizable agricultural foliage or crop structures. Automatic disease prediction was safely refused.")

        # Case 2: Untrained Route or Unsupported Crop
        elif not is_supp or v_status in ["unsupported", "unsupported_crop", "unsupported_viewpoint", "unsupported_crop_part", "needs_training_data", "pest_guidance_provided"]:
            st.warning(f"🛑 {prefix}**Untrained Route — Honest Guidance Provided**")
            st.markdown(f"• **Detected Crop**: `{v_crop}` | **Viewpoint**: `{v_part.replace('_', ' ').title()}`")
            st.markdown(f"• **Status**: Unsupported for automatic diagnosis.")
            if fallback:
                st.info(fallback)

            # Photo Guidance & KVK Referral
            c_guide, c_kvk = st.columns([1.5, 1.0])
            with c_guide:
                if pg and isinstance(pg, dict):
                    g_title = pg.get("title_kn") if active_lang == "kn" else pg.get("title_en", "Photo Guidance")
                    g_inst = pg.get("instructions_kn") if active_lang == "kn" else pg.get("instructions_en", "")
                    with st.expander(f"📷 {g_title}", expanded=True):
                        st.markdown(g_inst)
                        checklist = pg.get("checklist", [])
                        if checklist:
                            st.markdown("**Photo Checklist:**")
                            for item in checklist:
                                st.markdown(f"- [ ] {item}")

            with c_kvk:
                curr_dist = st.session_state.farmer_state.profile.district or "Dharwad"
                kvk = location_service.get_kvk_by_district(curr_dist)
                if kvk:
                    st.markdown(f"🏛️ **Nearest KVK**: **{kvk.name}**")
                    st.caption(f"📞 `{kvk.contact_phone}`")
                    st.caption(f"📍 {kvk.location}")
                    st.caption("Take a physical sample to the plant clinic for certified verification.")

        # Case 3: Supported Trained Route
        else:
            is_leaf_mode = (v_part == "leaf")
            mode_label = "Leaf Foliar Pathology" if is_leaf_mode else f"Multi-Part {v_part.replace('_', ' ').title()}"

            st.success(f"🌱 {prefix}**Diagnosis**: **{disease or 'Healthy Plant'}**")
            st.markdown(f"• **Crop**: `{v_crop}` | **Viewpoint**: `{v_part.replace('_', ' ').title()}` ({mode_label})")
            st.markdown(f"• **Confidence**: `{conf:.1f}%` (Decision Margin: `{margin:.1f}%`)")

            if chem_withheld:
                st.warning(
                    "⚠️ **Tentative Observation — Chemical Treatments Withheld**: "
                    "Diagnostic confidence or margin is below the release threshold. "
                    "Synthetic chemical sprays are withheld to prevent crop injury. "
                    "Practice cultural sanitation and consult your local KVK."
                )
            else:
                st.info("ℹ️ **Standard Observation**: ICAR package of practices recommendations referenced below.")

st.markdown("---")

# 6. Interactive Chat Stream & Single Input Area
st.markdown("### 💬 Conversation & Advisory")

# Render prior turns
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        c_text = msg["content_kn"] if active_lang == "kn" else msg["content_en"]
        if c_text:
            st.markdown(c_text)
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=240)

# Chat Input Area (also processes quick prompt buttons)
chat_input = st.chat_input("Ask a question about crops, weather, diseases, or soil...")
active_query = quick_query or chat_input

if active_query:
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
            "content_en": adv["en"],
            "content_kn": adv["kn"],
            "telemetry": telemetry,
            "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
        })
        st.rerun()

st.markdown("---")

# ==============================================================================
# TECHNICAL DETAILS & ADVANCED CONTROLS (IN COMPACT EXPANDERS ONLY)
# ==============================================================================

# Expander 1: 🔍 Diagnostic & Model Reasoning
with st.expander("🔍 Diagnostic & Model Reasoning", expanded=False):
    if st.session_state.last_diagnostic_result:
        t_data = st.session_state.last_diagnostic_result
        v_data = t_data.get("vision_result", {})
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**Inference Details**")
            st.write(f"• **Model Route**: `{v_data.get('selected_model', 'models/whole_plant/model.pt')}`")
            st.write(f"• **Checkpoint**: `{v_data.get('checkpoint_path', 'None')}`")
            st.write(f"• **OOD Refused**: `{v_data.get('is_ood', False)}`")
            st.write(f"• **Chemical Spray Withheld**: `{v_data.get('chemical_treatment_withheld', False)}`")
        with col_t2:
            preds = v_data.get("predictions", [])
            if preds:
                st.markdown("**Candidate Probabilities:**")
                for p in preds[:4]:
                    p_name = p.get("label", "Unknown")
                    p_val = float(p.get("prob", p.get("probability", 0.0)))
                    st.progress(p_val, text=f"{p_name}: {p_val*100:.1f}%")
            else:
                st.caption("No candidate predictions generated (safely abstained).")
    else:
        st.caption("Upload a photograph above to inspect model telemetry and candidate probabilities.")

# Expander 2: 📋 Supported Coverage & Safety
with st.expander("📋 Supported Coverage & Safety", expanded=False):
    st.markdown("#### Operational Route Matrix")
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
    st.caption("All 15 multi-part routes + foliar & pest routes are validated with zero cross-crop contamination (0.0%). Untrained crop-part combinations safely abstain with photo-capture guidance and KVK referral.")

# Expander 3: 📍 Location, KVK & Weather Details
with st.expander("📍 Location, KVK & Weather Details", expanded=False):
    loc_c1, loc_c2 = st.columns(2)

    with loc_c1:
        st.markdown("**Location & District Selection**")
        render_location_widget(location_service, st.session_state.farmer_state, lang=active_lang)

        districts = list(KARNATAKA_DISTRICTS.keys())
        curr_d = st.session_state.farmer_state.profile.district or "Dharwad"
        d_idx = districts.index(curr_d) if curr_d in districts else 0
        sel_d = st.selectbox("Selected District / Zone:", districts, index=d_idx)
        st.session_state.farmer_state.profile.district = sel_d

        d_info = KARNATAKA_DISTRICTS.get(sel_d)
        if d_info:
            st.caption(f"Zone {d_info.zone_number}: {d_info.agro_climatic_zone}")

        kvk_info = location_service.get_kvk_by_district(sel_d)
        if kvk_info:
            st.markdown(f"**Nearest KVK**: {kvk_info.name}")
            st.caption(f"• Phone: `{kvk_info.contact_phone}`")
            st.caption(f"• Location: {kvk_info.location}")

    with loc_c2:
        st.markdown("**Offline Weather & Spray Safety**")
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
        st.write(f"🌡️ Temp: `{temp_c}°C` | 💧 Humidity: `{rh}%` | 💨 Wind: `{wind} km/h` | 🌧️ Rain Prob: `{rain_prob}%`")
        st.info(f"Advisory: {adv_msg}")
        st.caption("Weather generated locally from offline Karnataka climatology (Zero API keys, zero external requests).")

# Expander 4: 📚 Model Cards & Dataset Information
with st.expander("📚 Model Cards & Dataset Information", expanded=False):
    st.markdown("#### Full 6-Dataset Audit Summary")
    audit_data = [
        {"Dataset": "plantwild_v2", "Scanned Images": "11,488", "Real Photos": "11,488", "Whole-Plant / Canopy": "3,212", "Exclusions": "Organ closeups, duplicates"},
        {"Dataset": "PlantDoc-Dataset-master", "Scanned Images": "2,573", "Real Photos": "2,573", "Whole-Plant / Canopy": "1,565", "Exclusions": "Foliar closeups, ambiguous background"},
        {"Dataset": "PlantVillage-Dataset-master", "Scanned Images": "182,215", "Real Photos": "54,306", "Whole-Plant / Canopy": "0", "Exclusions": "73.6k detached leaves; 108.6k synthetic cutouts/grayscale"},
        {"Dataset": "plant-pathology-2021-fgvc8", "Scanned Images": "18,635", "Real Photos": "18,635", "Whole-Plant / Canopy": "0", "Exclusions": "100% close-up apple leaves (orchards have no whole-plant disease labels)"},
        {"Dataset": "cassava-leaf-disease-classification", "Scanned Images": "21,398", "Real Photos": "21,398", "Whole-Plant / Canopy": "0", "Exclusions": "100% close-up cassava leaves"},
        {"Dataset": "IP102", "Scanned Images": "102", "Real Photos": "0", "Whole-Plant / Canopy": "0", "Exclusions": "Image directory missing (only classes.txt exists)"},
        {"Dataset": "Global Total", "Scanned Images": "236,309", "Real Photos": "108,400", "Whole-Plant / Canopy": "4,777", "Exclusions": "Quarantined into reproducible audit report"},
    ]
    st.dataframe(audit_data, use_container_width=True)

    mc1, mc2, mc3 = st.tabs(["Whole-Plant Model Card", "Leaf-Only Model Card", "Model Release Policy"])
    with mc1:
        doc1 = ROOT_DIR / "docs" / "MODEL_CARD_WHOLE_PLANT.md"
        if doc1.exists():
            st.markdown(doc1.read_text(encoding="utf-8"))
    with mc2:
        doc2 = ROOT_DIR / "docs" / "MODEL_CARD_DISEASE.md"
        if doc2.exists():
            st.markdown(doc2.read_text(encoding="utf-8"))
    with mc3:
        doc3 = ROOT_DIR / "docs" / "MODEL_RELEASE_POLICY.md"
        if doc3.exists():
            st.markdown(doc3.read_text(encoding="utf-8"))
