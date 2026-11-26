"""
Self-Learning Crop Intelligence System | Raitha Mitra
Production-Safe Local Farmer Advisory Application.
Enforces Strict Model-Release Policy:
- Active disease model has 58.33% held-out test accuracy; strictly labeled EXPERIMENTAL.
- High confidence (>=0.70) and margin (>=0.15) required for chemical guidance.
- Below threshold, chemical treatments are withheld, never presented as dependable diagnosis.
- Data-Blocked routes: stem/branch, fruit, canopy, whole plant, root, unsupported crops, and unverified pests.
- Consent-based live location with zero raw GPS persistence (lat/lon purged from memory).
- 100% keyless offline climatological weather engine.
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

# Page Configuration
st.set_page_config(
    page_title="Crop Intelligence System | Raitha Mitra",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main { background-color: #f8faf8; }
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
    .banner-experimental {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 16px;
        color: #78350f;
    }
    .card-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_startup_and_services():
    """Run startup checks and initialize singletons."""
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
        "👋 **Namaskara! Welcome to the Crop Intelligence System (Raitha Mitra).**\n\n"
        "I am your local offline agricultural advisor. I can help you with:\n"
        "- 📷 **Field Photo Analysis**: Experimental foliar disease screening with safe abstention.\n"
        "- 📍 **Location & KVK Matching**: Nearest extension center and district agro-climate.\n"
        "- 🌦️ **Offline Spraying Weather**: Keyless monthly weather model and spraying safety.\n"
        "- 🌾 **Crop Recommendations**: Fertilizer and crop suitability based on soil NPK."
    )
    welcome_kn = (
        "👋 **ನಮಸ್ಕಾರ! ಬೆಳೆ ಬುದ್ಧಿಮತ್ತೆ ವ್ಯವಸ್ಥೆಗೆ (ರೈತ ಮಿತ್ರ) ಸುಸ್ವಾಗತ.**\n\n"
        "ನಾನು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಕೃಷಿ ಸಲಹೆಗಾರ. ನಾನು ನಿಮಗೆ ಈ ಕೆಳಗಿನವುಗಳಲ್ಲಿ ಸಹಾಯ ಮಾಡುತ್ತೇನೆ:\n"
        "- 📷 **ಗಿಡದ ಫೋಟೋ ವಿಶ್ಲೇಷಣೆ**: ಪ್ರಾಯೋಗಿಕ ರೋಗ ತಪಾಸಣೆ ಮತ್ತು ಸುರಕ್ಷಿತ ಮಾರ್ಗದರ್ಶನ.\n"
        "- 📍 **ಸ್ಥಳ ಮತ್ತು ಕೆವಿಕೆ ಮಾಹಿತಿ**: ಸಮೀಪದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರದ ವಿಳಾಸ ಮತ್ತು ಸಂಪರ್ಕ.\n"
        "- 🌦️ **ಸ್ಥಳೀಯ ಹವಾಮಾನ**: ಮಳೆ ಮುನ್ಸೂಚನೆ ಮತ್ತು ಕೀಟನಾಶಕ ಸಿಂಪಡಣಾ ಸುರಕ್ಷತೆ.\n"
        "- 🌾 **ಬೆಳೆ ಶಿಫಾರಸು**: ಮಣ್ಣಿನ ಪೋಷಕಾಂಶಗಳಿಗೆ ಅನುಗುಣವಾಗಿ ಬೆಳೆ ಆಯ್ಕೆ."
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


# ==============================================================================
# SIDEBAR: Farmer Context & Validation Gates Configuration
# ==============================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/eco/materialicons/48dp/2x/baseline_eco_black_48dp.png",
        width=44,
    )
    st.title("🌾 Raitha Mitra")
    st.caption("Local Production-Safe Agricultural Advisory")

    # Language Selector
    lang_choice = st.radio(
        "Language / ಭಾಷೆ:",
        options=["English", "ಕನ್ನಡ (Kannada)"],
        index=0 if st.session_state.farmer_state.profile.language == "en" else 1,
    )
    active_lang = "kn" if "ಕನ್ನಡ" in lang_choice else "en"
    st.session_state.farmer_state.profile.language = active_lang

    st.markdown("---")
    st.subheader("⚙️ Validation Gates & Policy")
    st.caption("Strict safety thresholds for experimental model tier (58.33% Accuracy)")

    conf_threshold = st.slider(
        "Confidence Threshold (Min):",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Diagnoses below this probability withhold chemical prescriptions completely.",
    )

    margin_threshold = st.slider(
        "Margin Threshold (Top1 - Top2):",
        min_value=0.05,
        max_value=0.40,
        value=0.15,
        step=0.05,
        help="If the difference between top two candidate diagnoses is below this margin, symptoms are treated as ambiguous and chemical guidance is withheld.",
    )

    st.markdown("---")
    st.subheader("📍 Location & Farm Profile")

    # Location Widget (Consent-based, zero GPS persistence)
    render_location_widget(location_service, st.session_state.farmer_state, lang=active_lang)

    districts = list(KARNATAKA_DISTRICTS.keys())
    current_dist = st.session_state.farmer_state.profile.district
    matched_idx = 0
    for i, d in enumerate(districts):
        if current_dist.lower() in d.lower() or d.lower() in current_dist.lower():
            matched_idx = i
            break

    selected_district = st.selectbox(
        "Active District / Zone:",
        districts,
        index=matched_idx,
        help="Resolved from nearest coordinates upon consent or selected manually.",
    )
    st.session_state.farmer_state.profile.district = selected_district

    farmer_name = st.text_input(
        "Farmer Name:",
        value=st.session_state.farmer_state.profile.name,
    )
    st.session_state.farmer_state.profile.name = farmer_name

    land_acres = st.number_input(
        "Land Holding (Acres):",
        min_value=0.25,
        max_value=100.0,
        value=float(st.session_state.farmer_state.profile.land_area_acres),
        step=0.5,
    )
    st.session_state.farmer_state.profile.land_area_acres = land_acres

    # Soil NPK
    with st.expander("🧪 Soil Health Card (NPK & pH)", expanded=False):
        c_n, c_p = st.columns(2)
        with c_n:
            n_val = st.number_input("N (kg/ha):", value=float(st.session_state.farmer_state.profile.soil_n), step=5.0)
        with c_p:
            p_val = st.number_input("P (kg/ha):", value=float(st.session_state.farmer_state.profile.soil_p), step=5.0)
        c_k, c_ph = st.columns(2)
        with c_k:
            k_val = st.number_input("K (kg/ha):", value=float(st.session_state.farmer_state.profile.soil_k), step=5.0)
        with c_ph:
            ph_val = st.number_input("pH:", value=float(st.session_state.farmer_state.profile.soil_ph), step=0.1, min_value=3.5, max_value=9.5)

        st.session_state.farmer_state.profile.soil_n = n_val
        st.session_state.farmer_state.profile.soil_p = p_val
        st.session_state.farmer_state.profile.soil_k = k_val
        st.session_state.farmer_state.profile.soil_ph = ph_val

    st.markdown("---")
    st.subheader("🔄 Responsible Learning Gate")
    q_stats = get_queue_statistics()
    st.caption(f"Interactions logged: {q_stats.get('total_interactions', 0)} | Verified: {q_stats.get('verified_samples', 0)}")
    if st.button("Evaluate Retraining Gate", use_container_width=True):
        with st.spinner("Checking candidate models against production accuracy..."):
            gate_res = run_batch_retrain_gate(min_verified_samples=2)
            st.info(f"Gate Status: {gate_res.get('status')} ({gate_res.get('reason', '')})")


# ==============================================================================
# MAIN VIEW: 3 Dedicated Application Tabs
# ==============================================================================
tab_home, tab_coverage, tab_audit = st.tabs([
    "🏡 Farmer Advisory Journey",
    "📋 Supported Coverage & Data-Required",
    "🔍 System Audit & Model Cards",
])

# ------------------------------------------------------------------------------
# TAB 1: Farmer Advisory Journey (Home Flow)
# ------------------------------------------------------------------------------
with tab_home:
    # Mandatory Model-Release Policy Banner
    st.markdown(
        """
        <div class="banner-experimental">
            <strong>⚠️ EXPERIMENTAL RESEARCH PREVIEW (Held-Out Accuracy: 58.33%)</strong><br/>
            This application uses local neural network models for agricultural research demonstration.
            <strong>It is NEVER a dependable or certified laboratory diagnosis.</strong>
            Below strict confidence (70%) and margin (15%) thresholds, chemical treatments are completely withheld.
            Always physically inspect symptoms with your local Krishi Vigyan Kendra (KVK) before applying chemical sprays.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # STEP 1: Location, KVK & Weather Intelligence Card
    st.markdown("### 1. 📍 Agro-Climatic & Local Extension Context")
    loc_col1, loc_col2 = st.columns([1.2, 1.8])

    with loc_col1:
        dist_info = KARNATAKA_DISTRICTS.get(selected_district)
        zone_num = dist_info.zone_number if dist_info else 3
        zone_name = dist_info.agro_climatic_zone if dist_info else "Northern Dry Zone"
        st.markdown(f"**District**: `{selected_district}`")
        st.caption(f"Agro-Climatic Zone {zone_num}: {zone_name}")

        # Nearest KVK Card
        kvk = location_service.get_kvk_by_district(selected_district)
        if kvk:
            st.markdown(f"🏛️ **Nearest KVK**: **{kvk.name}**")
            st.caption(f"• **Institution**: {kvk.nodal_institution}")
            st.caption(f"• **Contact**: `{kvk.contact_phone}`")
            st.caption(f"• **Location**: {kvk.location}")
            st.caption(f"• **Services**: {', '.join(kvk.services_offered[:3])}")

    with loc_col2:
        # Offline Weather & Spray-Safety Card
        weather_res = location_service.get_weather(selected_district)
        temp_c = weather_res.get("temperature_c")
        rh = weather_res.get("humidity_pct")
        wind = weather_res.get("wind_speed_kmh")
        rain_prob = weather_res.get("today_rain_prob_pct", 0)
        is_safe = weather_res.get("spray_safe", True)
        adv_text = weather_res.get("spray_advisory_kn") if active_lang == "kn" else weather_res.get("spray_advisory_en")

        badge_class = "badge-confident" if is_safe else "badge-ood"
        badge_label = "🟢 SPRAY SAFE" if is_safe else "🔴 SPRAY UNSAFE (RAIN/WIND RISK)"

        st.markdown(f"**Offline Weather Conditions ({selected_district})**:")
        st.markdown(
            f'<span class="metric-badge {badge_class}">{badge_label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"🌡️ **Temp**: `{temp_c}°C` | 💧 **Humidity**: `{rh}%` | 💨 **Wind**: `{wind} km/h` | 🌧️ **Rain Probability**: `{rain_prob}%`")
        st.info(f"🚜 **Spraying Advisory**: {adv_text}")
        st.caption("ℹ️ **Offline Weather Disclaimer**: Generated from local historical Karnataka agro-climatic climatology. Zero external API calls or secret keys.")

    st.markdown("---")

    # STEP 2: Crop & Viewpoint Selection
    st.markdown("### 2. 🌿 Crop & Viewpoint Selection (Route Gating)")
    g_col1, g_col2, g_col3 = st.columns([1.2, 1.2, 1.6])

    with g_col1:
        supported_crops_list = ["Auto-detect", "Tomato", "Corn (Maize)", "Apple", "Cassava", "Grape (Unsupported)", "Potato (Unsupported)", "Citrus (Unsupported)", "Other"]
        sel_crop = st.selectbox(
            "Target Crop Species:",
            supported_crops_list,
            index=0,
            help="Apple, Cassava, Corn, and Tomato have foliar training data. Other crops safely abstain.",
        )

    with g_col2:
        viewpoint_options = [
            "Whole Plant (Active Primary Diagnosis)",
            "Canopy / Field View (Data-Blocked)",
            "Leaf (Optional Leaf-Only Experimental)",
            "Stem / Branch (Data-Blocked - Guidance Only)",
            "Fruit / Pod / Ear (Data-Blocked - Guidance Only)",
            "Root / Tuber (Data-Blocked - Guidance Only)",
            "Auto-detect",
        ]
        sel_viewpoint = st.selectbox(
            "Anatomical Viewpoint:",
            viewpoint_options,
            index=0,
            help="Whole-plant diagnosis is the active primary route for Corn and Tomato. Leaf diagnosis is a separate optional experimental feature.",
        )

    with g_col3:
        user_symptom_inquiry = st.text_input(
            "Inquiry or Observed Symptoms (Optional):",
            value="",
            placeholder="e.g., lower leaves yellowing with spots, or are there spider mites?",
            help="Mentioning pests or symptoms triggers specialized gating.",
        )

    st.markdown("---")

    # STEP 3: Photo Upload & Pipeline Execution
    st.markdown("### 3. 📷 Photo Upload & Diagnostic Pipeline")

    sample_image_bytes = None
    sample_image_label = None

    u_col1, u_col2 = st.columns([1.5, 1.5])
    with u_col1:
        uploaded_file = st.file_uploader(
            "Upload crop photo (Leaf, field view, or insect close-up):",
            type=["jpg", "jpeg", "png", "webp"],
            key="home_uploader",
        )

    with u_col2:
        st.markdown("**Or test with benchmark camera samples:**")
        b_cols = st.columns(3)
        if b_cols[0].button("🌱 Field Tomato", use_container_width=True):
            p = config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg"
            if p.exists():
                sample_image_bytes = open(p, "rb").read()
                sample_image_label = "Analyze this field tomato plant."
        if b_cols[1].button("🍂 Apple Scab", use_container_width=True):
            p = config.RAW_DATA_DIR / "sample_apple_scab.jpg"
            if p.exists():
                sample_image_bytes = open(p, "rb").read()
                sample_image_label = "Examine this apple leaf for scab lesions."
        if b_cols[2].button("🍅 Tomato Blight", use_container_width=True):
            p = config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg"
            if p.exists():
                sample_image_bytes = open(p, "rb").read()
                sample_image_label = "Check tomato leaf for blight."
        b_cols2 = st.columns(3)
        if b_cols2[0].button("🔍 Blurry Leaf", use_container_width=True):
            p = config.DATA_DIR / "processed" / "eval_samples" / "sample_blurred_leaf.jpg"
            if p.exists():
                sample_image_bytes = open(p, "rb").read()
                sample_image_label = "Evaluate blurred leaf."
        if b_cols2[1].button("🐱 Cat (Non-Plant)", use_container_width=True):
            p = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
            if p.exists():
                sample_image_bytes = open(p, "rb").read()
                sample_image_label = "Evaluate this non-plant image."
        if b_cols2[2].button("🕷️ Spider Mite", use_container_width=True):
            mite_dir = Path("data/raw/pest_detection/test/Tomato_Spider_Mites")
            mite_files = list(mite_dir.glob("*.jpg")) if mite_dir.exists() else []
            if mite_files:
                sample_image_bytes = open(mite_files[0], "rb").read()
                sample_image_label = "Check leaf for spider mites."

    # Determine active image bytes
    active_bytes = None
    if uploaded_file is not None:
        active_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    elif sample_image_bytes:
        active_bytes = sample_image_bytes

    # Execution Trigger Button
    if active_bytes is not None:
        img_hash = compute_image_hash(active_bytes)
        st.image(active_bytes, caption="Active Inspected Photograph", width=340)

        run_diag_btn = st.button("🚀 Execute Crop-Aware Vision Pipeline", type="primary", use_container_width=True)

        if run_diag_btn or (img_hash != st.session_state.last_processed_image_hash):
            st.session_state.last_processed_image_hash = img_hash

            with st.spinner("Executing modular vision and routing through validation gates..."):
                raw_pil = Image.open(io.BytesIO(active_bytes))
                pil_image = ImageOps.exif_transpose(raw_pil).convert("RGB")

                # Parse clean hints
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
                else:
                    viewpoint_hint_clean = None

                # Call Autonomous Router
                query_text = user_symptom_inquiry or sample_image_label or "Analyze this crop image for diseases or pests."
                telemetry = router.route_and_execute(
                    user_text=query_text,
                    image_input=pil_image,
                    is_new_image=True,
                    image_hash=img_hash,
                    crop_hint=crop_hint_clean,
                    part_hint=viewpoint_hint_clean,
                    confidence_threshold=conf_threshold,
                    state=st.session_state.farmer_state,
                )

                # Generate bilingual advisory
                advisory = synthesis_service.generate_advisory(
                    router_payload=telemetry,
                    language=active_lang,
                )
                telemetry["advisory"] = advisory
                st.session_state.last_diagnostic_result = telemetry

                # Append to chat messages
                st.session_state.messages.append({
                    "role": "user",
                    "content_en": query_text,
                    "content_kn": query_text,
                    "image_bytes": active_bytes,
                    "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content_en": advisory["en"],
                    "content_kn": advisory["kn"],
                    "telemetry": telemetry,
                    "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
                })

    # STEP 4: Calibrated Supported Result or Honest Abstention Card
    if st.session_state.last_diagnostic_result:
        res = st.session_state.last_diagnostic_result
        vision = res.get("vision_result")

        if vision:
            st.markdown("### 4. 🔬 Diagnostic & Route Gating Assessment")
            v_crop = vision.get("crop", "Unknown")
            v_part = vision.get("viewpoint", "leaf")
            v_status = vision.get("status", "unsupported")
            is_supp = vision.get("supported", False)
            disease = vision.get("disease")
            conf = vision.get("confidence", 0.0) * 100.0
            margin = vision.get("margin", 0.0) * 100.0
            chem_withheld = vision.get("chemical_treatment_withheld", False) or (conf < conf_threshold * 100.0) or (margin < margin_threshold * 100.0)
            fallback = vision.get("fallback_reason", "")
            pg = vision.get("photo_guidance")

            card_col1, card_col2 = st.columns([2, 1])

            with card_col1:
                # Case 1: Non-Plant Rejection
                if not vision.get("plant_detected", True) or v_status == "out_of_distribution":
                    st.error("🚫 **NON-PLANT / OUT-OF-DISTRIBUTION DETECTED**")
                    st.write("The uploaded photograph does not match recognizable agricultural foliage or crop structures.")
                    st.caption("Automatic disease prediction was safely refused to prevent false diagnosis.")

                # Case 2: Data-Blocked Route (Non-Leaf Viewpoints or Unsupported Crops)
                elif not is_supp or v_status in ["unsupported", "unsupported_crop", "unsupported_viewpoint", "unsupported_crop_part", "needs_training_data", "pest_guidance_provided"]:
                    st.warning("🛑 **DATA-BLOCKED ROUTE — SAFE ABSTENTION ENFORCED**")
                    st.markdown(f"- **Detected Crop**: `{v_crop}`")
                    st.markdown(f"- **Detected Viewpoint**: `{v_part.replace('_', ' ').title()}`")
                    st.markdown(f"- **Supported Scope**: Foliar (`leaf`) view for Apple, Cassava, Corn, and Tomato.")
                    st.markdown(f"- **Abstention Reason**: *{fallback or 'Untrained route. Real camera data required.'}*")
                    st.markdown(f"👉 *Check the **'Supported Coverage & Data-Required'** tab to see what authentic training datasets are needed to unlock this route.*")

                    # Bilingual Photo Guidance
                    if pg and isinstance(pg, dict):
                        g_title = pg.get("title_kn") if active_lang == "kn" else pg.get("title_en", "Photo Guidance")
                        g_inst = pg.get("instructions_kn") if active_lang == "kn" else pg.get("instructions_en", "")
                        with st.expander(f"📷 {g_title}", expanded=True):
                            st.markdown(g_inst)
                            checklist = pg.get("checklist", [])
                            if checklist:
                                st.markdown("**Photo-Capture Checklist:**")
                                for item in checklist:
                                    st.markdown(f"- [ ] {item}")

                # Case 3: Supported Foliar Route (Experimental Tier)
                else:
                    st.success(f"🧪 **EXPERIMENTAL OBSERVATION**: **{disease or 'Healthy Leaf'}**")
                    st.markdown(f"- **Target Crop**: `{v_crop}` | **Viewpoint**: `Leaf (Foliar)`")
                    st.markdown(f"- **Predicted Probability**: `{conf:.1f}%` (Threshold: > `{conf_threshold*100:.0f}%`)")
                    st.markdown(f"- **Candidate Margin**: `{margin:.1f}%` (Threshold: > `{margin_threshold*100:.0f}%`)")

                    # Check chemical withholding gate
                    if chem_withheld:
                        st.warning(
                            "🛑 **CHEMICAL TREATMENTS WITHHELD**: Confidence or margin is below the strict safety release threshold. "
                            "To protect crop health, automatic pesticide/fungicide prescriptions are safely withheld. "
                            "Practice cultural sanitation and take a physical leaf sample to your nearest KVK center."
                        )
                    else:
                        st.info(
                            "ℹ️ **High-Confidence Observation**: Chemical options from ICAR guidelines may be referenced below. "
                            "Always verify symptoms with your local agricultural officer and wear protective equipment."
                        )

                    # Top Candidates Bar
                    preds = vision.get("predictions", [])
                    if preds:
                        st.markdown("**Candidate Diagnosis Probabilities:**")
                        for p in preds[:3]:
                            p_prob = p.get("probability", 0.0)
                            p_name = p.get("label", "Unknown")
                            st.progress(float(p_prob), text=f"{p_name}: {p_prob*100:.1f}%")

            with card_col2:
                # KVK Referral Card
                kvk_ref = location_service.get_kvk_by_district(selected_district)
                if kvk_ref:
                    st.markdown("### 🏛️ KVK Referral")
                    st.markdown(f"**{kvk_ref.name}**")
                    st.caption(f"📞 `{kvk_ref.contact_phone}`")
                    st.caption(f"📍 {kvk_ref.location}")
                    st.caption("✅ Take a fresh specimen bag of affected leaves directly to the plant clinic.")

    st.markdown("---")

    # STEP 5: Interactive Chat Stream (Advisory turns)
    st.markdown("### 5. 💬 Agricultural Advisory Stream")
    for msg in st.session_state.messages[1:]:  # skip initial welcome in stream
        with st.chat_message(msg["role"]):
            c_text = msg["content_kn"] if active_lang == "kn" else msg["content_en"]
            if c_text:
                st.markdown(c_text)
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], width=220)

    # Chat Input
    chat_prompt = st.chat_input("Ask a follow-up question regarding crops, soil NPK, weather, or pest management...")
    if chat_prompt:
        st.session_state.messages.append({
            "role": "user",
            "content_en": chat_prompt,
            "content_kn": chat_prompt,
            "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
        })
        with st.spinner("Formulating agricultural advisory..."):
            c_telemetry = router.route_and_execute(
                user_text=chat_prompt,
                image_input=None,
                is_new_image=False,
                state=st.session_state.farmer_state,
            )
            c_adv = synthesis_service.generate_advisory(c_telemetry, language=active_lang)
            st.session_state.messages.append({
                "role": "assistant",
                "content_en": c_adv["en"],
                "content_kn": c_adv["kn"],
                "telemetry": c_telemetry,
                "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
            })
            st.rerun()


# ------------------------------------------------------------------------------
# TAB 2: Supported Coverage & Data-Required Screen
# ------------------------------------------------------------------------------
with tab_coverage:
    st.markdown("## 📋 Supported Diagnostic Coverage & Data-Required Matrix")
    st.markdown(
        "To ensure production safety and eliminate AI hallucinations, this system operates under a strict "
        "**Zero-Guessing Policy**. Routes are activated **ONLY** when verified real-camera training datasets and held-out evaluations exist."
    )

    # Coverage Matrix Table
    st.markdown("### 1. Operational Route Matrix")
    matrix_data = [
        {"Organ / Viewpoint": "Whole Plant", "Apple": "🛑 Data-Blocked (Foliar Only)", "Cassava": "🛑 Data-Blocked (Foliar Only)", "Corn (Maize)": "✅ Active Primary Diagnosis", "Tomato": "✅ Active Primary Diagnosis", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Canopy / Field", "Apple": "🛑 Data-Blocked", "Cassava": "🛑 Data-Blocked", "Corn (Maize)": "🛑 Data-Blocked", "Tomato": "🛑 Data-Blocked", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Leaf (Foliar)", "Apple": "✅ Leaf-Only Experimental", "Cassava": "✅ Leaf-Only Experimental", "Corn (Maize)": "✅ Leaf-Only Experimental", "Tomato": "✅ Leaf-Only Experimental", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Stem / Branch", "Apple": "🛑 Data-Blocked", "Cassava": "🛑 Data-Blocked", "Corn (Maize)": "🛑 Data-Blocked", "Tomato": "🛑 Data-Blocked", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Fruit / Pod", "Apple": "🛑 Data-Blocked", "Cassava": "🛑 Data-Blocked", "Corn (Maize)": "🛑 Data-Blocked", "Tomato": "🛑 Data-Blocked", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Root / Tuber", "Apple": "🛑 Data-Blocked", "Cassava": "🛑 Data-Blocked", "Corn (Maize)": "🛑 Data-Blocked", "Tomato": "🛑 Data-Blocked", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Pests (Spider Mites)", "Apple": "🛑 Data-Blocked", "Cassava": "🛑 Data-Blocked", "Corn (Maize)": "🛑 Data-Blocked", "Tomato": "✅ Pilot (100.0%)", "Other Crops": "🛑 Data-Blocked"},
        {"Organ / Viewpoint": "Pests (Armyworm/Borer)", "Apple": "🛑 Data-Blocked", "Cassava": "🛑 Data-Blocked", "Corn (Maize)": "🛑 Data-Blocked", "Tomato": "🛑 Data-Blocked", "Other Crops": "🛑 Data-Blocked"},
    ]
    st.dataframe(matrix_data, use_container_width=True)

    st.markdown("---")

    # Data-Required Screen
    st.markdown("### 2. 🛑 Data-Required Specifications to Unlock Blocked Routes")
    st.markdown(
        "The following organ viewpoints, crops, and insect species are currently **Data-Blocked**. "
        "Here is the exact authentic dataset specification required to activate them:"
    )

    req_cols1, req_cols2 = st.columns(2)

    with req_cols1:
        with st.expander("🪵 Stem & Branch Cankers (Data Blocked)", expanded=True):
            st.markdown(
                """
                - **Current Status**: 🛑 Disabled.
                - **Reason**: No authentic camera dataset for stem fissures or cankers exists locally.
                - **Data Required**:
                  - Minimum 250 verified high-resolution photographs of stem lesions across crops.
                  - Longitudinal and cross-sectional stem cuts showing internal vascular browning.
                  - Laboratory isolation confirmation (e.g. *Fusarium*, *Ralstonia solanacearum*).
                """
            )

        with req_cols2:
            with st.expander("🍎 Fruit Rots & Boreholes (Data Blocked)", expanded=True):
                st.markdown(
                    """
                    - **Current Status**: 🛑 Disabled.
                    - **Reason**: Fruit rot datasets are currently absent from verified local manifests.
                    - **Data Required**:
                      - 200+ authenticated field close-ups of fruit calyxes and boreholes.
                      - Clear differentiation between physical sunburn/blossom-end rot vs fungal rot (*Colletotrichum*).
                      - Multi-angle shots showing larval frass.
                    """
                )

    req_cols3, req_cols4 = st.columns(2)

    with req_cols3:
        with st.expander("🌱 Canopy & Whole Plant Field Overviews (Data Blocked)", expanded=True):
            st.markdown(
                """
                - **Current Status**: 🛑 Disabled.
                - **Reason**: Field canopy photographs suffer from severe distance variance and background weed clutter.
                - **Data Required**:
                  - 300+ full-plant field photos taken at 1.0–1.5 meter distance under natural sunlight.
                  - Paired ground-truth foliar annotations to avoid weed misrouting.
                  - Verified standing crop stages (vegetative, tasseling, harvest).
                """
            )

    with req_cols4:
        with st.expander("🥔 Roots, Tubers & Nematodes (Data Blocked)", expanded=True):
            st.markdown(
                """
                - **Current Status**: 🛑 Disabled.
                - **Reason**: Sub-surface soil pathogens cannot be reliably classified from surface foliage without destructive sampling.
                - **Data Required**:
                  - 200+ photos of washed root systems showing root-knot nematode galls (*Meloidogyne* spp.).
                  - Certified soil nematode laboratory assay reports.
                  - Safe laboratory referral instructions.
                """
            )

    req_cols5, req_cols6 = st.columns(2)

    with req_cols5:
        with st.expander("🐛 Fall Armyworm, Stem Borers & Aphids (Data Blocked)", expanded=True):
            st.markdown(
                """
                - **Current Status**: 🛑 Disabled (Chemicals Withheld).
                - **Reason**: IP102 image files are missing on disk (only classes.txt exists). Synthetic diagrams have been quarantined.
                - **Data Required**:
                  - 200+ authentic macro photos of larval instars inside maize whorls.
                  - Verified pheromone trap catch photos.
                  - Economic threshold level (ETL) field scouting counts.
                """
            )

    with req_cols6:
        with st.expander("🍇 Unsupported Crops: Grape, Potato, Citrus, Rice, Wheat", expanded=True):
            st.markdown(
                """
                - **Current Status**: 🛑 Disabled.
                - **Reason**: Local manifests currently exclude these crops to prevent cross-crop hallucination.
                - **Data Required**:
                  - Minimum 300 real camera samples per crop species across 3 disease classes + 1 healthy control.
                  - Deterministic train/val/test splits with zero cross-split leakage.
                  - Held-out evaluation achieving >= 65% macro F1.
                """
            )


# ------------------------------------------------------------------------------
# TAB 3: System Startup Audit & Model Cards
# ------------------------------------------------------------------------------
with tab_audit:
    st.markdown("## 🔍 Startup Artifact Verification & Model Cards")

    # Artifact Readiness Status
    st.markdown("### 1. Startup Artifact Integrity Check")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    s_col1.metric("Disease Model", "Ready (Experimental)" if startup_report.disease_model_ready else "Degraded")
    s_col2.metric("Pest Model", "Ready (Pilot Mites)" if startup_report.pest_model_ready else "Degraded")
    s_col3.metric("Manifests", "592 Samples" if startup_report.manifests_ready else "Missing")
    s_col4.metric("KVK Registry", "20 Centers" if startup_report.location_registry_ready else "Degraded")

    with st.expander("🔎 View Detailed Artifact Inspection Report", expanded=False):
        st.json(startup_report.to_dict())

    st.markdown("---")

    # Real Photo Contact Sheet
    st.markdown("### 2. Optical Realism & Real-Photo Gallery")
    st.caption("Visual proof of camera sensor noise, texture entropy, and organic optical capture (Synthetic diagrams quarantined).")
    contact_sheet_path = ROOT_DIR / "models" / "real_photo_gallery_contact_sheet.jpg"
    if contact_sheet_path.exists():
        st.image(str(contact_sheet_path), caption="Deterministic 16-Class Real-Camera Contact Sheet", use_container_width=True)
    else:
        st.warning("Contact sheet artifact not found on disk.")

    st.markdown("---")

    # Model Cards Viewer
    st.markdown("### 3. Official Model Cards")
    mc_tab0, mc_tab1, mc_tab2, mc_tab3 = st.tabs([
        "Whole-Plant & Canopy Model Card",
        "Optional Leaf-Only Model Card",
        "Pest Detection CNN Model Card",
        "Model Release Policy",
    ])

    with mc_tab0:
        doc_p0 = ROOT_DIR / "docs" / "MODEL_CARD_WHOLE_PLANT.md"
        if doc_p0.exists():
            st.markdown(doc_p0.read_text(encoding="utf-8"))

    with mc_tab1:
        doc_p = ROOT_DIR / "docs" / "MODEL_CARD_DISEASE.md"
        if doc_p.exists():
            st.markdown(doc_p.read_text(encoding="utf-8"))

    with mc_tab2:
        doc_p2 = ROOT_DIR / "docs" / "MODEL_CARD_PEST.md"
        if doc_p2.exists():
            st.markdown(doc_p2.read_text(encoding="utf-8"))

    with mc_tab3:
        doc_p3 = ROOT_DIR / "docs" / "MODEL_RELEASE_POLICY.md"
        if doc_p3.exists():
            st.markdown(doc_p3.read_text(encoding="utf-8"))
