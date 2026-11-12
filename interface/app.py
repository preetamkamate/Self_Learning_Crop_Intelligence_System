"""
Self-Learning Crop Intelligence System
Unified Conversational Agricultural Advisory Interface (Streamlit).
Pairs Multimodal Computer Vision (Whole-Plant DINOv2 + Multi-Scale Pathology),
Tabular ML (RandomForest on Indian Agriculture Dataset), Open-Meteo Weather API,
and FAISS RAG over ICAR/IIHR Guidelines with Responsible Self-Learning.
"""
import hashlib
import io
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.feedback.batch_retrain import run_batch_retrain_gate
from src.feedback.feedback_logger import (
    get_queue_statistics,
    log_interaction,
    update_feedback,
)
from src.orchestrator.llm_service import AdvisorySynthesisService
from src.orchestrator.router import AutonomousRouter
from src.orchestrator.state import FarmerProfile, SessionState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set Page Config
st.set_page_config(
    page_title="Self-Learning Crop Intelligence System | Raitha Mitra",
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
    .badge-module { background-color: #e0f2fe; color: #0369a1; border: 1px solid #38bdf8; }
    .sample-btn { margin-bottom: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_orchestrator():
    """Cache the router and advisory synthesis service for fast turns."""
    router = AutonomousRouter()
    synthesis = AdvisorySynthesisService()
    return router, synthesis


router, synthesis_service = load_orchestrator()

# Initialize Session State
if "farmer_state" not in st.session_state:
    st.session_state.farmer_state = SessionState()

if "last_processed_image_hash" not in st.session_state:
    st.session_state.last_processed_image_hash = None

if "pending_sample_image" not in st.session_state:
    st.session_state.pending_sample_image = None

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    welcome_en = (
        "👋 **Namaskara! Welcome to the Self-Learning Crop Intelligence System.**\n\n"
        "I am your autonomous agricultural advisor. You can interact with me naturally:\n"
        "- 📷 **Upload a field photo** (whole plant, canopy, or leaf) for disease diagnosis with DINOv2 & Shannon entropy gating.\n"
        "- 🌱 **Ask for crop recommendations** based on your soil NPK & district climate.\n"
        "- 🌦️ **Check real-time weather & spray safety** for your district.\n"
        "- 📖 **Get authoritative ICAR / IIHR management guidelines** and safe chemical dosages."
    )
    welcome_kn = (
        "👋 **ನಮಸ್ಕಾರ! ಸ್ವಯಂ-ಕಲಿಕೆಯ ಬೆಳೆ ಬುದ್ಧಿಮತ್ತೆ ವ್ಯವಸ್ಥೆಗೆ (Crop Intelligence System) ಸುಸ್ವಾಗತ.**\n\n"
        "ನಾನು ನಿಮ್ಮ ಕೃಷಿ ಸಲಹೆಗಾರ. ನೀವು ನನ್ನೊಂದಿಗೆ ಮುಕ್ತವಾಗಿ ಸಂವಹನ ನಡೆಸಬಹುದು:\n"
        "- 📷 **ಗಿಡದ ಫೋಟೋ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ**: DINOv2 ಮತ್ತು ಎಂಟ್ರೊಪಿ ಸುರಕ್ಷತೆಯೊಂದಿಗೆ ರೋಗ ಪತ್ತೆ.\n"
        "- 🌱 **ಬೆಳೆ ಶಿಫಾರಸು ಪಡೆಯಿರಿ**: ಮಣ್ಣಿನ NPK ಮತ್ತು ಹವಾಮಾನಕ್ಕೆ ತಕ್ಕಂತೆ.\n"
        "- 🌦️ **ಲೈವ್ ಹವಾಮಾನ ಮತ್ತು ಸಿಂಪಡಣಾ ಮಾಹಿತಿ**: ನಿಮ್ಮ ಜಿಲ್ಲೆಯ ಮಳೆ ಮುನ್ಸೂಚನೆ.\n"
        "- 📖 **ICAR / IIHR ಪ್ರಾಧಿಕಾರ ಮಾರ್ಗಸೂಚಿಗಳು**: ಸರಿಯಾದ ಕೀಟನಾಶಕ ಪ್ರಮಾಣ."
    )
    st.session_state.messages.append({
        "role": "assistant",
        "content_en": welcome_en,
        "content_kn": welcome_kn,
        "telemetry": None,
        "turn_id": "welcome_turn",
    })


def compute_image_hash(data: bytes) -> str:
    """Compute MD5 hash for image deduplication and multi-turn state tracking."""
    return hashlib.md5(data).hexdigest()


# ==============================================================================
# SIDEBAR: Unified Farmer Profile & Responsible Retraining Gate
# ==============================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/eco/materialicons/48dp/2x/baseline_eco_black_48dp.png",
        width=48,
    )
    st.title("🌾 Farmer Profile")
    st.caption("NHCE 5th-Sem AI & ML Mini-Project")

    # Language Toggle
    lang_choice = st.radio(
        "Language / ಭಾಷೆ:",
        options=["English", "ಕನ್ನಡ (Kannada)"],
        index=0 if st.session_state.farmer_state.profile.language == "en" else 1,
    )
    active_lang = "kn" if "ಕನ್ನಡ" in lang_choice else "en"
    st.session_state.farmer_state.profile.language = active_lang

    st.markdown("---")
    st.subheader("📍 Location & Farm Specs")
    districts = list(config.DISTRICT_COORDINATES.keys())
    selected_district = st.selectbox(
        "District / Zone:",
        districts,
        index=districts.index(st.session_state.farmer_state.profile.district)
        if st.session_state.farmer_state.profile.district in districts
        else 1,
    )
    st.session_state.farmer_state.profile.district = selected_district

    farmer_name = st.text_input(
        "Farmer Name:",
        value=st.session_state.farmer_state.profile.name,
    )
    st.session_state.farmer_state.profile.name = farmer_name

    land_area = st.number_input(
        "Land Holding (Acres):",
        min_value=0.25,
        max_value=100.0,
        value=float(st.session_state.farmer_state.profile.land_area_acres),
        step=0.5,
    )
    st.session_state.farmer_state.profile.land_area_acres = land_area

    st.markdown("---")
    st.subheader("🧪 Soil Health Card (NPK & pH)")
    st.caption("Auto-updated if mentioned in chat")
    col_n, col_p = st.columns(2)
    with col_n:
        n_val = st.number_input("N (kg/ha):", value=float(st.session_state.farmer_state.profile.soil_n), step=5.0)
    with col_p:
        p_val = st.number_input("P (kg/ha):", value=float(st.session_state.farmer_state.profile.soil_p), step=5.0)

    col_k, col_ph = st.columns(2)
    with col_k:
        k_val = st.number_input("K (kg/ha):", value=float(st.session_state.farmer_state.profile.soil_k), step=5.0)
    with col_ph:
        ph_val = st.number_input("Soil pH:", value=float(st.session_state.farmer_state.profile.soil_ph), step=0.1, min_value=3.5, max_value=9.5)

    st.session_state.farmer_state.profile.soil_n = n_val
    st.session_state.farmer_state.profile.soil_p = p_val
    st.session_state.farmer_state.profile.soil_k = k_val
    st.session_state.farmer_state.profile.soil_ph = ph_val

    st.markdown("---")
    # Responsible Self-Learning Retraining Section
    st.subheader("🔄 Self-Learning Model Gate")
    stats = get_queue_statistics()
    st.markdown(
        f"""
        - **Total Queue Items**: `{stats.get('total_interactions', 0)}`
        - **Verified Samples**: `{stats.get('verified_samples', 0)}`
        - **Pending Review**: `{stats.get('pending_verification', 0)}`
        """
    )

    if st.button("Trigger Batch Retraining Gate"):
        with st.spinner("Evaluating candidate model against production accuracy..."):
            gate_res = run_batch_retrain_gate(min_verified_samples=2)
            if gate_res["status"] == "promoted":
                st.success(
                    f"🎉 Gate Passed! Candidate Accuracy ({gate_res['candidate_accuracy']*100:.2f}%) "
                    f">= Production ({gate_res['production_accuracy']*100:.2f}%). Model Promoted!"
                )
            elif gate_res["status"] == "rejected":
                st.warning(
                    f"🛑 Gate Rejected: Candidate Accuracy ({gate_res['candidate_accuracy']*100:.2f}%) "
                    f"< Production ({gate_res['production_accuracy']*100:.2f}%). Production model preserved."
                )
            else:
                st.info(f"ℹ️ {gate_res.get('reason', 'Retraining gate deferred.')}")

# ==============================================================================
# MAIN VIEW: Unified Continuous Chat Stream
# ==============================================================================
st.markdown("## 🌾 Self-Learning Crop Intelligence System")
st.markdown(
    "*Autonomous Multimodal Agro-Advisory Engine with DINOv2 Whole-Plant Analysis & ICAR RAG Knowledge Base*"
)

# Quick Action / Sample Query Pills
st.markdown("**Quick Prompts / ತ್ವರಿತ ಪ್ರಶ್ನೆಗಳು:**")
qp_cols = st.columns(4)
sample_prompt = None

if qp_cols[0].button("🌱 Recommend Crops", use_container_width=True):
    sample_prompt = "What crops are most suitable for my soil and district?"
if qp_cols[1].button("🌦️ Spraying Weather", use_container_width=True):
    sample_prompt = "Will it rain tomorrow for spraying pesticides?"
if qp_cols[2].button("🍅 Tomato Blight Guide", use_container_width=True):
    sample_prompt = "What organic spray works for Tomato Early Blight?"
if qp_cols[3].button("🌽 Maize Pest Management", use_container_width=True):
    sample_prompt = "What is the ICAR chemical dosage for Fall Armyworm in maize?"

# Render Chat History
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Pick content based on active language
        content = msg["content_kn"] if active_lang == "kn" else msg["content_en"]
        if content:  # CHANGED: Render text only if non-empty (avoids empty box for image-only turn)
            st.markdown(content)

        # Show uploaded image if exists for this turn
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], caption="Inspected Crop Photograph", width=320)

        # Diagnostic & Model Reasoning Expander
        telemetry = msg.get("telemetry")
        if telemetry:
            with st.expander("🔍 Diagnostic & Model Reasoning (NHCE Mini-Project Telemetry)"):
                # CHANGED: Prominently show live LLM execution status in telemetry expander
                llm_status = telemetry.get("llm_status", "Gemini API Active")
                status_class = "badge-confident" if "Active" in llm_status else "badge-ood"
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span class="metric-badge {status_class}">🤖 LLM Status: {llm_status}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                mod_badges = "".join(
                    [f'<span class="metric-badge badge-module">⚙️ {m}</span>' for m in telemetry.get("modules_run", [])]
                )
                st.markdown(mod_badges, unsafe_allow_html=True)
                st.markdown("<br/>", unsafe_allow_html=True)

                t_cols = st.columns(3)

                # Vision Diagnostics
                vision = telemetry.get("vision_result")
                with t_cols[0]:
                    st.markdown("**Whole-Plant Vision Diagnostics**")
                    if vision:
                        status = vision.get("status")
                        entropy = vision.get("entropy", 0.0)
                        dino_sim = vision.get("dinov2_plant_similarity", 1.0)
                        aff_reg = vision.get("affected_regions", "Whole Canopy")

                        if status == "out_of_distribution":
                            st.markdown(
                                f'<span class="metric-badge badge-ood">🚫 NON-PLANT OOD (DINOv2={dino_sim:.3f})</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Refusal Gate Active: {vision.get('top1_label')}")
                        elif status == "uncertain":
                            st.markdown(
                                f'<span class="metric-badge badge-ood">⚠️ UNCERTAIN (H={entropy:.3f})</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Conflict across regions. Top1: {vision.get('top1_label')} ({vision.get('top1_confidence')}%)")
                        else:
                            st.markdown(
                                f'<span class="metric-badge badge-confident">✅ CONFIDENT (DINOv2={dino_sim:.3f})</span>',
                                unsafe_allow_html=True,
                            )
                            st.write(f"**Top Condition**: {vision.get('top1_label')}")
                            st.write(f"**Confidence**: {vision.get('top1_confidence')}% | **Entropy**: `{entropy:.3f}`")
                            st.write(f"**Affected Region**: 📍 *{aff_reg}*")

                        breakdown = vision.get("region_breakdown", [])
                        if breakdown:
                            with st.expander("🔎 View 5-Region Spatial Breakdown"):
                                for reg in breakdown:
                                    st.caption(f"• **{reg['region']}**: {reg['top_label']} ({reg['confidence_percent']}%) [H={reg['entropy']}]")
                    else:
                        st.caption("No image evaluated in this turn.")

                # Tabular ML Diagnostics
                tabular = telemetry.get("tabular_result")
                with t_cols[1]:
                    st.markdown("**Tabular ML (RandomForest)**")
                    if tabular:
                        st.write(f"**Top Pick**: {tabular.get('top_crop').title()} ({tabular.get('top_confidence')}%)")
                        recs = tabular.get("recommendations", [])
                        for r in recs[:3]:
                            st.caption(f"• {r['crop'].title()}: {r['confidence_percent']}%")
                    else:
                        st.caption("Tabular crop model not requested.")

                # Weather & RAG Diagnostics
                weather = telemetry.get("weather_result")
                rag_docs = telemetry.get("rag_results", [])
                with t_cols[2]:
                    st.markdown("**Weather & RAG Citations**")
                    if weather:
                        spray_status = "🟢 Safe to spray" if weather.get("spray_safe") else "🔴 Rain / Wind Warning"
                        st.write(f"**Live Weather**: {weather.get('temperature_c')}°C | {weather.get('humidity_pct')}% RH")
                        st.write(f"**Spray Gate**: {spray_status}")
                    elif rag_docs:
                        st.write(f"**RAG Citations**: {len(rag_docs)} chunks")
                        for d in rag_docs:
                            st.caption(f"📄 {d.get('source')} (Score: {d.get('relevance_percent')}%)")
                    else:
                        st.caption("Weather / RAG not requested.")

                # Farmer Feedback Action
                turn_id = msg.get("turn_id")
                if turn_id and turn_id != "welcome_turn":
                    st.markdown("---")
                    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 3])
                    if fb_col1.button("👍 Helpful", key=f"up_{turn_id}"):
                        update_feedback(turn_id, "up")
                        st.toast("Thank you! Feedback logged into responsible self-learning queue.")
                    if fb_col2.button("👎 Needs Review", key=f"down_{turn_id}"):
                        update_feedback(turn_id, "down")
                        st.toast("Logged for agronomist verification queue.")

# ==============================================================================
# UNIFIED INPUT AREA: Text + Image Upload
# ==============================================================================
st.markdown("---")

sample_image_trigger = False
sample_image_bytes = None
sample_image_label = None

# Compact Multimodal Attachment Row
with st.expander("📷 Upload Plant Photo (Whole-Plant Field View, Canopy, Stem, or Leaf/Fruit)", expanded=False):
    st.info(
        "💡 **Whole-Plant Field Understanding Active**: Upload full-plant photographs, field canopy shots, "
        "or leaf lesions. The DINOv2 backbone verifies agricultural context, and multi-scale tiling pinpoints affected regions."
    )
    uploaded_file = st.file_uploader(
        "Select crop photograph from your device:",
        type=["jpg", "jpeg", "png"],
        key="plant_file_uploader",
    )
    # Quick Test Images
    st.markdown("**Or test with benchmark field & leaf samples:**")
    samp_cols = st.columns(5)
    if samp_cols[0].button("🌱 Field Tomato", key="btn_field_tomato"):
        p = config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Please analyze this full-plant tomato crop."
            sample_image_trigger = True
    if samp_cols[1].button("🍂 Apple Scab", key="btn_apple"):
        p = config.RAW_DATA_DIR / "sample_apple_scab.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Please diagnose the disease on this leaf."
            sample_image_trigger = True
    if samp_cols[2].button("🍅 Tomato Blight", key="btn_tomato"):
        p = config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Please identify the disease on this tomato plant."
            sample_image_trigger = True
    if samp_cols[3].button("🐱 Non-Plant (Cat)", key="btn_cat"):
        p = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Please analyze this image."
            sample_image_trigger = True
    if samp_cols[4].button("⚠️ Noise (OOD)", key="btn_ood"):
        import numpy as np
        noise = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(noise).save(buf, format="JPEG")
        sample_image_bytes = buf.getvalue()
        sample_image_label = "Please check this sample."
        sample_image_trigger = True

# ==============================================================================
# UNIFIED CHAT INPUT: Simultaneous Image Attachment + Text in One Turn
# ==============================================================================
# CHANGED: Unified Multimodal Chat Input accepting text and/or images
prompt_data = st.chat_input(
    "Ask a question or upload a plant photo...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"],
)

# CHANGED: Extract both user text and attached files together in the same turn
user_text = prompt_data.text.strip() if prompt_data and hasattr(prompt_data, "text") and prompt_data.text else (str(prompt_data).strip() if isinstance(prompt_data, str) else "")
chat_files = prompt_data.files if prompt_data and hasattr(prompt_data, "files") and prompt_data.files else []
chat_image_bytes = chat_files[0].read() if chat_files else None

# CHANGED: Check fallback attachment sources (Quick Test samples or expander file uploader)
active_img_bytes = None
if chat_image_bytes:
    active_img_bytes = chat_image_bytes
elif sample_image_trigger and sample_image_bytes:
    active_img_bytes = sample_image_bytes
    if not user_text and sample_image_label:
        user_text = sample_image_label
elif uploaded_file is not None:
    active_img_bytes = uploaded_file.getvalue()

if not user_text and sample_prompt:
    user_text = sample_prompt

# CHANGED: Check image novelty via MD5 hash
current_hash = compute_image_hash(active_img_bytes) if active_img_bytes else None
is_new_image = (
    current_hash is not None 
    and current_hash != st.session_state.last_processed_image_hash
)

# CHANGED: Trigger turn if farmer entered text OR provided a new image
should_run_turn = bool(user_text) or (is_new_image and active_img_bytes is not None)

# Process turn
if should_run_turn:
    turn_id = f"turn_{uuid.uuid4().hex[:8]}"

    # CHANGED: Assign default diagnostic prompt internally if user submitted image with empty text
    internal_query = user_text if user_text else "Analyze this plant image and identify any health issues or diseases."
    display_text = user_text  # Clean display in chat bubble (blank if image-only)

    # Attach image thumbnail only if a NEW image is uploaded in this turn
    img_for_turn = active_img_bytes if is_new_image else None
    pil_image = Image.open(io.BytesIO(img_for_turn)) if img_for_turn else None

    # CHANGED: Record User Message (both image thumbnail and text inside the same chat bubble)
    st.session_state.messages.append({
        "role": "user",
        "content_en": display_text,
        "content_kn": display_text,
        "image_bytes": img_for_turn,
        "turn_id": turn_id,
    })

    # Execute Autonomous Internal Router
    with st.spinner("Processing agricultural inquiry..."):
        telemetry = router.route_and_execute(  # CHANGED: Pass both user query and image input together
            user_text=internal_query,
            image_input=pil_image,
            is_new_image=is_new_image,  # CHANGED: Explicit boolean novelty flag
            image_hash=current_hash if is_new_image else None,
            state=st.session_state.farmer_state,
        )

        # If a new image was evaluated, update the session state hash tracker
        if is_new_image and current_hash:
            st.session_state.last_processed_image_hash = current_hash

        # Synthesize Dual Bilingual Advisory (Concise 120-180 words)
        advisories = synthesis_service.generate_advisory(
            router_payload=telemetry,
            language=active_lang,
        )
        # CHANGED: Capture live LLM execution status in telemetry payload
        telemetry["llm_status"] = advisories.get("llm_status", getattr(synthesis_service, "last_llm_status", "Gemini API Active"))

        # Log to Responsible Self-Learning Queue
        input_type = "multimodal" if is_new_image else ("tabular" if telemetry.get("tabular_result") else "text_rag")
        pred_label = ""
        conf = 0.0
        entropy = 0.0
        feats_or_img = internal_query  # CHANGED: Log internal query

        if telemetry.get("vision_result") and is_new_image:
            v = telemetry["vision_result"]
            pred_label = v.get("top1_label", "")
            conf = v.get("top1_confidence", 0.0)
            entropy = v.get("entropy", 0.0)
            feats_or_img = f"image_{turn_id}.jpg"
        elif telemetry.get("tabular_result"):
            t = telemetry["tabular_result"]
            pred_label = t.get("top_crop", "")
            conf = t.get("top_confidence", 0.0)
            feats_or_img = json.dumps(t.get("input_features", {}))

        log_interaction(
            interaction_id=turn_id,
            input_type=input_type,
            features_or_image=feats_or_img,
            predicted_label=pred_label,
            confidence=conf,
            entropy=entropy,
            farmer_feedback="neutral",
            verified_label="",
        )

        # Append Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content_en": advisories["en"],
            "content_kn": advisories["kn"],
            "telemetry": telemetry,
            "turn_id": turn_id,
        })

    st.rerun()
