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
from PIL import Image, ImageOps
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
    st.subheader("🌿 Crop & Plant Viewpoint Context")
    crop_options = ["Auto-detect", "Tomato", "Potato", "Corn (Maize)", "Cotton", "Paddy (Rice)", "Chilli", "Citrus", "Coconut", "Apple", "Grape", "Other"]
    confirmed_crop = st.selectbox(
        "Crop / Tree Species:",
        crop_options,
        index=0,
        help="Confirm crop/tree species for tailored diagnosis and ICAR package of practices.",
    )
    part_options = [
        "Auto-detect",
        "Leaf",
        "Stem / Branch",
        "Fruit",
        "Root / Tuber",
        "Whole Plant",
        "Canopy / Field",
        "Bark / Trunk",
    ]
    confirmed_part = st.selectbox(
        "Plant Part / Viewpoint:",
        part_options,
        index=0,
        help="Select the plant part shown in the photo to route to specialized multi-part diagnostic models.",
    )
    col_stg, col_dur = st.columns(2)
    with col_stg:
        confirmed_stage = st.selectbox(
            "Growth Stage:",
            ["Vegetative", "Seedling", "Flowering", "Fruiting/Maturity", "Perennial/Tree"],
            index=0,
        )
    with col_dur:
        confirmed_duration = st.selectbox(
            "Symptom Duration:",
            ["< 3 days", "3-7 days", "1-2 weeks", "> 1 month"],
            index=1,
        )

    symptom_description = st.text_area(
        "Optional Symptom Description:",
        value="",
        placeholder="e.g., lower leaves yellowing with concentric brown rings, wilting during mid-day",
        help="Field-observed symptom details and lesion patterns.",
    )

    st.markdown("---")
    st.subheader("⚙️ Vision & OOD Tuning")
    ood_threshold_val = st.slider(
        "DINOv2 Plant OOD Threshold:",
        min_value=0.05,
        max_value=0.50,
        value=float(config.DINOV2_PLANT_OOD_THRESHOLD),
        step=0.01,
        help="Cosine similarity below this threshold flags non-plant/unrelated objects. Configurable via DINOV2_PLANT_OOD_THRESHOLD env var.",
    )

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
                        dbg = vision.get("debug_info", {})
                        crop_sp = vision.get("crop", vision.get("crop_species", "Unknown"))
                        p_part = vision.get("viewpoint", vision.get("plant_part", "whole_plant")).replace("_", " ").title()
                        selected_model = vision.get("selected_model", dbg.get("selected_model", "models/whole_plant/model.pt"))
                        model_ckpt = vision.get("checkpoint_path", vision.get("model_path", "models/whole_plant/model.pt"))
                        conf_val = vision.get("confidence", vision.get("top1_confidence", 0.0))
                        conf_pct = conf_val * 100 if conf_val <= 1.0 else conf_val
                        fallback_reason = vision.get("fallback_reason", dbg.get("fallback_reason", ""))

                        # Required Telemetry Display
                        st.markdown(f"• **Detected Crop**: `{crop_sp}`")
                        st.markdown(f"• **Detected Viewpoint**: `{p_part}`")
                        st.markdown(f"• **Selected Model Route**: `{selected_model}`")
                        st.markdown(f"• **Checkpoint Path**: `{model_ckpt}`")
                        st.markdown(f"• **Confidence**: `{conf_pct:.2f}%`")
                        if fallback_reason:
                            st.caption(f"• **Exact Fallback Reason**: `{fallback_reason}`")

                        # Check if Whole-Plant model and held-out evaluation report exist
                        wp_ckpt_path = ROOT_DIR / "models" / "whole_plant" / "model.pt"
                        wp_eval_path = ROOT_DIR / "models" / "whole_plant" / "evaluation_report.json"
                        is_whole_plant_deployed = wp_ckpt_path.exists() and wp_eval_path.exists()

                        if not is_whole_plant_deployed or status in ["needs_training_data", "whole_plant_not_trained"]:
                            st.info("Valid plant image detected, but this crop/viewpoint diagnosis is not trained yet.")
                        elif status == "out_of_distribution" or not vision.get("plant_detected", True):
                            st.markdown(
                                f'<span class="metric-badge badge-ood">🚫 NON-PLANT</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"**Safe Result**: Non-plant image detected. No disease predicted.")
                        elif status in ["unsupported", "unsupported_crop_part", "whole_plant_unsupported"] or not vision.get("supported", False):
                            st.info("Valid plant image detected, but this crop/viewpoint diagnosis is not trained yet.")
                        elif status in ["uncertain_disease", "uncertain"]:
                            st.markdown(
                                f'<span class="metric-badge badge-ood">⚠️ UNCERTAIN SYMPTOMS</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Top Candidate: {vision.get('top1_label')} ({conf_pct:.1f}%) — confirmation advised.")
                        else:
                            st.markdown(
                                f'<span class="metric-badge badge-confident">✅ VALIDATED WHOLE-PLANT DIAGNOSIS</span>'
                                f'<span class="metric-badge badge-confident">✅ HELD-OUT TEST VERIFIED</span>',
                                unsafe_allow_html=True,
                            )
                            st.write(f"**Diagnosed Condition**: {vision.get('disease', vision.get('top1_label'))}")
                            st.write(f"**Confidence**: {conf_pct:.2f}%")

                        # Hierarchical Diagnosis Card
                        hierarchical = vision.get("hierarchical_diagnosis")
                        if hierarchical:
                            st.markdown(
                                f"**Hierarchy**: `{hierarchical.get('crop_species')}` ➔ `{hierarchical.get('plant_part')}` ➔ `{hierarchical.get('issue_category')}` "
                                f"(`{hierarchical.get('evidence_level')}`)"
                            )

                        # Authoritative Direct HTTPS Source Reference Card (Requirements 9 & 10)
                        source_ev = vision.get("source_evidence")
                        if source_ev:
                            if isinstance(source_ev, dict):
                                st.markdown("---")
                                st.markdown(f"📚 **Authoritative Source**: [{source_ev.get('source_title', 'Agricultural Extension Protocol')}]({source_ev.get('url', 'https://icar.org.in')})")
                                st.caption(f"🏛️ **Organization**: {source_ev.get('organization', 'ICAR & Agricultural Extension')}")
                                st.caption(f"🔗 **Direct HTTPS Link**: [{source_ev.get('url')}]({source_ev.get('url')})")
                                st.caption(f"📅 **Access Date**: {source_ev.get('access_date', 'November 2026')}")
                                st.caption(f"📋 **Supports**: {source_ev.get('supported_recommendations', 'Diagnostic & Treatment Package')}")
                            else:
                                st.caption(f"📚 **Source Evidence**: {source_ev}")

                        # Nutrient Screening Expander
                        nut_screen = vision.get("nutrient_screening")
                        if nut_screen:
                            with st.expander("🌱 Nutrient Screening Hypothesis (Mobility & Soil pH)"):
                                ph_info = nut_screen.get("ph_evaluation", {})
                                st.caption(f"• **Soil pH Status**: {ph_info.get('message_en', 'Normal')}")
                                cands = nut_screen.get("candidate_deficiencies", [])
                                for c in cands[:2]:
                                    st.caption(f"• **{c['name']}**: {c['visual_symptoms'][:100]}... (Mobility: {c['mobility']})")
                                st.warning("⚠️ Visual screening hypothesis only. Confirm via certified soil and tissue tests.")

                        # Safe Development Debug Panel (Requirement 7)
                        if dbg:
                            with st.expander("🛠️ Development Debug Panel (Vision Decision Telemetry)", expanded=False):
                                st.markdown(f"**Exact Decision Rule**: `{dbg.get('decision_rule', 'N/A')}`")
                                st.markdown(f"**Final Decision State**: `{dbg.get('final_decision', 'N/A')}`")

                                st.markdown(
                                    f"• **Stage 1 (Validity)**: `{'Valid Plant' if dbg.get('stage1_valid', True) else 'Refused'}` "
                                    f"(Sharpness: `{dbg.get('blur_metric', 0.0):.1f}`, DINOv2: `{dbg.get('dinov2_similarity', 0.0):.4f}`)\n"
                                    f"• **Stage 2 (Plant Part)**: `{dbg.get('stage2_plant_part', vision.get('plant_part', 'N/A'))}`\n"
                                    f"• **Stage 3 (Crop Species)**: `{dbg.get('stage3_crop_species', vision.get('crop_species', 'N/A'))}` "
                                    f"(Confidence: `{dbg.get('stage3_crop_confidence_pct', 0.0)}%`)\n"
                                    f"• **Stage 4 (Route Coverage)**: `{'SUPPORTED & DEPLOYED' if dbg.get('stage4_route_supported', False) else 'UNSUPPORTED / NOT DEPLOYED'}`"
                                )

                                d_col1, d_col2 = st.columns(2)
                                with d_col1:
                                    st.caption(f"• **Raw Shannon Entropy H(p)**: `{dbg.get('raw_entropy', dbg.get('entropy', 0.0)):.4f}`")
                                    st.caption(f"• **Normalized Entropy Ĥ(p)**: `{dbg.get('norm_entropy', 0.0):.4f}`")
                                    st.caption(f"• **Crop Subspace Classes**: `{dbg.get('class_count', 38)}`")
                                    st.caption(f"• **Applied Entropy Threshold**: `{dbg.get('entropy_threshold', 2.20):.2f}`")
                                with d_col2:
                                    st.caption(f"• **Top-1 Probability**: `{dbg.get('top1_confidence_pct', 0.0):.1f}%` ({dbg.get('predicted_class', 'N/A')})")
                                    st.caption(f"• **DINOv2 Plant Similarity**: `{dbg.get('dinov2_similarity', 0.0):.4f}` (Threshold: `{dbg.get('dinov2_threshold', 0.16):.2f}`)")
                                    st.caption(f"• **Focus Sharpness (Laplacian)**: `{dbg.get('blur_metric', 0.0):.1f}` (Threshold: `{dbg.get('blur_threshold', 40.0):.1f}`)")
                                    st.caption(f"• **OOD Gate Result**: `{'OOD (Non-Plant)' if dbg.get('is_ood') else 'In-Distribution (Plant Recognized)'}`")

                                top3 = dbg.get("top3_probabilities", dbg.get("top_probabilities", []))
                                if top3:
                                    st.markdown("**Top-3 Probabilities:**")
                                    for t in top3:
                                        if not isinstance(t, dict):
                                            continue
                                        label = t.get("class") or t.get("label") or "Unknown candidate"
                                        raw_prob = t.get("prob") if t.get("prob") is not None else t.get("probability")

                                        try:
                                            prob = float(raw_prob)
                                        except (TypeError, ValueError):
                                            prob = None

                                        if prob is not None and 0.0 <= prob <= 1.0:
                                            st.caption(f"- `{label}`: **{prob * 100:.2f}%**")
                                        else:
                                            st.caption(f"- `{label}`: confidence unavailable")

                                q_preds = dbg.get("quadrant_predictions", [])
                                if q_preds:
                                    st.markdown("**Quadrant Predictions:**")
                                    for q in q_preds:
                                        st.caption(f"  - **{q.get('quadrant')}**: {q.get('predicted')} ({q.get('confidence_pct')}%)")

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
    samp_cols = st.columns(6)
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
    if samp_cols[3].button("🔍 Blurry Leaf", key="btn_blurry"):
        p = config.DATA_DIR / "processed" / "eval_samples" / "sample_blurred_leaf.jpg"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Please analyze this blurred plant photograph."
            sample_image_trigger = True
    if samp_cols[4].button("🐱 Non-Plant (Cat)", key="btn_cat"):
        p = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
        if p.exists():
            sample_image_bytes = open(p, "rb").read()
            sample_image_label = "Please analyze this image."
            sample_image_trigger = True
    if samp_cols[5].button("⚠️ Noise (OOD)", key="btn_ood"):
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
chat_image_bytes = None
if chat_files:
    chat_image_bytes = chat_files[0].getvalue() if hasattr(chat_files[0], "getvalue") else chat_files[0].read()

# CHANGED: Check fallback attachment sources (Quick Test samples or expander file uploader)
active_img_bytes = None
if chat_image_bytes:
    active_img_bytes = chat_image_bytes
elif sample_image_trigger and sample_image_bytes:
    active_img_bytes = sample_image_bytes
    if not user_text and sample_image_label:
        user_text = sample_image_label
elif uploaded_file is not None:
    active_img_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

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
    if symptom_description and symptom_description.strip():
        internal_query += f" (Observed Symptoms: {symptom_description.strip()})"
    display_text = user_text  # Clean display in chat bubble (blank if image-only)

    # Attach image thumbnail only if a NEW image is uploaded in this turn
    img_for_turn = active_img_bytes if is_new_image else None
    if img_for_turn:
        raw_pil = Image.open(io.BytesIO(img_for_turn))
        pil_image = ImageOps.exif_transpose(raw_pil)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
    else:
        pil_image = None

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
            dinov2_threshold=ood_threshold_val,
            crop_hint=confirmed_crop,
            part_hint=confirmed_part,
            growth_stage=confirmed_stage,
            symptom_duration=confirmed_duration,
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
