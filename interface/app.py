"""
Crop Intelligence System — Vibrant, Responsive Agritech Streamlit Interface.

Features:
- Clean, high-contrast header ("Crop Intelligence System") with top-right 3-dot (⋮)
  NPK & soil pH configuration popover.
- Responsive centered layout (max-width: 820px) tailored for both laptop and mobile.
- High-contrast assistant cards (#1E293B on #F0FDF4/#FFFFFF with #10B981 border)
  and user bubbles (#FFFFFF on emerald gradient).
- Fixed bottom chat input bar with embedded `+` attachment & drag-and-drop image upload,
  free of dark-mode clashes or double-box artifacts.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.orchestrator.router import CropRouter
from src.rag.kb_retriever import KnowledgeBaseRetriever

st.set_page_config(
    page_title="Crop Intelligence System",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# High-contrast, responsive agritech CSS overriding dark/light browser defaults
st.markdown(
    """
    <style>
    :root {
        --cis-bg: #F8FAFC;
        --cis-surface: #FFFFFF;
        --cis-mint: #F0FDF4;
        --cis-emerald: #10B981;
        --cis-emerald-dark: #047857;
        --cis-teal: #0D9488;
        --cis-amber: #F59E0B;
        --cis-amber-bg: #FEF3C7;
        --cis-amber-text: #92400E;
        --cis-text: #1E293B;
        --cis-muted: #475569;
        --cis-border: #CBD5E1;
    }

    /* Hide sidebar control and footer while keeping header clean and light */
    [data-testid="collapsedControl"],
    [data-testid="stSidebar"],
    footer {
        display: none !important;
    }

    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: linear-gradient(180deg, #ECFDF5 0%, #F8FAFC 26%, #F8FAFC 100%) !important;
        color: #1E293B !important;
        overflow-x: hidden !important;
    }

    /* Remove harsh black top bar; blend cleanly with botanical background */
    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.88) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(16, 185, 129, 0.14) !important;
        color: #1E293B !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] svg {
        color: #334155 !important;
        fill: #334155 !important;
    }

    /* Responsive centered main container */
    .block-container,
    [data-testid="stMainBlockContainer"] {
        max-width: 820px !important;
        width: 100% !important;
        padding: 3.6rem 1rem 7.5rem 1rem !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }

    /* Force legible text colors globally so dark-mode overrides never hide text */
    .stApp p,
    .stApp li,
    .stApp span,
    .stApp label,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong {
        color: #1E293B;
    }

    /* Keep header horizontal row side-by-side on all viewports */
    [data-testid="stHorizontalBlock"]:has(.cis-header-anchor) {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: space-between !important;
        flex-wrap: nowrap !important;
        gap: 0.75rem !important;
        background: #FFFFFF !important;
        border: 1px solid #D1FAE5 !important;
        border-radius: 16px !important;
        padding: 0.85rem 1.1rem !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08) !important;
        margin-bottom: 0.75rem !important;
    }

    [data-testid="stHorizontalBlock"]:has(.cis-header-anchor) > div:first-child {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }

    [data-testid="stHorizontalBlock"]:has(.cis-header-anchor) > div:last-child {
        flex: 0 0 auto !important;
        width: auto !important;
        display: flex !important;
        justify-content: flex-end !important;
    }

    .cis-title {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #064E3B !important;
        margin: 0;
        line-height: 1.25;
    }

    .cis-subtitle {
        font-size: 0.8rem;
        color: #475569 !important;
        margin-top: 0.2rem;
        line-height: 1.35;
    }

    .cis-npk-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.45rem;
    }

    .cis-amber-badge {
        display: inline-flex;
        align-items: center;
        background-color: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #F59E0B !important;
        border-radius: 999px;
        padding: 0.14rem 0.6rem;
        font-size: 0.74rem;
        font-weight: 600;
    }

    /* Top-right 3-dot (⋮) popover button */
    div[data-testid="stPopover"] > button {
        border-radius: 999px !important;
        border: 1px solid #10B981 !important;
        background-color: #ECFDF5 !important;
        color: #065F46 !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        min-height: 2.4rem !important;
        min-width: 2.4rem !important;
        padding: 0 0.65rem !important;
        box-shadow: 0 1px 4px rgba(16, 185, 129, 0.15) !important;
        transition: all 0.15s ease !important;
    }

    div[data-testid="stPopover"] > button:hover {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stPopover"] > button p,
    div[data-testid="stPopover"] > button span {
        color: inherit !important;
    }

    /* Popover body contrast & inputs */
    [data-testid="stPopoverBody"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #D1FAE5 !important;
        border-radius: 16px !important;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12) !important;
    }

    [data-testid="stPopoverBody"] label,
    [data-testid="stPopoverBody"] p,
    [data-testid="stPopoverBody"] h4,
    [data-testid="stPopoverBody"] span {
        color: #1E293B !important;
    }

    [data-testid="stPopoverBody"] input {
        background-color: #F8FAFC !important;
        color: #1E293B !important;
        border-color: #CBD5E1 !important;
    }

    /* Welcome Hero Card */
    .cis-welcome {
        text-align: center;
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%);
        border: 1px solid #A7F3D0;
        border-radius: 16px;
        padding: 2rem 1.25rem;
        margin: 0.75rem 0 1.1rem 0;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.08);
    }

    .cis-welcome-pill {
        display: inline-block;
        background-color: #D1FAE5;
        color: #065F46 !important;
        border: 1px solid #6EE7B7;
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }

    .cis-welcome h2 {
        font-size: 1.85rem;
        font-weight: 700;
        color: #064E3B !important;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }

    .cis-welcome p {
        color: #334155 !important;
        font-size: 0.96rem;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Primary & Quick-Action Buttons */
    .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        padding: 0.55rem 1rem !important;
        min-height: 2.65rem !important;
        transition: transform 0.12s ease, box-shadow 0.15s ease, background-color 0.15s ease !important;
    }

    .stButton > button[kind="primary"] {
        background-color: #10B981 !important;
        border: 1px solid #059669 !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #059669 !important;
        color: #FFFFFF !important;
    }

    .stButton > button[kind="primary"] p {
        color: #FFFFFF !important;
    }

    /* 3 Vibrant Quick-Action Prompt Chips */
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) {
        gap: 0.65rem !important;
        margin-bottom: 1rem !important;
    }

    /* Chip 1: Emerald */
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(1) .stButton > button {
        background: linear-gradient(135deg, #ECFDF5, #D1FAE5) !important;
        border: 1.5px solid #10B981 !important;
        color: #065F46 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(1) .stButton > button p {
        color: #065F46 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(1) .stButton > button:hover {
        background: #10B981 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(1) .stButton > button:hover p {
        color: #FFFFFF !important;
    }

    /* Chip 2: Teal */
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(2) .stButton > button {
        background: linear-gradient(135deg, #F0FDFA, #CCFBF1) !important;
        border: 1.5px solid #0D9488 !important;
        color: #115E59 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(2) .stButton > button p {
        color: #115E59 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(2) .stButton > button:hover {
        background: #0D9488 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(2) .stButton > button:hover p {
        color: #FFFFFF !important;
    }

    /* Chip 3: Amber */
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(3) .stButton > button {
        background: linear-gradient(135deg, #FFFBEB, #FEF3C7) !important;
        border: 1.5px solid #F59E0B !important;
        color: #92400E !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(3) .stButton > button p {
        color: #92400E !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(3) .stButton > button:hover {
        background: #F59E0B !important;
        color: #FFFFFF !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div:nth-child(3) .stButton > button:hover p {
        color: #FFFFFF !important;
    }

    /* Chat Message Containers */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.4rem 0 !important;
        border: none !important;
    }

    /* Assistant Message Card: Dark-slate text on mint/off-white card with emerald border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    .cis-assistant-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F0FDF4 100%) !important;
        border: 1.5px solid #10B981 !important;
        border-radius: 16px !important;
        padding: 1rem 1.15rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 3px 12px rgba(16, 185, 129, 0.08) !important;
        color: #1E293B !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) *,
    .cis-assistant-card * {
        color: #1E293B !important;
    }

    .cis-assistant-meta {
        margin-top: 0.65rem;
        padding-top: 0.5rem;
        border-top: 1px solid #D1FAE5;
        font-size: 0.76rem;
        color: #065F46 !important;
        font-weight: 500;
    }

    /* User Message Bubble: Crisp white text on rich forest/emerald gradient */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    .cis-user-bubble {
        background: linear-gradient(135deg, #059669, #047857) !important;
        border: 1px solid #065F46 !important;
        border-radius: 16px !important;
        padding: 0.9rem 1.15rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 12px rgba(4, 120, 87, 0.18) !important;
        color: #FFFFFF !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) strong,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div,
    .cis-user-bubble,
    .cis-user-bubble p,
    .cis-user-bubble span {
        color: #FFFFFF !important;
    }

    /* Responsive uploaded image thumbnails */
    [data-testid="stImage"] img {
        max-height: 260px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: cover !important;
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.45) !important;
    }

    /* Remove harsh black bottom bar & eliminate double-box artifacts */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: rgba(248, 250, 252, 0.94) !important;
        backdrop-filter: blur(10px) !important;
        border-top: 1px solid #E2E8F0 !important;
        box-shadow: none !important;
    }

    [data-testid="stBottom"] [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #10B981 !important;
        border-radius: 26px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08) !important;
        padding: 0.25rem 0.5rem !important;
        max-width: 820px !important;
        margin: 0 auto !important;
    }

    /* Strip inner wrapper borders/backgrounds inside stChatInput to prevent double boxes */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInputTextArea"] {
        background: transparent !important;
        color: #1E293B !important;
        caret-color: #059669 !important;
        -webkit-text-fill-color: #1E293B !important;
    }

    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInputTextArea"]::placeholder {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #ECFDF5 !important;
        color: #059669 !important;
        border-radius: 999px !important;
        border: 1px solid #A7F3D0 !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #10B981 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stChatInput"] button svg {
        fill: currentColor !important;
        color: inherit !important;
    }

    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .block-container,
        [data-testid="stMainBlockContainer"] {
            padding: 3.15rem 0.75rem 7rem 0.75rem !important;
        }

        [data-testid="stHorizontalBlock"]:has(.cis-header-anchor) {
            padding: 0.7rem 0.85rem !important;
        }

        .cis-title {
            font-size: 1.15rem !important;
        }

        .cis-subtitle {
            font-size: 0.72rem !important;
        }

        .cis-welcome {
            padding: 1.35rem 0.9rem !important;
        }

        .cis-welcome h2 {
            font-size: 1.45rem !important;
        }

        /* Stack quick-action prompt chips cleanly on mobile */
        [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) {
            flex-direction: column !important;
            gap: 0.5rem !important;
        }

        [data-testid="stHorizontalBlock"]:has(.cis-chips-anchor) > div {
            width: 100% !important;
            flex: 1 1 100% !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            padding: 0.8rem 0.9rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_router() -> CropRouter:
    """Initialize and cache the direct Gemini API router."""
    return CropRouter()


def _init_session_state() -> None:
    """Initialize chat messages and soil NPK/pH defaults."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "soil_n" not in st.session_state:
        st.session_state.soil_n = 90.0
    if "soil_p" not in st.session_state:
        st.session_state.soil_p = 42.0
    if "soil_k" not in st.session_state:
        st.session_state.soil_k = 43.0
    if "soil_ph" not in st.session_state:
        st.session_state.soil_ph = 6.5
    if "crop_hint" not in st.session_state:
        st.session_state.crop_hint = ""
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def _render_header() -> None:
    """Render the clean header and top-right 3-dot (⋮) NPK/soil settings popover."""
    col_title, col_menu = st.columns([0.88, 0.12], vertical_alignment="center")
    with col_title:
        n_val = int(round(st.session_state.soil_n))
        p_val = int(round(st.session_state.soil_p))
        k_val = int(round(st.session_state.soil_k))
        ph_val = round(float(st.session_state.soil_ph), 1)
        st.markdown(
            f"""
            <div class="cis-header-anchor">
                <div class="cis-title">🌱 Crop Intelligence System</div>
                <div class="cis-subtitle">
                    Gemini 2.5 Flash · PlantVillage (Corn) · FGVC8 (Apple) · Cassava · ICAR · IIHR
                </div>
                <div class="cis-npk-strip">
                    <span class="cis-amber-badge">N: {n_val}</span>
                    <span class="cis-amber-badge">P: {p_val}</span>
                    <span class="cis-amber-badge">K: {k_val}</span>
                    <span class="cis-amber-badge">pH: {ph_val}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_menu:
        with st.popover("⋮", help="Soil & NPK Configuration"):
            st.markdown("#### Soil & NPK Parameters")
            st.caption("Used to ground diagnoses and agronomic advisories.")

            st.session_state.soil_n = st.number_input(
                "Nitrogen (N) kg/ha",
                min_value=0.0,
                max_value=400.0,
                value=float(st.session_state.soil_n),
                step=5.0,
            )
            st.session_state.soil_p = st.number_input(
                "Phosphorus (P) kg/ha",
                min_value=0.0,
                max_value=300.0,
                value=float(st.session_state.soil_p),
                step=5.0,
            )
            st.session_state.soil_k = st.number_input(
                "Potassium (K) kg/ha",
                min_value=0.0,
                max_value=400.0,
                value=float(st.session_state.soil_k),
                step=5.0,
            )
            st.session_state.soil_ph = st.slider(
                "Soil pH",
                min_value=3.5,
                max_value=9.5,
                value=float(st.session_state.soil_ph),
                step=0.1,
            )
            st.session_state.crop_hint = st.text_input(
                "Target Crop (optional)",
                value=st.session_state.crop_hint,
                placeholder="e.g., Corn, Apple, Cassava, Tomato",
            )

            advisory = KnowledgeBaseRetriever.get_soil_advisory(
                n=st.session_state.soil_n,
                p=st.session_state.soil_p,
                k=st.session_state.soil_k,
                ph=st.session_state.soil_ph,
            )
            st.markdown(
                (
                    f'<div class="cis-npk-strip">'
                    f'<span class="cis-amber-badge">N: {advisory["n_status"]}</span>'
                    f'<span class="cis-amber-badge">P: {advisory["p_status"]}</span>'
                    f'<span class="cis-amber-badge">K: {advisory["k_status"]}</span>'
                    f'<span class="cis-amber-badge">pH: {advisory["ph_status"]}</span>'
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

            if st.button("Clear Chat", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


def _render_welcome() -> None:
    """Render vibrant, responsive welcome card and 3 quick-action prompt chips."""
    st.markdown(
        """
        <div class="cis-welcome">
            <div class="cis-welcome-pill">5-Point Agronomic Vision &amp; Soil Advisory</div>
            <h2>How can I help your crop today?</h2>
            <p>
                Attach or drag-and-drop a crop photo below for an instant 5-point diagnosis,
                or tap a quick prompt to explore ICAR &amp; IIHR crop and soil guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<span class="cis-chips-anchor"></span>', unsafe_allow_html=True)
        if st.button("🌽 Corn Gray Leaf Spot", use_container_width=True):
            st.session_state.pending_prompt = "Diagnose rectangular gray lesions on corn leaves and advise treatment."
    with c2:
        if st.button("🍎 Apple Scab Plan", use_container_width=True):
            st.session_state.pending_prompt = "Diagnose olive-green velvety scab spots on apple leaves."
    with c3:
        if st.button("🧪 Evaluate Soil NPK & pH", use_container_width=True):
            st.session_state.pending_prompt = "Evaluate my current soil NPK and pH parameters and suggest improvements."


def _render_assistant_message(content: str, caption: Optional[str] = None) -> None:
    """Render assistant response inside a high-contrast mint/slate card."""
    with st.chat_message("assistant"):
        st.markdown(content)
        if caption:
            safe_caption = html.escape(caption)
            st.markdown(f'<div class="cis-assistant-meta">{safe_caption}</div>', unsafe_allow_html=True)


def _render_user_message(content: str, image_bytes: Optional[bytes] = None) -> None:
    """Render user prompt and optional image preview inside an emerald bubble."""
    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes)
        st.markdown(content)


def _process_turn(prompt_text: str, image_bytes: Optional[bytes]) -> None:
    """Execute routing via Gemini 2.5 Flash + ICAR/IIHR KB and append messages."""
    display_text = prompt_text.strip() or "Analyze the attached plant image."
    st.session_state.messages.append(
        {
            "role": "user",
            "content": display_text,
            "image_bytes": image_bytes,
        }
    )
    _render_user_message(display_text, image_bytes)

    router = get_router()
    soil_params: Dict[str, float] = {
        "n": float(st.session_state.soil_n),
        "p": float(st.session_state.soil_p),
        "k": float(st.session_state.soil_k),
        "ph": float(st.session_state.soil_ph),
    }

    with st.chat_message("assistant"):
        with st.spinner("Analyzing with Gemini 2.5 Flash & ICAR/IIHR Knowledge Base..."):
            result: Dict[str, Any] = router.route(
                query=display_text,
                image=image_bytes,
                soil_params=soil_params,
                crop_hint=st.session_state.crop_hint.strip() or None,
            )

        datasets_used = ", ".join(
            dict.fromkeys(m["dataset"] for m in result.get("kb_matches", []) if "dataset" in m)
        )
        mode_label = "Live Gemini 2.5 Flash" if result.get("used_live_api") else "Grounded KB Fallback"
        caption_parts = [f"Engine: {mode_label}"]
        if datasets_used:
            caption_parts.append(f"Sources: {datasets_used}")
        caption_text = " · ".join(caption_parts)

        st.markdown(result["response"])
        st.markdown(
            f'<div class="cis-assistant-meta">{html.escape(caption_text)}</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "caption": caption_text,
        }
    )


def main() -> None:
    """Render the Crop Intelligence System conversational UI."""
    _init_session_state()
    _render_header()

    if not st.session_state.messages and not st.session_state.pending_prompt:
        _render_welcome()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user_message(msg["content"], msg.get("image_bytes"))
        else:
            _render_assistant_message(msg["content"], msg.get("caption"))

    # Fixed bottom chat input with embedded + attachment button & drag-and-drop support
    submission = st.chat_input(
        "Ask about crop diseases, NPK fertility, or attach (+) / drop a plant photo...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp"],
    )

    if st.session_state.pending_prompt:
        queued = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        _process_turn(queued, None)
        st.rerun()

    if submission is not None:
        text_val = getattr(submission, "text", "") if not isinstance(submission, str) else submission
        files = getattr(submission, "files", []) if not isinstance(submission, str) else []
        image_bytes = files[0].getvalue() if files else None
        if (text_val and text_val.strip()) or image_bytes is not None:
            _process_turn(text_val or "", image_bytes)


if __name__ == "__main__":
    main()
