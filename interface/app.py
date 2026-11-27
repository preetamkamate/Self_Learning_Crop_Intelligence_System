"""
Crop Intelligence System — Minimalist Gemini-style Streamlit Interface.

Features:
- Clean header ("Crop Intelligence System") with top-right 3-dot (⋮) NPK & soil settings menu.
- Centered conversational stream displaying image previews and strict 5-point diagnoses.
- Fixed bottom chat input bar with embedded `+` attachment & drag-and-drop image upload.
"""
from __future__ import annotations

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

# Minimalist Gemini-inspired styling
st.markdown(
    """
    <style>
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="collapsedControl"],
    footer {
        display: none !important;
    }

    .stApp {
        background-color: #ffffff;
        color: #1f1f1f;
    }

    .block-container {
        max-width: 820px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 6.5rem !important;
    }

    .cis-title {
        font-size: 1.45rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1a1c1e;
        margin: 0;
        line-height: 1.25;
    }

    .cis-subtitle {
        font-size: 0.78rem;
        color: #5f6368;
        margin-top: 0.15rem;
    }

    .cis-welcome {
        text-align: center;
        padding: 3.2rem 1rem 1.8rem 1rem;
    }

    .cis-welcome h2 {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #1b5e20, #2e7d32, #0277bd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .cis-welcome p {
        color: #5f6368;
        font-size: 0.98rem;
        max-width: 540px;
        margin: 0 auto;
    }

    /* Sleek bottom-locked Gemini-style chat bar */
    [data-testid="stBottom"] {
        background: linear-gradient(to top, #ffffff 80%, rgba(255, 255, 255, 0));
        padding-top: 0.75rem;
        padding-bottom: 1rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1px solid #dadce0 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06) !important;
        background-color: #f8fafd !important;
    }

    /* Top-right 3-dot settings button polish */
    div[data-testid="stPopover"] > button {
        border-radius: 999px !important;
        border: 1px solid #e0e3e7 !important;
        background-color: #f8fafd !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        min-height: 2.35rem !important;
        padding: 0 0.7rem !important;
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
    col_title, col_menu = st.columns([0.90, 0.10], vertical_alignment="center")
    with col_title:
        st.markdown(
            """
            <div>
                <div class="cis-title">Crop Intelligence System</div>
                <div class="cis-subtitle">
                    Gemini 2.5 Flash · PlantVillage (Corn) · FGVC8 (Apple) · Cassava · ICAR · IIHR
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
            st.caption(
                f"Status — N: **{advisory['n_status']}** | P: **{advisory['p_status']}** | "
                f"K: **{advisory['k_status']}** | pH: **{advisory['ph_status']}**"
            )

            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()


def _render_welcome() -> None:
    """Render minimalist Gemini-style welcome state and starter prompts."""
    st.markdown(
        """
        <div class="cis-welcome">
            <h2>Crop Intelligence System</h2>
            <p>
                Attach or drag-and-drop a crop photo below for an instant 5-point diagnosis,
                or ask about crop diseases, ICAR/IIHR advisories, and soil NPK balance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Corn Gray Leaf Spot treatment", use_container_width=True):
            st.session_state.pending_prompt = "Diagnose rectangular gray lesions on corn leaves and advise treatment."
    with c2:
        if st.button("Apple Scab organic & chemical plan", use_container_width=True):
            st.session_state.pending_prompt = "Diagnose olive-green velvety scab spots on apple leaves."
    with c3:
        if st.button("Evaluate my Soil NPK & pH", use_container_width=True):
            st.session_state.pending_prompt = "Evaluate my current soil NPK and pH parameters and suggest improvements."


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

    with st.chat_message("user"):
        if image_bytes:
            st.image(image_bytes, width=260)
        st.markdown(display_text)

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

        st.markdown(result["response"])
        datasets_used = ", ".join(
            dict.fromkeys(m["dataset"] for m in result.get("kb_matches", []) if "dataset" in m)
        )
        mode_label = "Live Gemini 2.5 Flash" if result.get("used_live_api") else "Grounded KB Fallback"
        caption_parts = [f"Engine: {mode_label}"]
        if datasets_used:
            caption_parts.append(f"Sources: {datasets_used}")
        st.caption(" · ".join(caption_parts))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["response"],
            "caption": " · ".join(caption_parts),
        }
    )


def main() -> None:
    """Render the Crop Intelligence System conversational UI."""
    _init_session_state()
    _render_header()
    st.divider()

    if not st.session_state.messages and not st.session_state.pending_prompt:
        _render_welcome()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], width=260)
            st.markdown(msg["content"])
            if msg.get("caption"):
                st.caption(msg["caption"])

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
