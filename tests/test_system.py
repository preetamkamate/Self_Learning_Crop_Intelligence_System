"""
Pytest verification suite for the Zero-to-N Crop Intelligence System.

Verifies:
- Zero local `.pt`/`.pth` checkpoint dependencies across `src/` and `interface/`.
- 5-dataset ICAR/IIHR knowledge base coverage and soil NPK/pH advisory.
- In-memory multimodal image encoding and strict 5-point diagnostic formatting.
- Direct Gemini 2.5 Flash routing (with mocked client + offline resilience).
- Clean Streamlit UI structure without personal or semester branding.
"""
from __future__ import annotations

import ast
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.modular_vision.pipeline import (
    REQUIRED_SECTIONS,
    FivePointResponseFormatter,
    ModularVisionPipeline,
    MultimodalImageEncoder,
)
from src.orchestrator.router import CropRouter
from src.rag.kb_retriever import SUPPORTED_DATASETS, KnowledgeBaseRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent


def _make_sample_image(size: tuple[int, int] = (320, 240), mode: str = "RGB") -> Image.Image:
    """Create a synthetic in-memory leaf-colored test image."""
    color = (64, 140, 72, 255) if mode == "RGBA" else (64, 140, 72)
    return Image.new(mode, size, color=color)


def test_no_local_weight_checkpoints_or_torch_imports():
    """Ensure the codebase contains zero .pt/.pth checkpoints and zero torch imports."""
    excluded_parts = {".venv", "venv", ".git", "__pycache__"}
    checkpoint_files = [
        p
        for ext in ("*.pt", "*.pth")
        for p in ROOT_DIR.rglob(ext)
        if not any(part in excluded_parts for part in p.parts)
    ]
    assert checkpoint_files == [], f"Unexpected checkpoint files found: {checkpoint_files}"

    for folder in ("src", "interface"):
        for py_file in (ROOT_DIR / folder).rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            assert "import torch" not in source
            assert "from torch" not in source


def test_kb_retriever_covers_all_five_datasets():
    """Verify all 5 required taxonomies exist and return grounded matches."""
    retriever = KnowledgeBaseRetriever()
    datasets = retriever.list_datasets()
    assert datasets == [
        "PlantVillage (Corn)",
        "FGVC8 (Apple)",
        "Cassava Leaf Disease",
        "ICAR Research",
        "IIHR Advisory",
    ]
    assert tuple(datasets) == SUPPORTED_DATASETS

    for ds in datasets:
        entries = retriever.get_entries_by_dataset(ds)
        assert len(entries) >= 1, f"Missing entries for dataset: {ds}"

    corn_hits = retriever.retrieve("rectangular gray lesions on corn leaf")
    assert corn_hits[0]["dataset"] == "PlantVillage (Corn)"

    apple_hits = retriever.retrieve("olive velvety apple scab spot")
    assert apple_hits[0]["dataset"] == "FGVC8 (Apple)"

    cassava_hits = retriever.retrieve("cassava mosaic yellow mottling whitefly")
    assert cassava_hits[0]["dataset"] == "Cassava Leaf Disease"

    df = retriever.to_dataframe()
    assert not df.empty
    assert set(datasets).issubset(set(df["dataset"].unique()))


def test_soil_advisory_classification():
    """Verify ICAR soil NPK and pH threshold evaluations."""
    low_acidic = KnowledgeBaseRetriever.get_soil_advisory(n=20, p=10, k=15, ph=4.8)
    assert low_acidic["n_status"] == "Low"
    assert low_acidic["p_status"] == "Low"
    assert low_acidic["k_status"] == "Low"
    assert low_acidic["ph_status"] == "Acidic"
    assert len(low_acidic["recommendations"]) >= 3

    optimal = KnowledgeBaseRetriever.get_soil_advisory(n=90, p=42, k=43, ph=6.5)
    assert optimal["n_status"] == "Optimal"
    assert optimal["p_status"] == "Optimal"
    assert optimal["k_status"] == "Optimal"
    assert optimal["ph_status"] == "Neutral / Optimal"


def test_multimodal_image_encoder_normalizes_and_resizes():
    """Verify RGBA conversion, large-image downscaling, and Gemini Part creation."""
    encoder = MultimodalImageEncoder(max_dimension=512)
    large_rgba = _make_sample_image(size=(1600, 800), mode="RGBA")
    buf = io.BytesIO()
    large_rgba.save(buf, format="PNG")
    buf.seek(0)

    encoded = encoder.encode(buf)
    assert encoded.mime_type == "image/jpeg"
    assert max(encoded.width, encoded.height) == 512
    assert len(encoded.data) > 0
    part = encoded.to_genai_part()
    assert part is not None


def test_five_point_formatter_strict_enforcement():
    """Verify formatter enforces all 5 exact headings in order for arbitrary inputs."""
    partial_raw = (
        "1. Diagnosis: Northern Corn Leaf Blight\n"
        "3. Chemical Solution: Propiconazole 25% EC (1 mL/L)"
    )
    formatted = FivePointResponseFormatter.enforce_five_point_format(partial_raw)
    assert FivePointResponseFormatter.validate_five_point_format(formatted)

    for idx, heading in enumerate(REQUIRED_SECTIONS, start=1):
        assert f"{idx}. **{heading}:**" in formatted

    json_raw = (
        '{"Diagnosis": "Apple Scab", "Reason": "Venturia inaequalis", '
        '"Chemical Solution": "Captan 50% WP", "Organic Solution": "Wettable sulfur", '
        '"What to do next": "Remove fallen leaves."}'
    )
    from_json = FivePointResponseFormatter.enforce_five_point_format(json_raw)
    assert FivePointResponseFormatter.validate_five_point_format(from_json)
    assert "Apple Scab" in from_json


def test_router_multimodal_gemini_25_flash_with_mock_client():
    """Verify multimodal image routing calls gemini-2.5-flash and returns 5 points."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text=(
            "1. **Diagnosis:** PlantVillage (Corn) — Common Rust\n"
            "2. **Reason:** Puccinia sorghi windborne fungal spores under humid conditions.\n"
            "3. **Chemical Solution:** Mancozeb 75% WP at 2.5 g/L.\n"
            "4. **Organic Solution:** Wettable sulfur 2 g/L foliar spray.\n"
            "5. **What to do next:** Scout upper canopy every 5 days."
        )
    )

    router = CropRouter(client=mock_client, model_name="gemini-2.5-flash")
    result = router.route(
        query="Identify this corn leaf disease",
        image=_make_sample_image(),
        soil_params={"n": 80, "p": 40, "k": 45, "ph": 6.4},
    )

    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert len(call_kwargs["contents"]) == 2
    assert result["used_live_api"] is True
    assert result["mode"] == "multimodal_diagnosis"
    assert result["is_five_point"] is True
    assert FivePointResponseFormatter.validate_five_point_format(result["response"])


def test_router_offline_fallback_without_api_key(monkeypatch: pytest.MonkeyPatch):
    """Verify graceful 5-point fallback when GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    router = CropRouter(client=None, model_name="gemini-2.5-flash")

    img_result = router.route(
        query="Check cassava leaf mosaic symptoms",
        image=_make_sample_image(),
    )
    assert img_result["used_live_api"] is False
    assert img_result["is_five_point"] is True
    assert FivePointResponseFormatter.validate_five_point_format(img_result["response"])

    text_diag = router.route(query="Diagnose tomato early blight spots")
    assert text_diag["mode"] == "text_diagnosis"
    assert text_diag["is_five_point"] is True
    assert FivePointResponseFormatter.validate_five_point_format(text_diag["response"])


def test_streamlit_ui_structure_and_clean_branding():
    """Verify interface/app.py syntax, required UI controls, and clean branding."""
    app_path = ROOT_DIR / "interface" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    ast.parse(source)
    assert "Crop Intelligence System" in source
    assert "accept_file=True" in source
    assert 'st.popover("⋮"' in source

    forbidden_terms = ("semester", "b.e. project", "lead ai/ml", "preetam", "student")
    lowered = source.lower()
    for term in forbidden_terms:
        assert term not in lowered, f"Forbidden branding term found in UI: {term}"
