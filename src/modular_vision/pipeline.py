"""
Multimodal Vision Pipeline & 5-Point Response Formatter.

Zero local `.pt`/`.pth` weight bottlenecks: normalizes uploaded crop images in-memory
and sends multimodal payloads directly to Google Gemini (`gemini-2.5-flash`), enforcing
a strict 5-point diagnostic output format.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai import types
from PIL import Image, ImageOps

from src.rag.kb_retriever import SUPPORTED_DATASETS

REQUIRED_SECTIONS: Tuple[str, ...] = (
    "Diagnosis",
    "Reason",
    "Chemical Solution",
    "Organic Solution",
    "What to do next",
)

DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class EncodedImage:
    """Normalized image payload ready for multimodal Gemini inference."""

    data: bytes
    mime_type: str
    width: int
    height: int
    sha256: str

    def to_genai_part(self) -> types.Part:
        """Convert normalized image bytes into a Google GenAI multimodal Part."""
        return types.Part.from_bytes(data=self.data, mime_type=self.mime_type)

    def metadata(self) -> Dict[str, Any]:
        """Return lightweight non-binary metadata for logging and UI badges."""
        return {
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "bytes": len(self.data),
            "sha256": self.sha256[:12],
        }


class MultimodalImageEncoder:
    """In-memory multimodal image encoder using Pillow (no local neural checkpoints)."""

    def __init__(self, max_dimension: int = 1024, jpeg_quality: int = 90):
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality

    def encode(
        self,
        image_input: Union[Image.Image, bytes, bytearray, io.BytesIO, str, Path],
    ) -> EncodedImage:
        """Load, orient, resize, and JPEG-encode an input image cleanly."""
        if isinstance(image_input, Image.Image):
            img = image_input.copy()
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(image_input)))
        elif isinstance(image_input, (str, Path)):
            img = Image.open(Path(image_input))
        elif hasattr(image_input, "read"):
            raw = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            img = Image.open(io.BytesIO(raw))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)!r}")

        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        if max(width, height) > self.max_dimension:
            img.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
            width, height = img.size

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        encoded_bytes = buffer.getvalue()
        digest = hashlib.sha256(encoded_bytes).hexdigest()

        return EncodedImage(
            data=encoded_bytes,
            mime_type="image/jpeg",
            width=width,
            height=height,
            sha256=digest,
        )


class FivePointResponseFormatter:
    """Enforces the strict 5-point diagnostic output structure."""

    SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
        "Diagnosis": ("diagnosis", "condition", "identification", "disease"),
        "Reason": ("reason", "cause", "causes", "etiology", "why"),
        "Chemical Solution": ("chemical solution", "chemical control", "chemical treatment", "chemical"),
        "Organic Solution": ("organic solution", "organic control", "biological solution", "organic"),
        "What to do next": ("what to do next", "next steps", "action plan", "preventive measures", "next"),
    }

    @classmethod
    def format_from_dict(cls, sections: Dict[str, str]) -> str:
        """Render a canonical numbered 5-point markdown block from a dictionary."""
        lines: List[str] = []
        for idx, heading in enumerate(REQUIRED_SECTIONS, start=1):
            body = (sections.get(heading) or "").strip()
            if not body:
                body = "Consult local ICAR/IIHR extension guidelines for field verification."
            lines.append(f"{idx}. **{heading}:** {body}")
        return "\n\n".join(lines)

    @classmethod
    def _match_heading(cls, line: str) -> Optional[Tuple[str, str]]:
        """Detect if a line begins with one of the 5 required headings."""
        cleaned = re.sub(r"^[#*\-\s\d.)]+", "", line).strip()
        cleaned_no_md = re.sub(r"\*{1,2}|_{1,2}", "", cleaned).strip()
        lowered = cleaned_no_md.lower()

        for canonical, aliases in cls.SECTION_ALIASES.items():
            for alias in aliases:
                pattern = rf"^{re.escape(alias)}\s*(?:[:\-–—]\s*(.*))?$"
                match = re.match(pattern, lowered, flags=re.IGNORECASE)
                if match:
                    # Extract original-cased trailing text after separator
                    orig_match = re.match(
                        rf"^{re.escape(alias)}\s*(?:[:\-–—]\s*(.*))?$",
                        cleaned_no_md,
                        flags=re.IGNORECASE,
                    )
                    remainder = orig_match.group(1).strip() if orig_match and orig_match.group(1) else ""
                    return canonical, remainder
        return None

    @classmethod
    def enforce_five_point_format(
        cls,
        raw_text: str,
        fallback_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Normalize raw LLM or KB text into the strict 5-point output format."""
        raw_text = (raw_text or "").strip()
        parsed: Dict[str, str] = {key: "" for key in REQUIRED_SECTIONS}

        # 1. Try JSON payload if Gemini returned structured JSON
        if raw_text.startswith("{") and raw_text.endswith("}"):
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    lower_map = {str(k).strip().lower(): str(v).strip() for k, v in data.items()}
                    for canonical, aliases in cls.SECTION_ALIASES.items():
                        for alias in aliases:
                            if alias in lower_map and lower_map[alias]:
                                parsed[canonical] = lower_map[alias]
                                break
                    if any(parsed.values()):
                        return cls._fill_and_render(parsed, raw_text, fallback_context)
            except json.JSONDecodeError:
                pass

        # 2. Parse line-by-line section headers
        current_section: Optional[str] = None
        buckets: Dict[str, List[str]] = {key: [] for key in REQUIRED_SECTIONS}

        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            detected = cls._match_heading(stripped)
            if detected:
                current_section, remainder = detected
                if remainder:
                    buckets[current_section].append(remainder)
            elif current_section:
                buckets[current_section].append(stripped)

        for key in REQUIRED_SECTIONS:
            parsed[key] = " ".join(buckets[key]).strip()

        return cls._fill_and_render(parsed, raw_text, fallback_context)

    @classmethod
    def _fill_and_render(
        cls,
        parsed: Dict[str, str],
        raw_text: str,
        fallback_context: Optional[Dict[str, Any]],
    ) -> str:
        """Fill any missing sections from grounded fallback context and render."""
        ctx = fallback_context or {}
        defaults = {
            "Diagnosis": ctx.get("condition")
            or (raw_text[:240] if raw_text and not any(parsed.values()) else "Crop condition requires closer inspection."),
            "Reason": ctx.get("pathogen_or_cause")
            or "Environmental stress, nutrient imbalance, or foliar pathogen pressure may be contributing.",
            "Chemical Solution": ctx.get("chemical_solution")
            or "Apply a targeted, label-approved fungicide or pesticide only after confirming threshold levels.",
            "Organic Solution": ctx.get("organic_solution")
            or "Apply Neem oil (Azadirachtin 1500 ppm at 3–5 mL/L) or Trichoderma/Pseudomonas bio-agents.",
            "What to do next": ctx.get("next_steps")
            or "Capture a clear close-up photo in natural light, inspect neighboring plants, and monitor for 5–7 days.",
        }
        finalized = {key: (parsed.get(key) or defaults[key]).strip() for key in REQUIRED_SECTIONS}
        return cls.format_from_dict(finalized)

    @classmethod
    def validate_five_point_format(cls, formatted_text: str) -> bool:
        """Return True if all 5 required headings appear in strict order with content."""
        if not formatted_text:
            return False
        positions: List[int] = []
        for idx, heading in enumerate(REQUIRED_SECTIONS, start=1):
            pattern = re.compile(
                rf"(?im)^\s*{idx}\.\s+\*{{0,2}}{re.escape(heading)}\*{{0,2}}\s*:\s*\S+",
            )
            match = pattern.search(formatted_text)
            if not match:
                return False
            positions.append(match.start())
        return positions == sorted(positions)


class ModularVisionPipeline:
    """Multimodal Gemini vision pipeline grounded on the 5 supported taxonomies."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        encoder: Optional[MultimodalImageEncoder] = None,
    ):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        self.encoder = encoder or MultimodalImageEncoder()
        self._client = client

    def _get_client(self) -> Optional[genai.Client]:
        """Lazily construct a Gemini client if GEMINI_API_KEY is available."""
        if self._client is not None:
            return self._client
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        self._client = genai.Client(api_key=api_key)
        return self._client

    def build_diagnosis_prompt(
        self,
        user_query: str,
        kb_context: str,
        soil_assessment: Dict[str, Any],
    ) -> str:
        """Build a concise, grounded prompt enforcing the 5-point output format."""
        taxonomies = ", ".join(SUPPORTED_DATASETS)
        soil_line = (
            f"Soil Profile -> N: {soil_assessment.get('n')} ({soil_assessment.get('n_status')}), "
            f"P: {soil_assessment.get('p')} ({soil_assessment.get('p_status')}), "
            f"K: {soil_assessment.get('k')} ({soil_assessment.get('k_status')}), "
            f"pH: {soil_assessment.get('ph')} ({soil_assessment.get('ph_status')})."
        )
        return (
            "You are the Crop Intelligence System multimodal agronomist.\n"
            f"Evaluate the plant image against these 5 reference taxonomies: {taxonomies}.\n"
            f"User request: {user_query or 'Diagnose this plant image.'}\n"
            f"{soil_line}\n\n"
            f"Grounded ICAR / IIHR / Dataset Reference Context:\n{kb_context}\n\n"
            "INSTRUCTIONS:\n"
            "- If the image is NOT a plant/crop, state that clearly under Diagnosis and explain how to retake.\n"
            "- Respond STRICTLY using the exact 5 numbered points below (no extra sections):\n"
            "1. **Diagnosis:** <crop, taxonomy match if applicable, and condition>\n"
            "2. **Reason:** <pathogen/pest/nutrient/environmental cause>\n"
            "3. **Chemical Solution:** <targeted synthetic treatment & safe dosage guidance>\n"
            "4. **Organic Solution:** <bio-control, botanical, and cultural solution>\n"
            "5. **What to do next:** <field scouting, sanitation, and preventive follow-up>"
        )

    def analyze(
        self,
        image_input: Union[Image.Image, bytes, bytearray, io.BytesIO, str, Path],
        user_query: str = "Diagnose this crop image.",
        kb_context: str = "",
        soil_assessment: Optional[Dict[str, Any]] = None,
        fallback_entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Encode image, invoke Gemini `gemini-2.5-flash`, and enforce 5-point output."""
        encoded = self.encoder.encode(image_input)
        soil_info = soil_assessment or {"n": 90, "p": 42, "k": 43, "ph": 6.5}
        prompt = self.build_diagnosis_prompt(user_query, kb_context, soil_info)

        client = self._get_client()
        if client is None:
            fallback_text = FivePointResponseFormatter.enforce_five_point_format(
                "",
                fallback_context={
                    **(fallback_entry or {}),
                    "condition": (
                        "Live Gemini Vision API key is not configured; displaying grounded "
                        f"reference advisory ({(fallback_entry or {}).get('condition', 'General Crop Check')})."
                    ),
                },
            )
            return {
                "response": fallback_text,
                "image_metadata": encoded.metadata(),
                "used_live_api": False,
                "model": self.model_name,
                "error": "GEMINI_API_KEY not configured",
            }

        try:
            api_response = client.models.generate_content(
                model=self.model_name,
                contents=[encoded.to_genai_part(), prompt],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            raw_text = (getattr(api_response, "text", None) or "").strip()
            formatted = FivePointResponseFormatter.enforce_five_point_format(
                raw_text,
                fallback_context=fallback_entry,
            )
            return {
                "response": formatted,
                "image_metadata": encoded.metadata(),
                "used_live_api": True,
                "model": self.model_name,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - graceful resilience on network/API failures
            formatted = FivePointResponseFormatter.enforce_five_point_format(
                "",
                fallback_context={
                    **(fallback_entry or {}),
                    "condition": (
                        f"Vision API call unavailable ({type(exc).__name__}); showing grounded "
                        f"ICAR/IIHR fallback advisory."
                    ),
                },
            )
            return {
                "response": formatted,
                "image_metadata": encoded.metadata(),
                "used_live_api": False,
                "model": self.model_name,
                "error": str(exc),
            }
