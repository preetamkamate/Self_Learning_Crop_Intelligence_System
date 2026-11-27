"""
Direct Gemini API Orchestrator Router for Crop Intelligence System.

Routes text and multimodal image queries directly through Google GenAI (`gemini-2.5-flash`)
grounded with ICAR/IIHR knowledge retrieval and soil NPK/pH telemetry — with zero local
`.pt`/`.pth` model weight dependencies.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from src.modular_vision.pipeline import (
    DEFAULT_MODEL,
    FivePointResponseFormatter,
    ModularVisionPipeline,
)
from src.rag.kb_retriever import SUPPORTED_DATASETS, KnowledgeBaseRetriever

DIAGNOSIS_KEYWORDS_RE = re.compile(
    r"\b("
    r"diagnos\w*|disease\w*|spot\w*|blight|rust|rot|wilt|mosaic|mildew|scab|"
    r"curl|yellowing|chlorosis|lesion\w*|pustule\w*|pest\w*|thrips|mite\w*|"
    r"armyworm|aphid\w*|whitefly|fungus|bacterial|virus|symptom\w*|infected|"
    r"damage|cure|treat\w*|spray"
    r")\b",
    flags=re.IGNORECASE,
)


class CropRouter:
    """Autonomous router dispatching queries directly to Gemini 2.5 Flash + ICAR/IIHR KB."""

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model_name: Optional[str] = None,
        kb_retriever: Optional[KnowledgeBaseRetriever] = None,
        vision_pipeline: Optional[ModularVisionPipeline] = None,
    ):
        load_dotenv(override=False)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        self._client = client
        self.kb_retriever = kb_retriever or KnowledgeBaseRetriever()
        self.vision_pipeline = vision_pipeline or ModularVisionPipeline(
            client=client,
            model_name=self.model_name,
        )

    def _get_client(self) -> Optional[genai.Client]:
        """Lazily initialize Gemini API client when an API key is present."""
        if self._client is not None:
            return self._client
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _is_diagnosis_query(text: str) -> bool:
        """Detect whether a text-only query asks for plant disease/pest diagnosis."""
        return bool(DIAGNOSIS_KEYWORDS_RE.search(text or ""))

    def _build_text_prompt(
        self,
        query: str,
        kb_context: str,
        soil_assessment: Dict[str, Any],
        enforce_five_point: bool,
    ) -> str:
        """Construct grounded agronomic prompt for text-only Gemini routing."""
        taxonomies = ", ".join(SUPPORTED_DATASETS)
        soil_summary = (
            f"Soil Parameters -> N: {soil_assessment['n']} kg/ha ({soil_assessment['n_status']}), "
            f"P: {soil_assessment['p']} kg/ha ({soil_assessment['p_status']}), "
            f"K: {soil_assessment['k']} kg/ha ({soil_assessment['k_status']}), "
            f"pH: {soil_assessment['ph']} ({soil_assessment['ph_status']}).\n"
            f"Soil Notes: {' '.join(soil_assessment['recommendations'])}"
        )
        base = (
            "You are the Crop Intelligence System agricultural advisor.\n"
            f"Reference taxonomies: {taxonomies}.\n"
            f"{soil_summary}\n\n"
            f"ICAR / IIHR / Dataset Context:\n{kb_context}\n\n"
            f"Farmer Question: {query}\n\n"
        )
        if enforce_five_point:
            return base + (
                "Respond STRICTLY using the exact 5 numbered points below:\n"
                "1. **Diagnosis:** <identified condition/issue>\n"
                "2. **Reason:** <agronomic, pathogen, or nutrient cause>\n"
                "3. **Chemical Solution:** <targeted chemical control & safe dosage>\n"
                "4. **Organic Solution:** <organic, biological, and cultural management>\n"
                "5. **What to do next:** <monitoring, prevention, and follow-up>"
            )
        return base + (
            "Provide a clear, practical, science-backed agronomic answer grounded in the soil "
            "parameters and ICAR/IIHR context above."
        )

    def _offline_text_fallback(
        self,
        query: str,
        kb_matches: List[Dict[str, Any]],
        soil_assessment: Dict[str, Any],
        enforce_five_point: bool,
    ) -> str:
        """Produce a clean grounded response when the Gemini API key is unavailable or fails."""
        top_entry = kb_matches[0] if kb_matches else {}
        if enforce_five_point:
            return FivePointResponseFormatter.enforce_five_point_format(
                "",
                fallback_context=top_entry,
            )

        soil_notes = " ".join(soil_assessment.get("recommendations", []))
        lines = [
            f"**Soil Summary (N={soil_assessment['n']}, P={soil_assessment['p']}, "
            f"K={soil_assessment['k']}, pH={soil_assessment['ph']}):** {soil_notes}"
        ]
        if top_entry:
            lines.append(
                f"\n**{top_entry['dataset']} — {top_entry['condition']} ({top_entry['crop']}):**\n"
                f"- **Cause:** {top_entry['pathogen_or_cause']}\n"
                f"- **Organic Guidance:** {top_entry['organic_solution']}\n"
                f"- **Next Steps:** {top_entry['next_steps']}"
            )
        return "\n".join(lines)

    def route(
        self,
        query: str = "",
        image: Optional[Union[Image.Image, bytes, bytearray, str, Path]] = None,
        soil_params: Optional[Dict[str, float]] = None,
        crop_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Route a farmer query (with optional image and soil NPK/pH config) to Gemini 2.5 Flash.
        """
        clean_query = (query or "").strip()
        params = soil_params or {}
        soil_assessment = self.kb_retriever.get_soil_advisory(
            n=float(params.get("n", 90.0)),
            p=float(params.get("p", 42.0)),
            k=float(params.get("k", 43.0)),
            ph=float(params.get("ph", 6.5)),
        )

        kb_query = clean_query or crop_hint or "crop leaf disease diagnosis"
        kb_matches = self.kb_retriever.retrieve(kb_query, crop_hint=crop_hint, top_k=3)
        kb_context = self.kb_retriever.format_context(kb_matches)

        # 1. Multimodal Image Path -> always strict 5-point diagnosis
        if image is not None:
            vision_out = self.vision_pipeline.analyze(
                image_input=image,
                user_query=clean_query or "Diagnose this crop image and provide a 5-point advisory.",
                kb_context=kb_context,
                soil_assessment=soil_assessment,
                fallback_entry=kb_matches[0] if kb_matches else None,
            )
            response_text = vision_out["response"]
            return {
                "response": response_text,
                "mode": "multimodal_diagnosis",
                "model": vision_out["model"],
                "taxonomies": list(SUPPORTED_DATASETS),
                "kb_matches": kb_matches,
                "soil_assessment": soil_assessment,
                "image_metadata": vision_out["image_metadata"],
                "used_live_api": vision_out["used_live_api"],
                "is_five_point": FivePointResponseFormatter.validate_five_point_format(response_text),
                "error": vision_out.get("error"),
            }

        # 2. Text-Only Path
        if not clean_query:
            clean_query = "Provide a crop and soil health summary based on my current NPK parameters."

        enforce_five_point = self._is_diagnosis_query(clean_query)
        prompt = self._build_text_prompt(
            query=clean_query,
            kb_context=kb_context,
            soil_assessment=soil_assessment,
            enforce_five_point=enforce_five_point,
        )

        client = self._get_client()
        used_live_api = False
        error: Optional[str] = None
        response_text = ""

        if client is not None:
            try:
                api_response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.25),
                )
                raw_text = (getattr(api_response, "text", None) or "").strip()
                if raw_text:
                    response_text = (
                        FivePointResponseFormatter.enforce_five_point_format(
                            raw_text,
                            fallback_context=kb_matches[0] if kb_matches else None,
                        )
                        if enforce_five_point
                        else raw_text
                    )
                    used_live_api = True
            except Exception as exc:  # noqa: BLE001
                error = str(exc)

        if not response_text:
            response_text = self._offline_text_fallback(
                query=clean_query,
                kb_matches=kb_matches,
                soil_assessment=soil_assessment,
                enforce_five_point=enforce_five_point,
            )
            if error is None and client is None:
                error = "GEMINI_API_KEY not configured"

        return {
            "response": response_text,
            "mode": "text_diagnosis" if enforce_five_point else "agronomic_chat",
            "model": self.model_name,
            "taxonomies": list(SUPPORTED_DATASETS),
            "kb_matches": kb_matches,
            "soil_assessment": soil_assessment,
            "image_metadata": None,
            "used_live_api": used_live_api,
            "is_five_point": (
                FivePointResponseFormatter.validate_five_point_format(response_text)
                if enforce_five_point
                else False
            ),
            "error": error,
        }
