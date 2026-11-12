"""
Autonomous Internal Router for Crop Intelligence System.
Selectively triggers specialized components (Vision, Tabular ML, Weather, RAG)
strictly based on user intent and image novelty.
"""
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.orchestrator.state import SessionState
from src.orchestrator.weather import WeatherService
from src.rag.retriever import get_retriever
from src.tabular.predict import CropPredictor
from src.vision.predict import PlantDiseasePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AutonomousRouter:
    """Intelligent dispatcher that activates appropriate system modules selectively."""

    def __init__(self):
        logging.info("Initializing AutonomousRouter modules...")
        self.tabular_predictor = CropPredictor()
        self.vision_predictor = PlantDiseasePredictor()
        self.retriever = get_retriever()
        self.weather_service = WeatherService()

    def _extract_soil_overrides(self, text: str, state: SessionState):
        """Extract explicit NPK or pH values from user text to update profile."""
        n_match = re.search(r"\bN\s*[:=]?\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        p_match = re.search(r"\bP\s*[:=]?\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        k_match = re.search(r"\bK\s*[:=]?\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
        ph_match = re.search(r"\bpH\s*[:=]?\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)

        updates = {}
        if n_match:
            updates["soil_n"] = float(n_match.group(1))
        if p_match:
            updates["soil_p"] = float(p_match.group(1))
        if k_match:
            updates["soil_k"] = float(k_match.group(1))
        if ph_match:
            updates["soil_ph"] = float(ph_match.group(1))

        if updates:
            logging.info("Updating farmer profile with extracted soil values: %s", updates)
            state.update_profile(**updates)

    def route_and_execute(
        self,
        user_text: str,
        image_input: Optional[Union[Image.Image, bytes, str, Path]] = None,
        is_new_image: bool = False,  # CHANGED: Default is_new_image to False as specified
        image_hash: Optional[str] = None,
        state: Optional[SessionState] = None,
        **kwargs,  # CHANGED: Safely absorb any extra keyword arguments
    ) -> Dict[str, Any]:
        """
        Analyze input and trigger specialized components strictly based on intent.
        """
        if state is None:
            state = SessionState()

        # Update profile if user provided NPK/pH in text
        self._extract_soil_overrides(user_text, state)

        text_lower = user_text.lower().strip()
        modules_run: List[str] = []
        vision_result: Optional[Dict[str, Any]] = None
        tabular_result: Optional[Dict[str, Any]] = None
        weather_result: Optional[Dict[str, Any]] = None
        rag_results: List[Dict[str, Any]] = []
        is_followup = False

        # ----------------------------------------------------------------------
        # Strict Intent Classification
        # ----------------------------------------------------------------------

        # Tabular Crop Recommendation: User explicitly asks what to grow/plant
        crop_rec_patterns = [
            r"what\s+(crop|crops)\s+should\s+i\s+(grow|plant|sow)",
            r"what\s+to\s+(grow|plant|sow)",
            r"which\s+(crop|crops)\s+(to|should|can)\s+(grow|plant|sow)",
            r"recommend\s+(a\s+)?(crop|crops)",
            r"crop\s+recommendation",
            r"crop\s+suitability",
            r"suitable\s+crops?",
            r"best\s+crop\s+for",
            r"suggest\s+(a\s+)?crop",
            r"ಯಾವ\s+ಬೆಳೆ",
            r"ಬೆಳೆ\s+ಶಿಫಾರಸು",
            r"ಯಾವ\s+ಬೆಳೆ\s+ಬೆಳೆಯಬೇಕು",
        ]
        wants_crop_rec = any(re.search(pat, text_lower) for pat in crop_rec_patterns)

        # Weather Intelligence: User explicitly asks about weather, rain, forecast, or spray conditions
        weather_patterns = [
            r"weather",
            r"rain",
            r"raining",
            r"forecast",
            r"humidity",
            r"temperature",
            r"wind",
            r"safe\s+to\s+spray",
            r"spraying\s+weather",
            r"spray\s+conditions",
            r"can\s+i\s+spray",
            r"ಹವಾಮಾನ",
            r"ಮಳೆ",
            r"ಉಷ್ಣಾಂಶ",
            r"ಸಿಂಪಡಣೆಗೆ\s+(ಅನುಕೂಲ|ಸರಿ)",
        ]
        wants_weather = any(re.search(pat, text_lower) for pat in weather_patterns)

        # Disease / Pest / Treatment Question or Follow-up
        disease_patterns = [
            r"disease",
            r"blight",
            r"spot",
            r"rot",
            r"rust",
            r"wilt",
            r"pest",
            r"curling",
            r"caterpillar",
            r"worm",
            r"cure",
            r"medicine",
            r"treat",
            r"treatment",
            r"chemical",
            r"spray",
            r"fungicide",
            r"pesticide",
            r"organic\s+spray",
            r"dosage",
            r"how\s+to\s+(cure|control|manage|treat)",
            r"what\s+(medicine|spray|chemical)\s+should",
            r"contagious",
            r"ರೋಗ",
            r"ಕೀಟ",
            r"ಅಂಗಮಾರಿ",
            r"ಔಷಧಿ",
            r"ಚಿಕಿತ್ಸೆ",
            r"ಸಿಂಪಡಿಸಬೇಕು",
        ]
        wants_disease = any(re.search(pat, text_lower) for pat in disease_patterns)

        has_new_image = (image_input is not None) and (is_new_image is True)  # CHANGED: Vision only triggers when image is provided and is_new_image is True

        # ----------------------------------------------------------------------
        # Module Execution
        # ----------------------------------------------------------------------

        # 1. Vision Module: Run ONLY when a NEW image is uploaded
        if has_new_image:
            logging.info("AutonomousRouter: Running Whole-Plant Vision (New Image Uploaded)...")
            modules_run.append("Whole-Plant Vision (DINOv2 + Multi-Scale Pathology)")
            vision_result = self.vision_predictor.predict(image_input)
            state.set_vision_result(vision_result, image_hash=image_hash)
            primary_intent = "disease_diagnosis"
        elif wants_weather:
            primary_intent = "weather_query"
        elif wants_crop_rec:
            primary_intent = "crop_recommendation"
        else:
            primary_intent = "general_query"

        # 2. Conversational Follow-up: If no new image and not asking about weather/crop rec, reuse previous diagnosis
        last_diag = state.get_last_diagnosis()
        if not has_new_image and last_diag and not wants_weather and not wants_crop_rec:
            logging.info("AutonomousRouter: Multi-Turn Continuity -> Reusing previous diagnosis: %s", last_diag.get("top1_label"))
            vision_result = last_diag
            is_followup = True
            primary_intent = "disease_followup"
            modules_run.append("Conversational Context (Retained Previous Diagnosis)")

        # 3. Weather Module: Run ONLY when explicitly asked
        if wants_weather:
            logging.info("AutonomousRouter: Running Weather Module (User explicitly requested weather)...")
            modules_run.append("Live Weather API (Open-Meteo)")
            weather_result = self.weather_service.get_weather(
                district_or_city=state.profile.district
            )

        # 4. Tabular Crop ML: Run ONLY when explicitly asked to recommend crops
        if wants_crop_rec and not has_new_image:
            logging.info("AutonomousRouter: Running Tabular ML (User explicitly requested crop recommendation)...")
            modules_run.append("Tabular ML (RandomForest Suitability)")
            tabular_result = self.tabular_predictor.predict(
                n=state.profile.soil_n,
                p=state.profile.soil_p,
                k=state.profile.soil_k,
                temp=26.0,
                humidity=70.0,
                ph=state.profile.soil_ph,
                rainfall=200.0,
                top_k=3,
            )

        # 5. RAG Retrieval Module: Retrieve specific agronomic guidelines
        rag_query = ""
        if has_new_image and vision_result and vision_result.get("status") == "confident":
            # CHANGED: When image and specific text are submitted together, combine diagnosis with farmer query
            generic_prompts = [
                "analyze this plant image and identify any health issues or diseases.",
                "please inspect this uploaded plant photograph.",
                "please diagnose the disease on this leaf.",
                "please identify the disease on this tomato plant.",
                "please analyze this full-plant tomato crop.",
                "please analyze this image.",
                "please check this sample.",
            ]
            clean_text = user_text.strip().lower()
            if clean_text and clean_text not in generic_prompts:
                rag_query = f"{vision_result['top1_label']} {user_text} ICAR symptoms remedy chemical dosage"
            else:
                rag_query = f"{vision_result['top1_label']} ICAR symptoms, organic remedy and chemical dosage"
        elif is_followup and vision_result and vision_result.get("status") == "confident":
            rag_query = f"{vision_result['top1_label']} {user_text} ICAR dosage treatment"
        elif wants_crop_rec and tabular_result:
            rag_query = f"{tabular_result['top_crop']} cultivation practices soil requirement ICAR"
        elif wants_disease:
            rag_query = user_text

        if rag_query:
            logging.info("AutonomousRouter: Retrieving RAG knowledge for: '%s'", rag_query)
            modules_run.append("RAG Knowledge Engine (FAISS + MiniLM-L6)")
            rag_results = self.retriever.retrieve(rag_query, top_k=2)

        return {
            "modules_run": modules_run,
            "primary_intent": primary_intent,
            "user_text": user_text,  # CHANGED: Preserved user_text for multimodal synthesis
            "is_followup": is_followup,
            "vision_result": vision_result,
            "tabular_result": tabular_result,
            "weather_result": weather_result,
            "rag_results": rag_results,
            "farmer_profile": state.profile,
            "recent_history": state.get_recent_history(max_turns=6),
        }
