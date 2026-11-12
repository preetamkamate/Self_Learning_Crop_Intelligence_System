"""
LLM Synthesis & Bilingual Advisory Service.
Generates concise, scannable (120-180 words) agricultural advisories answering
strictly what the farmer asked in English or Kannada using Google GenAI SDK
with an offline rule-based fallback.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# CHANGED: Import dotenv and load environment variables
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import config
from src.orchestrator.state import FarmerProfile

# CHANGED: Import Google GenAI SDK with safe fallback
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AdvisorySynthesisService:
    """
    Synthesizes concise, intent-specific agricultural advisories.
    Uses real Google GenAI SDK (Gemini models) when an API key is available,
    with an offline rule-based synthesis fallback so the app never crashes.
    """

    def __init__(self):
        # CHANGED: Initialize GenAI Client and candidate models
        self.last_llm_status = "Initializing..."
        self._init_client()

    def _init_client(self):
        # CHANGED: Dynamically reload .env to ensure fresh API keys are read immediately
        load_dotenv(ROOT_DIR / ".env", override=True)
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip() or config.GEMINI_API_KEY
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or config.GEMINI_MODEL
        self.candidate_models = [
            self.model_name,
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]
        self.client = None

        if GENAI_AVAILABLE and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.last_llm_status = f"Gemini API Active ({self.model_name})"
                logging.info(
                    "AdvisorySynthesisService: Google GenAI Client initialized successfully (primary model: '%s').",
                    self.model_name,
                )
            except Exception as e:
                self.client = None
                self.last_llm_status = f"Client Init Failed ({e})"
                logging.warning(
                    "AdvisorySynthesisService: Failed to initialize Google GenAI Client: %s. Using local fallback.",
                    e,
                )
        else:
            self.client = None
            if not self.api_key:
                self.last_llm_status = "Local Fallback (GEMINI_API_KEY not set)"
            else:
                self.last_llm_status = "Local Fallback (google-genai unavailable)"

    def _build_system_instruction(self) -> str:
        """
        CHANGED: Structured system instruction adhering to all project requirements.
        """
        return (
            "You are an expert AI agricultural advisor for small and marginal Indian farmers in Karnataka.\n\n"
            "STRICT RULES YOU MUST FOLLOW:\n"
            "1. RELEVANCE & FOCUS: Answer ONLY what the farmer asked in their inquiry. Do NOT mention weather, rainfall, or crop recommendations unless the farmer specifically asked for them or uploaded an image whose management requires spraying weather.\n"
            "2. CONCISE & SCANNABLE: Keep each language advisory between 120 and 180 words. Use clean Markdown with short, easily readable bullet points (e.g., Diagnosis, Key Symptoms, Immediate Action/Dosage, Cultural Management).\n"
            "3. HONESTY & REFUSAL GATING:\n"
            "   - If Vision Status is 'OUT_OF_DISTRIBUTION' (non-plant image detected), do NOT guess a crop disease. Politely inform the farmer that no plant or crop canopy was recognized (DINOv2 similarity failed), and advise uploading an actual plant photo.\n"
            "   - If Vision Status is 'UNCERTAIN' (high entropy / conflicting symptoms across plant quadrants), clearly state the uncertainty, explain what symptoms conflict, and advise taking a closer, well-lit photo of the leaf or stem rather than spraying blind chemicals.\n"
            "4. TREATMENT SPECIFICITY: For confirmed diseases, provide exact ICAR / IIHR recommended chemical dosages (with active ingredients, e.g. Mancozeb 75% WP @ 2.5 g/L) along with bio-control/organic alternatives (e.g. Neem Oil 10,000 ppm @ 3 ml/L or Pseudomonas fluorescens @ 5 g/L) and cultural practices.\n"
            "5. DUAL BILINGUAL OUTPUT FORMAT:\n"
            "   You must provide your response in EXACTLY two sections separated by markers:\n"
            "   ---ENGLISH---\n"
            "   [English advisory, 120-180 words]\n"
            "   ---KANNADA---\n"
            "   [Kannada (ಕನ್ನಡ) advisory, 120-180 words]\n"
        )

    def _build_context_prompt(self, router_payload: Dict[str, Any]) -> str:
        """
        CHANGED: Assemble rich context inputs (profile, history, telemetry, vision, tabular, weather, RAG).
        """
        profile: Optional[FarmerProfile] = router_payload.get("farmer_profile")
        district = profile.district if profile else "Mandya (Cauvery Basin)"
        name = profile.name if profile else "Farmer"
        soil_n = profile.soil_n if profile else 90.0
        soil_p = profile.soil_p if profile else 42.0
        soil_k = profile.soil_k if profile else 43.0
        soil_ph = getattr(profile, "soil_ph", 6.5) if profile else 6.5
        land_acres = getattr(profile, "land_area_acres", 2.5) if profile else 2.5
        active_crop = getattr(profile, "active_crop", "Not specified") if profile and profile.active_crop else "Not specified"

        # Recent conversation history
        history = router_payload.get("recent_history", [])
        history_str = ""
        if history:
            for turn in history[-4:]:
                history_str += f"Farmer: {turn.get('user', '')}\nAdvisor: {turn.get('assistant', '')}\n"
        else:
            history_str = "No prior history in this session."

        user_query = router_payload.get("user_text", "Please assist me with my crop.")
        modules_run = ", ".join(router_payload.get("modules_run", ["Direct Query"]))
        primary_intent = router_payload.get("primary_intent", "general_query")
        is_followup = router_payload.get("is_followup", False)

        # Computer Vision Telemetry
        vision = router_payload.get("vision_result")
        if vision:
            v_status = vision.get("status", "unknown")
            v_label = vision.get("top1_label", "Unknown")
            v_label_kn = vision.get("top1_label_kannada", v_label)
            v_conf = vision.get("top1_confidence", 0.0)
            v_entropy = vision.get("entropy", 0.0)
            v_dino = vision.get("dinov2_plant_similarity", 0.0)
            v_reg = vision.get("affected_regions", "Whole Canopy")
            v_healthy = vision.get("is_healthy", False)
            vision_block = (
                f"- Status: {v_status.upper()}\n"
                f"- Top Condition: {v_label} (Kannada: {v_label_kn})\n"
                f"- Confidence: {v_conf:.1f}%\n"
                f"- Shannon Entropy H(p): {v_entropy:.3f} (Threshold: >1.80 is uncertain)\n"
                f"- DINOv2 Plant Match: {v_dino:.3f} (Threshold: <0.35 is Non-Plant OOD)\n"
                f"- Affected Plant Regions: {v_reg}\n"
                f"- Healthy Crop Flag: {v_healthy}"
            )
        else:
            vision_block = "None (Vision module did not run in this turn)."

        # Tabular ML Telemetry
        tabular = router_payload.get("tabular_result")
        if tabular:
            t_crop = tabular.get("top_crop", "Unknown")
            t_conf = tabular.get("top_confidence", 0.0)
            recs = tabular.get("recommendations", [])
            rec_str = ", ".join([f"{r.get('crop')} ({r.get('confidence_percent')}%)" for r in recs[:3]])
            tabular_block = (
                f"- Primary Recommended Crop: {t_crop} ({t_conf:.1f}% suitability)\n"
                f"- Top Candidates: {rec_str}\n"
                f"- Input Features: {tabular.get('input_features', {})}"
            )
        else:
            tabular_block = "None (Tabular model did not run in this turn)."

        # Live Weather Telemetry
        weather = router_payload.get("weather_result")
        if weather:
            w_temp = weather.get("temperature_c")
            w_rh = weather.get("humidity_pct")
            w_wind = weather.get("wind_speed_kmh")
            w_rain = weather.get("today_rain_prob_pct", 0)
            w_safe = weather.get("spray_safe", True)
            w_adv_en = weather.get("spray_advisory_en", "Normal.")
            w_adv_kn = weather.get("spray_advisory_kn", "ಸಾಮಾನ್ಯ.")
            weather_block = (
                f"- Temperature: {w_temp}°C | Humidity: {w_rh}% | Wind: {w_wind} km/h\n"
                f"- Rain Probability Today: {w_rain}%\n"
                f"- Spraying Safe Gate: {'SAFE (Green)' if w_safe else 'UNSAFE (Red Risk)'}\n"
                f"- Spray Advisory (EN): {w_adv_en}\n"
                f"- Spray Advisory (KN): {w_adv_kn}"
            )
        else:
            weather_block = "None (Weather module did not run in this turn)."

        # RAG Knowledge Chunks
        rag_results = router_payload.get("rag_results", [])
        if rag_results:
            rag_lines = []
            for i, doc in enumerate(rag_results, 1):
                rag_lines.append(
                    f"Chunk {i} [Source: {doc.get('source', 'ICAR')} | Score: {doc.get('relevance_percent', 0)}%]:\n"
                    f"{doc.get('content', '')}"
                )
            rag_block = "\n\n".join(rag_lines)
        else:
            rag_block = "No RAG knowledge chunks retrieved."

        return (
            f"### FARMER CONTEXT\n"
            f"- Farmer Name: {name}\n"
            f"- District/Zone: {district}\n"
            f"- Land Holding: {land_acres} acres\n"
            f"- Soil Profile: N={soil_n}, P={soil_p}, K={soil_k}, pH={soil_ph}\n"
            f"- Active Crop: {active_crop}\n\n"
            f"### RECENT CONVERSATION HISTORY\n"
            f"{history_str}\n\n"
            f"### CURRENT USER INQUIRY\n"
            f"\"{user_query}\"\n\n"
            f"### SYSTEM MODULES EXECUTED IN THIS TURN\n"
            f"- Modules Run: {modules_run}\n"
            f"- Primary Intent: {primary_intent}\n"
            f"- Follow-up Turn: {is_followup}\n\n"
            f"### 1. COMPUTER VISION TELEMETRY\n{vision_block}\n\n"
            f"### 2. TABULAR ML TELEMETRY\n{tabular_block}\n\n"
            f"### 3. LIVE WEATHER TELEMETRY\n{weather_block}\n\n"
            f"### 4. RETRIEVED ICAR / IIHR GUIDELINES (RAG)\n{rag_block}\n"
        )

    def _generate_with_gemini(
        self,
        router_payload: Dict[str, Any],
        language: str = "en",
    ) -> Optional[Dict[str, str]]:
        """
        CHANGED: Real LLM generation with candidate model fallback and dual bilingual parsing.
        """
        if not self.client:
            return None

        system_instruction = self._build_system_instruction()
        user_content = self._build_context_prompt(router_payload)

        # Build ordered list of candidate models without duplicates
        models_to_try = []
        for m in self.candidate_models:
            if m and m not in models_to_try:
                models_to_try.append(m)

        for model in models_to_try:
            try:
                logging.info("AdvisorySynthesisService: Generating advisory with Gemini model '%s'...", model)
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                        max_output_tokens=800,
                    ),
                )
                if response and response.text:
                    raw_text = response.text.strip()
                    en_text = ""
                    kn_text = ""

                    if "---KANNADA---" in raw_text:
                        parts = raw_text.split("---KANNADA---")
                        en_text = parts[0].replace("---ENGLISH---", "").strip()
                        kn_text = parts[1].strip()
                    elif "---ENGLISH---" in raw_text:
                        en_text = raw_text.replace("---ENGLISH---", "").strip()
                        fallback = self._local_fallback_synthesis(router_payload, language="kn")
                        kn_text = fallback["kn"]
                    else:
                        en_text = raw_text
                        fallback = self._local_fallback_synthesis(router_payload, language="kn")
                        kn_text = fallback["kn"]

                    if en_text:
                        self.last_llm_status = f"Gemini API Active ({model})"
                        logging.info("AdvisorySynthesisService: Successfully generated advisory via Gemini ('%s').", model)
                        return {
                            "en": en_text,
                            "kn": kn_text,
                            "chosen_response": kn_text if language == "kn" else en_text,
                            "llm_status": self.last_llm_status,
                        }
            except Exception as e:
                last_err = str(e)
                logging.warning("AdvisorySynthesisService: Gemini model '%s' failed: %s", model, e)
                continue

        self.last_llm_status = f"API Error ({last_err[:50]})" if 'last_err' in locals() else "API Offline"
        logging.warning("AdvisorySynthesisService: All Gemini models failed or offline (%s). Falling back to local synthesis.", self.last_llm_status)
        return None

    def _extract_rag_treatments(self, rag_docs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract organic, chemical, and cultural recommendations from RAG chunks."""
        organic = "Spray Cold-pressed Neem Oil (10,000 ppm) @ 3 ml/L or Pseudomonas fluorescens @ 5 g/L."
        chemical = "Apply Mancozeb 75% WP @ 2.0-2.5 g/L or Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1.0 ml/L (Wait period: 5-7 days)."
        cultural = "Remove and burn severely infected leaves; avoid wetting foliage during irrigation."

        organic_kn = "ಬೇವಿನ ಎಣ್ಣೆ (10,000 ppm) 3 ml/ಲೀ ಅಥವಾ ಸೂಡೋಮೊನಾಸ್ 5 ಗ್ರಾಂ/ಲೀ ನೀರಿಗೆ ಬೆರೆಸಿ ಸಿಂಪಡಿಸಿ."
        chemical_kn = "ಮ್ಯಾಂಕೋಜೆಬ್ 75% WP 2.5 ಗ್ರಾಂ/ಲೀ ಅಥವಾ ಅಜಾಕ್ಸಿಸ್ಟ್ರೋಬಿನ್ + ಡೈಫೆನೊಕೊನಜೋಲ್ 1.0 ml/ಲೀ ಸಿಂಪಡಿಸಿ (ಕಾಯುವ ಅವಧಿ: 7 ದಿನ)."
        cultural_kn = "ರೋಗಬಾಧಿತ ಎಲೆಗಳನ್ನು ಕಿತ್ತು ನಾಶಪಡಿಸಿ; ಗಿಡದ ಬುಡದಲ್ಲಿ ನೀರು ನಿಲ್ಲದಂತೆ ಬಸಿದು ಹೋಗಲು ವ್ಯವಸ್ಥೆ ಮಾಡಿ."

        for doc in rag_docs:
            content = doc.get("content", "")
            lines = content.split("\n")
            for line in lines:
                l_lower = line.lower()
                if "organic" in l_lower or "neem" in l_lower or "pseudomonas" in l_lower or "trichoderma" in l_lower:
                    clean = line.replace("-", "").replace("*", "").strip()
                    if len(clean) > 20:
                        organic = clean
                elif "mancozeb" in l_lower or "copper" in l_lower or "chemical" in l_lower or "spray" in l_lower:
                    clean = line.replace("-", "").replace("*", "").strip()
                    if len(clean) > 20:
                        chemical = clean
                elif "cultural" in l_lower or "spacing" in l_lower or "rotation" in l_lower:
                    clean = line.replace("-", "").replace("*", "").strip()
                    if len(clean) > 20:
                        cultural = clean

        return {
            "organic_en": organic,
            "chemical_en": chemical,
            "cultural_en": cultural,
            "organic_kn": organic_kn,
            "chemical_kn": chemical_kn,
            "cultural_kn": cultural_kn,
        }

    def _local_fallback_synthesis(
        self,
        router_payload: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, str]:
        """
        Local rule-based template synthesis used as an offline fallback when
        GEMINI_API_KEY is not set or the network is unavailable.
        """
        intent = router_payload.get("primary_intent", "general_query")
        is_followup = router_payload.get("is_followup", False)
        vision = router_payload.get("vision_result")
        tabular = router_payload.get("tabular_result")
        weather = router_payload.get("weather_result")
        rag_docs = router_payload.get("rag_results", [])
        profile: FarmerProfile = router_payload.get("farmer_profile")
        district = profile.district if profile else "Karnataka"

        treatments = self._extract_rag_treatments(rag_docs)

        # Case 1: Weather & Spray Advisory Query
        if intent == "weather_query" or (weather and not vision and not tabular):
            temp = weather.get("temperature_c") if weather else 26.0
            humidity = weather.get("humidity_pct") if weather else 70.0
            cond_en = weather.get("condition_en", "Partly Cloudy") if weather else "Clear"
            cond_kn = weather.get("condition_kn", "ಭಾಗಶಃ ಮೋಡ") if weather else "ಸ್ವಚ್ಛ"
            spray_safe = weather.get("spray_safe", True) if weather else True
            spray_en = weather.get("spray_advisory_en", "Normal spraying conditions.") if weather else "Normal."
            spray_kn = weather.get("spray_advisory_kn", "ಸಾಮಾನ್ಯ ಸಿಂಪಡಣಾ ವಾತಾವರಣ.") if weather else "ಸಾಮಾನ್ಯ."
            wind = weather.get("wind_speed_kmh", 5.0) if weather else 5.0
            rain_prob = weather.get("today_rain_prob_pct", 0) if weather else 0
            status_icon = "🟢" if spray_safe else "🔴"

            en = (
                f"### 🌦️ Live Weather & Spray Advisory ({district})\n\n"
                f"- **Current Weather**: {temp}°C, {humidity}% Humidity, Wind: {wind} km/h ({cond_en})\n"
                f"- **Rain Forecast**: {rain_prob}% probability of rain today.\n"
                f"- {status_icon} **Spraying Safety**: {spray_en}\n\n"
                f"*Tip: Spraying during high winds (>15 km/h) or before rain causes pesticide drift and loss.*"
            )
            kn = (
                f"### 🌦️ ನೈಜ ಸಮಯದ ಹವಾಮಾನ ಮತ್ತು ಸಿಂಪಡಣಾ ಮುನ್ಸೂಚನೆ ({district})\n\n"
                f"- **ಪ್ರಸ್ತುತ ಹವಾಮಾನ**: {temp}°C ಉಷ್ಣಾಂಶ, {humidity}% ತೇವಾಂಶ, ಗಾಳಿ: {wind} km/h ({cond_kn})\n"
                f"- **ಮಳೆ ಸಂಭವನೀಯತೆ**: ಇಂದು {rain_prob}% ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆ.\n"
                f"- {status_icon} **ಸಿಂಪಡಣಾ ಸಲಹೆ**: {spray_kn}\n\n"
                f"*ಸಲಹೆ: ಜೋರಾದ ಗಾಳಿ ಅಥವಾ ಮಳೆಯ ಸಮಯದಲ್ಲಿ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ.*"
            )
            return {"en": en, "kn": kn, "chosen_response": kn if language == "kn" else en}

        # Case 2: Tabular Crop Recommendation Query ONLY
        if intent == "crop_recommendation" or (tabular and not vision):
            top_crop = tabular.get("top_crop", "").title()
            top_crop_kn = tabular.get("top_crop_kannada", top_crop)
            top_prob = tabular.get("top_confidence", 0.0)
            recs = tabular.get("recommendations", [])
            feats = tabular.get("input_features", {})

            rec_list_en = "\n".join([f"  1. **{r['crop'].title()}** ({r['confidence_percent']}%) - Season: {r['recommended_season']}" for r in recs])
            rec_list_kn = "\n".join([f"  - **{r['kannada_name']}**: {r['confidence_percent']}% ({r['recommended_season']} ಹಂಗಾಮು)" for r in recs])

            en = (
                f"### 🌱 Crop Recommendation for {district}\n\n"
                f"Based on Soil N={feats.get('N')}, P={feats.get('P')}, K={feats.get('K')}, pH={feats.get('ph')}:\n\n"
                f"- **Primary Best Crop**: **{top_crop}** ({top_prob:.1f}% match)\n"
                f"- **Top-3 Suitable Crops**:\n{rec_list_en}\n\n"
                f"- **Agronomic Tip**: Ensure balanced basal fertilizer application and adequate soil organic matter."
            )
            kn = (
                f"### 🌱 {district} ಜಿಲ್ಲೆಗೆ ಬೆಳೆ ಶಿಫಾರಸು\n\n"
                f"ಮಣ್ಣಿನ ಪೋಷಕಾಂಶಗಳ ಆಧಾರದ ಮೇಲೆ (N={feats.get('N')}, P={feats.get('P')}, K={feats.get('K')}, pH={feats.get('ph')}):\n\n"
                f"- **ಅತ್ಯಂತ ಸೂಕ್ತ ಬೆಳೆ**: **{top_crop_kn}** ({top_prob:.1f}% ಸೂಕ್ತತೆ)\n"
                f"- **ಟಾಪ್-3 ಶಿಫಾರಸು ಬೆಳೆಗಳು**:\n{rec_list_kn}\n\n"
                f"- **ಕೃಷಿ ಸಲಹೆ**: ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಶಿಫಾರಸು ಮಾಡಿದ ಪ್ರಮಾಣದ ಸಾವಯವ ಗೊಬ್ಬರ ಮತ್ತು ಪೋಷಕಾಂಶಗಳನ್ನು ನೀಡಿ."
            )
            return {"en": en, "kn": kn, "chosen_response": kn if language == "kn" else en}

        # Case 3: Plant Pathology Diagnosis
        if (intent == "disease_diagnosis" and vision) or (vision and not is_followup):
            status = vision.get("status", "confident") if vision else "confident"

            # OOD Non-Plant Refusal
            if status == "out_of_distribution":
                dino_sim = vision.get("dinov2_plant_similarity", 0.0)
                en = (
                    f"⚠️ **Non-Plant Image Detected**\n\n"
                    f"- **DINOv2 Agricultural Match**: `{dino_sim:.3f}` (Threshold: > 0.35)\n"
                    f"- **Status**: The uploaded photo does not contain a recognizable crop, plant canopy, or field foliage.\n"
                    f"- **Safety Notice**: Automatic disease diagnosis was safely withheld to prevent incorrect chemical application.\n\n"
                    f"👉 **Action**: Please upload a clear photo of an actual crop, leaf, stem, or plant canopy."
                )
                kn = (
                    f"⚠️ **ಕೃಷಿಯೇತರ ಚಿತ್ರ ಪತ್ತೆಯಾಗಿದೆ**\n\n"
                    f"- **DINOv2 ಸಸ್ಯ ಹೋಲಿಕೆ**: `{dino_sim:.3f}` (ಅಗತ್ಯ: > 0.35)\n"
                    f"- **ಸ್ಥಿತಿ**: ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವುದೇ ಕೃಷಿ ಬೆಳೆ, ಎಲೆ ಅಥವಾ ಕಾಂಡ ಕಂಡುಬಂದಿಲ್ಲ.\n"
                    f"- **ಸೂಚನೆ**: ತಪ್ಪಾದ ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆಯನ್ನು ತಪ್ಪಿಸಲು ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ.\n\n"
                    f"👉 **ಸಲಹೆ**: ದಯವಿಟ್ಟು ಕೃಷಿ ಬೆಳೆ, ಗಿಡದ ಕಾಂಡ, ಅಥವಾ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಕಳುಹಿಸಿ."
                )
                return {"en": en, "kn": kn, "chosen_response": kn if language == "kn" else en}

            # Uncertain / Conflicted Multi-Quadrant Pattern
            if status == "uncertain":
                entropy = vision.get("entropy", 0.0)
                conf = vision.get("top1_confidence", 0.0)
                aff_reg = vision.get("affected_regions", "Multiple quadrants")
                en = (
                    f"⚠️ **Diagnostic Uncertainty Detected**\n\n"
                    f"- **Top Signal**: {vision.get('top1_label')} ({conf:.1f}% confidence)\n"
                    f"- **Predictive Entropy**: `{entropy:.3f}` (High uncertainty across quadrants)\n"
                    f"- **Inspected Region**: {aff_reg}\n\n"
                    f"> **Notice**: Conflicting symptoms observed across plant quadrants. Refusing blind chemical recommendation.\n"
                    f"👉 **Action**: Please take a sharp, well-lit close-up photo of the transition between healthy and damaged leaves."
                )
                kn = (
                    f"⚠️ **ರೋಗ ಲಕ್ಷಣಗಳಲ್ಲಿ ಅನಿಶ್ಚಿತತೆ**\n\n"
                    f"- **ಪತ್ತೆಯಾದ ಲಕ್ಷಣ**: {vision.get('top1_label_kannada')} ({conf:.1f}% ಖಚಿತತೆ)\n"
                    f"- **ಎಂಟ್ರೊಪಿ ಅನಿಶ್ಚಿತತೆ**: `{entropy:.3f}`\n"
                    f"- **ಪರಿಶೀಲಿಸಿದ ಭಾಗ**: {vision.get('affected_regions_kn', aff_reg)}\n\n"
                    f"> **ಸೂಚನೆ**: ಗಿಡದ ವಿವಿಧ ಭಾಗಗಳಲ್ಲಿ ಲಕ್ಷಣಗಳು ಅಸ್ಪಷ್ಟವಾಗಿರುವುದರಿಂದ ಖಚಿತ ರೋಗನಿರ್ಣಯ ಸಾಧ್ಯವಾಗಿಲ್ಲ.\n"
                    f"👉 **ಸಲಹೆ**: ದಯವಿಟ್ಟು ರೋಗಬಾಧಿತ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋವನ್ನು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ತೆಗೆದು ಕಳುಹಿಸಿ."
                )
                return {"en": en, "kn": kn, "chosen_response": kn if language == "kn" else en}

            # Confident Diagnosis
            top1_en = vision.get("top1_label", "Unknown Condition")
            top1_kn = vision.get("top1_label_kannada", top1_en)
            conf = vision.get("top1_confidence", 0.0)
            aff_reg = vision.get("affected_regions", "Whole Canopy")
            aff_reg_kn = vision.get("affected_regions_kn", "ಸಂಪೂರ್ಣ ಗಿಡದ ಮೇಲ್ಭಾಗ")
            is_healthy = vision.get("is_healthy", False)

            if is_healthy:
                en = (
                    f"### ✅ Plant Health Check: **{top1_en}**\n\n"
                    f"- **Status**: Healthy crop foliage ({conf:.1f}% confidence).\n"
                    f"- **Canopy Scope**: {aff_reg}.\n"
                    f"- **Assessment**: No active fungal lesions, viral mottling, or pest blights detected.\n"
                    f"- **Maintenance**: Continue balanced irrigation and scheduled weeding. No chemical spray required."
                )
                kn = (
                    f"### ✅ ಬೆಳೆ ಆರೋಗ್ಯ ತಪಾಸಣೆ: **{top1_kn}**\n\n"
                    f"- **ಸ್ಥಿತಿ**: ಆರೋಗ್ಯಕರ ಬೆಳೆ ({conf:.1f}% ಖಚಿತತೆ).\n"
                    f"- **ತಪಾಸಿತ ಭಾಗ**: {aff_reg_kn}.\n"
                    f"- **ವಿಶ್ಲೇಷಣೆ**: ಯಾವುದೇ ಶಿಲೀಂಧ್ರ ಕಲೆಗಳು ಅಥವಾ ಕೀಟಬಾಧೆ ಕಂಡುಬಂದಿಲ್ಲ.\n"
                    f"- **ಸಲಹೆ**: ಸರಿಯಾದ ಸಮಯಕ್ಕೆ ನೀರು ಹಾಯಿಸಿ. ಯಾವುದೇ ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆಯ ಅಗತ್ಯವಿಲ್ಲ."
                )
            else:
                en = (
                    f"### 🔬 Diagnosis: **{top1_en}** ({conf:.1f}% Confidence)\n\n"
                    f"- **Kannada**: {top1_kn}\n"
                    f"- **Affected Plant Region**: 📍 {aff_reg}\n\n"
                    f"**Immediate Management (ICAR/IIHR Protocol):**\n"
                    f"- **Organic / Bio-control**: {treatments['organic_en']}\n"
                    f"- **Chemical Treatment**: {treatments['chemical_en']}\n"
                    f"- **Cultural Action**: {treatments['cultural_en']}\n\n"
                    f"*Follow 7-day pre-harvest interval (PHI) for chemical sprays.*"
                )
                kn = (
                    f"### 🔬 ರೋಗ ಪತ್ತೆ: **{top1_kn}** (ಖಚಿತತೆ: {conf:.1f}%)\n\n"
                    f"- **ಇಂಗ್ಲಿಷ್ ಹೆಸರು**: {top1_en}\n"
                    f"- **ಬಾಧಿತ ಭಾಗ**: 📍 {aff_reg_kn}\n\n"
                    f"**ನಿರ್ವಹಣಾ ಕ್ರಮಗಳು (ICAR / IIHR ಶಿಫಾರಸು):**\n"
                    f"- **ಜೈವಿಕ / ಸಾವಯವ ನಿಯಂತ್ರಣ**: {treatments['organic_kn']}\n"
                    f"- **ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆ**: {treatments['chemical_kn']}\n"
                    f"- **ಮುನ್ನೆಚ್ಚರಿಕೆ**: {treatments['cultural_kn']}\n\n"
                    f"*ಔಷಧಿ ಸಿಂಪಡಿಸುವಾಗ ಮುಖಗವಸು ಧರಿಸಿ ಮತ್ತು ಕಾಯುವ ಅವಧಿಯನ್ನು ಪಾಲಿಸಿ.*"
                )

            # Append spray weather advisory if weather was also requested with image
            if weather:
                spray_safe = weather.get("spray_safe", True)
                spray_icon = "🟢" if spray_safe else "🔴"
                en += (
                    f"\n\n**Weather Alert ({district}):**\n"
                    f"- Current: {weather.get('temperature_c')}°C, {weather.get('humidity_pct')}% RH, Wind: {weather.get('wind_speed_kmh')} km/h.\n"
                    f"- Spraying Safety: {spray_icon} {weather.get('spray_advisory_en')}"
                )
                kn += (
                    f"\n\n**ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ({district}):**\n"
                    f"- ಪ್ರಸ್ತುತ: {weather.get('temperature_c')}°C, {weather.get('humidity_pct')}% ತೇವಾಂಶ, ಗಾಳಿ: {weather.get('wind_speed_kmh')} km/h.\n"
                    f"- ಸಿಂಪಡಣಾ ಸಲಹೆ: {spray_icon} {weather.get('spray_advisory_kn')}"
                )

            return {"en": en, "kn": kn, "chosen_response": kn if language == "kn" else en}

        # Case 4: Multi-Turn Conversational Follow-up
        if is_followup and vision:
            disease_name = vision.get("top1_label", "the diagnosed disease")
            disease_kn = vision.get("top1_label_kannada", disease_name)

            en = (
                f"### 🌿 Treatment Guide for **{disease_name}**\n\n"
                f"- **Context**: Continuing from previous diagnosis ({vision.get('top1_confidence', 0)}% confidence).\n"
                f"- **Organic Remedy**: {treatments['organic_en']}\n"
                f"- **Standard Chemical Dose**: {treatments['chemical_en']}\n"
                f"- **Cultural Sanity**: {treatments['cultural_en']}\n\n"
                f"*Spray during early morning or late evening. Avoid spraying under strong sunlight.*"
            )
            kn = (
                f"### 🌿 **{disease_kn}** ರೋಗದ ಚಿಕಿತ್ಸಾ ಮಾರ್ಗಸೂಚಿ\n\n"
                f"- **ಹಿಂದಿನ ರೋಗ ಪತ್ತೆ**: {disease_name}\n"
                f"- **ಸಾವಯವ / ಜೈವಿಕ ಪರಿಹಾರ**: {treatments['organic_kn']}\n"
                f"- **ರಾಸಾಯನಿಕ ಸಿಂಪಡಣೆ**: {treatments['chemical_kn']}\n"
                f"- **ಮುನ್ನೆಚ್ಚರಿಕೆ ಕ್ರಮ**: {treatments['cultural_kn']}\n\n"
                f"*ಔಷಧವನ್ನು ಮುಂಜಾನೆ ಅಥವಾ ಸಂಜೆಯ ವೇಳೆ ಸಿಂಪಡಿಸಿ. ಬಿಸಿಲಿನಲ್ಲಿ ಸಿಂಪಡಿಸಬೇಡಿ.*"
            )
            return {"en": en, "kn": kn, "chosen_response": kn if language == "kn" else en}

        # Case 5: Contextual Agricultural Guidance (DELETED hardcoded menu completely)
        # CHANGED: Genuine agricultural advisory based on farmer profile and agro-climatic zone
        crops_sample = "Maize, Cotton, Sorghum, Chickpea, and Pulses"
        if tabular:
            crops_sample = f"**{tabular.get('top_crop', '').title()}** along with {', '.join([r.get('crop', '').title() for r in tabular.get('recommendations', [])[:3]])}"
        en = (
            f"### 🌾 Crop Advisory ({district})\n\n"
            f"Based on your profile for **{district}** (Soil: N={profile.soil_n if profile else 75}, P={profile.soil_p if profile else 40}, K={profile.soil_k if profile else 42}, pH={profile.soil_ph if profile else 6.8}):\n\n"
            f"- **Agronomic Recommendation**: {crops_sample} perform reliably under the agro-climatic conditions of {district}.\n"
            f"- **Soil Health & Moisture**: Apply well-decomposed FYM (5 tonnes/acre) before sowing to enhance organic carbon and water-holding capacity.\n"
            f"- **Field Management**: Use certified seeds with recommended spacing and conduct regular field scouting for early pest detection."
        )
        kn = (
            f"### 🌾 ಕೃಷಿ ಸಲಹೆಗಾರ ({district})\n\n"
            f"ನಿಮ್ಮ **{district}** ಪ್ರದೇಶದ ಮಣ್ಣಿನ ಪೋಷಕಾಂಶಗಳ ಆಧಾರದ ಮೇಲೆ (N={profile.soil_n if profile else 75}, P={profile.soil_p if profile else 40}, K={profile.soil_k if profile else 42}, pH={profile.soil_ph if profile else 6.8}):\n\n"
            f"- **ಶಿಫಾರಸು ಬೆಳೆಗಳು**: {district} ಹವಾಮಾನಕ್ಕೆ ಸೂಕ್ತವಾದ ಬೆಳೆಗಳು: {crops_sample}.\n"
            f"- **ಮಣ್ಣಿನ ನಿರ್ವಹಣೆ**: ಬಿತ್ತನೆಗೆ ಮುನ್ನ ಸಾಕಷ್ಟು ಸಾವಯವ ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರವನ್ನು ಮಣ್ಣಿಗೆ ಸೇರಿಸಿ.\n"
            f"- **ಮುಂದಿನ ಕ್ರಮ**: ಗುಣಮಟ್ಟದ ಬೀಜೋಪಚಾರ ಮಾಡಿದ ಪ್ರಮಾಣೀಕೃತ ಬೀಜಗಳನ್ನು ಬಳಸಿ ಸೂಕ್ತ ಸಮಯದಲ್ಲಿ ಬಿತ್ತನೆ ಮಾಡಿ."
        )
        return {
            "en": en,
            "kn": kn,
            "chosen_response": kn if language == "kn" else en,
            "llm_status": getattr(self, "last_llm_status", "Local Fallback Active"),
        }

    def generate_advisory(
        self,
        router_payload: Dict[str, Any],
        language: str = "en",
    ) -> Dict[str, str]:
        """
        Synthesize concise response tailored strictly to what the farmer asked.
        Uses real Google GenAI SDK (Gemini models) when GEMINI_API_KEY is available,
        with an offline rule-based fallback to guarantee stability.
        Word count: 120-180 words.
        """
        # CHANGED: Ensure client is dynamically refreshed from environment if not yet ready
        if not self.client:
            self._init_client()

        # CHANGED: Force live Gemini LLM generation for ALL queries when client is available
        if self.client:
            llm_res = self._generate_with_gemini(router_payload, language=language)
            if llm_res and llm_res.get("en"):
                return llm_res

        # CHANGED: Seamless local rule-based synthesis fallback
        fallback_res = self._local_fallback_synthesis(router_payload, language=language)
        fallback_res["llm_status"] = getattr(self, "last_llm_status", "Local Fallback Active")
        return fallback_res
