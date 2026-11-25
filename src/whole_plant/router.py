"""
Crop-Aware Whole-Plant Vision Router.
Directs image inputs strictly through the crop-aware whole-plant pipeline.
Supported viewpoints:
  - whole_plant
  - canopy
  - stem
  - fruit
  - root
  - bark
  - leaf
  - unknown
  - non_plant

Enforces:
1. Never allows non-whole-plant views to fall back to a leaf model or guess diseases.
2. Never defaults unknown crops to tomato.
3. Never produces disease labels for untrained crops or viewpoints.
4. Returns all 9 required fields in prediction result:
   plant_detected, crop, viewpoint, supported, disease, confidence, status, fallback_reason, checkpoint_path.
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.whole_plant.dataset import RAW_WHOLE_PLANT_DIR, SUPPORTED_VIEWPOINTS
from src.modular_vision.photo_guidance import get_photo_guidance
from src.whole_plant.predict import (
    NON_PLANT_MSG,
    UNSUPPORTED_MSG,
    WholePlantPredictionResult,
    WholePlantPredictor,
)
from src.whole_plant.train import DEFAULT_MODEL_DIR, DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class WholePlantRouter:
    """
    Independent Whole-Plant Router.
    Routes visual crop queries to the trained whole-plant model or executes safe abstention.
    """

    def __init__(
        self,
        predictor: Optional[WholePlantPredictor] = None,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.eval_report_path = Path(DEFAULT_MODEL_DIR) / "evaluation_report.json"
        self.raw_dir = Path(raw_dir)
        self.predictor = predictor or WholePlantPredictor(
            model_path=self.model_path,
            raw_dir=self.raw_dir,
            device=device,
        )

    @property
    def is_deployed(self) -> bool:
        """Returns True only if both the trained checkpoint and evaluation report exist."""
        return self.model_path.exists() and self.eval_report_path.exists() and self.predictor.is_trained

    def normalize_viewpoint(self, raw_viewpoint: Optional[str]) -> str:
        """Map raw viewpoint strings to canonical supported viewpoints."""
        if not raw_viewpoint or str(raw_viewpoint).strip() in ["", "Auto-detect", "auto", "None"]:
            return "whole_plant"

        v = str(raw_viewpoint).lower().strip()
        if "whole" in v or "field" in v or "plant" in v and "part" not in v and "non" not in v:
            return "whole_plant"
        elif "canopy" in v:
            return "canopy"
        elif "stem" in v or "branch" in v:
            return "stem"
        elif "fruit" in v or "pod" in v:
            return "fruit"
        elif "root" in v or "tuber" in v:
            return "root"
        elif "bark" in v or "trunk" in v:
            return "bark"
        elif "leaf" in v or "foliar" in v:
            return "leaf"
        elif "non" in v:
            return "non_plant"
        elif v in SUPPORTED_VIEWPOINTS:
            return v
        return "unknown"

    def normalize_crop(self, raw_crop: Optional[str]) -> str:
        """Normalize crop string using the predictor's canonical mapper."""
        return self.predictor.normalize_crop_name(raw_crop)

    def route(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop_hint: Optional[str] = None,
        viewpoint_hint: Optional[str] = None,
        confidence_threshold: float = 0.40,
        user_text: Optional[str] = None,
        symptom_desc: Optional[str] = None,
        **kwargs,
    ) -> WholePlantPredictionResult:
        """
        Main entry point for routing an image query.
        Validates viewpoint and crop, verifies checkpoint and evaluation report presence,
        and enforces zero cross-viewpoint / cross-crop leakage.
        Provides honest photo guidance when a route cannot be diagnosed.
        """
        crop = self.normalize_crop(crop_hint)
        viewpoint = self.normalize_viewpoint(viewpoint_hint)

        # 1. Non-plant rejection
        if viewpoint == "non_plant":
            return WholePlantPredictionResult(
                plant_detected=False,
                crop="None",
                viewpoint="non_plant",
                supported=False,
                disease=None,
                confidence=0.0,
                status="out_of_distribution",
                fallback_reason="Non-plant image detected",
                checkpoint_path=str(self.model_path),
                is_ood=True,
                top1_label=NON_PLANT_MSG,
                message=NON_PLANT_MSG,
                photo_guidance=get_photo_guidance(viewpoint="leaf", is_non_plant=True),
            )

        # 2. Checkpoint & Deployment Gate
        # If model checkpoint or evaluation report does not exist, return safe abstention
        if not self.is_deployed:
            return WholePlantPredictionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=viewpoint,
                supported=False,
                disease=None,
                confidence=0.0,
                status="needs_training_data",
                fallback_reason=f"Whole-plant diagnosis is not trained or deployed yet (checkpoint={self.model_path.exists()}, report={self.eval_report_path.exists()}).",
                checkpoint_path=str(self.model_path),
                top1_label=UNSUPPORTED_MSG,
                message=UNSUPPORTED_MSG,
                photo_guidance=get_photo_guidance(viewpoint=viewpoint, crop=crop),
            )

        # 3. Viewpoint & Crop Routing Gate
        # Validate that the crop is supported and the viewpoint is trained for this crop.
        crop_vp_map = getattr(self.predictor, "crop_viewpoint_map", {
            "Corn (Maize)": ["whole_plant"],
            "Tomato": ["whole_plant"],
            "Apple": ["leaf"],
            "Cassava": ["leaf"],
        })
        supported_crops = getattr(self.predictor, "supported_crops", ["Corn (Maize)", "Tomato", "Apple", "Cassava"])

        if crop == "Unknown" or crop not in supported_crops:
            return WholePlantPredictionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=viewpoint,
                supported=False,
                disease=None,
                confidence=0.0,
                status="unsupported",
                fallback_reason="Valid plant image detected, but this crop/viewpoint diagnosis is not trained yet.",
                checkpoint_path=str(self.model_path),
                top1_label=UNSUPPORTED_MSG,
                message=UNSUPPORTED_MSG,
                photo_guidance=get_photo_guidance(viewpoint=viewpoint, crop=crop, is_unsupported_crop=True),
            )

        valid_vps = crop_vp_map.get(crop, ["whole_plant"])
        v_mapped = "leaf" if viewpoint in ["leaf", "foliar", "single_leaf", "close_up_leaf"] else viewpoint

        if v_mapped not in valid_vps:
            reason = f"Viewpoint '{viewpoint}' diagnosis is not trained yet for crop '{crop}'."
            return WholePlantPredictionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=viewpoint,
                supported=False,
                disease=None,
                confidence=0.0,
                status="unsupported",
                fallback_reason=reason,
                checkpoint_path=str(self.model_path),
                top1_label=UNSUPPORTED_MSG,
                message=UNSUPPORTED_MSG,
                photo_guidance=get_photo_guidance(viewpoint=viewpoint, crop=crop),
            )

        # 4. Whole-Plant Viewpoint Route
        result = self.predictor.predict(
            image_input=image_input,
            crop_hint=crop,
            viewpoint_hint=viewpoint,
            confidence_threshold=confidence_threshold,
        )
        return result
