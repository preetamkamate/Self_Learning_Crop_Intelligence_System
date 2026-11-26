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

        # 1B. Pest Query Screening & IPM Routing
        from src.modular_vision.pest_detector import PestDetector
        pest_detector = PestDetector()
        if pest_detector.is_pest_query(user_text, symptom_desc):
            pest_res = pest_detector.detect_pest(
                image_input=image_input,
                crop=crop,
                viewpoint=viewpoint,
                user_text=user_text,
                symptom_desc=symptom_desc,
            )
            return WholePlantPredictionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=viewpoint,
                supported=pest_res.supported,
                disease=pest_res.pest_label,
                confidence=pest_res.confidence,
                status=pest_res.status,
                fallback_reason=pest_res.status_reason,
                checkpoint_path=str(pest_detector.model_path) if pest_res.supported else "None",
                top1_label=pest_res.pest_label or pest_res.status_reason,
                message=pest_res.status_reason,
                photo_guidance=pest_res.photo_guidance,
                chemical_treatment_withheld=pest_res.metadata.get("chemical_pesticides_withheld", True),
                debug_info={"pest_info": pest_res.to_dict(), "ipm_advice": pest_res.ipm_scouting_advice},
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

        # 3. Optional Leaf-Only Experimental Route
        # Enforces: Leaf diagnosis is a separate optional feature and must NEVER be a fallback
        # for whole_plant, canopy, stem, fruit, or root photos.
        if viewpoint == "leaf":
            from src.modular_vision.disease_classifier import ModularDiseaseClassifier
            leaf_classifier = ModularDiseaseClassifier()
            leaf_res = leaf_classifier.classify_disease(
                image_input=image_input,
                crop=crop,
                viewpoint="leaf",
                confidence_threshold=confidence_threshold,
            )
            return WholePlantPredictionResult(
                plant_detected=True,
                crop=leaf_res.crop,
                viewpoint="leaf",
                supported=leaf_res.supported,
                disease=leaf_res.disease,
                confidence=leaf_res.confidence,
                status=leaf_res.status,
                fallback_reason=leaf_res.fallback_reason or "Optional Leaf-only experimental feature executed.",
                checkpoint_path=str(leaf_res.checkpoint_path),
                top1_label=leaf_res.top1_label,
                message=leaf_res.top1_label,
                photo_guidance=leaf_res.photo_guidance,
                margin=leaf_res.margin,
                chemical_treatment_withheld=leaf_res.chemical_treatment_withheld,
                model_release_tier="Leaf-Only Experimental (Research Prototype)",
                is_experimental=True,
                debug_info=leaf_res.debug_info,
            )

        # 4. Multi-Part Organ Routes (stem, fruit, root, bark, canopy)
        if viewpoint in ["stem", "fruit", "root", "bark", "canopy"]:
            from src.vision.predict import PlantDiseasePredictor
            part_pred = PlantDiseasePredictor().predict(
                image_input=image_input,
                user_crop_hint=crop if crop != "Unknown" else None,
                user_part_hint=viewpoint,
            )
            if part_pred.get("supported"):
                dis = part_pred.get("disease") or part_pred.get("top1_label")
                conf = float(part_pred.get("top1_confidence", 0.0)) / 100.0
                return WholePlantPredictionResult(
                    plant_detected=True,
                    crop=part_pred.get("crop_species", crop),
                    viewpoint=viewpoint,
                    supported=True,
                    disease=dis,
                    confidence=conf,
                    status=part_pred.get("status", "confident"),
                    fallback_reason="",
                    checkpoint_path=part_pred.get("model_path", f"models/vision/{viewpoint}_classifier.pt"),
                    top1_label=part_pred.get("top1_label", dis),
                    message=part_pred.get("top1_label", dis),
                    predictions=part_pred.get("predictions", []),
                    source_evidence=part_pred.get("source_evidence"),
                    margin=part_pred.get("margin", 0.0),
                    chemical_treatment_withheld=part_pred.get("chemical_treatment_withheld", False),
                    model_release_tier="VALIDATED MULTI-PART (Held-Out Benchmark)",
                    is_experimental=False,
                    debug_info=part_pred.get("debug_info", {}),
                )
            else:
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

        # 5. Whole-Plant Viewpoint Route
        crop_vp_map = {
            "Tomato": ["whole_plant", "canopy"],
            "Corn (Maize)": ["whole_plant", "canopy"],
        }
        supported_crops = list(crop_vp_map.keys())

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

        valid_vps = crop_vp_map.get(crop, ["whole_plant", "canopy"])
        v_mapped = viewpoint

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

        # 6. Whole-Plant Predictor
        result = self.predictor.predict(
            image_input=image_input,
            crop_hint=crop,
            viewpoint_hint=viewpoint,
            confidence_threshold=confidence_threshold,
        )
        return result
