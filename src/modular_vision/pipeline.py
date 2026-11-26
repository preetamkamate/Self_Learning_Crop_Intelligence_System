"""
Unified Modular Crop Vision Pipeline.
Coordinates:
1. Modular Crop ID (CropIdentifier)
2. Viewpoint & Organ Router (ViewpointRouter for leaf, stem/branch, fruit, canopy, whole plant, root)
3. Pest Detection (PestDetector)
4. Disease Classification (ModularDiseaseClassifier for leaf & field pathology)
5. Honest Photo Guidance (PhotoGuidanceEngine)

Core Rule:
Enables diagnosis ONLY with verified real labeled data and a tested exact route;
otherwise provides honest photo guidance. Never guesses disease or chemical treatment.
"""
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

from src.modular_vision.crop_id import CropIdentifier
from src.modular_vision.disease_classifier import ModularDiseaseClassifier
from src.modular_vision.pest_detector import PestDetector
from src.modular_vision.photo_guidance import get_photo_guidance
from src.modular_vision.viewpoint_router import ViewpointRouter

logger = logging.getLogger(__name__)


@dataclass
class ModularVisionResult:
    """Unified diagnosis and guidance output structure."""
    plant_detected: bool = True
    crop: str = "Unknown"
    viewpoint: str = "leaf"
    supported: bool = False
    disease: Optional[str] = None
    confidence: float = 0.0
    status: str = "unsupported"
    fallback_reason: str = ""
    checkpoint_path: str = "None"
    selected_model: str = "None"
    top1_label: str = ""
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    pest_info: Optional[Dict[str, Any]] = None
    photo_guidance: Optional[Dict[str, Any]] = None
    source_evidence: Optional[Dict[str, Any]] = None
    is_healthy: bool = False
    margin: float = 0.0
    chemical_treatment_withheld: bool = False
    model_release_tier: str = "EXPERIMENTAL (58.33% Held-Out Accuracy)"
    is_experimental: bool = True
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full backward-compatibility fields."""
        data = {
            "plant_detected": self.plant_detected,
            "crop": self.crop,
            "crop_species": self.crop,
            "viewpoint": self.viewpoint,
            "plant_part": self.viewpoint,
            "supported": self.supported,
            "disease": self.disease,
            "confidence": self.confidence,
            "top1_confidence": round(self.confidence * 100, 2),
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "checkpoint_path": self.checkpoint_path,
            "model_path": self.checkpoint_path,
            "selected_model": self.selected_model,
            "top1_label": self.top1_label or (self.disease if self.supported and self.disease else self.fallback_reason),
            "predictions": self.predictions,
            "pest_info": self.pest_info,
            "photo_guidance": self.photo_guidance,
            "source_evidence": self.source_evidence,
            "is_healthy": self.is_healthy,
            "margin": self.margin,
            "chemical_treatment_withheld": self.chemical_treatment_withheld,
            "model_release_tier": self.model_release_tier,
            "is_experimental": self.is_experimental,
            "debug_info": {
                "crop_route": self.crop,
                "viewpoint_route": self.viewpoint,
                "selected_model": self.selected_model,
                "checkpoint_path": self.checkpoint_path,
                "stage4_route_supported": self.supported,
                "fallback_reason": self.fallback_reason,
                "disease": self.disease,
                "status": self.status,
                "margin": self.margin,
                "chemical_treatment_withheld": self.chemical_treatment_withheld,
                "model_release_tier": self.model_release_tier,
                **self.debug_info,
            },
        }
        return data


class ModularCropVisionPipeline:
    """Unified Modular Crop Vision Architecture."""

    def __init__(
        self,
        crop_identifier: Optional[CropIdentifier] = None,
        viewpoint_router: Optional[ViewpointRouter] = None,
        pest_detector: Optional[PestDetector] = None,
        disease_classifier: Optional[ModularDiseaseClassifier] = None,
    ):
        self.crop_identifier = crop_identifier or CropIdentifier()
        self.viewpoint_router = viewpoint_router or ViewpointRouter()
        self.pest_detector = pest_detector or PestDetector()
        self.disease_classifier = disease_classifier or ModularDiseaseClassifier()

    def diagnose(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop_hint: Optional[str] = None,
        viewpoint_hint: Optional[str] = None,
        user_text: Optional[str] = None,
        symptom_desc: Optional[str] = None,
        confidence_threshold: float = 0.25,
    ) -> ModularVisionResult:
        """
        Execute modular agricultural vision evaluation.
        1. Crop ID
        2. Viewpoint & Organ Route
        3. Pest Screening
        4. Verified Pathology Diagnosis (or Honest Photo Guidance)
        """
        # Step 1: Crop Identification
        crop_res = self.crop_identifier.identify_crop(
            image_input=image_input,
            crop_hint=crop_hint,
        )

        if not crop_res.is_plant:
            guidance = get_photo_guidance(viewpoint="leaf", is_non_plant=True)
            return ModularVisionResult(
                plant_detected=False,
                crop="None",
                viewpoint="non_plant",
                supported=False,
                disease=None,
                confidence=0.0,
                status="out_of_distribution",
                fallback_reason="Non-plant image detected.",
                top1_label="Non-plant image detected",
                checkpoint_path="None",
                selected_model="None",
                photo_guidance=guidance,
            )

        crop = crop_res.canonical_name

        # Step 2: Viewpoint Routing across the 6 anatomical viewpoints
        # (leaf, stem_branch, fruit, canopy, whole_plant, root)
        vp_res = self.viewpoint_router.route_viewpoint(
            crop=crop,
            viewpoint_hint=viewpoint_hint,
        )
        canonical_vp = vp_res.canonical_viewpoint

        # Step 3: Pest Detection Screening
        is_pest_query = self.pest_detector.is_pest_query(user_text, symptom_desc)
        if is_pest_query:
            pest_res = self.pest_detector.detect_pest(
                image_input=image_input,
                crop=crop,
                viewpoint=canonical_vp,
                user_text=user_text,
                symptom_desc=symptom_desc,
            )
            return ModularVisionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=canonical_vp,
                supported=pest_res.supported,
                disease=pest_res.pest_label,
                confidence=pest_res.confidence,
                status=pest_res.status,
                fallback_reason=pest_res.status_reason,
                top1_label="Pest scouting photo guidance provided",
                checkpoint_path="None",
                selected_model="PestScoutingEngine",
                pest_info=pest_res.to_dict(),
                photo_guidance=pest_res.photo_guidance,
                debug_info={"pest_suspected": True, "ipm_advice": pest_res.ipm_scouting_advice},
            )

        # Step 4: Route-Specific Disease Diagnosis
        # Only diagnose if exact verified route is deployed; otherwise provide honest guidance

        # Case A: Unsupported crop
        if not crop_res.is_verified_dataset_crop or crop == "Unknown":
            guidance = get_photo_guidance(viewpoint=canonical_vp, crop=crop, is_unsupported_crop=True)
            reason = (
                f"Valid plant image detected, but diagnosis is not trained yet for crop '{crop}'."
                if crop != "Unknown"
                else "Plant image detected, but crop species could not be verified."
            )
            return ModularVisionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=canonical_vp,
                supported=False,
                disease=None,
                confidence=0.0,
                status="unsupported_crop",
                fallback_reason=reason,
                top1_label=reason,
                checkpoint_path="None",
                selected_model="None",
                photo_guidance=guidance,
                debug_info={"crop_verified": False, "crop": crop},
            )

        # Case B: Verified crop + Leaf viewpoint -> Call verified foliar model
        if canonical_vp == "leaf":
            leaf_res = self.disease_classifier.classify_leaf_disease(
                image_input=image_input,
                crop=crop,
                confidence_threshold=confidence_threshold,
            )
            top1 = (
                leaf_res.disease
                if (leaf_res.supported and leaf_res.disease)
                else (f"Healthy {crop} Leaf" if leaf_res.is_healthy else leaf_res.status_reason)
            )
            return ModularVisionResult(
                plant_detected=True,
                crop=crop,
                viewpoint="leaf",
                supported=leaf_res.supported,
                disease=leaf_res.disease,
                confidence=leaf_res.confidence,
                status=leaf_res.status,
                fallback_reason=leaf_res.status_reason,
                checkpoint_path=leaf_res.checkpoint_path,
                selected_model=leaf_res.selected_model,
                top1_label=top1,
                predictions=leaf_res.predictions,
                source_evidence=leaf_res.source_evidence,
                photo_guidance=leaf_res.photo_guidance,
                is_healthy=leaf_res.is_healthy,
                margin=getattr(leaf_res, "margin", 0.0),
                chemical_treatment_withheld=getattr(leaf_res, "chemical_treatment_withheld", False),
                model_release_tier=getattr(leaf_res, "model_release_tier", "EXPERIMENTAL (58.33% Held-Out Accuracy)"),
                is_experimental=getattr(leaf_res, "is_experimental", True),
                debug_info=leaf_res.debug_info,
            )

        # Case C: Field views (whole_plant, canopy)
        if canonical_vp in ["whole_plant", "canopy"]:
            field_res = self.disease_classifier.classify_field_disease(
                image_input=image_input,
                crop=crop,
                viewpoint=canonical_vp,
                confidence_threshold=confidence_threshold,
            )
            top1 = (
                field_res.disease
                if (field_res.supported and field_res.disease)
                else (f"Healthy {crop} {canonical_vp.replace('_', ' ').title()}" if field_res.is_healthy else field_res.status_reason)
            )
            return ModularVisionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=canonical_vp,
                supported=field_res.supported,
                disease=field_res.disease,
                confidence=field_res.confidence,
                status=field_res.status,
                fallback_reason=field_res.status_reason,
                top1_label=top1,
                checkpoint_path=field_res.checkpoint_path,
                selected_model=field_res.selected_model,
                predictions=field_res.predictions,
                source_evidence=field_res.source_evidence,
                photo_guidance=field_res.photo_guidance,
                is_healthy=field_res.is_healthy,
                margin=getattr(field_res, "margin", 0.0),
                chemical_treatment_withheld=getattr(field_res, "chemical_treatment_withheld", False),
                model_release_tier=getattr(field_res, "model_release_tier", "VALIDATED MULTI-PART (Held-Out Benchmark)"),
                is_experimental=getattr(field_res, "is_experimental", False),
                debug_info=field_res.debug_info,
            )

        # Case D: Specialized organ views (stem_branch, fruit, root, bark_trunk)
        if canonical_vp in ["stem_branch", "fruit", "root", "bark_trunk"]:
            organ_res = self.disease_classifier.classify_stem_fruit_root_disease(
                image_input=image_input,
                crop=crop,
                viewpoint=canonical_vp,
            )
            top1 = (
                organ_res.disease
                if (organ_res.supported and organ_res.disease)
                else (f"Healthy {crop} {canonical_vp.replace('_', ' ').title()}" if organ_res.is_healthy else organ_res.status_reason)
            )
            return ModularVisionResult(
                plant_detected=True,
                crop=crop,
                viewpoint=canonical_vp,
                supported=organ_res.supported,
                disease=organ_res.disease,
                confidence=organ_res.confidence,
                status=organ_res.status,
                fallback_reason=organ_res.status_reason,
                top1_label=top1,
                checkpoint_path=organ_res.checkpoint_path,
                selected_model=organ_res.selected_model,
                predictions=organ_res.predictions,
                source_evidence=organ_res.source_evidence,
                photo_guidance=organ_res.photo_guidance,
                is_healthy=organ_res.is_healthy,
                margin=getattr(organ_res, "margin", 0.0),
                chemical_treatment_withheld=getattr(organ_res, "chemical_treatment_withheld", False),
                model_release_tier=getattr(organ_res, "model_release_tier", "VALIDATED MULTI-PART (Held-Out Benchmark)"),
                is_experimental=getattr(organ_res, "is_experimental", False),
                debug_info=organ_res.debug_info,
            )

        # Case E: Unknown viewpoint / Auto-detect
        from src.vision.predict import PlantDiseasePredictor
        auto_pred = PlantDiseasePredictor().predict(
            image_input=image_input,
            user_crop_hint=crop if crop != "Unknown" else None,
        )
        if auto_pred.get("supported"):
            return ModularVisionResult(
                plant_detected=True,
                crop=auto_pred.get("crop_species", crop),
                viewpoint=auto_pred.get("plant_part", "whole_plant"),
                supported=True,
                disease=auto_pred.get("disease"),
                confidence=auto_pred.get("top1_confidence", 0.0) / 100.0,
                status=auto_pred.get("status", "confident"),
                fallback_reason="",
                top1_label=auto_pred.get("top1_label", ""),
                checkpoint_path=auto_pred.get("model_path", "models/vision/whole_plant_classifier.pt"),
                selected_model=auto_pred.get("selected_model", "DINOv2 Multi-Part Probe"),
                predictions=auto_pred.get("predictions", []),
                source_evidence=auto_pred.get("source_evidence"),
                is_healthy=auto_pred.get("is_healthy", False),
                margin=auto_pred.get("margin", 0.0),
                chemical_treatment_withheld=auto_pred.get("chemical_treatment_withheld", False),
                model_release_tier="VALIDATED MULTI-PART (Auto-Detected)",
                is_experimental=False,
                debug_info=auto_pred.get("debug_info", {}),
            )

        guidance = get_photo_guidance(viewpoint="whole_plant", crop=crop)
        reason = f"Viewpoint '{vp_res.raw_viewpoint}' diagnosis is not trained yet for crop '{crop}'."
        return ModularVisionResult(
            plant_detected=True,
            crop=crop,
            viewpoint="unknown",
            supported=False,
            disease=None,
            confidence=0.0,
            status="unsupported",
            fallback_reason=reason,
            top1_label=reason,
            checkpoint_path="None",
            selected_model="None",
            photo_guidance=guidance,
        )
