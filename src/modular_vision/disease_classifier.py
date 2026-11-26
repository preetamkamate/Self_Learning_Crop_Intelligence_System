"""
Modular Crop Disease Classification Engine.
Handles fine-grained foliar (leaf) and field (canopy / whole plant) pathology.
Enforces strict rule:
Diagnoses ONLY with verified real labeled data and an exact tested route.
Never guesses diseases or chemical treatments on unverified routes.
"""
from dataclasses import dataclass, field
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src.modular_vision.photo_guidance import get_photo_guidance
from src.whole_plant.predict import (
    DEFAULT_MODEL_PATH,
    WHOLE_PLANT_AUTHORITATIVE_SOURCES,
    WholePlantPredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class DiseaseClassificationResult:
    """Standardized disease classification result."""
    disease: Optional[str]
    confidence: float
    supported: bool
    status: str
    status_reason: str
    crop: str
    viewpoint: str
    checkpoint_path: str
    selected_model: str
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    source_evidence: Optional[Dict[str, Any]] = None
    photo_guidance: Optional[Dict[str, Any]] = None
    is_healthy: bool = False
    margin: float = 0.0
    chemical_treatment_withheld: bool = False
    model_release_tier: str = "EXPERIMENTAL (58.33% Held-Out Accuracy)"
    is_experimental: bool = True
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disease": self.disease,
            "confidence": self.confidence,
            "supported": self.supported,
            "status": self.status,
            "status_reason": self.status_reason,
            "crop": self.crop,
            "viewpoint": self.viewpoint,
            "checkpoint_path": self.checkpoint_path,
            "selected_model": self.selected_model,
            "predictions": self.predictions,
            "source_evidence": self.source_evidence,
            "photo_guidance": self.photo_guidance,
            "is_healthy": self.is_healthy,
            "margin": self.margin,
            "chemical_treatment_withheld": self.chemical_treatment_withheld,
            "model_release_tier": self.model_release_tier,
            "is_experimental": self.is_experimental,
            "debug_info": self.debug_info,
        }


class ModularDiseaseClassifier:
    """
    Modular Disease Classifier covering Leaf, Field (Whole Plant / Canopy),
    Stem, Fruit, and Root pathology routes.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.predictor = WholePlantPredictor(model_path=self.model_path, device=device)

    @property
    def is_trained(self) -> bool:
        return self.predictor.is_trained

    @property
    def supported_crops(self) -> List[str]:
        return self.predictor.supported_crops or ["Apple", "Cassava", "Corn (Maize)", "Tomato"]

    def classify_leaf_disease(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop: str,
        confidence_threshold: float = 0.40,
    ) -> DiseaseClassificationResult:
        """
        Diagnose foliar (leaf) pathology using verified real labeled model.
        """
        # 1. Gate: Verify crop is in supported training list
        if crop not in self.supported_crops:
            guidance = get_photo_guidance(viewpoint="leaf", crop=crop, is_unsupported_crop=True)
            return DiseaseClassificationResult(
                disease=None,
                confidence=0.0,
                supported=False,
                status="unsupported_crop",
                status_reason=f"Foliar diagnosis is not trained yet for crop '{crop}'.",
                crop=crop,
                viewpoint="leaf",
                checkpoint_path="None",
                selected_model="None",
                predictions=[],
                source_evidence=None,
                photo_guidance=guidance,
            )

        # 2. Gate: Verify model checkpoint is loaded
        if not self.predictor.is_trained:
            guidance = get_photo_guidance(viewpoint="leaf", crop=crop)
            return DiseaseClassificationResult(
                disease=None,
                confidence=0.0,
                supported=False,
                status="needs_training_data",
                status_reason="Verified foliar model checkpoint is not loaded.",
                crop=crop,
                viewpoint="leaf",
                checkpoint_path=str(self.model_path),
                selected_model="None",
                predictions=[],
                source_evidence=None,
                photo_guidance=guidance,
            )

        # 3. Predict foliar pathology strictly within confirmed crop subspace
        pred_res = self.predictor.predict(
            image_input=image_input,
            crop_hint=crop,
            viewpoint_hint="leaf",
            confidence_threshold=confidence_threshold,
        )

        return DiseaseClassificationResult(
            disease=pred_res.disease,
            confidence=pred_res.confidence,
            supported=pred_res.supported,
            status=pred_res.status,
            status_reason=pred_res.fallback_reason or ("Verified diagnosis." if pred_res.supported else "Abstained."),
            crop=pred_res.crop,
            viewpoint="leaf",
            checkpoint_path=pred_res.checkpoint_path,
            selected_model=pred_res.selected_model,
            predictions=pred_res.predictions,
            source_evidence=pred_res.source_evidence,
            photo_guidance=None if pred_res.supported else get_photo_guidance(viewpoint="leaf", crop=crop),
            is_healthy=(pred_res.status == "healthy"),
            margin=getattr(pred_res, "margin", 0.0),
            chemical_treatment_withheld=getattr(pred_res, "chemical_treatment_withheld", False),
            model_release_tier=getattr(pred_res, "model_release_tier", "EXPERIMENTAL (58.33% Held-Out Accuracy)"),
            is_experimental=getattr(pred_res, "is_experimental", True),
            debug_info=pred_res.debug_info,
        )

    def classify_field_disease(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop: str,
        viewpoint: str,  # 'whole_plant' or 'canopy'
    ) -> DiseaseClassificationResult:
        """
        Diagnose field whole-plant or canopy views.
        Requires verified real-camera field training dataset.
        If no verified field checkpoint exists, safely abstains with honest photo guidance.
        """
        guidance = get_photo_guidance(viewpoint=viewpoint, crop=crop)
        reason = (
            f"Field {viewpoint.replace('_', ' ')} diagnosis for '{crop}' is not trained yet with verified real field data. "
            "To prevent incorrect chemical treatments, automatic diagnosis is safely withheld."
        )

        return DiseaseClassificationResult(
            disease=None,
            confidence=0.0,
            supported=False,
            status="unsupported_viewpoint",
            status_reason=reason,
            crop=crop,
            viewpoint=viewpoint,
            checkpoint_path="None",
            selected_model="None",
            predictions=[],
            source_evidence=None,
            photo_guidance=guidance,
            is_healthy=False,
            debug_info={
                "crop": crop,
                "viewpoint": viewpoint,
                "verified_field_checkpoint_present": False,
                "fallback_reason": reason,
            },
        )

    def classify_stem_fruit_root_disease(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop: str,
        viewpoint: str,  # 'stem_branch', 'fruit', or 'root'
    ) -> DiseaseClassificationResult:
        """
        Diagnose stem, fruit, or root views.
        Requires verified real labeled checkpoints for that exact organ.
        If unverified, safely abstains with honest photo guidance. Never guesses diseases.
        """
        guidance = get_photo_guidance(viewpoint=viewpoint, crop=crop)
        part_name = viewpoint.replace("_", " ").title()
        reason = (
            f"Plant image detected, but {part_name} diagnosis for '{crop}' is not trained yet with verified real labeled data. "
            "Automatic disease prescription is withheld to prevent crop injury."
        )

        return DiseaseClassificationResult(
            disease=None,
            confidence=0.0,
            supported=False,
            status="unsupported_organ_view",
            status_reason=reason,
            crop=crop,
            viewpoint=viewpoint,
            checkpoint_path="None",
            selected_model="None",
            predictions=[],
            source_evidence=None,
            photo_guidance=guidance,
            is_healthy=False,
            debug_info={
                "crop": crop,
                "viewpoint": viewpoint,
                "verified_organ_checkpoint_present": False,
                "fallback_reason": reason,
            },
        )
