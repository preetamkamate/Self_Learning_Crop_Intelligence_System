"""
Modular Pest & Insect Damage Detection Module.
Detects pest symptoms and infesting organisms strictly when verified real labeled data exists.
Never guesses pest species or prescribes chemical pesticides on unverified routes.
Provides honest macro photo guidance for insect scouting and Integrated Pest Management (IPM).
"""
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

from src.modular_vision.photo_guidance import get_photo_guidance

logger = logging.getLogger(__name__)

# Known agricultural pests taxonomy
AGRICULTURAL_PESTS: Dict[str, Dict[str, Any]] = {
    "Spider Mites": {
        "crops": ["Tomato", "Apple", "Bell Pepper (Capsicum)"],
        "parts": ["leaf"],
        "symptoms": "Fine silvery stippling on leaf surface, dense webbing on leaf underside, bronzing.",
        "verified_in_active_model": False,  # Not in active 14-class verified checkpoint
    },
    "Fall Armyworm": {
        "crops": ["Corn (Maize)"],
        "parts": ["leaf", "canopy", "whole_plant"],
        "symptoms": "Large ragged shot-holes in whorl leaves, sawdust-like frass inside funnel.",
        "verified_in_active_model": False,  # Quarantined synthetic diagrams; awaiting real field dataset
    },
    "Fruit Borer / Helicoverpa": {
        "crops": ["Tomato", "Bell Pepper (Capsicum)"],
        "parts": ["fruit"],
        "symptoms": "Circular boreholes near calyx, premature fruit rotting, caterpillar entry.",
        "verified_in_active_model": False,
    },
    "Stem Borer": {
        "crops": ["Corn (Maize)", "Rice (Paddy)", "Sugarcane"],
        "parts": ["stem_branch"],
        "symptoms": "Dead heart in seedling stage, white earhead at reproductive stage, stem boreholes.",
        "verified_in_active_model": False,
    },
    "Aphids / Whiteflies": {
        "crops": ["Tomato", "Cassava", "Bell Pepper (Capsicum)"],
        "parts": ["leaf", "stem_branch"],
        "symptoms": "Sticky honeydew, black sooty mold, curling of apical leaves, insect clusters.",
        "verified_in_active_model": False,
    },
}

PEST_KEYWORDS = [
    "pest", "insect", "mite", "spider mite", "caterpillar", "worm", "armyworm",
    "borer", "aphid", "whitefly", "thrips", "hopper", "bug", "grub", "larva",
    "pupa", "frass", "honeydew", "webbing", "infestation", "eating leaf",
]


@dataclass
class PestDetectionResult:
    """Standardized pest detection and scouting result."""
    pest_suspected: bool
    pest_label: Optional[str]
    confidence: float
    supported: bool
    status: str
    status_reason: str
    photo_guidance: Dict[str, Any]
    ipm_scouting_advice: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pest_suspected": self.pest_suspected,
            "pest_label": self.pest_label,
            "confidence": self.confidence,
            "supported": self.supported,
            "status": self.status,
            "status_reason": self.status_reason,
            "photo_guidance": self.photo_guidance,
            "ipm_scouting_advice": self.ipm_scouting_advice,
            "metadata": self.metadata,
        }


class PestDetector:
    """Modular Pest Detection Component."""

    def __init__(self, verified_pest_classes: Optional[List[str]] = None):
        # Currently, active 14-class verified model contains foliar pathology (fungal/bacterial/viral)
        self.verified_pest_classes = verified_pest_classes or []

    def is_pest_query(self, user_text: Optional[str] = None, symptom_desc: Optional[str] = None) -> bool:
        """Check whether user text or symptom description suggests an insect pest."""
        combined = f"{user_text or ''} {symptom_desc or ''}".lower()
        for kw in PEST_KEYWORDS:
            if kw in combined:
                return True
        return False

    def detect_pest(
        self,
        image_input: Optional[Union[Image.Image, str, Any]] = None,
        crop: str = "Unknown",
        viewpoint: str = "leaf",
        user_text: Optional[str] = None,
        symptom_desc: Optional[str] = None,
    ) -> PestDetectionResult:
        """
        Screen for pests and evaluate against verified real labeled pest models.
        """
        pest_indicated = self.is_pest_query(user_text, symptom_desc)
        guidance = get_photo_guidance(viewpoint=viewpoint, crop=crop, is_pest=True)

        # If a verified real labeled pest model is loaded and deployed for this (crop, viewpoint)
        if self.verified_pest_classes:
            # Here real pest model inference would run if deployed
            pass

        # Honest safety gate: No verified real-camera pest checkpoint is active
        # Never guess chemical insecticides or specific pest species
        reason = (
            f"Pest detection for '{crop}' ({viewpoint}) requires verified real labeled insect/damage training data. "
            "To prevent chemical pesticide misuse, automatic pest diagnosis is safely withheld."
        )

        ipm_advice = (
            "Integrated Pest Management (IPM) Scouting:\n"
            "• Turn over leaves to inspect for active insect clusters or webbing.\n"
            "• Install yellow sticky traps (for whiteflies/aphids) or pheromone traps.\n"
            "• Spray a 5% neem seed kernel extract (NSKE) or azadirachtin as a safe botanical deterrent.\n"
            "• Take a macro close-up of the actual insect or collect a specimen for KVK identification."
        )

        return PestDetectionResult(
            pest_suspected=pest_indicated,
            pest_label=None,
            confidence=0.0,
            supported=False,
            status="pest_guidance_provided",
            status_reason=reason,
            photo_guidance=guidance,
            ipm_scouting_advice=ipm_advice,
            metadata={
                "chemical_pesticides_withheld": True,
                "requires_macro_photo": True,
            },
        )
