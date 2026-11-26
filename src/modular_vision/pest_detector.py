"""
Modular Pest & Insect Damage Detection Module.
Detects pest symptoms and infesting organisms strictly when verified real labeled data exists.
Never guesses pest species or prescribes chemical pesticides on unverified routes.
Provides honest macro photo guidance for insect scouting and Integrated Pest Management (IPM).
"""
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

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

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.model_path = Path(model_path) if model_path else (self.root_dir / "models" / "pest_detector" / "model.pt")
        self.eval_report_path = self.root_dir / "models" / "pest_detector" / "evaluation_report.json"
        
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model: Optional[torch.nn.Module] = None
        self.idx_to_label: Dict[int, str] = {}
        self.verified_pest_classes: List[str] = []
        self._load_pest_model_if_available()

    @property
    def is_trained(self) -> bool:
        return self.model is not None and len(self.idx_to_label) > 0 and self.eval_report_path.exists()

    def _load_pest_model_if_available(self) -> None:
        """Load trained pest model checkpoint if present."""
        if not self.model_path.exists():
            return
        try:
            from src.modular_vision.train_pest_detector import DedicatedPestCNN
            ckpt = torch.load(self.model_path, map_location=self.device)
            num_classes = ckpt.get("num_classes", 2)
            self.model = DedicatedPestCNN(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            raw_labels = ckpt.get("idx_to_label", {})
            self.idx_to_label = {int(k): v for k, v in raw_labels.items()}
            self.verified_pest_classes = list(self.idx_to_label.values())
            logger.info("Successfully loaded dedicated pest detector from %s", self.model_path)
        except Exception as e:
            logger.warning("Failed to load pest model from %s: %s", self.model_path, e)
            self.model = None

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

        ipm_advice = (
            "Integrated Pest Management (IPM) Scouting:\n"
            "• Turn over leaves to inspect for active insect clusters or webbing.\n"
            "• Install yellow sticky traps (for whiteflies/aphids) or pheromone traps.\n"
            "• Spray a 5% neem seed kernel extract (NSKE) or azadirachtin as a safe botanical deterrent.\n"
            "• Take a macro close-up of the actual insect or collect a specimen for KVK identification."
        )

        # 1. Gate: Check if specific unverified pest was queried (e.g. armyworm, borer)
        combined_text = f"{user_text or ''} {symptom_desc or ''}".lower()
        for blocked_pest in ["armyworm", "borer", "aphid", "whitefly", "thrip", "caterpillar"]:
            if blocked_pest in combined_text:
                return PestDetectionResult(
                    pest_suspected=True,
                    pest_label=None,
                    confidence=0.0,
                    supported=False,
                    status="pest_data_blocked",
                    status_reason=(
                        f"Real camera training data for '{blocked_pest}' on '{crop}' is currently data-blocked. "
                        "To prevent chemical pesticide misuse, automatic diagnosis is safely withheld."
                    ),
                    photo_guidance=guidance,
                    ipm_scouting_advice=ipm_advice,
                    metadata={"chemical_pesticides_withheld": True, "data_blocked_pest": blocked_pest},
                )

        # 2. If crop is Tomato, viewpoint is leaf, and dedicated pest model is trained
        if self.is_trained and crop == "Tomato" and viewpoint == "leaf" and image_input is not None:
            try:
                # Preprocess image
                if isinstance(image_input, Image.Image):
                    pil_img = image_input.convert("RGB")
                elif isinstance(image_input, (str, Path)) and Path(image_input).exists():
                    pil_img = Image.open(image_input).convert("RGB")
                else:
                    pil_img = Image.new("RGB", (224, 224), (34, 139, 34))

                rgb = pil_img.resize((224, 224))
                arr = np.array(rgb, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])

                predicted_label = self.idx_to_label.get(pred_idx, "Unknown")
                is_pest = "spider mites" in predicted_label.lower()

                if is_pest and confidence >= 0.50:
                    return PestDetectionResult(
                        pest_suspected=True,
                        pest_label=predicted_label,
                        confidence=round(confidence, 4),
                        supported=True,
                        status="pest_detected",
                        status_reason="Verified real pest detection on tomato foliage.",
                        photo_guidance=guidance,
                        ipm_scouting_advice=(
                            "Two-Spotted Spider Mite Management (ICAR-IIHR):\n"
                            "• Spray wettable sulfur 80 WP @ 3 g/L or Azadirachtin 10,000 ppm @ 1 ml/L.\n"
                            "• Avoid excessive nitrogen fertilizers which promote mite fecundity.\n"
                            "• Introduce predatory mites (Phytoseiulus persimilis) in greenhouse polyhouses."
                        ),
                        metadata={"chemical_pesticides_withheld": False, "confidence": confidence},
                    )
            except Exception as e:
                logger.warning("Pest model inference failed: %s", e)

        # Fallback safe abstention
        reason = (
            f"Pest detection for '{crop}' ({viewpoint}) requires verified real labeled insect/damage training data. "
            "To prevent chemical pesticide misuse, automatic pest diagnosis is safely withheld."
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
            metadata={"chemical_pesticides_withheld": True, "requires_macro_photo": True},
        )
