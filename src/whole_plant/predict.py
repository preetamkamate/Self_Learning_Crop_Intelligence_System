"""
Whole-Plant & Field Photo Disease Predictor.
Independent inference engine designed exclusively for full-plant and field photos.
Never delegates to or references the leaf pathology model.
Strictly isolates crops: maize never produces tomato labels, untrained crops return safe abstention.
"""
import io
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from src.whole_plant.dataset import (
    RAW_WHOLE_PLANT_DIR,
    has_training_images,
)
from src.whole_plant.train import (
    DEFAULT_MODEL_PATH,
    WholePlantModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Authoritative agricultural research citations for supported whole-plant diseases
WHOLE_PLANT_AUTHORITATIVE_SOURCES = {
    "Tomato Fusarium Wilt": {
        "source_title": "ICAR-IIHR Technical Bulletin on Tomato Soil-borne Diseases",
        "organization": "ICAR - Indian Institute of Horticultural Research (IIHR)",
        "url": "https://iihr.res.in",
        "management_summary": "Soil solarization, bio-agent Trichoderma viride seed treatment, and resistant varieties.",
    },
    "Tomato Heat and Drought Stress": {
        "source_title": "ICAR Advisory on Abiotic Stress Management in Solanaceous Crops",
        "organization": "Indian Council of Agricultural Research (ICAR)",
        "url": "https://icar.org.in",
        "management_summary": "Drip irrigation scheduling, mulching, and anti-transpirant potassium foliar sprays.",
    },
    "Corn Stunt Disease": {
        "source_title": "ICAR-IIMR Protocol for Maize Spiroplasma Stunt Management",
        "organization": "ICAR - Indian Institute of Maize Research (IIMR)",
        "url": "https://iimr.icar.gov.in",
        "management_summary": "Vector control (leafhoppers) using thiamethoxam seed treatment and field sanitation.",
    },
    "Corn Fall Armyworm Damage": {
        "source_title": "ICAR-IIMR Integrated Management Guidelines for Spodoptera frugiperda",
        "organization": "ICAR - Indian Institute of Maize Research (IIMR)",
        "url": "https://iimr.icar.gov.in",
        "management_summary": "Pheromone traps, push-pull intercropping with Desmodium, and neem-based botanical spray.",
    },
    "Corn Nitrogen Deficiency Chlorosis": {
        "source_title": "ICAR Nutrient Management & Deficiency Diagnosis in Maize",
        "organization": "ICAR - Indian Institute of Maize Research (IIMR)",
        "url": "https://iimr.icar.gov.in",
        "management_summary": "Split application of urea along with bio-fertilizers (Azotobacter) based on leaf color chart.",
    },
}

UNSUPPORTED_CROP_MSG = "Valid whole plant detected, but this crop is not trained for diagnosis yet."


@dataclass
class WholePlantPredictionResult:
    """Standardized prediction structure for whole-plant field photos."""
    crop: str
    status: str  # 'healthy', 'disease', 'unsupported', 'uncertain', 'needs_training_data', 'out_of_distribution'
    confidence: float
    supported: bool
    disease: Optional[str] = None
    message: str = ""
    is_ood: bool = False
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    top1_label: str = ""
    viewpoint: str = "whole_plant"
    selected_model: str = "models/whole_plant/model.pt"
    model_path: str = "models/whole_plant/model.pt"
    fallback_reason: str = ""
    source_evidence: Optional[Dict[str, Any]] = None
    debug_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        label_display = self.top1_label or (self.disease if self.supported and self.disease else self.message)
        data["top1_label"] = label_display
        data["top1_confidence"] = round(self.confidence * 100, 2)
        data["plant_part"] = "whole_plant"
        data["crop_species"] = self.crop
        data["disease"] = self.disease if self.supported else None
        data["selected_model"] = self.selected_model if self.supported else "None"
        data["model_path"] = self.model_path if self.supported else "None"
        data["fallback_reason"] = self.fallback_reason or ("" if self.supported else self.message)
        data["predictions"] = self.predictions if self.supported else []
        data["is_healthy"] = (self.status == "healthy")
        data["source_evidence"] = self.source_evidence if self.supported else None
        data["debug_info"] = {
            "crop_route": self.crop,
            "viewpoint_route": self.viewpoint,
            "selected_model": data["selected_model"],
            "model_path": data["model_path"],
            "fallback_reason": data["fallback_reason"],
            "stage4_route_supported": self.supported,
            "disease": data["disease"],
            "is_ood": self.is_ood,
            "top1_confidence_pct": data["top1_confidence"],
            "status": self.status,
            "predicted_class": label_display,
        }
        return data


class WholePlantPredictor:
    """Inference engine for full-plant and field photographs."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.raw_dir = Path(raw_dir)

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        self.model: Optional[WholePlantModel] = None
        self.supported_crops: List[str] = []
        self.supported_diseases: List[str] = []
        self.crop_disease_map: Dict[str, List[str]] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.label_to_idx: Dict[str, int] = {}
        self._load_model_if_available()

    @property
    def is_trained(self) -> bool:
        """Returns True only when a local model file exists and is loaded."""
        return self.model is not None and len(self.idx_to_label) > 0

    def _load_model_if_available(self) -> None:
        """Load trained whole-plant model checkpoint if present locally."""
        if not self.model_path.exists():
            logging.info("No whole-plant model checkpoint found at %s", self.model_path)
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            num_classes = checkpoint.get("num_classes", len(checkpoint.get("classes", [])) or 9)
            self.model = WholePlantModel(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            self.idx_to_label = checkpoint.get("idx_to_label", {})
            # Ensure integer keys
            self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
            self.label_to_idx = checkpoint.get("label_to_idx", {})
            self.crop_disease_map = checkpoint.get("crop_disease_map", {})
            raw_crops = checkpoint.get("supported_crops", list(self.crop_disease_map.keys()))
            self.supported_crops = [c for c in raw_crops if c not in ["None", "Unknown", "none", "unknown"]]
            self.supported_diseases = checkpoint.get("supported_diseases", [])
            logging.info("Successfully loaded whole-plant model with %d classes from %s", num_classes, self.model_path)
        except Exception as e:
            logging.error("Failed to load whole-plant model from %s: %s", self.model_path, e)
            self.model = None

    def _preprocess(self, image_input: Union[Image.Image, Path, str, bytes]) -> Tuple[torch.Tensor, Image.Image]:
        """Convert input to normalized 224x224 RGB tensor and return PIL image."""
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = Image.new("RGB", (224, 224), color=(34, 139, 34))

        resized = pil_img.resize((224, 224))
        arr = torch.tensor(list(getattr(resized, "get_flattened_data", resized.getdata)()), dtype=torch.float32).view(224, 224, 3)
        tensor = arr.permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device), pil_img

    def normalize_crop_name(self, raw_crop: Optional[str]) -> str:
        """Normalize user crop string into canonical crop name."""
        if not raw_crop or str(raw_crop).strip() in ["", "None", "Unknown", "auto", "Auto-detect"]:
            return "Unknown"
        s = str(raw_crop).strip().lower()
        if "corn" in s or "maize" in s:
            return "Corn (Maize)"
        elif "tomato" in s:
            return "Tomato"
        elif "grape" in s:
            return "Grape"
        elif "potato" in s:
            return "Potato"
        elif "apple" in s:
            return "Apple"
        elif "cotton" in s:
            return "Cotton"
        elif "banana" in s:
            return "Banana"
        elif "sugarcane" in s:
            return "Sugarcane"
        elif "wheat" in s:
            return "Wheat"
        return raw_crop.strip().title()

    def predict(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop_hint: Optional[str] = None,
        viewpoint_hint: Optional[str] = None,
        confidence_threshold: float = 0.40,
    ) -> WholePlantPredictionResult:
        """
        Execute whole-plant inference with strict crop routing and zero cross-crop contamination.
        Step 1: Check training data and checkpoint presence.
        Step 2: Check Plant vs Non-Plant.
        Step 3: Check Viewpoint (Full-plant/field vs close-up single leaf).
        Step 4: Determine crop and verify against trained classes.
        Step 5: Diagnose only if crop was trained; otherwise return safe abstention.
        """
        norm_crop = self.normalize_crop_name(crop_hint)

        # 1. Training data & Checkpoint Gate
        if not has_training_images(self.raw_dir) or not self.is_trained:
            return WholePlantPredictionResult(
                crop=norm_crop,
                status="needs_training_data",
                confidence=0.0,
                supported=False,
                disease=None,
                top1_label="Whole-plant diagnosis is not trained yet. Add approved labeled whole-plant images before diagnosis.",
                message="Whole-plant diagnosis is not trained yet. Add approved labeled whole-plant images before diagnosis.",
                fallback_reason=f"No trained checkpoint found at {self.model_path}",
            )

        # 2. Run Forward Pass
        try:
            tensor, pil_img = self._preprocess(image_input)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        except Exception as e:
            logging.error("Inference exception in WholePlantPredictor: %s", e)
            return WholePlantPredictionResult(
                crop=norm_crop,
                status="uncertain",
                confidence=0.0,
                supported=False,
                disease=None,
                top1_label=UNSUPPORTED_CROP_MSG,
                message=UNSUPPORTED_CROP_MSG,
                fallback_reason=f"Execution error: {e}",
            )

        # 3. Determine Plant vs Non-Plant
        # Class 8 is Non-Plant Background Object if present in classes
        non_plant_idx = self.label_to_idx.get("Non-Plant Background Object", -1)
        p_non_plant = float(probs[non_plant_idx]) if non_plant_idx != -1 else 0.0

        # Quality / Blank screen check
        arr_img = np.array(pil_img, dtype=np.float32)
        if arr_img.std() < 3.0 or p_non_plant > 0.45:
            return WholePlantPredictionResult(
                crop="None",
                status="out_of_distribution",
                confidence=round(p_non_plant, 4),
                supported=False,
                disease=None,
                is_ood=True,
                top1_label="Non-plant image detected",
                message="Non-plant image detected",
                fallback_reason=f"Model detected Non-Plant Background Object ({p_non_plant*100:.1f}%)",
            )

        # 4. Determine Viewpoint (Full-plant/field vs close-up single leaf)
        vp_clean = str(viewpoint_hint).lower().strip() if viewpoint_hint else "whole_plant"
        if vp_clean in ["leaf", "foliar", "single_leaf", "close_up_leaf"]:
            return WholePlantPredictionResult(
                crop=norm_crop,
                status="unsupported",
                confidence=0.0,
                supported=False,
                disease=None,
                top1_label=UNSUPPORTED_CROP_MSG,
                message=UNSUPPORTED_CROP_MSG,
                fallback_reason=f"Non-whole-plant viewpoint '{vp_clean}' provided; whole-plant model only accepts field/whole-plant photos",
            )

        # 5. Determine Crop
        # Group model classes by crop
        corn_classes = [c for c in self.crop_disease_map.get("Corn (Maize)", []) if c in self.label_to_idx]
        tomato_classes = [c for c in self.crop_disease_map.get("Tomato", []) if c in self.label_to_idx]

        corn_prob = sum(float(probs[self.label_to_idx[c]]) for c in corn_classes)
        tomato_prob = sum(float(probs[self.label_to_idx[c]]) for c in tomato_classes)

        if norm_crop != "Unknown":
            active_crop = norm_crop
        else:
            if corn_prob >= 0.35 and corn_prob > tomato_prob:
                active_crop = "Corn (Maize)"
            elif tomato_prob >= 0.35 and tomato_prob > corn_prob:
                active_crop = "Tomato"
            else:
                active_crop = "Unknown"

        # 6. Verify Crop is Supported by Trained Model (Requirement 6, 8, 9, 10)
        # Check against self.supported_crops: ['Corn (Maize)', 'Tomato']
        is_supported_crop = any(
            active_crop.lower() in sc.lower() or sc.lower() in active_crop.lower()
            for sc in self.supported_crops
        )

        if not is_supported_crop or active_crop == "Unknown":
            # STRICT REQUIREMENT 9:
            # If crop is untrained, return ONLY:
            # "Valid whole plant detected, but this crop is not trained for diagnosis yet."
            return WholePlantPredictionResult(
                crop=active_crop,
                status="unsupported",
                confidence=0.0,
                supported=False,
                disease=None,  # NEVER return a disease for untrained crop!
                predictions=[],  # STRICT: Empty candidate list
                top1_label=UNSUPPORTED_CROP_MSG,
                message=UNSUPPORTED_CROP_MSG,
                fallback_reason=f"Crop '{active_crop}' was not in whole-plant training data. Safe abstention enforced.",
            )

        # 7. Strict Crop-Isolated Diagnosis (Requirement 10: Zero Cross-Crop Contamination)
        # Maize can NEVER access Tomato classes; Tomato can NEVER access Maize classes
        if active_crop == "Corn (Maize)":
            crop_candidate_classes = corn_classes
        else:
            crop_candidate_classes = tomato_classes

        if not crop_candidate_classes:
            return WholePlantPredictionResult(
                crop=active_crop,
                status="unsupported",
                confidence=0.0,
                supported=False,
                disease=None,
                predictions=[],
                top1_label=UNSUPPORTED_CROP_MSG,
                message=UNSUPPORTED_CROP_MSG,
                fallback_reason=f"No trained pathology classes for crop '{active_crop}'",
            )

        # Conditional Subspace Normalization: P(class | crop) = P(class) / sum_{k in crop} P(k)
        sub_probs = np.array([float(probs[self.label_to_idx[c]]) for c in crop_candidate_classes], dtype=np.float32)
        total_sub = float(np.sum(sub_probs))
        norm_sub_probs = sub_probs / total_sub if total_sub > 1e-9 else (np.ones(len(crop_candidate_classes)) / len(crop_candidate_classes))

        sorted_indices = np.argsort(norm_sub_probs)[::-1]
        top1_sub_idx = int(sorted_indices[0])
        top1_prob = float(norm_sub_probs[top1_sub_idx])
        top1_class = crop_candidate_classes[top1_sub_idx]

        # Candidate predictions strictly for confirmed crop
        ranked_predictions: List[Dict[str, Any]] = []
        for s_idx in sorted_indices:
            c_name = crop_candidate_classes[int(s_idx)]
            c_prob = float(norm_sub_probs[int(s_idx)])
            ranked_predictions.append({
                "label": c_name,
                "class": c_name,
                "prob": round(c_prob, 4),
                "probability": round(c_prob, 4),
                "confidence_percent": round(c_prob * 100, 2),
            })

        # Confidence Gate
        if top1_prob < confidence_threshold:
            return WholePlantPredictionResult(
                crop=active_crop,
                status="uncertain",
                confidence=round(top1_prob, 4),
                supported=False,
                disease=None,
                predictions=[],
                top1_label="Valid whole plant detected, but symptoms are uncertain.",
                message="Valid whole plant detected, but symptoms are uncertain.",
                fallback_reason=f"Top candidate '{top1_class}' confidence {top1_prob*100:.1f}% < threshold {confidence_threshold*100:.1f}%",
            )

        # Confirmed condition
        is_healthy = ("healthy" in top1_class.lower())
        disease_name = None if is_healthy else top1_class
        status_val = "healthy" if is_healthy else "disease"
        source_evidence = WHOLE_PLANT_AUTHORITATIVE_SOURCES.get(top1_class)

        return WholePlantPredictionResult(
            crop=active_crop,
            status=status_val,
            confidence=round(top1_prob, 4),
            supported=True,
            disease=disease_name,
            predictions=ranked_predictions,
            top1_label=top1_class,
            message=f"Whole-plant diagnosis: {top1_class}" if not is_healthy else f"Healthy {active_crop} whole plant",
            source_evidence=source_evidence,
            fallback_reason="",
        )
