"""
Whole-Plant & Field Photo Disease Predictor.
Independent inference engine designed exclusively for full-plant and canopy views.
Never delegates to or references the leaf pathology model.
"""
import io
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


@dataclass
class WholePlantPredictionResult:
    """Standardized prediction structure for whole-plant field photos."""
    crop: str
    status: str  # 'healthy', 'disease', 'unsupported', 'uncertain', 'needs_training_data'
    confidence: float
    supported: bool
    disease: Optional[str] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Compatibility fields for unified UI telemetry
        selected_model = str(self.model_path if hasattr(self, "model_path") else DEFAULT_MODEL_PATH) if self.supported else "None"
        fallback_reason = "" if self.supported else self.message
        data["top1_label"] = self.disease if self.supported and self.disease else self.message
        data["top1_confidence"] = round(self.confidence * 100, 2)
        data["plant_part"] = "whole_plant"
        data["crop_species"] = self.crop
        data["disease"] = self.disease if self.supported else None
        data["selected_model"] = selected_model
        data["model_path"] = selected_model
        data["fallback_reason"] = fallback_reason
        data["predictions"] = [{"label": self.disease, "prob": round(self.confidence, 4)}] if (self.supported and self.disease) else []
        data["source_evidence"] = None if not self.supported else data.get("source_evidence")
        data["debug_info"] = {
            "crop_route": self.crop,
            "viewpoint_route": "whole_plant",
            "selected_model": selected_model,
            "model_path": selected_model,
            "fallback_reason": fallback_reason,
            "stage4_route_supported": self.supported,
            "disease": data["disease"],
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
            num_classes = checkpoint.get("num_classes", 2)
            self.model = WholePlantModel(num_classes=num_classes).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()

            self.idx_to_label = checkpoint.get("idx_to_label", {})
            self.label_to_idx = checkpoint.get("label_to_idx", {})
            self.crop_disease_map = checkpoint.get("crop_disease_map", {})
            self.supported_crops = checkpoint.get("supported_crops", list(self.crop_disease_map.keys()))
            self.supported_diseases = checkpoint.get("supported_diseases", [])
            logging.info("Successfully loaded whole-plant model with %d classes.", num_classes)
        except Exception as e:
            logging.error("Failed to load whole-plant model from %s: %s", self.model_path, e)
            self.model = None

    def _preprocess(self, image_input: Union[Image.Image, Path, str, bytes]) -> torch.Tensor:
        """Convert input to normalized 224x224 RGB tensor."""
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            pil_img = Image.new("RGB", (224, 224), color=(34, 139, 34))

        resized = pil_img.resize((224, 224))
        arr = torch.tensor(list(getattr(resized, "get_flattened_data", resized.getdata)()), dtype=torch.float32).view(224, 224, 3)
        tensor = arr.permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image_input: Union[Image.Image, Path, str, bytes],
        crop_hint: Optional[str] = None,
        confidence_threshold: float = 0.50,
    ) -> WholePlantPredictionResult:
        """
        Predict whole-plant condition from a full plant/field photo.
        - If raw images are missing or model is untrained: returns needs_training_data notice.
        - If unsupported or uncertain: returns no disease name.
        - Only returns confirmed disease if crop and disease are supported by this model.
        """
        # 1. Check training data availability and model existence
        if not has_training_images(self.raw_dir) or not self.is_trained:
            return WholePlantPredictionResult(
                crop=crop_hint or "Unknown",
                status="needs_training_data",
                confidence=0.0,
                supported=False,
                disease=None,
                message="Whole-plant model needs training data",
            )

        # 2. Run inference
        try:
            tensor = self._preprocess(image_input)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0)
                conf, pred_idx = torch.max(probs, dim=0)

            confidence = float(conf.item())
            pred_class = self.idx_to_label.get(int(pred_idx.item()), "unknown")
        except Exception as e:
            logging.error("Inference error in whole-plant model: %s", e)
            return WholePlantPredictionResult(
                crop=crop_hint or "Unknown",
                status="uncertain",
                confidence=0.0,
                supported=False,
                disease=None,
                message="Whole plant detected, but this crop or disease is not supported yet.",
            )

        # 3. Determine crop and disease from class mapping
        detected_crop = "Unknown"
        detected_disease = None
        status = "disease"

        # Check against supported crop disease map
        for c, d_list in self.crop_disease_map.items():
            if pred_class in d_list:
                detected_crop = c
                detected_disease = pred_class
                status = "disease"
                break
            elif pred_class.lower() in [f"healthy {c.lower()} whole plant", f"{c.lower()} healthy"]:
                detected_crop = c
                detected_disease = None
                status = "healthy"
                break

        if detected_crop == "Unknown":
            if "_" in pred_class:
                parts = pred_class.split("_", 1)
                detected_crop = parts[0]
                detected_disease = parts[1].replace("_", " ")
            else:
                detected_disease = pred_class

        active_crop = crop_hint if (crop_hint and crop_hint != "Auto-detect") else detected_crop

        # 4. Verification against supported crops and diseases
        is_supported_crop = any(active_crop.lower() in sc.lower() or sc.lower() in active_crop.lower() for sc in self.supported_crops)
        is_supported_disease = (detected_disease in self.supported_diseases) if detected_disease else (status == "healthy")
        is_confident = confidence >= confidence_threshold

        # If unsupported crop, unsupported disease, or uncertain: return NO disease name
        if not is_supported_crop or not is_supported_disease or not is_confident or status in ["unknown", "non_plant"]:
            return WholePlantPredictionResult(
                crop=active_crop,
                status="unsupported",
                confidence=confidence,
                supported=False,
                disease=None,  # NEVER return disease name for unsupported or uncertain cases
                message="Whole plant detected, but this crop or disease is not supported yet.",
            )

        # Supported and confident
        return WholePlantPredictionResult(
            crop=active_crop,
            status=status,
            confidence=confidence,
            supported=True,
            disease=detected_disease,
            message=f"Whole-plant diagnosis: {detected_disease} ({active_crop})" if detected_disease else f"Healthy {active_crop} plant",
        )
