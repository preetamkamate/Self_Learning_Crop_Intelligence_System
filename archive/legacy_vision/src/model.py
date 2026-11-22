"""
Vision Model Loaders for Whole-Plant Field Understanding.
Combines:
1. facebook/dinov2-small (~85 MB) for whole-plant global structural representation & plant/field OOD detection.
2. linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification (~14 MB) for fine-grained multi-region plant pathology.
Optimized for Apple Silicon M1 (MPS/CPU) with low memory footprint (<150 MB).
"""
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    MobileNetV2ImageProcessor,
)

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class WholePlantDINOv2Backbone:
    """Extracts whole-plant global structural embeddings and detects plant vs non-plant OOD."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = "facebook/dinov2-small", device: torch.device = config.DEVICE):
        if self._initialized:
            return

        self.model_id = model_id
        self.device = device
        logging.info("Loading Whole-Plant DINOv2 Backbone: %s on %s", self.model_id, self.device)

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(model_id)

        try:
            self.model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self.model = AutoModel.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
        logging.info("DINOv2 Backbone loaded successfully.")


class PlantDiseaseModel:
    """Fine-tuned PlantVillage MobileNetV2 for localized plant pathology classification."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_id: str = config.VISION_MODEL_ID, device: torch.device = config.DEVICE):
        if self._initialized:
            return

        self.model_id = model_id
        self.device = device
        logging.info("Loading Plant Pathology Classifier: %s on %s", self.model_id, self.device)

        try:
            self.processor = MobileNetV2ImageProcessor.from_pretrained(model_id, local_files_only=True)
        except Exception:
            try:
                self.processor = MobileNetV2ImageProcessor.from_pretrained(model_id)
            except Exception:
                self.processor = AutoImageProcessor.from_pretrained(model_id)

        try:
            self.model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self.model = AutoModelForImageClassification.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        self._initialized = True
        logging.info("Plant Pathology Classifier loaded with %d classes.", len(self.id2label))


def get_vision_models():
    """Retrieve initialized instances of DINOv2 backbone and MobileNetV2 classifier."""
    dinov2 = WholePlantDINOv2Backbone()
    classifier = PlantDiseaseModel()
    return dinov2, classifier


def get_vision_model():
    """Legacy compatibility helper returning the classifier."""
    return PlantDiseaseModel()


if __name__ == "__main__":
    dino, clf = get_vision_models()
    print("DINOv2 and MobileNetV2 models initialized successfully.")
