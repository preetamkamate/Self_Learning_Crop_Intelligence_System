"""
Modular Crop Intelligence Vision Package.
Implements:
- Modular Crop Identification (CropIdentifier)
- Viewpoint & Organ Routing (ViewpointRouter) for leaf, stem/branch, fruit, canopy, whole plant, root
- Pest Detection & Scouting (PestDetector)
- Leaf & Field Disease Classification (ModularDiseaseClassifier)
- Honest Photo Guidance Engine (PhotoGuidanceEngine)
- Unified Pipeline (ModularCropVisionPipeline)
"""
from src.modular_vision.crop_id import (
    CropIdentificationResult,
    CropIdentifier,
    VERIFIED_DATASET_CROPS,
)
from src.modular_vision.disease_classifier import (
    DiseaseClassificationResult,
    ModularDiseaseClassifier,
)
from src.modular_vision.pest_detector import (
    PestDetectionResult,
    PestDetector,
)
from src.modular_vision.photo_guidance import (
    PHOTO_GUIDANCE_CATALOG,
    get_photo_guidance,
)
from src.modular_vision.pipeline import (
    ModularCropVisionPipeline,
    ModularVisionResult,
)
from src.modular_vision.viewpoint_router import (
    CANONICAL_VIEWPOINTS,
    VIEWPOINT_ALIASES,
    ViewpointRouteResult,
    ViewpointRouter,
)

__all__ = [
    "CropIdentifier",
    "CropIdentificationResult",
    "VERIFIED_DATASET_CROPS",
    "ViewpointRouter",
    "ViewpointRouteResult",
    "CANONICAL_VIEWPOINTS",
    "VIEWPOINT_ALIASES",
    "PestDetector",
    "PestDetectionResult",
    "ModularDiseaseClassifier",
    "DiseaseClassificationResult",
    "get_photo_guidance",
    "PHOTO_GUIDANCE_CATALOG",
    "ModularCropVisionPipeline",
    "ModularVisionResult",
]
