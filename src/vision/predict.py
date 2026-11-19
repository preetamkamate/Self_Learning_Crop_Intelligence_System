"""
Whole-Plant & Multi-Part Field Understanding Vision Predictor.
4-Stage Staged Hierarchical Architecture:
Stage 1: Image Validity & Quality Gating (Laplacian sharpness & DINOv2 non-plant OOD)
Stage 2: Plant Part Identification (leaf, whole_plant, canopy, stem, fruit, root, bark)
Stage 3: Crop Species Identification (constrained hierarchical evidence aggregation)
Stage 4: Gated Pathology Diagnosis (strictly constrained to confirmed crop subspace;
         abstains completely from untrained viewpoints and unsupported crops).
"""
import io
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

import config
from src.rag.authoritative_sources import get_authoritative_source_citation
from src.vision.hierarchical_classifier import (
    HierarchicalVisionPipeline,
    build_top_candidates,
    compute_shannon_entropy,
    measure_image_sharpness,
    normalize_candidate,
)
from src.vision.model import get_vision_models
from src.vision.nutrient import (
    SCREENING_DISCLAIMER_EN,
    SCREENING_DISCLAIMER_KN,
    analyze_nutrient_symptoms,
)
from src.vision.taxonomy import (
    EvidenceLevel,
    HierarchicalDiagnosis,
    IssueCategory,
    PlantPart,
    parse_legacy_label,
)
from src.vision.training_registry import (
    CROP_DISEASE_TAXONOMY,
    DEPLOYED_DIAGNOSTIC_ROUTES,
    LABEL_TO_CROP_MAP,
    get_crop_for_label,
    get_valid_classes_for_crop,
    is_route_supported,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PlantDiseasePredictor(HierarchicalVisionPipeline):
    """
    Production-grade Staged Hierarchical Crop Health Pipeline.
    Prevents cross-crop label mixing and untrained viewpoint disease diagnosis.
    """
    pass


if __name__ == "__main__":
    predictor = PlantDiseasePredictor()
    print("\n--- Testing Whole-Plant Tomato Image (Should Safely Abstain) ---")
    res1 = predictor.predict(
        config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg",
        user_crop_hint="Tomato",
        user_part_hint="whole_plant",
    )
    print("Status:", res1["status"])
    print("Top1 Label:", res1["top1_label"])
    print("Candidates:", res1["predictions"])
    print("Source Evidence:", res1["source_evidence"])

    print("\n--- Testing Tomato Leaf Image (Should Diagnose Tomato Early Blight) ---")
    res2 = predictor.predict(
        config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg",
        user_crop_hint="Tomato",
        user_part_hint="leaf",
    )
    print("Status:", res2["status"])
    print("Top1 Label:", res2["top1_label"])
    print("Candidates:", [c["label"] for c in res2["predictions"]])
    print("Source Evidence URL:", res2["source_evidence"]["url"] if res2["source_evidence"] else "None")
