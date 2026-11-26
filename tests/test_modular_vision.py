"""
Comprehensive Test Suite for Modular Crop ID, Viewpoint Routing, Pest Detection,
and Honest Photo Guidance Engine.
Verifies all 6 anatomical viewpoints: leaf, stem/branch, fruit, canopy, whole plant, root.
Asserts that diagnosis is enabled ONLY with verified real labeled data on tested exact routes;
otherwise provides honest photo guidance and NEVER guesses disease or chemical treatment.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import numpy as np

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


@pytest.fixture
def dummy_leaf_image(tmp_path):
    """Generate a clean synthetic 224x224 RGB test image."""
    img_path = tmp_path / "test_leaf.jpg"
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:, :, 1] = 160  # Green leaf hue
    img = Image.fromarray(arr)
    img.save(img_path)
    return img_path


# ==============================================================================
# 1. Modular Crop ID Tests
# ==============================================================================
def test_crop_identifier_normalization():
    """Verify crop identifier maps variations and synonyms to canonical names."""
    identifier = CropIdentifier()

    test_cases = {
        "corn": "Corn (Maize)",
        "maize": "Corn (Maize)",
        "makkajola": "Corn (Maize)",
        "tomato": "Tomato",
        "tamata": "Tomato",
        "apple": "Apple",
        "cassava": "Cassava",
        "maragenasu": "Cassava",
        "grape": "Grape",
        "potato": "Potato",
        "capsicum": "Bell Pepper (Capsicum)",
        "bell pepper": "Bell Pepper (Capsicum)",
        "citrus": "Citrus",
        "orange": "Citrus",
        "unknown": "Unknown",
        "": "Unknown",
        None: "Unknown",
    }

    for raw, expected in test_cases.items():
        assert identifier.normalize_crop_name(raw) == expected


def test_crop_identifier_verified_dataset_status():
    """Verify crop identifier accurately distinguishes verified dataset crops."""
    identifier = CropIdentifier()

    # Verified crops
    for crop in ["Apple", "Cassava", "Corn (Maize)", "Tomato"]:
        res = identifier.identify_crop(crop_hint=crop)
        assert res.is_verified_dataset_crop is True
        assert res.canonical_name == crop
        assert res.is_plant is True
        assert "leaf" in res.supported_viewpoints

    # Unsupported agricultural crops (recognized but not in verified training split)
    for crop in ["Grape", "Potato", "Citrus", "Banana", "Wheat"]:
        res = identifier.identify_crop(crop_hint=crop)
        assert res.is_verified_dataset_crop is False
        assert res.canonical_name == identifier.normalize_crop_name(crop)
        assert res.is_plant is True

    # Non-plant input
    res_np = identifier.identify_crop(crop_hint="non_plant")
    assert res_np.is_plant is False
    assert res_np.canonical_name == "None"


# ==============================================================================
# 2. Viewpoint & Organ Router Tests (All 6 Canonical Viewpoints)
# ==============================================================================
def test_viewpoint_router_normalizes_all_6_viewpoints():
    """Verify router correctly normalizes all 6 canonical viewpoints and their aliases."""
    router = ViewpointRouter()

    mapping_cases = {
        # 1. Leaf
        "leaf": "leaf",
        "leaves": "leaf",
        "foliar": "leaf",
        "single_leaf": "leaf",
        # 2. Stem / Branch
        "stem": "stem_branch",
        "branch": "stem_branch",
        "stem_branch": "stem_branch",
        "stem / branch": "stem_branch",
        "stalk": "stem_branch",
        "cane": "stem_branch",
        "twig": "stem_branch",
        "bark": "stem_branch",
        # 3. Fruit
        "fruit": "fruit",
        "pod": "fruit",
        "berry": "fruit",
        "grain": "fruit",
        "cob": "fruit",
        # 4. Canopy
        "canopy": "canopy",
        "canopy / field": "canopy",
        "field": "canopy",
        "plot": "canopy",
        "row": "canopy",
        # 5. Whole Plant
        "whole_plant": "whole_plant",
        "whole plant": "whole_plant",
        "full plant": "whole_plant",
        "entire plant": "whole_plant",
        # 6. Root
        "root": "root",
        "tuber": "root",
        "root / tuber": "root",
        "rhizome": "root",
        # Non-plant and unknown
        "non_plant": "non_plant",
        "not_plant": "non_plant",
        "unknown": "unknown",
    }

    for raw, expected in mapping_cases.items():
        assert router.normalize_viewpoint(raw) == expected, f"Failed for raw input: {raw}"


def test_viewpoint_router_route_gating():
    """Verify route gating across verified leaf vs. non-leaf viewpoints."""
    router = ViewpointRouter()

    # Leaf on verified crop (Tomato) must be supported
    res_leaf = router.route_viewpoint(crop="Tomato", viewpoint_hint="leaf")
    assert res_leaf.is_supported_route is True
    assert res_leaf.canonical_viewpoint == "leaf"
    assert res_leaf.requires_photo_guidance is False

    # All non-leaf viewpoints on Tomato must be unsupported and require photo guidance
    non_leaf_views = ["stem_branch", "fruit", "canopy", "whole_plant", "root"]
    for vp in non_leaf_views:
        res = router.route_viewpoint(crop="Tomato", viewpoint_hint=vp)
        assert res.is_supported_route is False, f"Viewpoint {vp} was unexpectedly marked supported!"
        assert res.canonical_viewpoint == vp
        assert res.requires_photo_guidance is True
        assert "not trained yet" in res.status_reason

    # Any viewpoint on an unsupported crop (e.g. Grape) must be unsupported
    res_grape = router.route_viewpoint(crop="Grape", viewpoint_hint="leaf")
    assert res_grape.is_supported_route is False
    assert res_grape.requires_photo_guidance is True


# ==============================================================================
# 3. Pest Detection Module Tests
# ==============================================================================
def test_pest_detector_query_detection():
    """Verify pest detector recognizes pest terminology in questions and symptoms."""
    detector = PestDetector()

    assert detector.is_pest_query(user_text="Are there armyworms on this maize?") is True
    assert detector.is_pest_query(user_text="What pesticide kills spider mites?") is True
    assert detector.is_pest_query(symptom_desc="boreholes in the stem with frass") is True
    assert detector.is_pest_query(symptom_desc="aphids clustered under leaves") is True
    assert detector.is_pest_query(user_text="Recommend best fertilizer for soil") is False


def test_pest_detector_safe_abstention_and_guidance():
    """Verify unverified pest routes abstain safely without guessing chemicals."""
    detector = PestDetector()

    # Case 1: Specific unverified pest query (e.g. armyworm)
    res_blocked = detector.detect_pest(
        crop="Corn (Maize)",
        viewpoint="leaf",
        user_text="Is this fall armyworm damage?",
        symptom_desc="ragged holes in whorl",
    )

    assert res_blocked.pest_suspected is True
    assert res_blocked.pest_label is None  # Never guess specific pest without verified checkpoint
    assert res_blocked.supported is False
    assert res_blocked.status == "pest_data_blocked"
    assert "photo_guidance" in res_blocked.to_dict()
    assert res_blocked.photo_guidance["viewpoint_category"] == "pest"
    assert "IPM" in res_blocked.ipm_scouting_advice
    assert res_blocked.metadata["chemical_pesticides_withheld"] is True

    # Case 2: General insect query without verified checkpoint
    res_general = detector.detect_pest(
        crop="Corn (Maize)",
        viewpoint="leaf",
        user_text="What insect is eating this plant?",
        symptom_desc="webbing on surface",
    )

    assert res_general.pest_suspected is True
    assert res_general.pest_label is None
    assert res_general.supported is False
    assert res_general.status == "pest_guidance_provided"
    assert res_general.metadata["chemical_pesticides_withheld"] is True


# ==============================================================================
# 4. Honest Photo Guidance Engine Tests
# ==============================================================================
def test_photo_guidance_catalog_coverage():
    """Verify photo guidance engine covers all 6 canonical viewpoints + special cases."""
    required_keys = ["leaf", "stem_branch", "fruit", "canopy", "whole_plant", "root", "pest", "blurry", "non_plant", "unsupported_crop"]

    for key in required_keys:
        assert key in PHOTO_GUIDANCE_CATALOG
        item = PHOTO_GUIDANCE_CATALOG[key]
        assert "title_en" in item and len(item["title_en"]) > 5
        assert "title_kn" in item and len(item["title_kn"]) > 5
        assert "instructions_en" in item and len(item["instructions_en"]) > 20
        assert "instructions_kn" in item and len(item["instructions_kn"]) > 20
        assert "checklist" in item and len(item["checklist"]) >= 2


def test_photo_guidance_retrieval_for_each_organ():
    """Verify get_photo_guidance retrieves tailored guidance for each viewpoint."""
    organs = ["leaf", "stem", "fruit", "canopy", "whole_plant", "root"]
    for organ in organs:
        guidance = get_photo_guidance(viewpoint=organ, crop="Tomato")
        assert guidance is not None
        assert "title_en" in guidance
        assert "title_kn" in guidance
        assert "checklist" in guidance

    # Root and stem cankers recommend certified lab testing
    assert get_photo_guidance("root")["lab_test_recommended"] is True
    assert get_photo_guidance("stem")["lab_test_recommended"] is True


# ==============================================================================
# 5. Modular Crop Vision Pipeline End-to-End Tests
# ==============================================================================
def test_modular_pipeline_verified_leaf_route(dummy_leaf_image):
    """Verify pipeline allows diagnosis when crop and leaf route are verified."""
    pipeline = ModularCropVisionPipeline()

    result = pipeline.diagnose(
        image_input=dummy_leaf_image,
        crop_hint="Tomato",
        viewpoint_hint="leaf",
    )

    assert result.plant_detected is True
    assert result.crop == "Tomato"
    assert result.viewpoint == "leaf"
    assert result.supported is True
    assert result.disease is not None or result.is_healthy is True
    assert "grape" not in result.top1_label.lower()
    assert "corn" not in result.top1_label.lower()


def test_modular_pipeline_non_leaf_viewpoints_abstain_with_photo_guidance(dummy_leaf_image):
    """
    Assert that all 5 non-leaf viewpoints (stem_branch, fruit, canopy, whole_plant, root)
    SAFELY ABSTAIN from disease diagnosis and provide honest photo guidance.
    """
    pipeline = ModularCropVisionPipeline()

    non_leaf_views = ["stem_branch", "fruit", "canopy", "whole_plant", "root"]

    for vp in non_leaf_views:
        result = pipeline.diagnose(
            image_input=dummy_leaf_image,
            crop_hint="Tomato",
            viewpoint_hint=vp,
        )

        # 1. Must never predict a disease
        assert result.supported is False, f"Viewpoint {vp} was unexpectedly marked supported!"
        assert result.disease is None, f"Viewpoint {vp} predicted a disease: {result.disease}!"
        assert result.predictions == []

        # 2. Must return honest photo guidance
        assert result.photo_guidance is not None, f"Viewpoint {vp} missing photo guidance!"
        assert "instructions_en" in result.photo_guidance
        assert "instructions_kn" in result.photo_guidance
        assert len(result.photo_guidance["checklist"]) > 0

        # 3. Status reason must clearly explain abstention
        assert "not trained yet" in result.fallback_reason.lower() or "withheld" in result.fallback_reason.lower()


def test_modular_pipeline_unsupported_crop_abstains_with_guidance(dummy_leaf_image):
    """Assert that unsupported crops (e.g. Grape, Potato, Citrus) safely abstain."""
    pipeline = ModularCropVisionPipeline()

    for crop in ["Grape", "Potato", "Citrus"]:
        result = pipeline.diagnose(
            image_input=dummy_leaf_image,
            crop_hint=crop,
            viewpoint_hint="leaf",
        )

        assert result.supported is False
        assert result.disease is None
        assert result.predictions == []
        assert result.photo_guidance is not None
        assert result.photo_guidance["viewpoint_category"] == "unsupported_crop"


def test_modular_pipeline_non_plant_rejection(dummy_leaf_image):
    """Assert that non-plant input is immediately rejected."""
    pipeline = ModularCropVisionPipeline()

    result = pipeline.diagnose(
        image_input=dummy_leaf_image,
        crop_hint="non_plant",
        viewpoint_hint="non_plant",
    )

    assert result.plant_detected is False
    assert result.supported is False
    assert result.disease is None
    assert result.status == "out_of_distribution"
    assert result.photo_guidance["viewpoint_category"] == "non_plant"


def test_modular_pipeline_never_guesses_chemical_treatment_for_unverified(dummy_leaf_image):
    """
    CRITICAL SAFETY ASSERTION:
    Ensure no chemical dosage, pesticide, or disease candidate is returned
    for unverified crop / viewpoint routes.
    """
    pipeline = ModularCropVisionPipeline()

    # Adversarial test across multiple unsupported combinations
    adversarial_cases = [
        ("Corn (Maize)", "root"),
        ("Tomato", "canopy"),
        ("Apple", "fruit"),
        ("Cassava", "stem_branch"),
        ("Unknown", "whole_plant"),
    ]

    for crop, vp in adversarial_cases:
        res = pipeline.diagnose(
            image_input=dummy_leaf_image,
            crop_hint=crop,
            viewpoint_hint=vp,
        )
        d = res.to_dict()

        assert d["supported"] is False
        assert d["disease"] is None
        assert d["predictions"] == []
        assert "photo_guidance" in d and d["photo_guidance"] is not None
        # Telemetry verification
        assert d["debug_info"]["checkpoint_path"] == "None"
        assert d["debug_info"]["stage4_route_supported"] is False


# ==============================================================================
# 6. Missing Model Checkpoint & Graceful Abstention Tests
# ==============================================================================
def test_disease_classifier_missing_checkpoint_abstains_safely(tmp_path, dummy_leaf_image):
    """Verify ModularDiseaseClassifier safely abstains when checkpoint is missing."""
    nonexistent_model = tmp_path / "nonexistent_model.pt"
    classifier = ModularDiseaseClassifier(model_path=nonexistent_model)

    assert classifier.is_trained is False

    res = classifier.classify_leaf_disease(
        image_input=dummy_leaf_image,
        crop="Tomato",
    )

    assert res.supported is False
    assert res.disease is None
    assert res.status == "needs_training_data"
    assert "not loaded" in res.status_reason.lower() or "missing" in res.status_reason.lower()
    assert res.photo_guidance is not None
    assert res.predictions == []


def test_pest_detector_missing_checkpoint_abstains_safely(tmp_path, dummy_leaf_image):
    """Verify PestDetector safely abstains when pest checkpoint is missing."""
    nonexistent_pest_model = tmp_path / "nonexistent_pest.pt"
    detector = PestDetector(model_path=nonexistent_pest_model)

    assert detector.is_trained is False

    res = detector.detect_pest(
        image_input=dummy_leaf_image,
        crop="Tomato",
        viewpoint="leaf",
        user_text="Are there two-spotted spider mites on this leaf?",
    )

    assert res.supported is False
    assert res.pest_label is None
    assert res.metadata["chemical_pesticides_withheld"] is True
    assert res.photo_guidance is not None


def test_modular_pipeline_missing_model_abstains_cleanly(tmp_path, dummy_leaf_image):
    """Verify full ModularCropVisionPipeline handles missing models gracefully without crashing."""
    nonexistent_model = tmp_path / "nonexistent_model.pt"
    mock_disease_classifier = ModularDiseaseClassifier(model_path=nonexistent_model)
    pipeline = ModularCropVisionPipeline(disease_classifier=mock_disease_classifier)

    res = pipeline.diagnose(
        image_input=dummy_leaf_image,
        crop_hint="Tomato",
        viewpoint_hint="leaf",
    )

    assert res.plant_detected is True
    assert res.crop == "Tomato"
    assert res.supported is False
    assert res.disease is None
    assert res.status == "needs_training_data"
    assert res.photo_guidance is not None


def test_dedicated_pest_detector_verified_tomato_spider_mite_inference(dummy_leaf_image):
    """Verify dedicated pest detector loads verified checkpoint and infers cleanly."""
    detector = PestDetector()

    if detector.is_trained:
        assert len(detector.verified_pest_classes) >= 2
        assert any("spider mite" in c.lower() for c in detector.verified_pest_classes)
        assert any("healthy" in c.lower() for c in detector.verified_pest_classes)

        res = detector.detect_pest(
            image_input=dummy_leaf_image,
            crop="Tomato",
            viewpoint="leaf",
            user_text="Check leaf for spider mites",
        )
        assert res.pest_suspected is True
        assert res.status in ["pest_detected", "pest_guidance_provided"]
        assert res.photo_guidance is not None

