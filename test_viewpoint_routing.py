"""
Unit tests for vision architecture viewpoint routing and safe abstention.
Verifies routing across all 9 views using synthetic metadata and mocks only.
No external images or downloads required.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.vision.taxonomy import PlantPart
from src.vision.hierarchical_classifier import build_top_candidates, normalize_candidate


def test_viewpoint_enum_and_parser():
    """Verify all 9 image viewpoints parse accurately."""
    expected_views = {
        "leaf": PlantPart.LEAF,
        "whole_plant": PlantPart.WHOLE_PLANT,
        "whole plant": PlantPart.WHOLE_PLANT,
        "canopy": PlantPart.CANOPY,
        "stem": PlantPart.STEM,
        "branch": PlantPart.STEM,
        "fruit": PlantPart.FRUIT,
        "root": PlantPart.ROOT,
        "tuber": PlantPart.ROOT,
        "bark": PlantPart.BARK,
        "trunk": PlantPart.BARK,
        "non_plant": PlantPart.NON_PLANT,
        "not_plant": PlantPart.NON_PLANT,
        "unknown": PlantPart.UNKNOWN,
    }

    for input_str, expected_part in expected_views.items():
        parsed = PlantPart.from_str(input_str)
        assert parsed == expected_part, f"Failed for {input_str}: got {parsed}"


def test_build_top_candidates_numerical_safety():
    """Verify candidate formatter handles missing/non-numeric inputs without crashing."""
    malformed_inputs = [
        None,
        [],
        {"class": "Tomato Early Blight", "prob": None},
        {"label": "Apple Scab", "prob": "invalid"},
        {"class": "Corn Rust", "probability": 0.85},
        ("Grape Black Rot", None),
        ("Potato Late Blight", "0.92"),
    ]

    for item in malformed_inputs:
        res = build_top_candidates(item)
        assert isinstance(res, list)
        for cand in res:
            assert "class" in cand
            assert "prob" in cand
            assert isinstance(cand["prob"], float)
            assert 0.0 <= cand["prob"] <= 1.0


def test_non_leaf_views_safe_abstention():
    """Verify non-leaf views return safe structured result with zero disease guesses."""
    from src.vision.hierarchical_classifier import HierarchicalVisionPipeline

    non_leaf_parts = [
        PlantPart.WHOLE_PLANT,
        PlantPart.CANOPY,
        PlantPart.STEM,
        PlantPart.FRUIT,
        PlantPart.ROOT,
        PlantPart.BARK,
        PlantPart.UNKNOWN,
    ]

    expected_msg = "Plant image detected, but this crop/viewpoint diagnosis is not supported yet."

    for part in non_leaf_parts:
        with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(None, [])), \
             patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=None), \
             patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=None), \
             patch.object(HierarchicalVisionPipeline, "_get_multipart_model", return_value=None), \
             patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
            mock_dinov2 = MagicMock()
            mock_dinov2.device = "cpu"
            mock_clf = MagicMock()
            mock_clf.id2label = {0: "Tomato___Early_blight"}
            mock_clf.label2id = {"Tomato___Early_blight": 0}
            mock_models.return_value = (mock_dinov2, mock_clf)

            pipeline = HierarchicalVisionPipeline()

            dummy_img = MagicMock()
            dummy_img.size = (224, 224)
            dummy_img.mode = "RGB"

            with patch.object(pipeline, "_check_stage1_image_validity", return_value=(True, True, 60.0, 0.95, MagicMock())):
                with patch.object(pipeline, "_identify_stage2_plant_part", return_value=part):
                    with patch.object(pipeline, "_identify_stage3_crop_species", return_value=("Tomato", 0.90)):
                        result = pipeline.predict(dummy_img, user_crop_hint="Tomato", user_part_hint=part.value)

                        assert result["status"] == "unsupported_crop_part"
                        assert result["top1_label"] == expected_msg
                        assert result["predictions"] == []
                        assert result["source_evidence"] is None
                        assert result["is_ood"] is False


def test_non_plant_view_safe_rejection():
    """Verify non_plant view is rejected safely as out-of-distribution."""
    from src.vision.hierarchical_classifier import HierarchicalVisionPipeline

    with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(None, [])):
        with patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=None):
            with patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=None):
                with patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
                    mock_dinov2 = MagicMock()
                    mock_dinov2.device = "cpu"
                    mock_clf = MagicMock()
                    mock_clf.id2label = {0: "Tomato___Early_blight"}
                    mock_clf.label2id = {"Tomato___Early_blight": 0}
                    mock_models.return_value = (mock_dinov2, mock_clf)

                    pipeline = HierarchicalVisionPipeline()

                    dummy_img = MagicMock()
                    dummy_img.size = (224, 224)
                    dummy_img.mode = "RGB"

                    with patch.object(pipeline, "_check_stage1_image_validity", return_value=(True, True, 60.0, 0.95, MagicMock())):
                        with patch.object(pipeline, "_identify_stage2_plant_part", return_value=PlantPart.NON_PLANT):
                            result = pipeline.predict(dummy_img, user_part_hint="non_plant")

                            assert result["status"] == "out_of_distribution"
                            assert result["top1_label"] == "Non-plant image detected"
                            assert result["predictions"] == []
                            assert result["source_evidence"] is None
                            assert result["is_ood"] is True
