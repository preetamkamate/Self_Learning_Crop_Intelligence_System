"""
Comprehensive Vision Fallback Safety & Routing Regression Test Suite.
Verifies the global non-leaf vision safeguard across all crops and viewpoints.
Works strictly with local project code and synthetic/mocked metadata. No internet/downloads.
"""
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import torch
import numpy as np

from src.vision.taxonomy import PlantPart
from src.vision.predict import PlantDiseasePredictor
from src.vision.hierarchical_classifier import HierarchicalVisionPipeline


@pytest.fixture
def dummy_image():
    """Create a synthetic 224x224 RGB image for testing."""
    # Synthetic gradient so it passes sharpness/Laplacian checks
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        for j in range(224):
            arr[i, j] = [(i * 2) % 255, (j * 3 + 40) % 255, 60]
    return Image.fromarray(arr)


def test_rule4_mocked_maize_whole_plant_regression(dummy_image):
    """
    Rule 4 Regression Test:
    Input with mocked metadata: crop=maize, viewpoint=whole_plant.
    Expected:
    - No tomato label anywhere in results
    - No foliar backbone (MobileNetV2) selected or called
    - disease=None (when unsupported / unconfirmed)
    - predictions empty or strictly non-tomato
    """
    with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(None, [])):
        with patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=torch.randn(1, 384)):
            with patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=torch.randn(1, 384)):
                with patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
                    mock_dino = MagicMock()
                    mock_dino.device = "cpu"
                    mock_clf = MagicMock()
                    mock_clf.id2label = {0: "Tomato___Early_blight"}
                    mock_clf.label2id = {"Tomato___Early_blight": 0}
                    mock_models.return_value = (mock_dino, mock_clf)

                    predictor = PlantDiseasePredictor()

                    # Mock stage 1 as sharp plant
                    with patch.object(predictor, "_check_stage1_image_validity", return_value=(True, True, 80.0, 0.90, torch.randn(1, 384))):
                        # Force multi-part model to None to test unavailable whole-plant route
                        with patch.object(predictor, "_get_multipart_model", return_value=None):
                            # Spy on _raw_infer_probabilities (foliar backbone)
                            with patch.object(predictor, "_raw_infer_probabilities", side_effect=RuntimeError("MobileNetV2 called illegally!")) as mock_foliar:
                                result = predictor.predict(
                                    dummy_image,
                                    user_crop_hint="maize",
                                    user_part_hint="whole_plant",
                                )

                                # 1. Assert MobileNetV2 was NEVER called
                                mock_foliar.assert_not_called()

                                # 2. Assert no tomato label
                                assert "tomato" not in result["top1_label"].lower(), f"Unexpected tomato in top1_label: {result['top1_label']}"
                                for c in result["predictions"]:
                                    assert "tomato" not in c["label"].lower(), f"Unexpected tomato in predictions: {c}"

                                # 3. Assert disease is None
                                assert result["disease"] is None, f"Expected disease=None, got {result['disease']}"

                                # 4. Assert telemetry model is NOT MobileNetV2 Foliar Backbone
                                debug = result["debug_info"]
                                assert debug["selected_model"] != "MobileNetV2 Foliar Backbone"
                                assert "mobilenet" not in str(debug.get("model_path", "")).lower()
                                assert debug["crop_route"] in ["Corn (Maize)", "maize"]
                                assert debug["viewpoint_route"] == "whole_plant"
                                assert debug["fallback_reason"] != ""
                                assert result["status"] == "unsupported_crop_part"
                                assert result["top1_label"] == "Plant image detected, but this crop/viewpoint diagnosis is not supported yet."


def test_no_non_leaf_viewpoint_can_reach_foliar_backbone(dummy_image):
    """
    Assert that every non-leaf viewpoint (whole_plant, canopy, stem, fruit, root, bark, unknown, non_plant)
    NEVER calls MobileNetV2 Foliar Backbone under any circumstances.
    """
    non_leaf_views = [
        PlantPart.WHOLE_PLANT,
        PlantPart.CANOPY,
        PlantPart.STEM,
        PlantPart.FRUIT,
        PlantPart.ROOT,
        PlantPart.BARK,
        PlantPart.UNKNOWN,
        PlantPart.NON_PLANT,
    ]

    with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(None, [])):
        with patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=torch.randn(1, 384)):
            with patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=torch.randn(1, 384)):
                with patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
                    mock_dino = MagicMock()
                    mock_dino.device = "cpu"
                    mock_clf = MagicMock()
                    mock_clf.id2label = {0: "Tomato___Early_blight"}
                    mock_clf.label2id = {"Tomato___Early_blight": 0}
                    mock_models.return_value = (mock_dino, mock_clf)

                    predictor = PlantDiseasePredictor()

                    for view in non_leaf_views:
                        is_plant = (view != PlantPart.NON_PLANT)
                        with patch.object(predictor, "_check_stage1_image_validity", return_value=(True, is_plant, 80.0, 0.90 if is_plant else 0.05, torch.randn(1, 384))):
                            with patch.object(predictor, "_get_multipart_model", return_value=None):
                                with patch.object(predictor, "_raw_infer_probabilities", side_effect=RuntimeError(f"Foliar called on {view.value}!")) as mock_foliar:
                                    res = predictor.predict(
                                        dummy_image,
                                        user_crop_hint="Corn (Maize)",
                                        user_part_hint=view.value,
                                    )
                                    mock_foliar.assert_not_called()
                                    assert res["debug_info"]["selected_model"] != "MobileNetV2 Foliar Backbone"
                                    assert "mobilenet" not in str(res["debug_info"].get("model_path", "")).lower()
                                    assert res["disease"] is None
                                    assert len(res["predictions"]) == 0


def test_matrix_all_crops_and_viewpoints(dummy_image):
    """
    Test matrix across:
    Crops: maize, tomato, grape, potato, apple, unknown plant, non-plant
    Viewpoints: leaf, whole_plant, canopy, stem, fruit, root, bark, unknown, non_plant
    Verifies:
    1. No non-leaf calls MobileNetV2
    2. No cross-crop contamination (e.g. maize never gets tomato labels)
    3. Non-plant is always OOD refusal
    4. Telemetry fields are populated
    """
    crops = ["maize", "tomato", "grape", "potato", "apple", "unknown plant", "non-plant"]
    viewpoints = ["leaf", "whole_plant", "canopy", "stem", "fruit", "root", "bark", "unknown", "non_plant"]

    with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(None, [])):
        with patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=torch.randn(1, 384)):
            with patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=torch.randn(1, 384)):
                with patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
                    mock_dino = MagicMock()
                    mock_dino.device = "cpu"
                    mock_clf = MagicMock()
                    mock_clf.id2label = {
                        "0": "Tomato___Early_blight",
                        "1": "Corn_(maize)___Common_rust_",
                        "2": "Grape___Black_rot",
                        "3": "Apple___Apple_scab",
                        "4": "Potato___Early_blight",
                    }
                    mock_clf.label2id = {v: int(k) for k, v in mock_clf.id2label.items()}
                    mock_models.return_value = (mock_dino, mock_clf)

                    predictor = PlantDiseasePredictor()

                    for crop in crops:
                        for vp in viewpoints:
                            is_np = (crop == "non-plant" or vp == "non_plant")
                            with patch.object(predictor, "_check_stage1_image_validity", return_value=(True, not is_np, 80.0, 0.05 if is_np else 0.90, torch.randn(1, 384))):
                                with patch.object(predictor, "_get_multipart_model", return_value=None):
                                    with patch.object(predictor, "_raw_infer_probabilities", return_value=np.array([0.2, 0.2, 0.2, 0.2, 0.2])) as mock_foliar:
                                        res = predictor.predict(
                                            dummy_image,
                                            user_crop_hint=crop,
                                            user_part_hint=vp,
                                        )

                                        # 1. Non-leaf must NEVER call foliar
                                        if vp != "leaf":
                                            mock_foliar.assert_not_called()
                                            assert res["debug_info"]["selected_model"] != "MobileNetV2 Foliar Backbone"

                                        # 2. Non-plant must be OOD
                                        if is_np:
                                            assert res["status"] == "out_of_distribution"
                                            assert res["is_ood"] is True
                                            assert res["disease"] is None
                                            assert res["debug_info"]["selected_model"] == "facebook/dinov2-small"
                                            assert res["debug_info"]["fallback_reason"] != ""

                                        # 3. Non-tomato crops must NEVER receive tomato labels
                                        if crop not in ["tomato", "unknown plant", "non-plant"]:
                                            assert "tomato" not in res["top1_label"].lower()
                                            for cand in res["predictions"]:
                                                assert "tomato" not in cand["label"].lower()

                                        # 4. Telemetry fields must be present
                                        assert "crop_route" in res["debug_info"]
                                        assert "viewpoint_route" in res["debug_info"]
                                        assert "selected_model" in res["debug_info"]
                                        assert "fallback_reason" in res["debug_info"]


def test_weak_confidence_viewpoint_and_crop(dummy_image):
    """
    Test that when viewpoint or crop confidence is weak:
    1. Viewpoint designates PlantPart.UNKNOWN rather than defaulting to LEAF
    2. Crop designates Unknown Crop rather than defaulting to Tomato
    3. Safe abstention message is returned
    """
    with patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=torch.randn(1, 384)):
        with patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=torch.randn(1, 384)):
            with patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
                mock_dino = MagicMock()
                mock_dino.device = "cpu"
                mock_clf = MagicMock()
                mock_clf.id2label = {"0": "Tomato___Early_blight"}
                mock_clf.label2id = {"Tomato___Early_blight": 0}
                mock_models.return_value = (mock_dino, mock_clf)

                # Mock plant part model that produces weak / uniform probabilities (< 0.40)
                mock_part_model = MagicMock()
                mock_part_model.return_value = torch.zeros(1, 8)  # uniform softmax = 1/8 = 0.125 < 0.40
                part_classes = ['leaf', 'stem_branch', 'fruit', 'root_tuber', 'whole_plant', 'canopy', 'bark_trunk', 'non_plant']

                with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(mock_part_model, part_classes)):
                    predictor = PlantDiseasePredictor()

                    # Test _identify_stage2_plant_part directly with weak confidence
                    detected_part = predictor._identify_stage2_plant_part(
                        dummy_image,
                        global_emb=torch.randn(1, 384),
                        user_part_hint="auto",
                    )
                    assert detected_part == PlantPart.UNKNOWN, f"Expected PlantPart.UNKNOWN for weak part confidence, got {detected_part}"

                    # End-to-end predict with weak confidence
                    with patch.object(predictor, "_check_stage1_image_validity", return_value=(True, True, 80.0, 0.90, torch.randn(1, 384))):
                        with patch.object(predictor, "_raw_infer_probabilities", side_effect=RuntimeError("Should not call foliar!")):
                            res = predictor.predict(
                                dummy_image,
                                user_crop_hint="auto",
                                user_part_hint="auto",
                            )
                            assert res["status"] == "unsupported_crop_part"
                            assert res["top1_label"] == "Plant image detected, but this crop/viewpoint diagnosis is not supported yet."
                            assert res["disease"] is None
                            assert "tomato" not in res["top1_label"].lower()
                            assert res["debug_info"]["selected_model"] == "None"


def test_raw_infer_probabilities_raises_on_non_leaf(dummy_image):
    """
    Verify that HierarchicalVisionPipeline._raw_infer_probabilities raises a RuntimeError
    if called with any plant_part other than PlantPart.LEAF.
    """
    with patch.object(HierarchicalVisionPipeline, "_load_plant_part_model", return_value=(None, [])):
        with patch.object(HierarchicalVisionPipeline, "_build_reference_embeddings", return_value=torch.randn(1, 384)):
            with patch.object(HierarchicalVisionPipeline, "_build_leaf_prototypes", return_value=torch.randn(1, 384)):
                with patch("src.vision.hierarchical_classifier.get_vision_models") as mock_models:
                    mock_dino = MagicMock()
                    mock_dino.device = "cpu"
                    mock_clf = MagicMock()
                    mock_clf.id2label = {"0": "Tomato___Early_blight"}
                    mock_clf.label2id = {"Tomato___Early_blight": 0}
                    mock_models.return_value = (mock_dino, mock_clf)

                    predictor = PlantDiseasePredictor()

                    for part in [PlantPart.WHOLE_PLANT, PlantPart.CANOPY, PlantPart.STEM, PlantPart.FRUIT, PlantPart.ROOT, PlantPart.BARK, PlantPart.UNKNOWN, PlantPart.NON_PLANT]:
                        with pytest.raises(RuntimeError, match="CRITICAL SAFETY VIOLATION"):
                            predictor._raw_infer_probabilities(dummy_image, plant_part=part)
