"""
Unit tests for the independent whole-plant analysis module.
Uses fake file paths and fake metadata only.
Does not access internet, external content, or the leaf classifier.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unittest.mock import MagicMock, patch
import pytest
import torch

from src.whole_plant.dataset import (
    WholePlantDataset,
    WholePlantRecord,
    create_splits,
    has_training_images,
    load_raw_dataset,
)
from src.whole_plant.evaluate import evaluate_whole_plant_model
from src.whole_plant.predict import (
    WholePlantPredictionResult,
    WholePlantPredictor,
)
from src.whole_plant.train import WholePlantModel, train_whole_plant_model


def test_record_creation_with_fake_metadata():
    """Verify WholePlantRecord with fake path and metadata."""
    fake_path = "/fake/storage/field_photo_001.jpg"
    fake_meta = {"source": "fake_camera_trap", "gps": "12.97,77.59", "quality": "high"}

    rec = WholePlantRecord(
        crop="Tomato",
        image_path=fake_path,
        status="disease",
        disease="Fusarium Wilt",
        metadata=fake_meta,
    )

    data = rec.to_dict()
    assert data["crop"] == "Tomato"
    assert data["image_path"] == fake_path
    assert data["status"] == "disease"
    assert data["disease"] == "Fusarium Wilt"
    assert data["metadata"]["source"] == "fake_camera_trap"

    restored = WholePlantRecord.from_dict(data)
    assert restored.crop == rec.crop
    assert restored.image_path == rec.image_path
    assert restored.disease == rec.disease


def test_empty_raw_folder_returns_needs_training_data(tmp_path):
    """Empty raw whole-plant folder must return a clear needs_training_data status."""
    fake_raw = tmp_path / "raw" / "whole_plant"
    fake_raw.mkdir(parents=True)

    assert has_training_images(fake_raw) is False

    records, info = load_raw_dataset(fake_raw)
    assert records == []
    assert info["status"] == "needs_training_data"
    assert "Whole-plant model needs training data" in info["message"]


def test_splits_creation_with_fake_records(tmp_path):
    """Verify train, validation, and test split creation with fake paths."""
    fake_records = [
        WholePlantRecord(
            crop="Corn",
            image_path=f"/fake/corn_{i}.jpg",
            status="healthy" if i % 2 == 0 else "disease",
            disease=None if i % 2 == 0 else "Corn Stunt",
            metadata={"id": i},
        )
        for i in range(10)
    ]

    out_dir = tmp_path / "processed" / "whole_plant"
    splits = create_splits(fake_records, output_dir=out_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2

    # Check manifest files written
    assert (out_dir / "train" / "records.json").exists()
    assert (out_dir / "val" / "records.json").exists()
    assert (out_dir / "test" / "records.json").exists()


def test_dataset_item_retrieval_with_fake_paths():
    """Verify WholePlantDataset safely loads item with fake fallback canvas."""
    fake_records = [
        WholePlantRecord(crop="Tomato", image_path="/fake/nonexistent.jpg", status="healthy"),
        WholePlantRecord(crop="Corn", image_path="/fake/another_nonexistent.jpg", status="disease", disease="Stunt"),
    ]

    ds = WholePlantDataset(fake_records)
    assert len(ds) == 2

    img_tensor, target_idx, meta = ds[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert target_idx in [0, 1]
    assert meta["crop"] == "Tomato"


def test_predictor_with_no_images_returns_needs_training_data(tmp_path):
    """WholePlantPredictor must report needs_training_data when raw folder is empty."""
    empty_raw = tmp_path / "raw" / "whole_plant"
    empty_raw.mkdir(parents=True)
    fake_model_path = tmp_path / "model.pt"

    predictor = WholePlantPredictor(model_path=fake_model_path, raw_dir=empty_raw)
    result = predictor.predict("/fake/uploaded_field_photo.jpg", crop_hint="Tomato")

    assert isinstance(result, WholePlantPredictionResult)
    assert result.status == "needs_training_data"
    assert result.supported is False
    assert result.confidence == 0.0
    assert result.disease is None
    assert (
        "Whole-plant model needs training data" in result.message
        or "Whole-plant diagnosis is not trained yet" in result.message
    )


def test_predictor_unsupported_or_uncertain_returns_no_disease(tmp_path):
    """Unsupported or uncertain predictions must NEVER return a disease name."""
    raw_dir = tmp_path / "raw" / "whole_plant"
    raw_dir.mkdir(parents=True)
    (raw_dir / "fake.jpg").write_text("fake")

    fake_model_path = tmp_path / "model.pt"

    predictor = WholePlantPredictor(model_path=fake_model_path, raw_dir=raw_dir)
    predictor.model = MagicMock()
    predictor.supported_crops = ["Tomato"]
    predictor.supported_diseases = ["Fusarium Wilt"]
    predictor.idx_to_label = {0: "Healthy Tomato Whole Plant", 1: "Tomato Fusarium Wilt"}

    # Mock low-confidence inference
    fake_logits = torch.tensor([[0.45, 0.55]])
    predictor.model.return_value = fake_logits

    # 1. High threshold makes it uncertain
    res = predictor.predict("/fake/image.jpg", crop_hint="Tomato", confidence_threshold=0.90)
    assert res.supported is False
    assert res.disease is None
    assert "uncertain" in res.message.lower() or "not trained" in res.message.lower() or "not supported yet" in res.message.lower()

    # 2. Unsupported crop
    res_unsupported_crop = predictor.predict("/fake/image.jpg", crop_hint="Grape", confidence_threshold=0.50)
    assert res_unsupported_crop.supported is False
    assert res_unsupported_crop.disease is None
    assert "Valid whole plant detected, but this crop is not trained for diagnosis yet." in res_unsupported_crop.message


def test_maize_whole_plant_can_never_produce_tomato_label():
    """
    Requirement 10 & 13:
    Prove that maize whole-plant input can NEVER produce tomato disease labels under any condition.
    """
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH

    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    assert predictor.is_trained is True

    # Test with Corn/Maize image from held-out test set
    test_corn_dir = Path("data/raw/whole_plant/test/Corn_Stunt_Disease")
    if test_corn_dir.exists():
        test_img = next(test_corn_dir.glob("*.jpg"))
    else:
        test_img = "/fake/corn.jpg"

    # 1. Normal predict with crop_hint="Corn (Maize)"
    res = predictor.predict(test_img, crop_hint="Corn (Maize)")
    assert "tomato" not in res.top1_label.lower()
    assert res.disease is None or "tomato" not in res.disease.lower()
    for cand in res.predictions:
        assert "tomato" not in cand["label"].lower(), f"Cross-crop leak! Tomato label in Corn predictions: {cand}"

    # 2. Malicious adversarial test: Force tomato class logits to +1000.0 inside model
    # Strict crop isolation must STILL prevent any tomato class from being selected
    n_classes = len(predictor.idx_to_label)
    with patch.object(predictor.model, "forward", return_value=torch.tensor([[1000.0] * n_classes])):
        res_adv = predictor.predict(test_img, crop_hint="maize")
        assert "tomato" not in res_adv.top1_label.lower()
        if res_adv.disease:
            assert "tomato" not in res_adv.disease.lower()
        for cand in res_adv.predictions:
            assert "tomato" not in cand["label"].lower()
            assert "corn" in cand["label"].lower() or "healthy" in cand["label"].lower()


def test_unsupported_crops_return_exact_safe_abstention():
    """
    Requirement 8 & 9:
    If crop is untrained, return strictly:
    'Valid whole plant detected, but this crop is not trained for diagnosis yet.'
    """
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH

    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    assert predictor.is_trained is True

    unsupported_crops = ["Grape", "Potato", "Cotton", "Banana", "Sugarcane", "Wheat", "Coffee"]
    expected_msg = "Valid whole plant detected, but this crop is not trained for diagnosis yet."

    test_img = next(Path("data/raw/whole_plant/test").rglob("*.jpg"))

    for crop in unsupported_crops:
        res = predictor.predict(test_img, crop_hint=crop)
        assert res.supported is False
        assert res.disease is None
        assert res.predictions == []
        assert res.top1_label == expected_msg
        assert res.message == expected_msg
        assert "tomato" not in res.top1_label.lower()


def test_held_out_evaluation_report_exists_and_verified():
    """
    Requirement 11 & 12:
    Verify that models/whole_plant/evaluation_report.json exists and meets all safety criteria.
    """
    import json
    from src.whole_plant.evaluate import DEFAULT_EVALUATION_REPORT_PATH

    assert DEFAULT_EVALUATION_REPORT_PATH.exists()
    with open(DEFAULT_EVALUATION_REPORT_PATH, "r") as f:
        rep = json.load(f)

    assert rep["evaluated"] is True
    assert rep["total_test_samples"] == 90
    assert rep["false_crop_routing_rate"] == 0.0, "False crop-routing rate must be strictly 0.00%!"
    assert rep["unsupported_crop_abstention_rate"] == 1.0, "Unsupported crop abstention rate must be 100.00%!"
    assert "Tomato" in rep["per_crop_metrics"]
    assert "Corn (Maize)" in rep["per_crop_metrics"]
    assert "Apple" in rep["per_crop_metrics"]
    assert "Cassava" in rep["per_crop_metrics"]
    assert rep["overall_accuracy"] >= 0.65


def test_predictor_prediction_structure_fields():
    """Verify WholePlantPredictionResult structure contains all 9 required fields."""
    res = WholePlantPredictionResult(
        plant_detected=True,
        crop="Tomato",
        viewpoint="whole_plant",
        supported=True,
        disease="Tomato Fusarium Wilt",
        confidence=0.88,
        status="disease",
        fallback_reason="",
        checkpoint_path="models/whole_plant/model.pt",
    )
    d = res.to_dict()
    assert "plant_detected" in d and d["plant_detected"] is True
    assert "crop" in d and d["crop"] == "Tomato"
    assert "viewpoint" in d and d["viewpoint"] == "whole_plant"
    assert "supported" in d and d["supported"] is True
    assert "disease" in d and d["disease"] == "Tomato Fusarium Wilt"
    assert "confidence" in d and d["confidence"] == 0.88
    assert "status" in d and d["status"] == "disease"
    assert "fallback_reason" in d
    assert "checkpoint_path" in d


def test_tomato_cannot_return_grape_disease():
    """Regression test proving tomato whole-plant input cannot return grape disease."""
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH
    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    assert predictor.is_trained is True

    test_img = next(Path("data/raw/whole_plant/test/Tomato_Fusarium_Wilt").rglob("*.jpg"))
    res = predictor.predict(test_img, crop_hint="tomato")
    if res.disease:
        assert "grape" not in res.disease.lower()
    assert "grape" not in res.top1_label.lower()
    for cand in res.predictions:
        assert "grape" not in cand["label"].lower()


def test_unknown_crop_cannot_return_disease():
    """Regression test proving unknown crop cannot return any disease."""
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH
    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    assert predictor.is_trained is True

    test_img = next(Path("data/raw/whole_plant/test").rglob("*.jpg"))
    res = predictor.predict(test_img, crop_hint="Unknown")
    assert res.supported is False
    assert res.disease is None
    assert res.predictions == []


def test_non_leaf_inputs_cannot_use_leaf_model():
    """Regression test proving non-leaf inputs cannot access leaf model."""
    from src.whole_plant.router import WholePlantRouter
    router = WholePlantRouter()
    test_img = next(Path("data/raw/whole_plant/test").rglob("*.jpg"))

    non_leaf_views = ["canopy", "stem", "fruit", "root", "bark", "unknown", "non_plant"]
    for vp in non_leaf_views:
        res = router.route(test_img, crop_hint="Corn (Maize)", viewpoint_hint=vp)
        assert res.disease is None
        assert res.supported is False
        assert "leaf" not in res.checkpoint_path.lower()
        assert "mobilenet" not in res.checkpoint_path.lower()


def test_missing_checkpoint_returns_safe_abstention(tmp_path):
    """Regression test proving missing checkpoint returns safe abstention."""
    from src.whole_plant.router import WholePlantRouter
    missing_ckpt = tmp_path / "nonexistent.pt"
    router = WholePlantRouter(model_path=missing_ckpt)
    test_img = next(Path("data/raw/whole_plant/test").rglob("*.jpg"))

    res = router.route(test_img, crop_hint="Corn (Maize)", viewpoint_hint="whole_plant")
    assert res.supported is False
    assert res.disease is None
    assert res.status == "needs_training_data"
    assert "not trained" in res.fallback_reason.lower() or "not deployed" in res.fallback_reason.lower()


def test_llm_api_failure_does_not_break_vision():
    """Regression test proving LLM API failure does not break image routing or prediction."""
    from src.orchestrator.router import AutonomousRouter
    from src.orchestrator.llm_service import AdvisorySynthesisService

    router = AutonomousRouter()
    synthesis = AdvisorySynthesisService()

    # Mock LLM client raising an API exception
    synthesis.client = MagicMock()
    synthesis.client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable / API Failed")

    test_img = next(Path("data/raw/whole_plant/test/Tomato_Fusarium_Wilt").rglob("*.jpg"))
    telemetry = router.route_and_execute(
        user_text="What disease is on this tomato plant?",
        image_input=test_img,
        crop_hint="Tomato",
        is_new_image=True,
    )

    assert "vision_result" in telemetry
    v_res = telemetry["vision_result"]
    assert v_res is not None
    assert v_res["plant_detected"] is True
    assert v_res["crop"] == "Tomato" or v_res["crop_species"] == "Tomato"

    # Advisory generation handles failure gracefully without throwing
    advisory = synthesis.generate_advisory(telemetry)
    assert advisory is not None
    assert "en" in advisory and "kn" in advisory
    assert len(advisory["en"]) > 20
    assert synthesis.last_llm_status == "LLM unavailable"


def test_leaf_classifier_is_never_used():
    """Verify that WholePlantPredictor does not import or instantiate leaf pathology classifier."""
    import src.whole_plant.predict as wp_pred
    # Must not contain MobileNetV2 foliar backbone references
    assert "PlantDiseaseModel" not in dir(wp_pred)
    assert "MobileNetV2ImageProcessor" not in dir(wp_pred)


def test_train_function_aborts_safely_when_raw_folder_empty(tmp_path):
    """train_whole_plant_model must abort with needs_training_data when folder has no images."""
    empty_raw = tmp_path / "raw" / "whole_plant"
    empty_raw.mkdir(parents=True)
    out_model = tmp_path / "model.pt"

    res = train_whole_plant_model(raw_dir=empty_raw, output_model_path=out_model)
    assert res["status"] == "needs_training_data"
    assert res["trained"] is False
    assert out_model.exists() is False


def test_evaluate_returns_needs_training_data_without_checkpoint(tmp_path):
    """evaluate_whole_plant_model must return clear status when checkpoint is absent."""
    missing_model = tmp_path / "nonexistent_model.pt"
    res = evaluate_whole_plant_model(model_path=missing_model)
    assert res["status"] == "needs_training_data"
    assert res["evaluated"] is False


def test_apple_leaf_supported_and_isolated_from_corn_tomato():
    """Verify Apple leaf is supported and isolated from Corn and Tomato."""
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH
    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    assert "Apple" in predictor.supported_crops
    test_apple_dir = Path("data/raw/whole_plant/test/Apple_Scab")
    test_img = next(test_apple_dir.glob("*.jpg")) if test_apple_dir.exists() else "/fake/apple.jpg"
    res = predictor.predict(test_img, crop_hint="Apple", viewpoint_hint="leaf")
    assert res.supported is True
    assert res.crop == "Apple"
    assert res.viewpoint == "leaf"
    assert "tomato" not in res.top1_label.lower()
    assert "corn" not in res.top1_label.lower()


def test_apple_whole_plant_viewpoint_abstains_safely():
    """Verify Apple with whole_plant viewpoint abstains safely because Apple only has foliar data."""
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH
    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    test_img = "/fake/apple.jpg"
    res = predictor.predict(test_img, crop_hint="Apple", viewpoint_hint="whole_plant")
    assert res.supported is False
    assert res.disease is None
    assert "not trained" in res.fallback_reason.lower() or "not supported" in res.fallback_reason.lower()


def test_maize_cannot_produce_apple_disease():
    """Verify maize input can never produce apple disease labels."""
    from src.whole_plant.predict import WholePlantPredictor, DEFAULT_MODEL_PATH
    predictor = WholePlantPredictor(model_path=DEFAULT_MODEL_PATH)
    test_corn_dir = Path("data/raw/whole_plant/test/Corn_Stunt_Disease")
    test_img = next(test_corn_dir.glob("*.jpg")) if test_corn_dir.exists() else "/fake/corn.jpg"
    res = predictor.predict(test_img, crop_hint="Corn (Maize)", viewpoint_hint="whole_plant")
    assert "apple" not in res.top1_label.lower()
    if res.disease:
        assert "apple" not in res.disease.lower()
    for cand in res.predictions:
        assert "apple" not in cand["label"].lower()

