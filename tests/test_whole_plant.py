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
    assert "Whole-plant model needs training data" in result.message


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
    assert "Whole plant detected, but this crop or disease is not supported yet." in res.message

    # 2. Unsupported crop
    res_unsupported_crop = predictor.predict("/fake/image.jpg", crop_hint="Apple", confidence_threshold=0.50)
    assert res_unsupported_crop.supported is False
    assert res_unsupported_crop.disease is None
    assert "Whole plant detected, but this crop or disease is not supported yet." in res_unsupported_crop.message


def test_predictor_prediction_structure_fields():
    """Verify WholePlantPredictionResult structure contains all required fields."""
    res = WholePlantPredictionResult(
        crop="Tomato",
        status="disease",
        confidence=0.88,
        supported=True,
        disease="Fusarium Wilt",
        message="Whole-plant diagnosis: Fusarium Wilt",
    )
    d = res.to_dict()
    assert "crop" in d
    assert "status" in d
    assert "confidence" in d
    assert "supported" in d
    assert "disease" in d
    assert "message" in d
    assert d["top1_label"] == "Fusarium Wilt"


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
