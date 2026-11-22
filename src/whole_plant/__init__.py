"""
Whole-Plant & Field Photo Analysis Module.
Independent pipeline for full-plant and canopy diagnostics without relying on leaf pathology models.
"""
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

__all__ = [
    "WholePlantRecord",
    "WholePlantDataset",
    "WholePlantModel",
    "WholePlantPredictor",
    "WholePlantPredictionResult",
    "train_whole_plant_model",
    "evaluate_whole_plant_model",
    "has_training_images",
    "load_raw_dataset",
    "create_splits",
]
