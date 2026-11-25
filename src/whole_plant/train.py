"""
Whole-Plant Disease Model Training Engine.
Trains directly from local folders:
- data/raw/whole_plant/train/
- data/raw/whole_plant/val/
Saves artifacts:
- models/whole_plant/model.pt
- models/whole_plant/labels.json
- models/whole_plant/training_report.json
Never modifies, uses, or interacts with the leaf pathology model.
"""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.whole_plant.dataset import (
    PROCESSED_WHOLE_PLANT_DIR,
    RAW_WHOLE_PLANT_DIR,
    WholePlantDataset,
    create_splits,
    has_training_images,
    load_raw_dataset,
    load_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "whole_plant"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "model.pt"
DEFAULT_LABELS_PATH = DEFAULT_MODEL_DIR / "labels.json"
DEFAULT_TRAINING_REPORT_PATH = DEFAULT_MODEL_DIR / "training_report.json"


class WholePlantModel(nn.Module):
    """
    Dedicated Neural Architecture for Full-Plant & Field Photos.
    Combines multi-scale spatial pooling with a dense probe head.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


def train_whole_plant_model(
    raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR,
    processed_dir: Union[str, Path] = PROCESSED_WHOLE_PLANT_DIR,
    output_model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    output_labels_path: Union[str, Path] = DEFAULT_LABELS_PATH,
    output_report_path: Union[str, Path] = DEFAULT_TRAINING_REPORT_PATH,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a separate whole-plant model from local dataset folders.
    If no images exist or metadata is invalid, safely aborts and returns 'needs_training_data'.
    """
    raw_path = Path(raw_dir)
    all_records, dataset_summary = load_raw_dataset(raw_path)

    if not all_records or dataset_summary.get("status") != "success":
        logging.warning("Cannot train whole-plant model: %s has insufficient valid labeled images.", raw_path)
        return {
            "status": "needs_training_data",
            "message": "Whole-plant model needs training data",
            "trained": False,
            "exact_image_counts": dataset_summary.get("count", 0),
            "crops": dataset_summary.get("crop_counts", {}),
            "viewpoints": dataset_summary.get("viewpoint_counts", {}),
            "labels": dataset_summary.get("label_counts", {}),
            "missing_metadata": dataset_summary.get("total_rejected", 0),
        }

    # If train/ and val/ folders exist in raw_path, use them directly
    if (raw_path / "train").exists():
        train_records = load_split(raw_path, "train")
        val_records = load_split(raw_path, "val") or train_records
    else:
        splits = create_splits(all_records, output_dir=processed_dir)
        train_records = splits["train"]
        val_records = splits["val"] or splits["train"]

    if not train_records or len(train_records) < 5:
        return {
            "status": "needs_training_data",
            "message": "Not enough valid training records to train whole-plant model",
            "trained": False,
            "exact_image_counts": len(train_records),
            "crops": dataset_summary.get("crop_counts", {}),
            "viewpoints": dataset_summary.get("viewpoint_counts", {}),
            "labels": dataset_summary.get("label_counts", {}),
            "missing_metadata": dataset_summary.get("total_rejected", 0),
        }

    # Build unique labels and crop-disease mappings
    unique_labels = sorted({r.disease_label or r.disease or r.status for r in all_records})
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    crop_disease_map: Dict[str, List[str]] = {}
    crop_viewpoint_map: Dict[str, List[str]] = {}
    for r in all_records:
        c = r.crop
        d = r.disease_label or r.disease
        vp = r.viewpoint
        if c not in crop_disease_map:
            crop_disease_map[c] = []
        if d and d not in crop_disease_map[c]:
            crop_disease_map[c].append(d)
        if c not in crop_viewpoint_map:
            crop_viewpoint_map[c] = []
        if vp and vp not in crop_viewpoint_map[c]:
            crop_viewpoint_map[c].append(vp)

    def whole_plant_collate_fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        targets = torch.tensor([b[1] for b in batch], dtype=torch.long)
        meta = [b[2] for b in batch]
        return imgs, targets, meta

    train_ds = WholePlantDataset(train_records, label_to_idx=label_to_idx)
    val_ds = WholePlantDataset(val_records, label_to_idx=label_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=whole_plant_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, len(val_ds)),
        shuffle=False,
        collate_fn=whole_plant_collate_fn,
    )

    if device is None:
        target_device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        target_device = torch.device(device)

    model = WholePlantModel(num_classes=len(unique_labels)).to(target_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logging.info("Training WholePlantModel for %d epochs on device: %s", epochs, target_device)
    model.train()
    history = []
    for ep in range(epochs):
        total_loss = 0.0
        for imgs, targets, _ in train_loader:
            imgs = imgs.to(target_device)
            targets = targets.to(target_device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append({"epoch": ep + 1, "loss": round(total_loss / len(train_loader), 4)})

    # Validation evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets, _ in val_loader:
            imgs = imgs.to(target_device)
            targets = targets.to(target_device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_acc = (correct / total) if total > 0 else 0.0

    # Save artifacts
    out_file = Path(output_model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    clean_supported_crops = sorted([c for c in crop_disease_map.keys() if c not in ["None", "Unknown", "none", "unknown"]])
    clean_supported_diseases = sorted(list({
        d for c, d_list in crop_disease_map.items()
        if c in clean_supported_crops
        for d in d_list
        if "healthy" not in d.lower() and "non-plant" not in d.lower() and "unknown" not in d.lower()
    }))

    checkpoint = {
        "model_state": model.state_dict(),
        "num_classes": len(unique_labels),
        "classes": unique_labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "crop_disease_map": crop_disease_map,
        "crop_viewpoint_map": crop_viewpoint_map,
        "supported_crops": clean_supported_crops,
        "supported_diseases": clean_supported_diseases,
        "val_accuracy": val_acc,
    }
    torch.save(checkpoint, out_file)
    logging.info("Saved trained whole-plant model to: %s (Val Acc: %.2f%%)", out_file, val_acc * 100)

    # Save labels.json
    labels_file = Path(output_labels_path)
    labels_content = {
        "classes": unique_labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        "crop_disease_map": crop_disease_map,
        "crop_viewpoint_map": crop_viewpoint_map,
        "supported_crops": clean_supported_crops,
        "supported_diseases": clean_supported_diseases,
    }
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels_content, f, indent=2)
    logging.info("Saved whole-plant labels to: %s", labels_file)

    # Save training_report.json
    report_file = Path(output_report_path)
    training_report = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trained": True,
        "model_path": str(out_file),
        "labels_path": str(labels_file),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "device": str(target_device),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "val_accuracy": round(val_acc, 4),
        "num_classes": len(unique_labels),
        "supported_crops": clean_supported_crops,
        "supported_diseases": clean_supported_diseases,
        "training_history": history,
        "dataset_summary": {
            "total_records": len(all_records),
            "crops": dataset_summary.get("crop_counts", {}),
            "viewpoints": dataset_summary.get("viewpoint_counts", {}),
            "labels": dataset_summary.get("label_counts", {}),
        },
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)
    logging.info("Saved training report to: %s", report_file)

    return training_report


def main():
    parser = argparse.ArgumentParser(description="Train Whole-Plant Disease Model from Local Folders")
    parser.add_argument("--data-dir", type=str, default=str(RAW_WHOLE_PLANT_DIR), help="Path to data/raw/whole_plant")
    parser.add_argument("--output-model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to save model checkpoint")
    parser.add_argument("--output-labels", type=str, default=str(DEFAULT_LABELS_PATH), help="Path to save labels json")
    parser.add_argument("--output-report", type=str, default=str(DEFAULT_TRAINING_REPORT_PATH), help="Path to save training report")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    result = train_whole_plant_model(
        raw_dir=args.data_dir,
        output_model_path=args.output_model,
        output_labels_path=args.output_labels,
        output_report_path=args.output_report,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
