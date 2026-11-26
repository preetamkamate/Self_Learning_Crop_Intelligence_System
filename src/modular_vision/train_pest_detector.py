"""
Pest Detection Neural Classifier Training Engine.
Trains a dedicated binary/multi-class pest classifier strictly from validated real samples:
- data/raw/pest_detection/train/
- data/raw/pest_detection/val/
Saves artifacts:
- models/pest_detector/model.pt
- models/pest_detector/labels.json
- models/pest_detector/training_report.json
Completely independent from the plant disease pathology classifier.
"""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PEST_DATA_DIR = ROOT_DIR / "data" / "raw" / "pest_detection"
OUTPUT_MODEL_DIR = ROOT_DIR / "models" / "pest_detector"
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)


class RealPestDataset(Dataset):
    """Dataset for authentic pest photographs."""

    def __init__(self, split_dir: Path, class_to_idx: Dict[str, int]):
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx = class_to_idx

        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = split_dir / cls_name
            if cls_dir.exists():
                for p in sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))):
                    self.samples.append((p, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            rgb = img.convert("RGB").resize((224, 224))
            arr = np.array(rgb, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


class DedicatedPestCNN(nn.Module):
    """Dedicated architecture for fine-grained pest and insect scouting."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)


def train_pest_model(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.001,
) -> Dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Training dedicated pest model on device: %s", device)

    classes = ["Healthy_Pest_Free_Foliage", "Two_Spotted_Spider_Mites"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_label = {
        0: "Healthy Foliage (Pest-Free Control)",
        1: "Two-spotted Spider Mites (Tetranychus urticae)",
    }

    train_ds = RealPestDataset(PEST_DATA_DIR / "train", class_to_idx)
    val_ds = RealPestDataset(PEST_DATA_DIR / "val", class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DedicatedPestCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        avg_loss = total_loss / len(train_ds)
        history.append({"epoch": epoch, "loss": round(avg_loss, 4)})

    # Validation accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == targets).sum().item()

    val_acc = correct / len(val_ds) if len(val_ds) > 0 else 0.0
    logging.info("Pest model training complete. Val Accuracy: %.2f%%", val_acc * 100)

    # Save model checkpoint
    model_path = OUTPUT_MODEL_DIR / "model.pt"
    checkpoint = {
        "model_state": model.state_dict(),
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_label": idx_to_label,
        "pest_categories": list(idx_to_label.values()),
        "source_dataset": "PlantVillage Dataset (Mohanty et al., 2016)",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "val_accuracy": val_acc,
    }
    torch.save(checkpoint, model_path)

    # Save labels and report
    with open(OUTPUT_MODEL_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, indent=2)

    report = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "val_accuracy": round(val_acc, 4),
        "epochs": epochs,
        "classes": list(idx_to_label.values()),
        "source_dataset": "PlantVillage Dataset (Mohanty et al., 2016)",
        "history": history,
    }
    with open(OUTPUT_MODEL_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    train_pest_model()
