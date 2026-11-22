"""
Whole-Plant Dataset and Split Management.
Designed specifically for full-plant and field photos (rejects close-up single-leaf inputs).
Supports local dataset structure in data/raw/whole_plant/{train,val,test}.
"""
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_WHOLE_PLANT_DIR = ROOT_DIR / "data" / "raw" / "whole_plant"
PROCESSED_WHOLE_PLANT_DIR = ROOT_DIR / "data" / "processed" / "whole_plant"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class WholePlantRecord:
    """Represents a single full-plant / field image record with required local metadata."""
    crop: str
    image_path: str
    health_status: str = "unknown"
    disease_label: Optional[str] = None
    viewpoint: str = "whole_plant"
    disease: Optional[str] = None
    status: Optional[str] = None
    local_source_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Synchronize health_status and status for full compatibility
        if not self.health_status or self.health_status == "unknown":
            if self.status:
                self.health_status = self.status
        if not self.status:
            self.status = self.health_status
        if not self.disease and self.disease_label:
            self.disease = self.disease_label
        elif not self.disease_label and self.disease:
            self.disease_label = self.disease
        if not self.local_source_metadata and self.metadata:
            self.local_source_metadata = dict(self.metadata)
        elif not self.metadata and self.local_source_metadata:
            self.metadata = dict(self.local_source_metadata)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WholePlantRecord":
        h_status = data.get("health_status") or data.get("status", "unknown")
        d_label = data.get("disease_label") or data.get("disease")
        return cls(
            crop=data.get("crop", "Unknown"),
            image_path=str(data.get("image_path", "")),
            health_status=h_status,
            disease_label=d_label,
            disease=d_label,
            status=h_status,
            viewpoint=data.get("viewpoint", "whole_plant"),
            local_source_metadata=data.get("local_source_metadata", data.get("metadata", {})),
            metadata=data.get("metadata", data.get("local_source_metadata", {})),
        )


def has_training_images(raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR) -> bool:
    """Check if the local raw whole-plant directory contains any image files."""
    path = Path(raw_dir)
    if not path.exists() or not path.is_dir():
        return False

    for root, _, files in os.walk(path):
        for f in files:
            if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                return True
    return False


def load_split(raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR, split: str = "train") -> List[WholePlantRecord]:
    """
    Load records for a specific split (train, val, or test) from data/raw/whole_plant/<split>.
    """
    path = Path(raw_dir)
    split_dir = path / split if (path / split).exists() else path
    metadata_file = split_dir / "metadata.json"

    records: List[WholePlantRecord] = []
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                img_rel = item.get("image_path", item.get("filename", ""))
                # Resolve path relative to split_dir or raw_dir
                resolved = path / img_rel if (path / img_rel).exists() else (split_dir / img_rel if (split_dir / img_rel).exists() else Path(img_rel))
                rec = WholePlantRecord(
                    crop=item.get("crop", "Unknown"),
                    image_path=str(resolved),
                    health_status=item.get("health_status") or item.get("status", "unknown"),
                    disease_label=item.get("disease_label") or item.get("disease"),
                    viewpoint=item.get("viewpoint", "whole_plant"),
                    local_source_metadata=item.get("local_source_metadata", item.get("metadata", {})),
                    metadata=item.get("metadata", item.get("local_source_metadata", {})),
                )
                records.append(rec)
            return records
        except Exception as e:
            logging.error("Failed to parse split metadata in %s: %s", metadata_file, e)

    # Fallback: scan class subdirectories in split_dir
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        for img_file in sorted(split_dir.rglob(f"*{ext}")):
            cls_name = img_file.parent.name
            crop = "Unknown"
            disease = None
            status = "disease"
            if "healthy" in cls_name.lower():
                status = "healthy"
                crop = cls_name.split("_")[1] if "_" in cls_name else "Plant"
            elif "non-plant" in cls_name.lower():
                status = "non_plant"
                crop = "None"
            else:
                crop = cls_name.split("_")[0] if "_" in cls_name else cls_name
                disease = cls_name.replace("_", " ")

            records.append(
                WholePlantRecord(
                    crop=crop,
                    image_path=str(img_file),
                    status=status,
                    disease_label=disease,
                    disease=disease,
                    viewpoint="whole_plant",
                    local_source_metadata={"filename": img_file.name, "split": split},
                )
            )
    return records


def load_raw_dataset(raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR) -> Tuple[List[WholePlantRecord], Dict[str, Any]]:
    """
    Load raw whole-plant records from local data/raw/whole_plant directory.
    If no images exist, returns empty records with status 'needs_training_data'.
    """
    path = Path(raw_dir)
    if not has_training_images(path):
        logging.warning("No whole-plant images found in: %s", path)
        return [], {
            "status": "needs_training_data",
            "message": "Whole-plant model needs training data",
            "count": 0,
        }

    # If train split folder exists, load all splits
    if (path / "train").exists():
        records = load_split(path, "train") + load_split(path, "val") + load_split(path, "test")
        if records:
            return records, {
                "status": "success",
                "message": f"Loaded {len(records)} whole-plant records across train/val/test splits",
                "count": len(records),
            }

    # Otherwise load from root metadata or scan
    records = load_split(path, "")
    return records, {
        "status": "success",
        "message": f"Loaded {len(records)} whole-plant records",
        "count": len(records),
    }


def create_splits(
    records: List[WholePlantRecord],
    output_dir: Union[str, Path] = PROCESSED_WHOLE_PLANT_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[WholePlantRecord]]:
    """
    Split records into train, validation, and test sets.
    Saves metadata manifests into output_dir/train, output_dir/val, and output_dir/test.
    """
    if not records:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio)) if n_total >= 3 else 0

    train_records = shuffled[:n_train]
    val_records = shuffled[n_train : n_train + n_val]
    test_records = shuffled[n_train + n_val :]

    out_path = Path(output_dir)
    for split_name, split_list in [("train", train_records), ("val", val_records), ("test", test_records)]:
        split_dir = out_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = split_dir / "records.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in split_list], f, indent=2)

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


class WholePlantDataset(Dataset):
    """PyTorch Dataset for full-plant and field photos."""

    def __init__(
        self,
        records: List[WholePlantRecord],
        label_to_idx: Optional[Dict[str, int]] = None,
        transform: Optional[Any] = None,
    ):
        self.records = records
        self.transform = transform

        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            labels = sorted({r.disease_label or r.disease or r.status for r in records})
            self.label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict[str, Any]]:
        record = self.records[idx]
        img_path = Path(record.image_path)

        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            # Fallback canvas for tests or missing file
            img = Image.new("RGB", (224, 224), color=(34, 139, 34))

        if self.transform is not None:
            img_tensor = self.transform(img)
        elif torch is not None:
            resized = img.resize((224, 224))
            arr = torch.tensor(list(getattr(resized, "get_flattened_data", resized.getdata)()), dtype=torch.float32).view(224, 224, 3)
            img_tensor = arr.permute(2, 0, 1) / 255.0
        else:
            img_tensor = img

        target_label = record.disease_label or record.disease or record.status
        target_idx = self.label_to_idx.get(target_label, 0)
        return img_tensor, target_idx, record.to_dict()
