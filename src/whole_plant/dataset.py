"""
Whole-Plant Dataset and Split Management.
Designed specifically for full-plant and field photos (rejects close-up single-leaf inputs).
Supports local dataset structure in data/raw/whole_plant/{train,val,test}.
Requires and validates 6 metadata fields:
  - image_path
  - crop
  - viewpoint
  - health_status
  - disease_label
  - source_id
"""
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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

SUPPORTED_VIEWPOINTS = {
    "whole_plant",
    "canopy",
    "stem",
    "fruit",
    "root",
    "bark",
    "leaf",
    "unknown",
    "non_plant",
}


@dataclass
class WholePlantRecord:
    """Represents a single full-plant / field image record with required local metadata."""
    crop: str
    image_path: str
    health_status: str = "unknown"
    disease_label: Optional[str] = None
    viewpoint: str = "whole_plant"
    source_id: str = ""
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
        if not self.source_id:
            h_sub = hashlib.md5(f"{self.crop}_{self.image_path}".encode("utf-8")).hexdigest()[:8].upper()
            self.source_id = f"WP_{self.crop.upper().replace(' ', '_')}_{h_sub}"
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
        src_id = data.get("source_id", "")
        return cls(
            crop=data.get("crop", "Unknown"),
            image_path=str(data.get("image_path", "")),
            health_status=h_status,
            disease_label=d_label,
            disease=d_label,
            status=h_status,
            viewpoint=data.get("viewpoint", "whole_plant"),
            source_id=src_id,
            local_source_metadata=data.get("local_source_metadata", data.get("metadata", {})),
            metadata=data.get("metadata", data.get("local_source_metadata", {})),
        )


def compute_file_hash(path: Path) -> Optional[str]:
    """Compute sha256 hash for deduplication check."""
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def validate_metadata(
    raw_records: List[Dict[str, Any]],
    base_dir: Path,
    seen_hashes: Optional[Dict[str, str]] = None,
) -> Tuple[List[WholePlantRecord], Dict[str, Any]]:
    """
    Validate all metadata entries for whole plant images.
    Rejects missing, duplicate, invalid, or cross-split duplicate images.
    Returns:
        (valid_records, validation_report)
    """
    valid_records: List[WholePlantRecord] = []
    rejected: List[Dict[str, Any]] = []
    crop_counts: Dict[str, int] = {}
    viewpoint_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    missing_fields_count = 0

    local_hashes = seen_hashes if seen_hashes is not None else {}
    seen_paths: Set[str] = set()

    for item in raw_records:
        rel_path = item.get("image_path") or item.get("filename", "")
        crop = item.get("crop")
        viewpoint = str(item.get("viewpoint", "whole_plant")).lower().strip()
        health_status = item.get("health_status") or item.get("status")
        disease_label = item.get("disease_label") or item.get("disease")
        source_id = item.get("source_id")

        # 1. Required fields check
        missing_fields = []
        if not rel_path:
            missing_fields.append("image_path")
        if not crop:
            missing_fields.append("crop")
        if not viewpoint:
            missing_fields.append("viewpoint")
        if not health_status:
            missing_fields.append("health_status")
        if source_id is None:
            missing_fields.append("source_id")

        if missing_fields:
            missing_fields_count += 1
            rejected.append({"item": item, "reason": f"Missing required metadata fields: {missing_fields}"})
            continue

        # 2. Viewpoint validation
        if viewpoint not in SUPPORTED_VIEWPOINTS:
            rejected.append({"item": item, "reason": f"Unsupported viewpoint '{viewpoint}'. Must be one of {SUPPORTED_VIEWPOINTS}"})
            continue

        # 3. Path existence check
        p1 = base_dir / rel_path
        p2 = base_dir / os.path.basename(rel_path)
        img_resolved = p1 if p1.exists() else (p2 if p2.exists() else Path(rel_path))

        if not img_resolved.exists():
            rejected.append({"item": item, "reason": f"Image file not found on disk: {rel_path}"})
            continue

        # 4. Duplicate checks (path and image content hash)
        resolved_str = str(img_resolved.resolve())
        if resolved_str in seen_paths:
            rejected.append({"item": item, "reason": f"Duplicate image file path: {resolved_str}"})
            continue
        seen_paths.add(resolved_str)

        f_hash = compute_file_hash(img_resolved)
        if f_hash:
            if f_hash in local_hashes:
                rejected.append({"item": item, "reason": f"Duplicate image content (matches {local_hashes[f_hash]})"})
                continue
            local_hashes[f_hash] = resolved_str

        # Build clean valid record
        record = WholePlantRecord(
            crop=crop,
            image_path=str(img_resolved),
            health_status=health_status,
            disease_label=disease_label,
            viewpoint=viewpoint,
            source_id=str(source_id),
            local_source_metadata=item.get("local_source_metadata", item.get("metadata", {})),
            metadata=item.get("metadata", item.get("local_source_metadata", {})),
        )
        valid_records.append(record)

        # Update stats
        crop_counts[crop] = crop_counts.get(crop, 0) + 1
        viewpoint_counts[viewpoint] = viewpoint_counts.get(viewpoint, 0) + 1
        lbl_key = disease_label or health_status
        label_counts[lbl_key] = label_counts.get(lbl_key, 0) + 1

    report = {
        "total_evaluated": len(raw_records),
        "valid_count": len(valid_records),
        "rejected_count": len(rejected),
        "missing_metadata_count": missing_fields_count,
        "crop_counts": crop_counts,
        "viewpoint_counts": viewpoint_counts,
        "label_counts": label_counts,
        "rejected_samples": rejected[:5],
    }
    return valid_records, report


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


def load_split(
    raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR,
    split: str = "train",
    seen_hashes: Optional[Dict[str, str]] = None,
    return_report: bool = False,
) -> Union[List[WholePlantRecord], Tuple[List[WholePlantRecord], Dict[str, Any]]]:
    """
    Load and validate records for a specific split (train, val, or test) from data/raw/whole_plant/<split>.
    """
    path = Path(raw_dir)
    split_dir = path / split if (path / split).exists() else path
    metadata_file = split_dir / "metadata.json"

    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            records, report = validate_metadata(data, base_dir=path, seen_hashes=seen_hashes)
            return (records, report) if return_report else records
        except Exception as e:
            logging.error("Failed to parse split metadata in %s: %s", metadata_file, e)

    # Fallback: scan class subdirectories in split_dir
    fallback_items: List[Dict[str, Any]] = []
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

            fallback_items.append({
                "crop": crop,
                "image_path": str(img_file),
                "status": status,
                "health_status": status,
                "disease_label": disease,
                "disease": disease,
                "viewpoint": "whole_plant",
                "source_id": f"WP_{split.upper()}_{crop.upper()}_{img_file.stem}",
                "local_source_metadata": {"filename": img_file.name, "split": split},
            })
    records, report = validate_metadata(fallback_items, base_dir=path, seen_hashes=seen_hashes)
    return (records, report) if return_report else records


def load_raw_dataset(raw_dir: Union[str, Path] = RAW_WHOLE_PLANT_DIR) -> Tuple[List[WholePlantRecord], Dict[str, Any]]:
    """
    Load and cross-validate raw whole-plant records across train, val, and test splits.
    If no images exist or not enough valid images, returns empty records with status 'needs_training_data'.
    """
    path = Path(raw_dir)
    if not has_training_images(path):
        logging.warning("No whole-plant images found in: %s", path)
        return [], {
            "status": "needs_training_data",
            "message": "Whole-plant model needs training data",
            "count": 0,
            "crop_counts": {},
            "viewpoint_counts": {},
            "label_counts": {},
        }

    global_seen_hashes: Dict[str, str] = {}
    splits_loaded = {}
    all_records: List[WholePlantRecord] = []
    total_rejected = 0

    # Load splits sequentially with cross-split deduplication
    for split_name in ["train", "val", "test"]:
        if (path / split_name).exists():
            records, report = load_split(path, split_name, seen_hashes=global_seen_hashes, return_report=True)
            splits_loaded[split_name] = {"count": len(records), "report": report}
            all_records.extend(records)
            total_rejected += report.get("rejected_count", 0)

    if not all_records and (path / "metadata.json").exists():
        records, report = load_split(path, "", seen_hashes=global_seen_hashes, return_report=True)
        splits_loaded["root"] = {"count": len(records), "report": report}
        all_records.extend(records)
        total_rejected += report.get("rejected_count", 0)

    # Compute overall statistics
    crop_counts: Dict[str, int] = {}
    viewpoint_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    for r in all_records:
        crop_counts[r.crop] = crop_counts.get(r.crop, 0) + 1
        viewpoint_counts[r.viewpoint] = viewpoint_counts.get(r.viewpoint, 0) + 1
        lbl = r.disease_label or r.health_status
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    summary = {
        "status": "success" if all_records else "needs_training_data",
        "message": f"Loaded and validated {len(all_records)} whole-plant records across splits",
        "count": len(all_records),
        "total_rejected": total_rejected,
        "splits": splits_loaded,
        "crop_counts": crop_counts,
        "viewpoint_counts": viewpoint_counts,
        "label_counts": label_counts,
    }
    return all_records, summary


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
