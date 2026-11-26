#!/usr/bin/env python3
"""
Generates deterministic, verified real-camera manifests for Whole-Plant and Canopy Diagnosis.
Filters out:
  - Synthetic images / lab black-background segmentations
  - Leaf-only indoor lab specimens (PlantVillage)
  - Close-up foliar-only leaves (FGVC8, Cassava leaf close-ups)
  - Duplicate images (SHA-256 identical files)
  - Corrupted or unreadable files

Partitions data deterministically into no-leak train / validation / test splits (70% / 15% / 15%).
Outputs:
  - data/manifests/whole_plant_canopy_manifest.json
  - data/manifests/whole_plant_split_summary.json
"""
import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT_DIR / "data" / "manifests"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import audit runner
from scripts.reproducible_audit import run_full_dataset_audit


def build_whole_plant_manifests() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    logging.info("Building verified whole-plant and canopy manifests...")
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run full audit to retrieve verified records
    audit_report, verified_records = run_full_dataset_audit()

    logging.info("Total verified whole-plant/canopy records from audit: %d", len(verified_records))

    # Standardize crop and class names
    standardized_records: List[Dict[str, Any]] = []

    for r in verified_records:
        crop = r["crop"]
        dis = r["disease_label"]
        vp = r["viewpoint"]

        # Standardize crop name
        if crop in ["Corn", "Maize"]:
            crop_std = "Corn (Maize)"
        elif crop in ["Bell Pepper", "Pepper"]:
            crop_std = "Bell Pepper"
        else:
            crop_std = crop

        # Standardize disease label
        dis_clean = dis.replace("Leaf ", "").replace("  ", " ").strip()
        if "Late Blight" in dis_clean:
            dis_std = "Late Blight"
        elif "Early Blight" in dis_clean:
            dis_std = "Early Blight"
        elif "Yellow Leaf Curl" in dis_clean or "Yellow Curl" in dis_clean or "Yellow Virus" in dis_clean:
            dis_std = "Yellow Leaf Curl Virus"
        elif "Mosaic" in dis_clean:
            dis_std = "Mosaic Virus"
        elif "Bacterial Wilt" in dis_clean:
            dis_std = "Bacterial Wilt"
        elif "Bunchy Top" in dis_clean:
            dis_std = "Bunchy Top"
        elif "Panama" in dis_clean:
            dis_std = "Panama Disease"
        elif "Blast" in dis_clean:
            dis_std = "Blast"
        elif "Stripe Rust" in dis_clean:
            dis_std = "Stripe Rust"
        elif "Downy Mildew" in dis_clean:
            dis_std = "Downy Mildew"
        elif "Greening" in dis_clean:
            dis_std = "Greening Disease"
        else:
            dis_std = dis_clean

        full_class = f"{crop_std}___{dis_std}"
        src_id = f"WP_{crop_std.upper().replace(' ', '_').replace('(', '').replace(')', '')}_{r['sha256'][:8].upper()}"
        standardized_records.append({
            "sha256": r["sha256"],
            "image_path": r["image_path"],
            "source_dataset": r["source_dataset"],
            "relative_path": r["relative_path"],
            "source_id": src_id,
            "crop": crop_std,
            "disease_label": dis_std,
            "full_class": full_class,
            "viewpoint": vp,
            "health_status": "Healthy" if "healthy" in dis_std.lower() else "Disease",
            "real_photo": True,
        })

    # Group by class to ensure stratified, deterministic no-leak split
    class_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in standardized_records:
        class_groups[r["full_class"]].append(r)

    # Sort each class by SHA-256 for 100% deterministic reproducibility
    for cls in class_groups:
        class_groups[cls].sort(key=lambda x: x["sha256"])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    # Deterministic split: 70% train, 15% val, 15% test per class
    for cls, items in class_groups.items():
        n = len(items)
        if n == 1:
            items[0]["split"] = "train"
            train_records.append(items[0])
            continue
        elif n == 2:
            items[0]["split"] = "train"
            items[1]["split"] = "test"
            train_records.append(items[0])
            test_records.append(items[1])
            continue

        n_train = max(1, int(round(n * 0.70)))
        n_val = max(1, int(round(n * 0.15)))
        # Remaining to test
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1

        for idx, item in enumerate(items):
            if idx < n_train:
                item["split"] = "train"
                train_records.append(item)
            elif idx < n_train + n_val:
                item["split"] = "val"
                val_records.append(item)
            else:
                item["split"] = "test"
                test_records.append(item)

    all_split_records = train_records + val_records + test_records

    # Deduplication & Data Leakage Verification
    train_hashes = set(r["sha256"] for r in train_records)
    val_hashes = set(r["sha256"] for r in val_records)
    test_hashes = set(r["sha256"] for r in test_records)

    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)

    assert len(train_val_overlap) == 0, f"Leakage detected between train and val: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Leakage detected between train and test: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Leakage detected between val and test: {len(val_test_overlap)}"

    # Per-split, per-crop, per-viewpoint metrics
    split_summary = {
        "manifest_name": "whole_plant_canopy_manifest",
        "total_verified_samples": len(all_split_records),
        "split_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "crops_count": len(set(r["crop"] for r in all_split_records)),
        "classes_count": len(class_groups),
        "leakage_verification": {
            "train_val_overlap": len(train_val_overlap),
            "train_test_overlap": len(train_test_overlap),
            "val_test_overlap": len(val_test_overlap),
            "zero_leakage_guaranteed": True,
        },
        "viewpoint_distribution": dict(Counter(r["viewpoint"] for r in all_split_records)),
        "crop_distribution": dict(Counter(r["crop"] for r in all_split_records).most_common()),
        "split_distributions": {
            "train_crops": dict(Counter(r["crop"] for r in train_records).most_common()),
            "val_crops": dict(Counter(r["crop"] for r in val_records).most_common()),
            "test_crops": dict(Counter(r["crop"] for r in test_records).most_common()),
        },
    }

    # Save manifests
    manifest_path = MANIFESTS_DIR / "whole_plant_canopy_manifest.json"
    summary_path = MANIFESTS_DIR / "whole_plant_split_summary.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_split_records, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    logging.info("Manifests generated successfully!")
    logging.info("Train: %d, Val: %d, Test: %d | Total: %d", len(train_records), len(val_records), len(test_records), len(all_split_records))
    return split_summary, all_split_records


if __name__ == "__main__":
    summary, records = build_whole_plant_manifests()
    print("\n" + "=" * 60)
    print("WHOLE-PLANT & CANOPY MANIFEST SUMMARY:")
    print(f"Total Verified Real Samples: {summary['total_verified_samples']}")
    print(f"Split Counts: {summary['split_counts']}")
    print(f"Viewpoints: {summary['viewpoint_distribution']}")
    print(f"Crops: {len(summary['crop_distribution'])}")
    print(f"Zero Leakage: {summary['leakage_verification']['zero_leakage_guaranteed']}")
    print("=" * 60)
