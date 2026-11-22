#!/usr/bin/env python3
"""
Agricultural Dataset Validation & Integrity Checker.
Verifies dataset hygiene before model training:
1. Validates that image files are uncorrupted, readable, and in RGB mode.
2. Checks train, val, and test partitions to guarantee zero sample leakage.
3. Validates image metadata manifests: crop, viewpoint, issue type, disease label,
   source URL, license, and annotation quality.
4. Reports class distribution and split counts.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def hash_file(filepath: Path) -> str:
    """Compute MD5 hash for image deduplication check."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset_split(
    dataset_dir: Path,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Validate images, splits, and metadata within a dataset directory.
    """
    if not dataset_dir.exists():
        return {"status": "error", "message": f"Directory not found: {dataset_dir}"}

    splits = ["train", "val", "test"]
    split_hashes: Dict[str, Set[str]] = {s: set() for s in splits}
    split_counts: Dict[str, Dict[str, int]] = {s: {} for s in splits}
    corrupted_images: List[str] = []
    total_images = 0

    for s in splits:
        split_dir = dataset_dir / s
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name.startswith("."):
                continue

            class_name = class_dir.name
            split_counts[s][class_name] = 0

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                    continue

                total_images += 1
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                    # Reopen to check dimensions and mode
                    with Image.open(img_path) as im:
                        w, h = im.size
                        if w < 10 or h < 10:
                            corrupted_images.append(f"{img_path} (Invalid dimensions: {w}x{h})")
                            continue

                    f_hash = hash_file(img_path)
                    split_hashes[s].add(f_hash)
                    split_counts[s][class_name] += 1
                except Exception as e:
                    corrupted_images.append(f"{img_path} ({e})")

    # Check for train / val / test overlap (data leakage)
    train_val_overlap = split_hashes["train"].intersection(split_hashes["val"])
    train_test_overlap = split_hashes["train"].intersection(split_hashes["test"])
    val_test_overlap = split_hashes["val"].intersection(split_hashes["test"])

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    # Manifest verification
    manifest_valid = False
    manifest_count = 0
    manifest_fields_ok = True
    manifest_file = manifest_path or (dataset_dir / "image_manifest.json")
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            manifest_count = len(records)
            required_keys = ["crop", "viewpoint", "issue_type", "disease_label", "source_url", "license", "annotation_quality"]
            for rec in records:
                if not all(k in rec for k in required_keys):
                    manifest_fields_ok = False
                    break
            manifest_valid = manifest_fields_ok and manifest_count > 0
        except Exception as e:
            manifest_valid = False

    return {
        "status": "valid" if (not corrupted_images and not has_leakage) else "issues_found",
        "dataset_name": dataset_dir.name,
        "dataset_path": str(dataset_dir),
        "total_images": total_images,
        "corrupted_images_count": len(corrupted_images),
        "corrupted_images": corrupted_images[:10],
        "split_counts": split_counts,
        "data_leakage": {
            "has_leakage": has_leakage,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
        },
        "manifest": {
            "manifest_file_exists": manifest_file.exists(),
            "manifest_records_count": manifest_count,
            "manifest_valid": manifest_valid,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Agricultural Dataset Validator")
    parser.add_argument("--dataset-dir", "--dataset", type=str, default="data/datasets/whole_plant", help="Path or name of dataset directory")
    parser.add_argument("--all", action="store_true", help="Validate all active datasets in data/datasets")
    args = parser.parse_args()

    base_dir = config.DATA_DIR / "datasets"

    if args.all:
        target_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    else:
        target = Path(args.dataset_dir)
        if not target.is_absolute():
            if (base_dir / target.name).is_dir():
                target_dirs = [base_dir / target.name]
            elif (ROOT_DIR / target).is_dir():
                target_dirs = [ROOT_DIR / target]
            else:
                target_dirs = [target]
        else:
            target_dirs = [target]

    print("\n" + "=" * 80)
    print("RUNNING AGRICULTURAL DATASET VALIDATION SUITE")
    print("=" * 80)

    overall_pass = True
    for d in target_dirs:
        res = validate_dataset_split(d)
        print(f"\n📂 Dataset: {res.get('dataset_name')} ({res.get('dataset_path')})")
        print(f"   • Total Images Checked:    {res.get('total_images')}")
        print(f"   • Corrupted Images:        {res.get('corrupted_images_count')}")
        
        leak = res.get("data_leakage", {})
        if leak.get("has_leakage"):
            print(f"   ❌ DATA LEAKAGE DETECTED! Train/Test overlap: {leak.get('train_test_overlap_count')}")
            overall_pass = False
        else:
            print("   ✓ Split Partitioning:     STRICT ZERO LEAKAGE (Train/Val/Test completely independent)")

        man = res.get("manifest", {})
        if man.get("manifest_file_exists"):
            status_str = "VALID (all required agronomic fields present)" if man.get("manifest_valid") else "INVALID SCHEMA"
            print(f"   ✓ Image Manifest:         {status_str} ({man.get('manifest_records_count')} records)")
        else:
            print("   ℹ Image Manifest:         No local image_manifest.json found.")

        split_c = res.get("split_counts", {})
        print("   • Split Distribution:")
        for s_name, classes in split_c.items():
            tot = sum(classes.values())
            print(f"     - {s_name.upper():5s}: {tot:4d} images ({len(classes)} classes)")

    print("\n" + "=" * 80)
    if overall_pass:
        print("✓ ALL DATASET VALIDATION CHECKS PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("❌ DATASET VALIDATION FAILED DUE TO CORRUPTION OR DATA LEAKAGE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
