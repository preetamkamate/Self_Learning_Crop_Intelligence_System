#!/usr/bin/env python3
"""
Ingest balanced local datasets from ~/Downloads into data/raw/whole_plant/.
Draws exclusively from:
  - ~/Downloads/plant-pathology-2021-fgvc8 (Apple)
  - ~/Downloads/cassava-leaf-disease-classification (Cassava)
Does NOT use external downloads or internet.
Resizes images to 256x256 and generates valid 6-field metadata.
"""
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "whole_plant"
DOWNLOADS_DIR = Path(os.path.expanduser("~/Downloads"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_apple_manifest() -> Dict[str, List[Path]]:
    """Find images for Apple Scab, Apple Rust, and Healthy Apple."""
    pp_dir = DOWNLOADS_DIR / "plant-pathology-2021-fgvc8"
    csv_file = pp_dir / "train.csv"
    img_dir = pp_dir / "train_images"

    if not csv_file.exists() or not img_dir.exists():
        logging.warning("Plant Pathology 2021 dataset not found at %s", pp_dir)
        return {}

    df = pd.read_csv(csv_file)
    manifest = {
        "Apple_Scab": [],
        "Apple_Rust": [],
        "Healthy_Apple_Leaf": [],
    }

    # Clean exact label matching
    scab_df = df[df["labels"] == "scab"]
    rust_df = df[df["labels"] == "rust"]
    healthy_df = df[df["labels"] == "healthy"]

    for _, row in scab_df.iterrows():
        p = img_dir / row["image"]
        if p.exists():
            manifest["Apple_Scab"].append(p)

    for _, row in rust_df.iterrows():
        p = img_dir / row["image"]
        if p.exists():
            manifest["Apple_Rust"].append(p)

    for _, row in healthy_df.iterrows():
        p = img_dir / row["image"]
        if p.exists():
            manifest["Healthy_Apple_Leaf"].append(p)

    logging.info(
        "Found Apple images: Scab=%d, Rust=%d, Healthy=%d",
        len(manifest["Apple_Scab"]),
        len(manifest["Apple_Rust"]),
        len(manifest["Healthy_Apple_Leaf"]),
    )
    return manifest


def load_cassava_manifest() -> Dict[str, List[Path]]:
    """Find images for Cassava CBB, CMD, and Healthy."""
    cas_dir = DOWNLOADS_DIR / "cassava-leaf-disease-classification"
    csv_file = cas_dir / "train.csv"
    img_dir = cas_dir / "train_images"

    if not csv_file.exists() or not img_dir.exists():
        logging.warning("Cassava dataset not found at %s", cas_dir)
        return {}

    df = pd.read_csv(csv_file)
    manifest = {
        "Cassava_Bacterial_Blight_CBB": [],
        "Cassava_Mosaic_Disease_CMD": [],
        "Healthy_Cassava_Leaf": [],
    }

    cbb_df = df[df["label"] == 0]
    cmd_df = df[df["label"] == 3]
    healthy_df = df[df["label"] == 4]

    for _, row in cbb_df.iterrows():
        p = img_dir / row["image_id"]
        if p.exists():
            manifest["Cassava_Bacterial_Blight_CBB"].append(p)

    for _, row in cmd_df.iterrows():
        p = img_dir / row["image_id"]
        if p.exists():
            manifest["Cassava_Mosaic_Disease_CMD"].append(p)

    for _, row in healthy_df.iterrows():
        p = img_dir / row["image_id"]
        if p.exists():
            manifest["Healthy_Cassava_Leaf"].append(p)

    logging.info(
        "Found Cassava images: CBB=%d, CMD=%d, Healthy=%d",
        len(manifest["Cassava_Bacterial_Blight_CBB"]),
        len(manifest["Cassava_Mosaic_Disease_CMD"]),
        len(manifest["Healthy_Cassava_Leaf"]),
    )
    return manifest


def process_and_copy_images(
    src_images: List[Path],
    dest_dir: Path,
    num_train: int = 25,
    num_val: int = 6,
    num_test: int = 6,
) -> Dict[str, List[Path]]:
    """Resize and save images into train, val, and test splits."""
    total_needed = num_train + num_val + num_test
    selected = src_images[:total_needed]

    train_srcs = selected[:num_train]
    val_srcs = selected[num_train : num_train + num_val]
    test_srcs = selected[num_train + num_val : total_needed]

    result = {"train": [], "val": [], "test": []}

    for split_name, srcs in [("train", train_srcs), ("val", val_srcs), ("test", test_srcs)]:
        split_folder = dest_dir / split_name
        split_folder.mkdir(parents=True, exist_ok=True)
        for idx, src_p in enumerate(srcs):
            out_p = split_folder / f"img_{idx:03d}.jpg"
            try:
                with Image.open(src_p) as img:
                    rgb = img.convert("RGB")
                    resized = rgb.resize((256, 256), Image.Resampling.LANCZOS)
                    resized.save(out_p, "JPEG", quality=90)
                result[split_name].append(out_p)
            except Exception as e:
                logging.error("Failed to process %s: %s", src_p, e)

    return result


def build_class_record(
    class_name: str,
    crop: str,
    viewpoint: str,
    health_status: str,
    disease_label: str,
    split: str,
    filename: str,
    img_path_rel: str,
    source_dataset: str,
) -> Dict[str, Any]:
    """Construct a compliant 6-field metadata dictionary."""
    clean_label = disease_label
    src_hash = hashlib.md5(f"{split}_{class_name}_{filename}".encode("utf-8")).hexdigest()[:8].upper()
    source_id = f"{viewpoint.upper()}_{split.upper()}_{crop.upper()}_{src_hash}"

    return {
        "source_id": source_id,
        "image_path": img_path_rel,
        "crop": crop,
        "viewpoint": viewpoint,
        "health_status": health_status,
        "disease_label": clean_label,
        "status": health_status,
        "disease": clean_label,
        "split": split,
        "filename": filename,
        "local_source_metadata": {
            "source_dataset": source_dataset,
            "license": "Research Use Only",
            "annotation_quality": "expert_curated",
            "original_class": class_name,
        },
        "metadata": {
            "source_dataset": source_dataset,
            "license": "Research Use Only",
            "annotation_quality": "expert_curated",
            "original_class": class_name,
        },
    }


def main():
    logging.info("Starting local dataset ingestion from ~/Downloads...")

    apple_manifest = load_apple_manifest()
    cassava_manifest = load_cassava_manifest()

    all_targets = [
        # Apple (viewpoint: leaf)
        {
            "class_name": "Apple_Scab",
            "crop": "Apple",
            "viewpoint": "leaf",
            "health_status": "disease",
            "disease_label": "Apple Scab",
            "sources": apple_manifest.get("Apple_Scab", []),
            "dataset_name": "plant-pathology-2021-fgvc8",
        },
        {
            "class_name": "Apple_Rust",
            "crop": "Apple",
            "viewpoint": "leaf",
            "health_status": "disease",
            "disease_label": "Apple Rust",
            "sources": apple_manifest.get("Apple_Rust", []),
            "dataset_name": "plant-pathology-2021-fgvc8",
        },
        {
            "class_name": "Healthy_Apple_Leaf",
            "crop": "Apple",
            "viewpoint": "leaf",
            "health_status": "healthy",
            "disease_label": "Healthy Apple Leaf",
            "sources": apple_manifest.get("Healthy_Apple_Leaf", []),
            "dataset_name": "plant-pathology-2021-fgvc8",
        },
        # Cassava (viewpoint: leaf)
        {
            "class_name": "Cassava_Bacterial_Blight_CBB",
            "crop": "Cassava",
            "viewpoint": "leaf",
            "health_status": "disease",
            "disease_label": "Cassava Bacterial Blight (CBB)",
            "sources": cassava_manifest.get("Cassava_Bacterial_Blight_CBB", []),
            "dataset_name": "cassava-leaf-disease-classification",
        },
        {
            "class_name": "Cassava_Mosaic_Disease_CMD",
            "crop": "Cassava",
            "viewpoint": "leaf",
            "health_status": "disease",
            "disease_label": "Cassava Mosaic Disease (CMD)",
            "sources": cassava_manifest.get("Cassava_Mosaic_Disease_CMD", []),
            "dataset_name": "cassava-leaf-disease-classification",
        },
        {
            "class_name": "Healthy_Cassava_Leaf",
            "crop": "Cassava",
            "viewpoint": "leaf",
            "health_status": "healthy",
            "disease_label": "Healthy Cassava Leaf",
            "sources": cassava_manifest.get("Healthy_Cassava_Leaf", []),
            "dataset_name": "cassava-leaf-disease-classification",
        },
    ]

    new_records_by_split = {"train": [], "val": [], "test": []}

    for item in all_targets:
        c_name = item["class_name"]
        srcs = item["sources"]
        if len(srcs) < 37:
            logging.error("Not enough sources for %s: found %d, need 37", c_name, len(srcs))
            continue

        logging.info("Processing class %s (%d available sources)...", c_name, len(srcs))
        split_map = process_and_copy_images(
            src_images=srcs,
            dest_dir=RAW_DIR,
            num_train=25,
            num_val=6,
            num_test=6,
        )

        for split, out_paths in split_map.items():
            for p in out_paths:
                # e.g., train/Apple_Scab/img_000.jpg
                rel_p = f"{split}/{c_name}/{p.name}"
                # Move into class directory
                target_folder = RAW_DIR / split / c_name
                target_folder.mkdir(parents=True, exist_ok=True)
                final_p = target_folder / p.name
                if p != final_p:
                    shutil.move(p, final_p)

                rec = build_class_record(
                    class_name=c_name,
                    crop=item["crop"],
                    viewpoint=item["viewpoint"],
                    health_status=item["health_status"],
                    disease_label=item["disease_label"],
                    split=split,
                    filename=p.name,
                    img_path_rel=rel_p,
                    source_dataset=item["dataset_name"],
                )
                new_records_by_split[split].append(rec)

    # Now load existing metadata for train, val, test and merge
    all_combined = []
    for split in ["train", "val", "test"]:
        split_meta_p = RAW_DIR / split / "metadata.json"
        existing_recs = []
        if split_meta_p.exists():
            with open(split_meta_p, "r", encoding="utf-8") as f:
                existing_recs = json.load(f)

        # Remove any existing records for the newly ingested classes to avoid duplicates
        new_class_names = {t["class_name"] for t in all_targets}
        filtered_existing = [
            r for r in existing_recs
            if not any(f"/{c}/" in r.get("image_path", "") for c in new_class_names)
        ]
        merged = filtered_existing + new_records_by_split[split]
        with open(split_meta_p, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        logging.info("Updated %s split: %d existing + %d new = %d total records", split, len(filtered_existing), len(new_records_by_split[split]), len(merged))
        all_combined.extend(merged)

    # Update root metadata.json
    root_meta_p = RAW_DIR / "metadata.json"
    with open(root_meta_p, "w", encoding="utf-8") as f:
        json.dump(all_combined, f, indent=2)
    logging.info("Updated root metadata.json: %d total records across all splits.", len(all_combined))


if __name__ == "__main__":
    main()
