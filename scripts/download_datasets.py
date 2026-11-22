#!/usr/bin/env python3
"""
Safe Dataset Download & Registry Management Utility.
Strictly adheres to agricultural research data governance:
1. Automatically downloads ONLY datasets explicitly listed in dataset_registry.yaml.
2. Does NOT scrape Google Images or random websites.
3. Does NOT bypass registration, paywalls, access controls, licenses, or terms.
4. If manual approval/authentication is required, shows the official source page and instructions.
5. Stores raw downloads in data/raw/datasets/ separately from processed data and models.
6. Never commits large datasets, model weights, or API keys to Git.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REGISTRY_PATH = ROOT_DIR / "dataset_registry.yaml"
RAW_DATASETS_DIR = config.RAW_DATA_DIR / "datasets"
STATUS_CACHE_PATH = config.DATA_DIR / "datasets_status.json"


def load_registry() -> List[Dict[str, Any]]:
    """Load and validate the approved dataset registry."""
    if not REGISTRY_PATH.exists():
        logging.error("Approved dataset registry not found at %s", REGISTRY_PATH)
        sys.exit(1)
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("datasets", [])


def get_dataset_by_id(dataset_id: str, registry: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for an approved dataset ID."""
    clean_id = dataset_id.strip().lower()
    for ds in registry:
        if ds.get("id", "").lower() == clean_id:
            return ds
    return None


def list_datasets(registry: List[Dict[str, Any]]) -> None:
    """Print clean catalog of approved datasets."""
    print("\n" + "=" * 90)
    print("APPROVED AGRICULTURAL DATASET REGISTRY (dataset_registry.yaml)")
    print("=" * 90)
    for ds in registry:
        ds_id = ds.get("id")
        name = ds.get("dataset_name") or ds.get("name")
        license_name = ds.get("license")
        allowed_use = ds.get("allowed_use", "Research and Educational")
        crops = ", ".join(ds.get("crop", ds.get("crops", [])))
        parts = ", ".join(ds.get("plant_viewpoint", ds.get("supported_parts", [])))
        images = ds.get("image_count", ds.get("expected_images", 0))
        status = ds.get("download_status", ds.get("status", "not_downloaded"))
        auth = "Yes (Manual Auth Required)" if ds.get("requires_auth") else "No (Direct Open Access)"
        
        print(f"\n[ID: {ds_id}]")
        print(f"  • Name:             {name}")
        print(f"  • Official Page:    {ds.get('official_source_page', ds.get('official_page'))}")
        print(f"  • License:          {license_name}")
        print(f"  • Allowed Use:      {allowed_use}")
        print(f"  • Crops:            {crops}")
        print(f"  • Viewpoints:       {parts}")
        print(f"  • Image Count:      {images}")
        print(f"  • Auth Required:    {auth}")
        print(f"  • Download Status:  {status}")
    print("\n" + "=" * 90)


def download_dataset(ds: Dict[str, Any], force: bool = False) -> bool:
    """Safely download an approved dataset without bypassing access controls."""
    ds_id = ds.get("id")
    name = ds.get("dataset_name") or ds.get("name")
    url = ds.get("download_url")
    official_page = ds.get("official_source_page", ds.get("official_page"))
    requires_auth = ds.get("requires_auth", False)
    auth_instr = ds.get("auth_instructions", "")

    print(f"\n--- Processing Approved Dataset: [{ds_id}] {name} ---")

    # Safe access control check: NEVER bypass authentication or terms
    if requires_auth:
        print(f"🛑 [AUTHENTICATION REQUIRED] Dataset '{ds_id}' requires user authentication or terms acceptance.")
        print(f"   Official Source Page: {official_page}")
        print(f"   License / Terms:      {ds.get('license')}")
        print(f"   Instructions:         {auth_instr}")
        print("   -> Automated bypass is strictly disabled per safety and compliance guidelines.")
        return False

    if not url or not url.startswith("http"):
        print(f"ℹ️ Dataset '{ds_id}' is managed locally or through curated verified benchmarks.")
        print(f"   Official Documentation: {official_page}")
        return True

    target_dir = RAW_DATASETS_DIR / ds_id
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"{ds_id}.zip"

    if zip_path.exists() and not force:
        print(f"✓ Dataset archive already exists at {zip_path}. Use --force to re-download.")
        return True

    print(f"⬇️ Downloading verified archive from: {url}")
    print(f"   Destination: {zip_path}")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (CropIntelligence/2.0 Research Benchmark)"}
        )
        with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, "wb") as out_file:
            total_size = int(response.info().get("Content-Length", -1))
            downloaded = 0
            block_size = 65536
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    sys.stdout.write(f"\r   Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r   Downloaded: {downloaded / (1024*1024):.1f} MB")
                    sys.stdout.flush()
        print("\n✓ Download completed successfully.")

        # Extract if valid zip
        if zipfile.is_zipfile(zip_path):
            print(f"📦 Extracting archive to {target_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            print("✓ Extraction completed.")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Approved Agricultural Dataset Downloader")
    parser.add_argument("--list", action="store_true", help="List all approved datasets in registry")
    parser.add_argument("--dataset-id", type=str, help="Download a specific approved dataset by ID")
    parser.add_argument("--all", action="store_true", help="Download all approved public open datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download existing datasets")
    args = parser.parse_args()

    registry = load_registry()

    if args.list or (not args.dataset_id and not args.all):
        list_datasets(registry)
        return

    if args.dataset_id:
        ds = get_dataset_by_id(args.dataset_id, registry)
        if not ds:
            print(f"❌ Error: Dataset '{args.dataset_id}' is NOT listed in approved dataset_registry.yaml.")
            print("   Automatic downloading of unapproved or scraped sources is strictly forbidden.")
            sys.exit(1)
        download_dataset(ds, force=args.force)

    elif args.all:
        print("Starting batch download for all approved public datasets...")
        for ds in registry:
            download_dataset(ds, force=args.force)


if __name__ == "__main__":
    main()
