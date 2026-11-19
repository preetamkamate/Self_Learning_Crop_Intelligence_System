"""
Safe Dataset Download & Registry Management Utility.
Manages legally approved agricultural datasets for whole-plant and multi-part crop health.

Features:
- Reads from dataset_registry.yaml
- Validates downloads against the approved registry only
- Checks SHA-256 checksums if provided
- Prevents duplicate downloads
- Separates raw downloads into data/raw/datasets/
- Reports download progress and status
- Handles gated/authenticated sources responsibly (e.g. Kaggle CLI instructions)
- NEVER executes automatically at application startup
- NEVER bypasses logins, paywalls, or terms of use
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REGISTRY_PATH = ROOT_DIR / "dataset_registry.yaml"
STATUS_CACHE_PATH = config.DATA_DIR / "datasets_status.json"


def load_registry() -> List[Dict[str, Any]]:
    """Load the approved dataset registry from YAML."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Dataset registry not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("datasets", [])


def get_status_cache() -> Dict[str, str]:
    """Load local dataset download status cache."""
    if STATUS_CACHE_PATH.exists():
        try:
            with open(STATUS_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def update_status_cache(dataset_id: str, status: str) -> None:
    """Update status cache on disk."""
    cache = get_status_cache()
    cache[dataset_id] = status
    STATUS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_dataset_by_id(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve dataset record by ID."""
    registry = load_registry()
    for ds in registry:
        if ds["id"] == dataset_id:
            # Merge with dynamic status cache
            status_cache = get_status_cache()
            if dataset_id in status_cache:
                ds["status"] = status_cache[dataset_id]
            elif ds.get("status") != "ready":
                # Check if local dir exists and has content
                local_dir = ROOT_DIR / ds.get("local_dir", "")
                if local_dir.exists() and any(local_dir.iterdir()):
                    ds["status"] = "ready"
            return ds
    return None


def list_datasets() -> List[Dict[str, Any]]:
    """List all approved datasets with active status."""
    registry = load_registry()
    status_cache = get_status_cache()
    for ds in registry:
        ds_id = ds["id"]
        if ds_id in status_cache:
            ds["status"] = status_cache[ds_id]
        elif ds.get("status") != "ready":
            local_dir = ROOT_DIR / ds.get("local_dir", "")
            if local_dir.exists() and any(local_dir.iterdir()):
                ds["status"] = "ready"
    return registry


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DownloadProgressBar:
    """Console progress reporter for urllib downloads."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.last_reported_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, int(downloaded * 100 / total_size))
        if pct % 10 == 0 and pct != self.last_reported_pct:
            logging.info("[%s] Download progress: %d%% (%d / %d MB)",
                         self.dataset_name, pct, downloaded // (1024 * 1024), total_size // (1024 * 1024))
            self.last_reported_pct = pct


def download_dataset(dataset_id: str, force: bool = False) -> Dict[str, Any]:
    """
    Safely download an approved dataset by ID.
    Enforces registry verification, checksum check, auth handling, and duplicate prevention.
    """
    ds = get_dataset_by_id(dataset_id)
    if not ds:
        error_msg = f"Dataset ID '{dataset_id}' is not in the approved dataset registry."
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}

    # Check if already ready
    target_dir = ROOT_DIR / ds.get("local_dir", f"data/raw/datasets/{dataset_id}")
    if not force and ds.get("status") == "ready":
        logging.info("Dataset '%s' is already marked as ready in %s.", ds['name'], target_dir)
        return {"status": "ready", "message": f"Dataset already downloaded and ready at {target_dir}"}

    # Check if requires authentication or manual access
    if ds.get("requires_auth"):
        instructions = ds.get("auth_instructions", "Authentication required.")
        msg = (
            f"Dataset '{ds['name']}' requires user authentication / agreement.\n"
            f"Source Page: {ds['official_page']}\n"
            f"Instructions: {instructions}\n"
            "Automation cannot bypass logins or accept terms of service on your behalf."
        )
        logging.warning(msg)
        return {"status": "requires_auth", "message": msg, "instructions": instructions}

    download_url = ds.get("download_url")
    if not download_url:
        msg = f"No direct download URL provided for '{ds['name']}'. Please visit {ds['official_page']}."
        return {"status": "manual_action_required", "message": msg}

    target_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(download_url).name
    if not filename or filename == "master.zip" or "/" in filename:
        filename = f"{dataset_id}.zip"
    dest_file = target_dir / filename

    if dest_file.exists() and not force:
        logging.info("Destination archive %s already exists. Verifying...", dest_file)
        expected_sha = ds.get("checksum_sha256", "").strip()
        if expected_sha:
            computed_sha = calculate_sha256(dest_file)
            if computed_sha.lower() == expected_sha.lower():
                update_status_cache(dataset_id, "ready")
                return {"status": "ready", "message": "Archive exists and checksum verified."}
            else:
                logging.warning("Existing archive failed checksum! Re-downloading.")
        else:
            update_status_cache(dataset_id, "ready")
            return {"status": "ready", "message": "Archive already downloaded."}

    logging.info("Starting safe download for: %s", ds['name'])
    logging.info("License: %s", ds.get("license"))
    logging.info("Target: %s", dest_file)
    update_status_cache(dataset_id, "downloading")

    try:
        progress = DownloadProgressBar(ds["id"])
        urllib.request.urlretrieve(download_url, dest_file, reporthook=progress)

        # Checksum validation
        expected_sha = ds.get("checksum_sha256", "").strip()
        if expected_sha:
            computed_sha = calculate_sha256(dest_file)
            if computed_sha.lower() != expected_sha.lower():
                update_status_cache(dataset_id, "failed")
                dest_file.unlink(missing_ok=True)
                err_msg = f"Checksum mismatch for {dataset_id}! Expected {expected_sha}, got {computed_sha}."
                logging.error(err_msg)
                return {"status": "failed", "message": err_msg}

        update_status_cache(dataset_id, "ready")
        success_msg = f"Successfully downloaded '{ds['name']}' to {dest_file}."
        logging.info(success_msg)
        return {"status": "ready", "message": success_msg, "file_path": str(dest_file)}

    except Exception as e:
        update_status_cache(dataset_id, "failed")
        logging.error("Failed to download dataset %s: %s", dataset_id, e)
        return {"status": "failed", "message": f"Download failed: {str(e)}"}


def print_registry_table():
    """Print a clean CLI table of all datasets in registry."""
    datasets = list_datasets()
    print("=" * 110)
    print(f"{'ID':<26} | {'CROP COVERAGE':<25} | {'SUPPORTED PARTS':<22} | {'LICENSE':<18} | {'STATUS'}")
    print("=" * 110)
    for d in datasets:
        crops = ", ".join(d.get("crops", []))[:24]
        parts = ", ".join(d.get("supported_parts", []))[:21]
        lic = d.get("license", "").split("(")[0].strip()[:17]
        status = d.get("status", "not_downloaded")
        print(f"{d['id']:<26} | {crops:<25} | {parts:<22} | {lic:<18} | {status.upper()}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Agricultural Dataset Registry & Safe Downloader")
    parser.add_argument("--list", action="store_true", help="List all datasets in registry and their statuses")
    parser.add_argument("--download", type=str, help="Download a specific dataset by ID from the approved registry")
    parser.add_argument("--force", action="store_true", help="Force re-download even if already downloaded")
    parser.add_argument("--status", action="store_true", help="Check status of all datasets")

    args = parser.parse_args()

    if args.list or args.status:
        print_registry_table()
    elif args.download:
        res = download_dataset(args.download, force=args.force)
        print(f"\nResult: [{res['status'].upper()}] - {res['message']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
