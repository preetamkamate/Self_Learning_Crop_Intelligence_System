"""
Feedback Queue Logger.
Appends farmer predictions, feedback ratings, and verified agronomic labels
to data/processed/feedback_queue.csv for responsible self-learning.
"""
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FIELDNAMES = [
    "interaction_id",
    "timestamp",
    "input_type",
    "features_or_image",
    "predicted_label",
    "confidence",
    "entropy",
    "farmer_feedback",
    "verified_label",
    "verified_timestamp",
]


def ensure_feedback_file():
    """Ensure feedback_queue.csv exists with correct header."""
    config.FEEDBACK_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not config.FEEDBACK_QUEUE_PATH.exists() or config.FEEDBACK_QUEUE_PATH.stat().st_size == 0:
        with open(config.FEEDBACK_QUEUE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(FIELDNAMES)


def log_interaction(
    interaction_id: str,
    input_type: str,
    features_or_image: str,
    predicted_label: str,
    confidence: float,
    entropy: float = 0.0,
    farmer_feedback: str = "neutral",
    verified_label: str = "",
) -> None:
    """Log an interaction to the feedback queue."""
    ensure_feedback_file()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(config.FEEDBACK_QUEUE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({
            "interaction_id": interaction_id,
            "timestamp": now_str,
            "input_type": input_type,
            "features_or_image": features_or_image,
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "farmer_feedback": farmer_feedback,
            "verified_label": verified_label,
            "verified_timestamp": now_str if verified_label else "",
        })
    logging.info("Logged interaction %s to feedback queue.", interaction_id)


def update_feedback(interaction_id: str, feedback: str, verified_label: Optional[str] = None):
    """Update farmer feedback or add ground-truth verified label."""
    ensure_feedback_file()
    df = pd.read_csv(config.FEEDBACK_QUEUE_PATH)
    mask = df["interaction_id"] == interaction_id
    if not mask.any():
        logging.warning("Interaction ID %s not found in queue.", interaction_id)
        return False

    df.loc[mask, "farmer_feedback"] = feedback
    if verified_label:
        df.loc[mask, "verified_label"] = verified_label
        df.loc[mask, "verified_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(config.FEEDBACK_QUEUE_PATH, index=False)
    logging.info("Updated interaction %s (feedback=%s, verified=%s)", interaction_id, feedback, verified_label)
    return True


def get_queue_statistics() -> Dict[str, Any]:
    """Return summary of feedback staging queue."""
    ensure_feedback_file()
    try:
        df = pd.read_csv(config.FEEDBACK_QUEUE_PATH)
        total = len(df)
        verified = df["verified_label"].dropna().astype(str).str.strip()
        verified_count = len(verified[verified != ""])
        positive_feedback = len(df[df["farmer_feedback"] == "up"])
        negative_feedback = len(df[df["farmer_feedback"] == "down"])
        return {
            "total_interactions": total,
            "verified_samples": verified_count,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "pending_verification": total - verified_count,
        }
    except Exception as e:
        logging.error("Failed to read feedback queue: %s", e)
        return {"total_interactions": 0, "verified_samples": 0}

