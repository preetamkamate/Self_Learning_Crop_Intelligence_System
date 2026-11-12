"""
Responsible Batch Self-Learning & Retraining Gate.
Reads only confirmed verified labels from feedback_queue.csv, trains candidate model,
and enforces a strict accuracy improvement gate before production promotion.
"""
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_batch_retrain_gate(min_verified_samples: int = 5) -> Dict[str, Any]:
    """
    Executes the responsible self-learning retraining gate.

    Args:
        min_verified_samples: Minimum confirmed agronomist-verified samples required.

    Returns:
        Dict detailing candidate evaluation, comparison against production, and promotion status.
    """
    logging.info("Starting Batch Retraining Gate check...")

    if not config.FEEDBACK_QUEUE_PATH.exists():
        return {
            "status": "aborted",
            "reason": "Feedback queue does not exist yet.",
            "promoted": False,
        }

    queue_df = pd.read_csv(config.FEEDBACK_QUEUE_PATH)
    # Filter strictly for non-empty verified labels on tabular entries
    valid_verified = queue_df[
        (queue_df["verified_label"].notna())
        & (queue_df["verified_label"].astype(str).str.strip() != "")
        & (queue_df["input_type"] == "tabular")
    ]

    verified_count = len(valid_verified)
    logging.info("Found %d verified tabular samples in staging queue.", verified_count)

    if verified_count < min_verified_samples:
        msg = (
            f"Insufficient verified samples ({verified_count} < {min_verified_samples}). "
            "Batch retraining deferred to prevent model overfitting or drift."
        )
        logging.info(msg)
        return {
            "status": "deferred",
            "reason": msg,
            "verified_samples_found": verified_count,
            "min_required": min_verified_samples,
            "promoted": False,
        }

    # Load baseline dataset
    base_df = pd.read_csv(config.CROP_DATA_PATH)
    logging.info("Loaded baseline dataset with %d rows.", len(base_df))

    # Parse verified rows into training format
    new_rows = []
    for _, row in valid_verified.iterrows():
        try:
            # features_or_image contains json or key-value string
            feats_str = str(row["features_or_image"])
            if "{" in feats_str:
                feats = json.loads(feats_str)
            else:
                parts = [float(x.strip()) for x in feats_str.split(",")]
                feats = dict(zip(config.TABULAR_FEATURES, parts))

            row_dict = {feat: float(feats[feat]) for feat in config.TABULAR_FEATURES}
            row_dict[config.TABULAR_TARGET] = str(row["verified_label"]).strip().lower()
            new_rows.append(row_dict)
        except Exception as e:
            logging.warning("Skipping malformed feedback row: %s (error: %s)", row, e)

    if not new_rows:
        return {
            "status": "aborted",
            "reason": "Could not parse valid features from verified feedback rows.",
            "promoted": False,
        }

    augmented_df = pd.concat([base_df, pd.DataFrame(new_rows)], ignore_index=True)
    logging.info("Augmented dataset total rows: %d (+%d verified samples)", len(augmented_df), len(new_rows))

    # Prepare features and labels
    X = augmented_df[config.TABULAR_FEATURES]
    y = augmented_df[config.TABULAR_TARGET]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.TABULAR_TEST_SIZE,
        random_state=config.TABULAR_RANDOM_STATE,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train candidate model
    candidate_rf = RandomForestClassifier(
        n_estimators=130,
        max_depth=16,
        random_state=config.TABULAR_RANDOM_STATE,
        n_jobs=-1,
    )
    candidate_rf.fit(X_train_scaled, y_train)
    candidate_pred = candidate_rf.predict(X_test_scaled)
    candidate_acc = float(accuracy_score(y_test, candidate_pred))

    # Read current production metrics
    production_acc = 0.0
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, "r") as f:
            curr_metrics = json.load(f)
            production_acc = float(curr_metrics.get("test_accuracy", 0.0))

    logging.info("Candidate Model Test Accuracy: %.4f", candidate_acc)
    logging.info("Production Model Test Accuracy: %.4f", production_acc)

    # Evaluation Gate: Promote ONLY if candidate accuracy >= production accuracy
    if candidate_acc >= production_acc:
        logging.info("--> EVALUATION GATE PASSED: Candidate meets or exceeds production threshold!")
        archive_dir = config.TABULAR_MODEL_DIR / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Archive existing model
        if config.MODEL_PATH.exists():
            shutil.copy2(config.MODEL_PATH, archive_dir / f"crop_model_prev_{ts_str}.joblib")

        # Promote candidate
        joblib.dump(candidate_rf, config.MODEL_PATH)
        joblib.dump(scaler, config.SCALER_PATH)
        joblib.dump(encoder, config.ENCODER_PATH)

        new_metrics = {
            "model_type": "RandomForestClassifier (Retrained Candidate)",
            "retrained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": len(augmented_df),
            "new_verified_samples_incorporated": len(new_rows),
            "test_accuracy": round(candidate_acc, 4),
            "previous_accuracy": round(production_acc, 4),
            "classes": list(encoder.classes_),
            "status": "promoted",
        }
        with open(config.METRICS_PATH, "w") as f:
            json.dump(new_metrics, f, indent=2)

        return {
            "status": "promoted",
            "candidate_accuracy": round(candidate_acc, 4),
            "production_accuracy": round(production_acc, 4),
            "verified_samples_used": len(new_rows),
            "promoted": True,
            "message": "Candidate model successfully validated and promoted to production.",
        }
    else:
        logging.warning("--> EVALUATION GATE REJECTED: Candidate accuracy (%.4f) < Production (%.4f)", candidate_acc, production_acc)
        return {
            "status": "rejected",
            "candidate_accuracy": round(candidate_acc, 4),
            "production_accuracy": round(production_acc, 4),
            "verified_samples_used": len(new_rows),
            "promoted": False,
            "message": "Candidate accuracy degraded. Retaining current production model intact.",
        }


if __name__ == "__main__":
    result = run_batch_retrain_gate(min_verified_samples=1)
    print("Retrain Gate Result:", result)

