"""
Tabular Crop Recommendation Model Training.
Trains RandomForestClassifier and GradientBoostingClassifier on the real 2,200-row dataset,
selects the best model, evaluates actual test metrics, and serializes artifacts.
"""
import json
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_crop_model(data_path: Path = config.CROP_DATA_PATH):
    """
    Train tabular ML models on the real Crop Recommendation dataset.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run dataset downloader first.")

    logging.info("Loading dataset from: %s", data_path)
    df = pd.read_csv(data_path)
    logging.info("Dataset shape: %s", df.shape)

    # Validate schema
    for col in config.TABULAR_FEATURES + [config.TABULAR_TARGET]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")

    X = df[config.TABULAR_FEATURES]
    y = df[config.TABULAR_TARGET]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    logging.info("Identified %d unique crop classes: %s", num_classes, list(label_encoder.classes_))

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.TABULAR_TEST_SIZE,
        random_state=config.TABULAR_RANDOM_STATE,
        stratify=y_encoded,
    )
    logging.info("Training set: %d samples, Test set: %d samples", len(X_train), len(X_test))

    # Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Candidate 1: RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        random_state=config.TABULAR_RANDOM_STATE,
        n_jobs=-1,
    )
    logging.info("Training Candidate 1: RandomForestClassifier...")
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = float(accuracy_score(y_test, rf_pred))
    logging.info("RandomForest Test Accuracy: %.4f (%.2f%%)", rf_acc, rf_acc * 100)

    # Candidate 2: GradientBoostingClassifier
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=config.TABULAR_RANDOM_STATE,
    )
    logging.info("Training Candidate 2: GradientBoostingClassifier...")
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    gb_acc = float(accuracy_score(y_test, gb_pred))
    logging.info("GradientBoosting Test Accuracy: %.4f (%.2f%%)", gb_acc, gb_acc * 100)

    # Select best model
    if rf_acc >= gb_acc:
        best_model = rf
        best_name = "RandomForestClassifier"
        best_acc = rf_acc
        best_pred = rf_pred
    else:
        best_model = gb
        best_name = "GradientBoostingClassifier"
        best_acc = gb_acc
        best_pred = gb_pred

    logging.info("--> Selected Best Model: %s with Test Accuracy: %.4f", best_name, best_acc)

    # Compute actual classification report
    report_dict = classification_report(
        y_test,
        best_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        best_pred,
        target_names=label_encoder.classes_,
    )
    print("\n" + "=" * 60)
    print(f"VERIFIED TEST CLASSIFICATION REPORT ({best_name}):")
    print("=" * 60)
    print(report_text)
    print("=" * 60)

    # Feature Importances
    if hasattr(best_model, "feature_importances_"):
        importances = {
            feat: round(float(imp), 4)
            for feat, imp in zip(config.TABULAR_FEATURES, best_model.feature_importances_)
        }
    else:
        importances = {}

    metrics = {
        "model_type": best_name,
        "dataset_rows": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_classes": num_classes,
        "test_accuracy": round(best_acc, 4),
        "candidate_accuracies": {
            "RandomForestClassifier": round(rf_acc, 4),
            "GradientBoostingClassifier": round(gb_acc, 4),
        },
        "feature_importances": importances,
        "classification_summary": {
            "macro_avg": report_dict["macro avg"],
            "weighted_avg": report_dict["weighted avg"],
        },
        "classes": list(label_encoder.classes_),
    }

    # Save artifacts
    config.TABULAR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.ENCODER_PATH)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info("Model saved to: %s", config.MODEL_PATH)
    logging.info("Scaler saved to: %s", config.SCALER_PATH)
    logging.info("Encoder saved to: %s", config.ENCODER_PATH)
    logging.info("Metrics saved to: %s", config.METRICS_PATH)

    return metrics


if __name__ == "__main__":
    train_crop_model()

