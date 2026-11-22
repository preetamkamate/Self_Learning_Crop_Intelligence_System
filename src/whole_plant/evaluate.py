"""
Whole-Plant Model Held-Out Evaluation Engine.
Evaluates accuracy, precision, recall, F1-score, confusion matrix, and abstention rate
on the local test folder (data/raw/whole_plant/test/).
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from src.whole_plant.dataset import (
    PROCESSED_WHOLE_PLANT_DIR,
    RAW_WHOLE_PLANT_DIR,
    WholePlantRecord,
    load_split,
)
from src.whole_plant.predict import WholePlantPredictor
from src.whole_plant.train import DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_whole_plant_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    test_records_or_dir: Optional[Union[List[WholePlantRecord], str, Path]] = None,
    output_report_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the whole-plant model on a held-out test folder.
    Calculates precision, recall, F1, confusion matrix, and unknown/abstention rate.
    """
    m_path = Path(model_path)
    if not m_path.exists():
        logging.warning("Cannot evaluate: Model checkpoint not found at %s", m_path)
        return {
            "status": "needs_training_data",
            "message": "Whole-plant model needs training data and checkpoint",
            "evaluated": False,
        }

    predictor = WholePlantPredictor(model_path=m_path, device=device)
    if not predictor.is_trained:
        return {
            "status": "needs_training_data",
            "message": "Whole-plant model is not loaded or untrained",
            "evaluated": False,
        }

    # Load test records
    records: List[WholePlantRecord] = []
    if isinstance(test_records_or_dir, list):
        records = test_records_or_dir
    elif test_records_or_dir:
        t_path = Path(test_records_or_dir)
        if t_path.name == "test":
            records = load_split(t_path.parent, "test")
        else:
            records = load_split(t_path, "test") if (t_path / "test").exists() else load_split(t_path, "")
    else:
        # Default test location: data/raw/whole_plant/test
        records = load_split(RAW_WHOLE_PLANT_DIR, "test")

    if not records:
        logging.warning("No test records found for evaluation.")
        return {
            "status": "no_test_data",
            "message": "No test records found in local test folder",
            "evaluated": False,
        }

    classes = sorted(list(predictor.idx_to_label.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    y_true: List[int] = []
    y_pred: List[int] = []
    abstained_count = 0

    for r in records:
        ground_truth_label = r.disease_label or r.disease or r.status
        true_idx = class_to_idx.get(ground_truth_label, 0)
        y_true.append(true_idx)

        res = predictor.predict(r.image_path, crop_hint=r.crop)
        if not res.supported or res.status in ["unsupported", "uncertain", "needs_training_data"]:
            abstained_count += 1
            y_pred.append(-1)
        else:
            pred_lbl = res.disease or res.status
            pred_idx = class_to_idx.get(pred_lbl, -1)
            y_pred.append(pred_idx)

    # Compute metrics
    total_samples = len(records)
    abstention_rate = abstained_count / total_samples if total_samples > 0 else 0.0

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    correct = 0
    for t, p in zip(y_true, y_pred):
        if p != -1 and 0 <= p < num_classes:
            conf_matrix[t, p] += 1
            if t == p:
                correct += 1

    accuracy = correct / total_samples if total_samples > 0 else 0.0

    # Class-wise precision, recall, f1
    precision_per_class: Dict[str, float] = {}
    recall_per_class: Dict[str, float] = {}
    f1_per_class: Dict[str, float] = {}

    prec_list = []
    rec_list = []
    f1_list = []

    for idx, c_name in enumerate(classes):
        tp = conf_matrix[idx, idx]
        fp = sum(conf_matrix[r_idx, idx] for r_idx in range(num_classes) if r_idx != idx)
        fn = sum(conf_matrix[idx, c_idx] for c_idx in range(num_classes) if c_idx != idx)

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precision_per_class[c_name] = round(prec, 4)
        recall_per_class[c_name] = round(rec, 4)
        f1_per_class[c_name] = round(f1, 4)

        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
    macro_recall = float(np.mean(rec_list)) if rec_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    report = {
        "status": "success",
        "evaluated": True,
        "total_samples": total_samples,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "abstention_rate": round(abstention_rate, 4),
        "classes": classes,
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
        "per_class_f1": f1_per_class,
        "confusion_matrix": conf_matrix.tolist(),
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logging.info("Evaluation report saved to: %s", out_p)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whole-Plant Disease Model on Local Test Folder")
    parser.add_argument("--test-dir", type=str, default=str(RAW_WHOLE_PLANT_DIR / "test"), help="Path to test folder")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--output-report", type=str, default=str(PROCESSED_WHOLE_PLANT_DIR / "evaluation_report.json"), help="Output JSON path")
    args = parser.parse_args()

    report = evaluate_whole_plant_model(
        model_path=args.model_path,
        test_records_or_dir=args.test_dir,
        output_report_path=args.output_report,
    )
    print("\n================ Whole-Plant Evaluation Report ================")
    print(f"Total Test Samples: {report.get('total_samples')}")
    print(f"Accuracy:           {report.get('accuracy', 0.0)*100:.2f}%")
    print(f"Macro Precision:    {report.get('macro_precision', 0.0):.4f}")
    print(f"Macro Recall:       {report.get('macro_recall', 0.0):.4f}")
    print(f"Macro F1 Score:     {report.get('macro_f1', 0.0):.4f}")
    print(f"Abstention Rate:    {report.get('abstention_rate', 0.0)*100:.2f}%")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
