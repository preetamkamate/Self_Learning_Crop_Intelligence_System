"""
Whole-Plant Model Held-Out Evaluation Engine.
Evaluates accuracy, precision, recall, F1-score, confusion matrix, false crop-routing rate,
and unsupported-crop abstention rate on the local held-out test split (data/raw/whole_plant/test/).
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
from src.whole_plant.predict import (
    UNSUPPORTED_CROP_MSG,
    WholePlantPredictor,
)
from src.whole_plant.train import DEFAULT_MODEL_DIR, DEFAULT_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_EVALUATION_REPORT_PATH = DEFAULT_MODEL_DIR / "evaluation_report.json"


def evaluate_whole_plant_model(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    test_records_or_dir: Optional[Union[List[WholePlantRecord], str, Path]] = None,
    output_report_path: Optional[Union[str, Path]] = DEFAULT_EVALUATION_REPORT_PATH,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the whole-plant model strictly on the held-out local test split.
    Calculates sample counts, per-crop precision/recall/F1, confusion matrix,
    false crop-routing rate, and unsupported-crop abstention rate.
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

    # Load held-out test records
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

    # 1. Sample Counts
    total_samples = len(records)
    sample_counts_per_crop: Dict[str, int] = {}
    sample_counts_per_class: Dict[str, int] = {}
    for r in records:
        c = r.crop
        lbl = r.disease_label or r.disease or r.health_status
        sample_counts_per_crop[c] = sample_counts_per_crop.get(c, 0) + 1
        sample_counts_per_class[lbl] = sample_counts_per_class.get(lbl, 0) + 1

    # 2. Run Predictions & Track Metrics
    y_true: List[int] = []
    y_pred: List[int] = []
    crop_true_list: List[str] = []
    crop_pred_list: List[str] = []
    abstained_count = 0
    false_crop_routing_count = 0

    for r in records:
        ground_truth_label = r.disease_label or r.disease or r.health_status
        true_idx = class_to_idx.get(ground_truth_label, -1)
        y_true.append(true_idx)
        crop_true_list.append(r.crop)

        res = predictor.predict(r.image_path, crop_hint=r.crop)
        predicted_crop = res.crop
        crop_pred_list.append(predicted_crop)

        # Cross-crop routing safety check
        if r.crop in ["Corn (Maize)", "Tomato"] and predicted_crop in ["Corn (Maize)", "Tomato"]:
            if r.crop != predicted_crop:
                false_crop_routing_count += 1
                logging.error("CRITICAL CROSS-CROP ROUTING LEAK: True %s routed as %s!", r.crop, predicted_crop)

        # Non-plant or unsupported abstention
        if not res.supported or res.status in ["unsupported", "uncertain", "needs_training_data", "out_of_distribution"]:
            abstained_count += 1
            if res.is_ood and "Non-Plant" in ground_truth_label:
                # Correct non-plant rejection
                np_idx = class_to_idx.get("Non-Plant Background Object", -1)
                y_pred.append(np_idx)
            elif "Unknown" in ground_truth_label:
                unk_idx = class_to_idx.get("Unknown Plant Foliage", -1)
                y_pred.append(unk_idx)
            else:
                y_pred.append(-1)
        else:
            pred_lbl = res.disease or res.top1_label
            pred_idx = class_to_idx.get(pred_lbl, -1)
            y_pred.append(pred_idx)

    # 3. False Crop Routing Rate
    in_dist_samples = sum(1 for c in crop_true_list if c in ["Corn (Maize)", "Tomato"])
    false_crop_routing_rate = (false_crop_routing_count / in_dist_samples) if in_dist_samples > 0 else 0.0

    # 4. Unsupported Crop Abstention Rate Test
    # Test on synthetic/benchmark records from crops absent from training (e.g. Grape, Potato, Apple, Cotton, Banana)
    unsupported_crops_to_test = ["Grape", "Potato", "Apple", "Cotton", "Banana", "Sugarcane", "Wheat"]
    unsupported_abstentions = 0
    for u_crop in unsupported_crops_to_test:
        u_res = predictor.predict(records[0].image_path, crop_hint=u_crop)
        if (
            not u_res.supported
            and u_res.disease is None
            and len(u_res.predictions) == 0
            and u_res.top1_label == UNSUPPORTED_CROP_MSG
        ):
            unsupported_abstentions += 1

    unsupported_crop_abstention_rate = unsupported_abstentions / len(unsupported_crops_to_test)

    # 5. Confusion Matrix & Accuracy
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    correct = 0
    valid_eval_samples = 0
    for t, p in zip(y_true, y_pred):
        if t != -1:
            valid_eval_samples += 1
            if p != -1 and 0 <= p < num_classes:
                conf_matrix[t, p] += 1
                if t == p:
                    correct += 1

    accuracy = correct / valid_eval_samples if valid_eval_samples > 0 else 0.0
    abstention_rate = abstained_count / total_samples if total_samples > 0 else 0.0

    # 6. Per-Class Precision, Recall, F1
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

    # 7. Per-Crop Metrics (Tomato & Corn)
    per_crop_metrics: Dict[str, Dict[str, Any]] = {}
    for crop_name in ["Tomato", "Corn (Maize)"]:
        crop_classes = [c for c in predictor.crop_disease_map.get(crop_name, []) if c in class_to_idx]
        if crop_classes:
            c_precs = [precision_per_class.get(c, 0.0) for c in crop_classes]
            c_recs = [recall_per_class.get(c, 0.0) for c in crop_classes]
            c_f1s = [f1_per_class.get(c, 0.0) for c in crop_classes]
            per_crop_metrics[crop_name] = {
                "sample_count": sample_counts_per_crop.get(crop_name, 0),
                "precision": round(float(np.mean(c_precs)), 4),
                "recall": round(float(np.mean(c_recs)), 4),
                "f1_score": round(float(np.mean(c_f1s)), 4),
                "classes": crop_classes,
            }

    report = {
        "status": "success",
        "evaluated": True,
        "total_test_samples": total_samples,
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "false_crop_routing_rate": round(false_crop_routing_rate, 4),
        "unsupported_crop_abstention_rate": round(unsupported_crop_abstention_rate, 4),
        "test_abstention_rate": round(abstention_rate, 4),
        "sample_counts_per_crop": sample_counts_per_crop,
        "sample_counts_per_class": sample_counts_per_class,
        "per_crop_metrics": per_crop_metrics,
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
        "per_class_f1": f1_per_class,
        "classes": classes,
        "confusion_matrix": conf_matrix.tolist(),
        "model_checkpoint": str(m_path),
        "test_data_dir": str(RAW_WHOLE_PLANT_DIR / "test"),
    }

    if output_report_path:
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logging.info("Saved held-out evaluation report to: %s", out_p)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whole-Plant Model on Held-Out Test Split")
    parser.add_argument("--test-dir", type=str, default=str(RAW_WHOLE_PLANT_DIR / "test"), help="Path to test folder")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--output-report", type=str, default=str(DEFAULT_EVALUATION_REPORT_PATH), help="Output JSON path")
    args = parser.parse_args()

    report = evaluate_whole_plant_model(
        model_path=args.model_path,
        test_records_or_dir=args.test_dir,
        output_report_path=args.output_report,
    )
    print("\n" + "=" * 80)
    print("WHOLE-PLANT MODEL HELD-OUT TEST EVALUATION REPORT")
    print("=" * 80)
    print(f"Total Held-Out Test Samples:       {report.get('total_test_samples')}")
    print(f"Overall Accuracy:                  {report.get('overall_accuracy', 0.0)*100:.2f}%")
    print(f"Macro Precision:                   {report.get('macro_precision', 0.0):.4f}")
    print(f"Macro Recall:                      {report.get('macro_recall', 0.0):.4f}")
    print(f"Macro F1 Score:                    {report.get('macro_f1', 0.0):.4f}")
    print(f"False Crop Routing Rate:           {report.get('false_crop_routing_rate', 0.0)*100:.2f}% (Target: 0.00%)")
    print(f"Unsupported Crop Abstention Rate:  {report.get('unsupported_crop_abstention_rate', 0.0)*100:.2f}% (Target: 100.00%)")
    print("\n--- Per-Crop Breakdown ---")
    for crop, met in report.get("per_crop_metrics", {}).items():
        print(f"  [{crop}] Samples: {met['sample_count']} | Precision: {met['precision']:.4f} | Recall: {met['recall']:.4f} | F1: {met['f1_score']:.4f}")
    print("\nReport saved to:", args.output_report)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
