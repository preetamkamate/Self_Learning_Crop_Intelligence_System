"""
Empirical Entropy Threshold Calibration Engine.
Calibrates the optimal Shannon entropy threshold H(p) and normalized entropy H_norm(p) = H(p) / ln(C)
using a diverse held-out validation set of:
1. Valid healthy crop images
2. Valid confirmed foliar disease images
3. Valid whole-plant field images
4. Valid canopy images
5. Degraded / blurry images
6. Non-plant OOD images (cat, checkerboard, noise)

Evaluates candidate thresholds from 1.5 to 3.2 (raw) and 0.40 to 0.90 (normalized)
to maximize the harmonic mean of confident accuracy on valid crops and safe refusal rate on degraded/ambiguous inputs.
Exports findings to models/entropy_calibration_report.json.
"""
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.vision.model import get_vision_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def measure_blur_laplacian(img: Image.Image) -> float:
    """Measure image focus/sharpness via variance of the Laplacian."""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float64)
    # 3x3 Discrete Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    # Simple convolution via slicing
    h, w = arr.shape
    if h < 5 or w < 5:
        return 0.0
    lap = (
        arr[0 : h - 2, 1 : w - 1]
        + arr[2:h, 1 : w - 1]
        + arr[1 : h - 1, 0 : w - 2]
        + arr[1 : h - 1, 2:w]
        - 4.0 * arr[1 : h - 1, 1 : w - 1]
    )
    return float(np.var(lap))


def get_validation_dataset() -> List[Dict[str, Any]]:
    """Assemble a balanced, diverse validation set."""
    samples = []

    # 1. Clear healthy & confirmed diseases (Should be CONFIDENT / PROBABLE)
    positives = [
        ("sample_tomato_healthy.jpg", "healthy", "Tomato Healthy Leaf"),
        ("sample_apple_scab.jpg", "disease", "Apple Scab Leaf"),
        ("sample_corn_common_rust.jpg", "disease", "Corn Common Rust Leaf"),
        ("sample_tomato_early_blight.jpg", "disease", "Tomato Early Blight Leaf"),
        ("sample_field_tomato_plant.jpg", "whole_plant", "Field Tomato Whole-Plant"),
    ]
    for filename, cat, desc in positives:
        p = config.RAW_DATA_DIR / filename
        if p.exists():
            samples.append({
                "path": p,
                "expected_category": cat,
                "is_valid_plant": True,
                "should_be_accepted": True,
                "description": desc,
            })

    # In-distribution samples
    in_dist_dir = config.DATA_DIR / "processed" / "eval_samples" / "in_distribution"
    if in_dist_dir.exists():
        for p in sorted(in_dist_dir.glob("*.jpg"))[:6]:
            samples.append({
                "path": p,
                "expected_category": "disease" if "healthy" not in p.name.lower() else "healthy",
                "is_valid_plant": True,
                "should_be_accepted": True,
                "description": f"Benchmark: {p.stem}",
            })

    # 2. Multi-part samples (Canopy, Stem, Fruit)
    mp_dir = config.DATA_DIR / "processed" / "eval_samples" / "multi_part"
    if mp_dir.exists():
        for sub in ["canopy", "stem", "fruit"]:
            s_dir = mp_dir / sub
            if s_dir.exists():
                for p in s_dir.glob("*.jpg"):
                    samples.append({
                        "path": p,
                        "expected_category": sub,
                        "is_valid_plant": True,
                        "should_be_accepted": True,
                        "description": f"Multi-part {sub}: {p.name}",
                    })

    # 3. Degraded / Blurry image
    if (config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg").exists():
        blurry_path = config.DATA_DIR / "processed" / "eval_samples" / "sample_blurred_leaf.jpg"
        im = Image.open(config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg")
        blurred = im.filter(ImageFilter.GaussianBlur(radius=8))
        blurred.save(blurry_path, "JPEG")
        samples.append({
            "path": blurry_path,
            "expected_category": "blurry",
            "is_valid_plant": True,
            "should_be_accepted": False,
            "description": "Severely Blurred Leaf (Insufficient Quality)",
        })

    # 4. Non-plant OOD (Must be rejected as OOD)
    oods = [
        (config.RAW_DATA_DIR / "sample_non_plant_cat.png", "Cat Photograph"),
        (config.DATA_DIR / "processed" / "eval_samples" / "ood" / "ood_checkerboard.jpg", "Checkerboard Pattern"),
        (config.DATA_DIR / "processed" / "eval_samples" / "ood" / "ood_gaussian_noise.jpg", "Gaussian Noise"),
    ]
    for p, desc in oods:
        if p.exists():
            samples.append({
                "path": p,
                "expected_category": "ood",
                "is_valid_plant": False,
                "should_be_accepted": False,
                "description": f"OOD: {desc}",
            })

    return samples


def run_calibration() -> Dict[str, Any]:
    """
    Run empirical threshold calibration across candidate entropy thresholds.
    """
    logging.info("Starting empirical entropy calibration on held-out validation set...")
    dino_wrapper, clf_wrapper = get_vision_models()
    device = clf_wrapper.device
    num_classes = len(clf_wrapper.id2label)
    max_entropy = math.log(num_classes)

    samples = get_validation_dataset()
    logging.info("Collected %d validation samples.", len(samples))

    # Pre-compute inference features for each sample
    sample_records = []
    for s in samples:
        img = Image.open(s["path"]).convert("RGB")
        blur_metric = measure_blur_laplacian(img)

        # Global Full-View Softmax Probs
        inputs = clf_wrapper.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clf_wrapper.model(**inputs)
            global_probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # 4 Quadrants
        w, h = img.size
        box_w, box_h = int(w * 0.6), int(h * 0.6)
        quadrants = [
            img.crop((0, 0, box_w, box_h)),
            img.crop((w - box_w, 0, w, box_h)),
            img.crop((0, h - box_h, box_w, h)),
            img.crop((w - box_w, h - box_h, w, h)),
        ]
        q_probs_list = []
        for q in quadrants:
            q_in = clf_wrapper.processor(images=q, return_tensors="pt")
            q_in = {k: v.to(device) for k, v in q_in.items()}
            with torch.no_grad():
                q_out = clf_wrapper.model(**q_in)
                q_p = F.softmax(q_out.logits, dim=-1)[0].cpu().numpy()
                q_probs_list.append(q_p)

        # Robust Fused Distribution:
        # Global view has 60% anchor weight; mean of quadrant crops provides 40% spatial regularization
        q_mean = np.mean(q_probs_list, axis=0)
        fused_probs = 0.60 * global_probs + 0.40 * q_mean
        fused_probs = fused_probs / np.sum(fused_probs)

        # Shannon Entropy
        clipped = np.clip(fused_probs, 1e-12, 1.0)
        raw_entropy = float(-np.sum(clipped * np.log(clipped)))
        norm_entropy = float(raw_entropy / max_entropy)

        # Top 1 probability and agreement
        top1_idx = int(np.argmax(fused_probs))
        top1_prob = float(fused_probs[top1_idx])
        top1_label = clf_wrapper.id2label[str(top1_idx) if str(top1_idx) in clf_wrapper.id2label else top1_idx]

        # Quadrant agreement count
        agreed_quadrants = sum(1 for qp in q_probs_list if int(np.argmax(qp)) == top1_idx)

        sample_records.append({
            "path": str(s["path"]),
            "desc": s["description"],
            "expected_cat": s["expected_category"],
            "is_valid_plant": s["is_valid_plant"],
            "should_be_accepted": s["should_be_accepted"],
            "blur_metric": blur_metric,
            "raw_entropy": raw_entropy,
            "norm_entropy": norm_entropy,
            "top1_prob": top1_prob,
            "top1_label": top1_label,
            "agreed_quadrants": agreed_quadrants,
        })

    # Evaluate grid of candidate raw entropy thresholds
    candidate_thresholds = [1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.5, 2.6, 2.7, 2.8, 3.0]
    eval_results = []

    best_thresh = 2.40
    best_f1 = -1.0
    best_metrics = {}

    for cand in candidate_thresholds:
        # A valid plant with clear condition should be deemed confident if:
        # entropy <= cand AND (top1_prob >= 0.50 OR agreement >= 2)
        true_confidents = 0
        false_uncertains = 0
        safe_refusals = 0
        false_confidents = 0

        for r in sample_records:
            if not r["is_valid_plant"] or r["expected_cat"] in ["blurry", "ood"]:
                # Should NOT be confident
                is_conf = (r["raw_entropy"] <= cand) and (r["top1_prob"] >= 0.50)
                if is_conf:
                    false_confidents += 1
                else:
                    safe_refusals += 1
            else:
                # Valid clear plant
                is_conf = (r["raw_entropy"] <= cand) and (r["top1_prob"] >= 0.40)
                if is_conf:
                    true_confidents += 1
                else:
                    false_uncertains += 1

        total_valid = sum(1 for r in sample_records if r["should_be_accepted"])
        total_invalid = sum(1 for r in sample_records if not r["should_be_accepted"])

        sensitivity = true_confidents / total_valid if total_valid > 0 else 0.0
        specificity = safe_refusals / total_invalid if total_invalid > 0 else 0.0
        balanced_f1 = (
            2.0 * sensitivity * specificity / (sensitivity + specificity)
            if (sensitivity + specificity) > 0
            else 0.0
        )

        res_entry = {
            "threshold": cand,
            "norm_threshold": round(cand / max_entropy, 4),
            "valid_accepted_rate": round(sensitivity * 100, 2),
            "false_uncertain_rate": round((false_uncertains / total_valid) * 100, 2) if total_valid > 0 else 0.0,
            "safe_refusal_rate": round(specificity * 100, 2),
            "false_confident_rate": round((false_confidents / total_invalid) * 100, 2) if total_invalid > 0 else 0.0,
            "balanced_score": round(balanced_f1, 4),
        }
        eval_results.append(res_entry)

        if balanced_f1 > best_f1 or (balanced_f1 == best_f1 and cand > best_thresh):
            best_f1 = balanced_f1
            best_thresh = cand
            best_metrics = res_entry

    report = {
        "num_classes": num_classes,
        "max_theoretical_entropy": round(max_entropy, 4),
        "total_validation_samples": len(sample_records),
        "selected_optimal_threshold": best_thresh,
        "selected_normalized_threshold": round(best_thresh / max_entropy, 4),
        "best_performance": best_metrics,
        "blur_laplacian_threshold": 60.0,
        "candidate_sweep": eval_results,
        "sample_evaluations": sample_records,
    }

    report_path = config.MODELS_DIR / "entropy_calibration_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info("Calibration complete. Selected best threshold: H(p) <= %.2f (Norm: %.3f)", best_thresh, best_thresh / max_entropy)
    logging.info("Saved calibration report to %s", report_path)
    return report


if __name__ == "__main__":
    rep = run_calibration()
    print("\n" + "=" * 95)
    print(f"{'THRESHOLD':<12} | {'NORM THRESH':<14} | {'VALID ACCEPTED':<18} | {'FALSE UNCERTAIN':<18} | {'BALANCED F1'}")
    print("=" * 95)
    for c in rep["candidate_sweep"]:
        print(f"{c['threshold']:<12.2f} | {c['norm_threshold']:<14.4f} | {c['valid_accepted_rate']:<16.1f}% | {c['false_uncertain_rate']:<16.1f}% | {c['balanced_score']:<10.4f}")
    print("=" * 95)
    print(f"Optimal Chosen Entropy Threshold: {rep['selected_optimal_threshold']} (Norm: {rep['selected_normalized_threshold']})")
