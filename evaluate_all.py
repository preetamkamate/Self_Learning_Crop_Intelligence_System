"""
Comprehensive System Evaluation Suite: evaluate_all.py
Computes real, verifiable evaluation metrics across every AI/ML module:
1. Tabular Crop Suitability ML (Test Accuracy, Macro Precision/Recall/F1, 5-Fold CV, Latency)
2. Whole-Plant Computer Vision Engine (Top-1/Top-3 Accuracy, Entropy, OOD Rejection Rate, Latency)
3. Agricultural RAG Knowledge Base (Hit@1, Hit@3, MRR, Retrieval Latency)
Exports full JSON report to models/evaluation_report.json.
"""
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from src.rag.retriever import get_retriever
from src.vision.predict import PlantDiseasePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# 1. TABULAR CROP RECOMMENDATION EVALUATION
# ==============================================================================
def evaluate_tabular_model() -> Dict[str, Any]:
    """
    Evaluates trained tabular ML model on the held-out 440 test samples
    and computes 5-fold cross-validation metrics.
    """
    logging.info("Evaluating Tabular Crop Recommendation ML Model...")
    df = pd.read_csv(config.CROP_DATA_PATH)
    X = df[config.TABULAR_FEATURES]
    y = df[config.TABULAR_TARGET]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Exact 80/20 stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.TABULAR_TEST_SIZE,
        random_state=config.TABULAR_RANDOM_STATE,
        stratify=y_encoded,
    )

    # Load serialized production model, scaler, and encoder
    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    X_test_scaled = scaler.transform(X_test)

    # Latency measurement
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = model.predict(X_test_scaled[:1])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency_ms = float(np.mean(latencies))

    # Test set predictions
    y_pred = model.predict(X_test_scaled)
    test_acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # 5-Fold Cross-Validation on entire dataset
    logging.info("Running 5-Fold Cross-Validation on %d rows...", len(df))
    X_scaled = scaler.transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.TABULAR_RANDOM_STATE)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    return {
        "dataset_name": "Indian Agriculture Crop Recommendation Dataset",
        "total_samples": len(df),
        "test_samples": len(X_test),
        "num_classes": len(encoder.classes_),
        "test_accuracy_pct": round(test_acc * 100.0, 2),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "cv_5fold_mean_pct": round(cv_mean * 100.0, 2),
        "cv_5fold_std_pct": round(cv_std * 100.0, 2),
        "avg_inference_latency_ms": round(avg_latency_ms, 2),
    }


# ==============================================================================
# 2. COMPUTER VISION & OOD DETECTION EVALUATION
# ==============================================================================
def evaluate_vision_model() -> Dict[str, Any]:
    """
    Evaluates Whole-Plant Vision Engine on genuine in-distribution plant disease
    samples and real out-of-distribution non-plant samples.
    """
    logging.info("Evaluating Computer Vision & OOD Engine...")
    predictor = PlantDiseasePredictor()

    eval_base = config.DATA_DIR / "processed" / "eval_samples"
    in_dist_dir = eval_base / "in_distribution"
    ood_dir = eval_base / "ood"
    labels_file = eval_base / "in_distribution_labels.json"

    # Evaluate in-distribution samples
    in_dist_results = []
    latencies = []

    if labels_file.exists():
        with open(labels_file, "r") as f:
            samples = json.load(f)
    else:
        samples = []

    top1_correct = 0
    top3_correct = 0
    confidences = []
    entropies = []

    for item in samples:
        img_path = in_dist_dir / item["file"]
        if not img_path.exists():
            continue

        gt_label = item["ground_truth"]

        t0 = time.perf_counter()
        res = predictor.predict(img_path)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        preds = [p["label"] for p in res.get("predictions", [])]
        top1_label = res.get("top1_label", "")

        is_top1 = (top1_label.lower() == gt_label.lower())
        is_top3 = any(p.lower() == gt_label.lower() for p in preds[:3])

        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1

        confidences.append(res.get("top1_confidence", 0.0))
        entropies.append(res.get("entropy", 0.0))

        in_dist_results.append({
            "file": item["file"],
            "ground_truth": gt_label,
            "predicted_top1": top1_label,
            "top1_match": is_top1,
            "top3_match": is_top3,
            "confidence": res.get("top1_confidence", 0.0),
            "entropy": res.get("entropy", 0.0),
            "status": res.get("status"),
        })

    num_in_dist = len(in_dist_results)
    top1_acc = (top1_correct / num_in_dist * 100.0) if num_in_dist > 0 else 0.0
    top3_acc = (top3_correct / num_in_dist * 100.0) if num_in_dist > 0 else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    # Evaluate OOD samples
    ood_files = list(ood_dir.glob("*.*"))
    ood_rejected = 0
    ood_details = []

    for fpath in ood_files:
        res = predictor.predict(fpath)
        is_rejected = (res.get("status") in ["out_of_distribution", "uncertain"])
        if is_rejected:
            ood_rejected += 1
        ood_details.append({
            "file": fpath.name,
            "status": res.get("status"),
            "dinov2_similarity": res.get("dinov2_plant_similarity", 0.0),
            "is_rejected": is_rejected,
        })

    num_ood = len(ood_files)
    ood_rejection_rate = (ood_rejected / num_ood * 100.0) if num_ood > 0 else 0.0

    return {
        "backbone_model": "facebook/dinov2-small",
        "pathology_classifier": config.VISION_MODEL_ID,
        "in_distribution_samples": num_in_dist,
        "top1_accuracy_pct": round(top1_acc, 2),
        "top3_accuracy_pct": round(top3_acc, 2),
        "mean_confidence_pct": round(mean_conf, 2),
        "mean_shannon_entropy": round(mean_entropy, 4),
        "ood_samples_evaluated": num_ood,
        "ood_rejection_rate_pct": round(ood_rejection_rate, 2),
        "avg_multiscale_latency_ms": round(avg_latency, 2),
        "sample_evaluations": in_dist_results[:5],
        "ood_evaluations": ood_details,
    }


# ==============================================================================
# 3. RAG AGRICULTURAL KNOWLEDGE BASE EVALUATION
# ==============================================================================
def evaluate_rag_engine() -> Dict[str, Any]:
    """
    Evaluates semantic vector retrieval across 10 realistic farmer benchmark queries
    covering Tomato, Paddy, Ragi, Maize, and Cotton.
    """
    logging.info("Evaluating Agricultural RAG Knowledge Base...")
    retriever = get_retriever()

    # 10 realistic benchmark agricultural queries with ground-truth expected source guidelines
    benchmark_queries = [
        {
            "query": "How to identify and control early blight in tomato?",
            "expected_source": "tomato_icar_iihr_guidelines.md",
            "topic": "Tomato Early Blight",
        },
        {
            "query": "What are the recommended chemical dosages for tomato late blight?",
            "expected_source": "tomato_icar_iihr_guidelines.md",
            "topic": "Tomato Late Blight",
        },
        {
            "query": "How to manage rice blast disease in paddy fields?",
            "expected_source": "paddy_rice_icar_karnataka_guidelines.md",
            "topic": "Paddy Rice Blast",
        },
        {
            "query": "What causes bacterial leaf blight in rice and how to control kresek stage?",
            "expected_source": "paddy_rice_icar_karnataka_guidelines.md",
            "topic": "Paddy Bacterial Leaf Blight",
        },
        {
            "query": "Which varieties of ragi are resistant to finger blast in Karnataka?",
            "expected_source": "ragi_finger_millet_uas_guidelines.md",
            "topic": "Ragi Finger Blast",
        },
        {
            "query": "How to treat foot rot and damping off in finger millet seedlings?",
            "expected_source": "ragi_finger_millet_uas_guidelines.md",
            "topic": "Ragi Foot Rot",
        },
        {
            "query": "What is the ICAR recommendation for Fall Armyworm management in maize?",
            "expected_source": "maize_icar_iihr_guidelines.md",
            "topic": "Maize Fall Armyworm",
        },
        {
            "query": "Symptoms and fungicide treatment for common rust in corn",
            "expected_source": "maize_icar_iihr_guidelines.md",
            "topic": "Maize Common Rust",
        },
        {
            "query": "How to control bacterial blight and black arm in cotton?",
            "expected_source": "cotton_icar_uas_guidelines.md",
            "topic": "Cotton Black Arm Blight",
        },
        {
            "query": "Integrated management of cotton sucking pest complex whitefly and thrips",
            "expected_source": "cotton_icar_uas_guidelines.md",
            "topic": "Cotton Sucking Pests",
        },
    ]

    hit1_count = 0
    hit3_count = 0
    reciprocal_ranks = []
    latencies = []
    query_details = []

    for bq in benchmark_queries:
        q = bq["query"]
        expected = bq["expected_source"]

        t0 = time.perf_counter()
        matches = retriever.retrieve(q, top_k=3, min_score=0.1)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        retrieved_sources = [m["source"] for m in matches]

        hit1 = (len(retrieved_sources) > 0 and retrieved_sources[0] == expected)
        hit3 = (expected in retrieved_sources[:3])

        if hit1:
            hit1_count += 1
        if hit3:
            hit3_count += 1

        # Calculate Reciprocal Rank
        rr = 0.0
        for rank, src in enumerate(retrieved_sources[:3], start=1):
            if src == expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        top_score = matches[0]["score"] if matches else 0.0

        query_details.append({
            "query": q,
            "topic": bq["topic"],
            "expected_source": expected,
            "top1_source": retrieved_sources[0] if retrieved_sources else None,
            "top1_score": round(float(top_score), 4),
            "hit1": hit1,
            "hit3": hit3,
            "reciprocal_rank": round(rr, 3),
        })

    num_queries = len(benchmark_queries)
    hit1_acc = (hit1_count / num_queries) * 100.0
    hit3_acc = (hit3_count / num_queries) * 100.0
    mrr = float(np.mean(reciprocal_ranks))
    avg_latency = float(np.mean(latencies))

    return {
        "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
        "embedding_model": config.EMBEDDING_MODEL_ID,
        "total_queries_evaluated": num_queries,
        "hit1_accuracy_pct": round(hit1_acc, 2),
        "hit3_recall_pct": round(hit3_acc, 2),
        "mean_reciprocal_rank_mrr": round(mrr, 4),
        "avg_retrieval_latency_ms": round(avg_latency, 2),
        "query_evaluations": query_details,
    }


# ==============================================================================
# MAIN EXECUTION & SUMMARY TABLE
# ==============================================================================
def main():
    print("=" * 80)
    print("🌾 SELF-LEARNING CROP INTELLIGENCE SYSTEM: SYSTEM-WIDE EVALUATION")
    print("=" * 80)
    print("Computing real evaluation metrics across Tabular, Vision, and RAG modules...")
    print("Zero synthetic or hardcoded metrics. Real datasets & models only.\n")

    # Run evaluations
    tabular_metrics = evaluate_tabular_model()
    vision_metrics = evaluate_vision_model()
    rag_metrics = evaluate_rag_engine()

    # Compile comprehensive report
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": {
            "device": str(config.DEVICE),
            "platform": sys.platform,
        },
        "tabular_crop_recommendation": tabular_metrics,
        "computer_vision_and_ood": vision_metrics,
        "rag_knowledge_retrieval": rag_metrics,
    }

    # Save to models/evaluation_report.json
    out_path = config.MODELS_DIR / "evaluation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Full Evaluation Report exported to: {out_path}\n")

    # Formatted Summary Table
    print("=" * 80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY TABLE (NHCE MINI-PROJECT)")
    print("=" * 80)
    print(f"{'Subsystem':<26} | {'Key Evaluation Metric':<28} | {'Computed Value':<18}")
    print("-" * 80)
    # Tabular
    print(f"{'1. Tabular Crop ML':<26} | {'Test Accuracy (440 samples)':<28} | {tabular_metrics['test_accuracy_pct']}%")
    print(f"{'':<26} | {'Macro F1-Score':<28} | {tabular_metrics['macro_f1']}")
    print(f"{'':<26} | {'5-Fold Cross-Val Mean ± Std':<28} | {tabular_metrics['cv_5fold_mean_pct']}% ± {tabular_metrics['cv_5fold_std_pct']}%")
    print(f"{'':<26} | {'Inference Latency':<28} | {tabular_metrics['avg_inference_latency_ms']} ms/sample")
    print("-" * 80)
    # Vision
    print(f"{'2. Computer Vision Engine':<26} | {'Top-1 Accuracy':<28} | {vision_metrics['top1_accuracy_pct']}%")
    print(f"{'':<26} | {'Top-3 Accuracy':<28} | {vision_metrics['top3_accuracy_pct']}%")
    print(f"{'':<26} | {'Mean Confidence':<28} | {vision_metrics['mean_confidence_pct']}%")
    print(f"{'':<26} | {'Mean Shannon Entropy':<28} | {vision_metrics['mean_shannon_entropy']}")
    print(f"{'':<26} | {'OOD Rejection Rate':<28} | {vision_metrics['ood_rejection_rate_pct']}%")
    print(f"{'':<26} | {'Multi-Scale Tiling Latency':<28} | {vision_metrics['avg_multiscale_latency_ms']} ms/image")
    print("-" * 80)
    # RAG
    print(f"{'3. RAG Knowledge Base':<26} | {'Hit@1 Accuracy':<28} | {rag_metrics['hit1_accuracy_pct']}%")
    print(f"{'':<26} | {'Hit@3 Recall':<28} | {rag_metrics['hit3_recall_pct']}%")
    print(f"{'':<26} | {'Mean Reciprocal Rank (MRR)':<28} | {rag_metrics['mean_reciprocal_rank_mrr']}")
    print(f"{'':<26} | {'Retrieval Latency':<28} | {rag_metrics['avg_retrieval_latency_ms']} ms/query")
    print("=" * 80)


if __name__ == "__main__":
    main()

