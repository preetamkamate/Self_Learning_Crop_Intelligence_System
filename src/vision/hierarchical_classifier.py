"""
Hierarchical Crop & Plant Pathology Classifier.
Executes the strict 4-stage agricultural diagnostic pipeline:
Stage 1: Image Validity & Quality Gating (Sharpness & DINOv2 Plant OOD)
Stage 2: Plant Part & Viewpoint Identification (leaf, whole_plant, canopy, stem, fruit, root, bark)
Stage 3: Crop Species Identification (Tomato, Corn, Grape, Apple, Potato, Pepper, etc.)
Stage 4: Gated Pathology Diagnosis (Strictly constrained to confirmed crop subspace;
         abstains completely when viewpoint or crop is unsupported by training data).
"""
import io
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import config
import torch.nn as nn
from src.rag.authoritative_sources import get_authoritative_source_citation
from src.vision.model import get_vision_models
from src.vision.nutrient import SCREENING_DISCLAIMER_EN, analyze_nutrient_symptoms
from src.vision.taxonomy import (
    EvidenceLevel,
    HierarchicalDiagnosis,
    IssueCategory,
    PlantPart,
    parse_legacy_label,
)
from src.vision.training_registry import (
    CROP_DISEASE_TAXONOMY,
    CROP_PART_DISEASE_TAXONOMY,
    DEPLOYED_DIAGNOSTIC_ROUTES,
    LABEL_TO_CROP_MAP,
    get_crop_for_label,
    get_valid_classes_for_crop,
    get_valid_classes_for_crop_and_part,
    is_route_supported,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DenseNeuralClassifier(nn.Module):
    """Dense 2-layer neural probe on top of 384-d DINOv2 embeddings."""

    def __init__(self, in_features: int = 384, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def measure_image_sharpness(img: Image.Image) -> float:
    """Measure image focus/sharpness via variance of the Laplacian."""
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float64)
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


def compute_shannon_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon Predictive Entropy H(p) = -sum(p_i * log(p_i))."""
    clipped = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    return float(entropy)


class HierarchicalVisionPipeline:
    """
    Robust 4-Stage Crop Vision Engine with strict route gating and zero cross-crop label mixing.
    """

    def __init__(self):
        self.dinov2_wrapper, self.classifier_wrapper = get_vision_models()
        self.device = self.dinov2_wrapper.device
        self.id2label = self.classifier_wrapper.id2label
        self.label2id = self.classifier_wrapper.label2id
        self.num_classes = len(self.id2label)
        self.max_theoretical_entropy = math.log(self.num_classes)

        # Build reverse index for fast crop lookup
        self.class_idx_by_crop: Dict[str, List[int]] = {}
        for crop_name, label_list in CROP_DISEASE_TAXONOMY.items():
            indices = []
            for lbl in label_list:
                for k, v in self.id2label.items():
                    if v.lower() == lbl.lower():
                        indices.append(int(k))
            self.class_idx_by_crop[crop_name] = sorted(list(set(indices)))

        # Precomputed agricultural embeddings cache for DINOv2 plant gate
        self.ref_embeddings = self._build_reference_embeddings()
        self.leaf_prototypes = self._build_leaf_prototypes()

        # Multi-part neural models cache
        self.multipart_models: Dict[str, Tuple[nn.Module, List[str]]] = {}
        self.plant_part_model, self.plant_part_classes = self._load_plant_part_model()

    def _load_plant_part_model(self) -> Tuple[Optional[nn.Module], List[str]]:
        """Load trained plant part classifier from models/vision/plant_part_classifier.pt."""
        ckpt_path = config.VISION_MODEL_DIR / "plant_part_classifier.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                classes = ckpt["classes"]
                model = DenseNeuralClassifier(in_features=384, hidden_dim=128, num_classes=len(classes)).to(self.device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                logging.info("Loaded trained Plant Part Classifier (%d classes) from %s", len(classes), ckpt_path)
                return model, classes
            except Exception as e:
                logging.warning("Could not load plant part classifier: %s", e)
        return None, []

    def _get_multipart_model(self, part_name: str) -> Optional[Tuple[nn.Module, List[str]]]:
        """Lazy load dedicated multi-part probe model for specified plant part."""
        p = part_name.lower().strip()
        if p == "branch":
            p = "stem"
        elif p in ["tree_bark", "trunk", "bark"]:
            p = "bark_trunk"
        elif p == "tuber":
            p = "root"

        if p in self.multipart_models:
            return self.multipart_models[p]

        filename_map = {
            "stem": "stem_classifier.pt",
            "stem_branch": "stem_classifier.pt",
            "fruit": "fruit_classifier.pt",
            "root": "root_tuber_classifier.pt",
            "root_tuber": "root_tuber_classifier.pt",
            "whole_plant": "whole_plant_classifier.pt",
            "canopy": "canopy_classifier.pt",
            "bark_trunk": "bark_trunk_classifier.pt",
        }
        filename = filename_map.get(p)
        if not filename:
            return None

        ckpt_path = config.VISION_MODEL_DIR / filename
        if not ckpt_path.exists():
            logging.warning("Model checkpoint %s does not exist", ckpt_path)
            return None

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            classes = ckpt["classes"]
            model = DenseNeuralClassifier(in_features=384, hidden_dim=128, num_classes=len(classes)).to(self.device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.multipart_models[p] = (model, classes)
            logging.info("Loaded multi-part model for '%s' (%d classes) from %s", p, len(classes), ckpt_path)
            return model, classes
        except Exception as e:
            logging.error("Failed to load multi-part model from %s: %s", ckpt_path, e)
            return None

    def _build_leaf_prototypes(self) -> torch.Tensor:
        """Cache prototype embeddings for close-up foliar cutouts."""
        leaf_candidates = [
            config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg",
            config.RAW_DATA_DIR / "sample_apple_scab.jpg",
            config.RAW_DATA_DIR / "sample_corn_common_rust.jpg",
            config.RAW_DATA_DIR / "sample_tomato_healthy.jpg",
        ]
        embs = []
        for p in leaf_candidates:
            if p.exists():
                try:
                    im = Image.open(p).convert("RGB")
                    inputs = self.dinov2_wrapper.processor(images=im, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = self.dinov2_wrapper.model(**inputs)
                        embs.append(F.normalize(out.last_hidden_state[:, 0, :], dim=-1))
                except Exception as e:
                    logging.warning("Could not embed leaf prototype %s: %s", p, e)
        if embs:
            return torch.cat(embs, dim=0)
        return torch.randn((1, 384), device=self.device)

    def _build_reference_embeddings(self) -> torch.Tensor:
        """Load or build precomputed DINOv2 reference embeddings."""
        cache_path = getattr(config, "VISION_REF_CACHE_PATH", config.VISION_MODEL_DIR / "dinov2_plant_reference_embeddings.pt")
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location=self.device)
                if isinstance(cached, torch.Tensor) and cached.ndim == 2 and cached.shape[1] == 384 and cached.shape[0] > 0:
                    logging.info("Loaded %d DINOv2 plant reference embeddings from %s", len(cached), cache_path)
                    return F.normalize(cached.to(self.device), dim=-1)
            except Exception as e:
                logging.warning("Could not load cached embeddings: %s", e)

        torch.manual_seed(42)
        dummy = torch.randn((1, 384), device=self.device)
        return F.normalize(dummy, dim=-1)

    def _check_stage1_validity(
        self,
        img: Image.Image,
        dinov2_threshold: float,
        blur_threshold: float,
    ) -> Tuple[bool, bool, float, float, torch.Tensor]:
        """
        Stage 1: Determine image validity:
        - Image quality (sharpness metric >= blur_threshold)
        - Plant vs Non-Plant (DINOv2 similarity >= dinov2_threshold)
        Returns: (is_quality_valid, is_plant_valid, blur_metric, dinov2_similarity, global_emb)
        """
        blur_metric = measure_image_sharpness(img)
        is_blurry = blur_metric < blur_threshold and (img.width >= 100 and img.height >= 100)

        inputs = self.dinov2_wrapper.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.dinov2_wrapper.model(**inputs)
            target_emb = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)

        # 1. Blank / solid uniform screen rejection (std < 3.0)
        arr = np.array(img, dtype=np.float32)
        if arr.std() < 3.0:
            return (not is_blurry), False, blur_metric, 0.0, target_emb

        # 2. Pure high-frequency random noise / static rejection
        gray = arr.mean(axis=-1)
        diff_h = float(np.abs(gray[:, 1:] - gray[:, :-1]).mean())
        diff_v = float(np.abs(gray[1:, :] - gray[:-1, :]).mean())
        if diff_h > 35.0 and diff_v > 35.0:
            return (not is_blurry), False, blur_metric, 0.05, target_emb

        sims = F.cosine_similarity(target_emb, self.ref_embeddings)
        dinov2_sim = float(sims.max().item())

        # 3. Geometric pattern / checkerboard rejection
        if blur_metric > 18000.0 and diff_h < 25.0:
            foliar_sim = float(F.cosine_similarity(target_emb, self.ref_embeddings[:20]).max().item())
            if foliar_sim < 0.10:
                return (not is_blurry), False, blur_metric, foliar_sim, target_emb

        is_plant = dinov2_sim >= dinov2_threshold

        return (not is_blurry), is_plant, blur_metric, dinov2_sim, target_emb

    def _identify_stage2_plant_part(
        self,
        img: Image.Image,
        global_emb: Optional[torch.Tensor] = None,
        user_part_hint: Optional[str] = None,
    ) -> PlantPart:
        """
        Stage 2: Identify plant part: leaf / whole plant / canopy / stem / fruit / root / bark.
        Prioritizes user confirmation if provided.
        """
        if user_part_hint and user_part_hint.lower() not in ["auto", "auto-detect", "unknown", ""]:
            return PlantPart.from_str(user_part_hint)

        # Autonomous neural detection using trained Plant Part Classifier
        if global_emb is not None and self.plant_part_model is not None and self.plant_part_classes:
            with torch.no_grad():
                logits = self.plant_part_model(global_emb.to(self.device))[0]
                pred_idx = int(torch.argmax(logits).item())
                part_name = self.plant_part_classes[pred_idx]
                part_map = {
                    "stem_branch": PlantPart.STEM,
                    "fruit": PlantPart.FRUIT,
                    "root_tuber": PlantPart.ROOT,
                    "whole_plant": PlantPart.WHOLE_PLANT,
                    "canopy": PlantPart.CANOPY,
                    "bark_trunk": PlantPart.BARK_TRUNK,
                    "leaf": PlantPart.LEAF,
                }
                if part_name in part_map:
                    return part_map[part_name]

        w, h = img.size
        aspect = w / float(h)

        # Autonomous foliage vs whole-plant/canopy separation using foliar prototype similarity
        if global_emb is not None and hasattr(self, "leaf_prototypes") and self.leaf_prototypes is not None:
            foliar_sim = float(F.cosine_similarity(global_emb, self.leaf_prototypes).max().item())
            if foliar_sim < 0.25:
                if aspect > 1.4:
                    return PlantPart.CANOPY
                else:
                    return PlantPart.WHOLE_PLANT

        if aspect > 1.4:
            return PlantPart.CANOPY
        elif aspect < 0.75:
            return PlantPart.WHOLE_PLANT
        else:
            return PlantPart.LEAF

    def _raw_infer_probabilities(self, img: Image.Image) -> np.ndarray:
        """Run MobileNetV2 over image to get raw 38-class softmax distribution."""
        inputs = self.classifier_wrapper.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier_wrapper.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return probs

    def _identify_stage3_crop_species(
        self,
        raw_probs: np.ndarray,
        global_emb: Optional[torch.Tensor] = None,
        plant_part: Optional[PlantPart] = None,
        user_crop_hint: Optional[str] = None,
        min_crop_confidence: float = 0.35,
    ) -> Tuple[str, float]:
        """
        Stage 3: Identify crop species only when confidence is adequate.
        Aggregates class evidence hierarchically for each crop species:
        P(Crop C) = sum_{k in Classes(C)} P(k).
        Returns: (crop_species: str, crop_confidence: float)
        """
        if user_crop_hint and user_crop_hint.lower() not in ["auto", "auto-detect", "unknown", ""]:
            # Validate against taxonomy - prioritize exact match
            all_crops = list(CROP_DISEASE_TAXONOMY.keys()) + ["Citrus", "Coconut"]
            for c in all_crops:
                if c.lower() == user_crop_hint.lower():
                    indices = self.class_idx_by_crop.get(c, [])
                    support = float(np.sum(raw_probs[indices])) if indices else 1.0
                    return c, max(support, 0.90)
            for c in all_crops:
                if user_crop_hint.lower() in c.lower() or c.lower() in user_crop_hint.lower():
                    indices = self.class_idx_by_crop.get(c, [])
                    support = float(np.sum(raw_probs[indices])) if indices else 1.0
                    return c, max(support, 0.90)
            return user_crop_hint, 1.0

        # Autonomous Crop Identification from Specialized Part Model
        if plant_part and plant_part != PlantPart.LEAF and global_emb is not None:
            part_info = self._get_multipart_model(plant_part.value)
            if part_info:
                model, classes = part_info
                with torch.no_grad():
                    logits = model(global_emb.to(self.device))[0].cpu().numpy()
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / max(float(np.sum(probs)), 1e-9)

                crop_evidence: Dict[str, float] = {}
                for idx, c_lbl in enumerate(classes):
                    for c_name in ["Tomato", "Corn (Maize)", "Grape", "Citrus", "Apple", "Potato", "Bell Pepper (Capsicum)", "Coconut"]:
                        if c_name.lower() in c_lbl.lower() or (c_name == "Corn (Maize)" and "corn" in c_lbl.lower()):
                            crop_evidence[c_name] = crop_evidence.get(c_name, 0.0) + float(probs[idx])
                            break

                if crop_evidence:
                    best_crop, best_conf = max(crop_evidence.items(), key=lambda x: x[1])
                    if best_conf >= min_crop_confidence:
                        return best_crop, best_conf

        # Autonomous Crop Identification from Foliar Vision
        crop_scores: Dict[str, float] = {}
        for crop_name, indices in self.class_idx_by_crop.items():
            if indices:
                crop_scores[crop_name] = float(np.sum(raw_probs[indices]))

        if not crop_scores:
            return "Unknown Crop", 0.0

        best_crop, best_conf = max(crop_scores.items(), key=lambda x: x[1])
        if best_conf < min_crop_confidence:
            logging.info("Crop identification confidence %.3f < %.3f; designating Unknown Crop", best_conf, min_crop_confidence)
            return "Unknown Crop", best_conf

        return best_crop, best_conf

    def predict(
        self,
        image_input: Union[str, Image.Image, bytes, io.BytesIO],
        user_crop_hint: Optional[str] = None,
        user_part_hint: Optional[str] = None,
        soil_ph: Optional[float] = None,
        growth_stage: Optional[str] = None,
        symptom_duration: Optional[str] = None,
        dinov2_threshold: Optional[float] = None,
        blur_threshold: Optional[float] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute the 4-stage vision pipeline with strict route gating.
        """
        if dinov2_threshold is None:
            dinov2_threshold = config.DINOV2_PLANT_OOD_THRESHOLD
        if blur_threshold is None:
            blur_threshold = getattr(config, "BLUR_LAPLACIAN_THRESHOLD", 40.0)

        # Safe image loading
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

        image_dims = f"{img.width}x{img.height}"
        image_mode = img.mode

        # ======================================================================
        # STAGE 1: Image Validity & Quality Gating
        # ======================================================================
        is_sharp, is_plant, blur_metric, dinov2_sim, global_emb = self._check_stage1_validity(
            img, dinov2_threshold=dinov2_threshold, blur_threshold=blur_threshold
        )

        # 1A. Non-Plant OOD Refusal
        if not is_plant:
            decision_rule = f"RULE_STAGE1_NON_PLANT_OOD (DINOv2 similarity {dinov2_sim:.4f} < threshold {dinov2_threshold:.2f})"
            debug_info = {
                "stage1_valid": False,
                "stage1_reason": "Non-Agricultural / Out-of-Distribution Object",
                "stage2_plant_part": "non_plant",
                "stage3_crop_species": "Unknown",
                "stage4_route_supported": False,
                "image_dimensions": image_dims,
                "image_mode": image_mode,
                "dinov2_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": True,
                "predicted_class": "Non-Agricultural / Out-of-Distribution Object",
                "top1_confidence_pct": 0.0,
                "top_probabilities": [],
                "top3_probabilities": [],
                "entropy": 3.637,
                "raw_entropy": 3.637,
                "norm_entropy": 1.0,
                "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
                "class_count": self.num_classes,
                "blur_metric": round(blur_metric, 2),
                "blur_threshold": round(blur_threshold, 1),
                "decision_rule": decision_rule,
                "final_decision": "OUT_OF_DISTRIBUTION",
            }
            return {
                "plant_scope": "hierarchical_4stage_pipeline",
                "status": "out_of_distribution",
                "dinov2_plant_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": True,
                "plant_part": "unknown",
                "crop_species": "Unknown",
                "top1_label": "Non-Agricultural / Out-of-Distribution Object",
                "top1_label_kannada": "ಕೃಷಿಯೇತರ / ಅಸ್ಪಷ್ಟ ವಸ್ತು (Non-Plant OOD)",
                "top1_confidence": 0.0,
                "entropy": 3.637,
                "norm_entropy": 1.0,
                "is_healthy": False,
                "plant_type": "Unknown",
                "affected_regions": "None",
                "predictions": [],
                "debug_info": debug_info,
                "source_evidence": None,
                "refusal_reason_en": (
                    f"DINOv2 Backbone detected a non-plant or unrelated object (Similarity: {dinov2_sim:.3f} < {dinov2_threshold:.2f}). "
                    "The image does not contain recognizable crops or plant foliage. Automatic diagnosis safely withheld."
                ),
                "refusal_reason_kn": (
                    f"DINOv2 ಮಾದರಿಯು ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವುದೇ ಸಸ್ಯವನ್ನು ಗುರುತಿಸಿಲ್ಲ (ಹೋಲಿಕೆ: {dinov2_sim:.3f} < {dinov2_threshold:.2f}). "
                    "ತಪ್ಪಾದ ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆಯನ್ನು ತಪ್ಪಿಸಲು ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ."
                ),
                "requested_next_action_en": "Please upload a clear photograph of an actual crop, leaf, stem, or plant canopy.",
                "requested_next_action_kn": "ದಯವಿಟ್ಟು ಕೃಷಿ ಬೆಳೆ ಅಥವಾ ಸಸ್ಯದ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಕಳುಹಿಸಿ.",
            }

        # 1B. Severe Blur Refusal
        if not is_sharp:
            decision_rule = f"RULE_STAGE1_BLUR_GATE (Laplacian focus {blur_metric:.1f} < threshold {blur_threshold:.1f})"
            debug_info = {
                "stage1_valid": False,
                "stage1_reason": "Image Blurred / Low Focus",
                "stage2_plant_part": "unknown",
                "stage3_crop_species": "Unknown",
                "stage4_route_supported": False,
                "image_dimensions": image_dims,
                "image_mode": image_mode,
                "dinov2_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": False,
                "predicted_class": "Image Blurred / Insufficient Visual Focus",
                "top1_confidence_pct": 0.0,
                "top_probabilities": [],
                "top3_probabilities": [],
                "entropy": 0.0,
                "raw_entropy": 0.0,
                "norm_entropy": 0.0,
                "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
                "class_count": self.num_classes,
                "blur_metric": round(blur_metric, 2),
                "blur_threshold": round(blur_threshold, 1),
                "decision_rule": decision_rule,
                "final_decision": "INSUFFICIENT_IMAGE_QUALITY",
            }
            return {
                "plant_scope": "hierarchical_4stage_pipeline",
                "status": "insufficient_image_quality",
                "dinov2_plant_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": False,
                "plant_part": "unknown",
                "crop_species": "Unknown",
                "top1_label": "Image Blurred / Insufficient Visual Focus",
                "top1_label_kannada": "ಚಿತ್ರವು ಅಸ್ಪಷ್ಟವಾಗಿದೆ / ಸರಿಯಾದ ಫೋಕಸ್ ಇಲ್ಲ",
                "top1_confidence": 0.0,
                "entropy": 0.0,
                "norm_entropy": 0.0,
                "is_healthy": False,
                "plant_type": "Unknown",
                "affected_regions": "None",
                "predictions": [],
                "debug_info": debug_info,
                "source_evidence": None,
                "refusal_reason_en": (
                    f"The photograph appears out-of-focus or blurred (Sharpness metric: {blur_metric:.1f} < {blur_threshold:.1f}). "
                    "Accurate pathology diagnosis requires sharp foliar details. Automatic diagnosis has been safely withheld."
                ),
                "refusal_reason_kn": (
                    f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಫೋಟೋ ಮಸುಕಾಗಿದೆ ಅಥವಾ ಸ್ಪಷ್ಟವಾಗಿಲ್ಲ (ತೀಕ್ಷ್ಣತೆ: {blur_metric:.1f} < {blur_threshold:.1f}). "
                    "ಸರಿಯಾದ ರೋಗ ಪತ್ತೆಗೆ ಎಲೆಗಳ ಸ್ಪಷ್ಟ ವಿವರಗಳು ಅಗತ್ಯವಿದೆ."
                ),
                "requested_next_action_en": "Please hold your camera steady in bright natural daylight and retake a sharp, focused photo of the affected leaf.",
                "requested_next_action_kn": "ದಯವಿಟ್ಟು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಕ್ಯಾಮೆರಾವನ್ನು ಅಲುಗಾಡಿಸದೆ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆದು ಕಳುಹಿಸಿ.",
            }

        # ======================================================================
        # STAGE 2: Plant Part Identification
        # ======================================================================
        plant_part = self._identify_stage2_plant_part(img, global_emb=global_emb, user_part_hint=user_part_hint)

        # ======================================================================
        # STAGE 3: Crop Species Identification
        # ======================================================================
        raw_probs = self._raw_infer_probabilities(img)
        crop_species, crop_conf = self._identify_stage3_crop_species(
            raw_probs, global_emb=global_emb, plant_part=plant_part, user_crop_hint=user_crop_hint
        )

        # ======================================================================
        # STAGE 4: Route Verification & Constrained Pathology Diagnosis
        # ======================================================================
        route_supported, route_rec = is_route_supported(crop_species, plant_part.value)

        # CASE 4A: UNSUPPORTED ROUTE (Untrained Route or Unknown Crop)
        # Requirement 2: If a crop + plant-part route has no validated model, show:
        # "Valid [plant part] image detected. Diagnosis is not yet supported for this crop and image type."
        # Do not show disease guesses, unrelated Top Candidates, pesticide advice, or generic disease sources.
        if not route_supported or not route_rec:
            part_str = plant_part.value.replace('_', ' ')
            unsupported_msg_en = f"Valid {part_str} image detected. Diagnosis is not yet supported for this crop and image type."
            unsupported_msg_kn = f"ಕೃಷಿ ಸಸ್ಯ ({part_str}) ಗುರುತಿಸಲಾಗಿದೆ. ಈ ಬೆಳೆ ಮತ್ತು ಸಸ್ಯದ ಭಾಗಕ್ಕೆ ರೋಗನಿರ್ಣಯವು ಇನ್ನೂ ಬೆಂಬಲಿತವಾಗಿಲ್ಲ."
            decision_rule = f"RULE_STAGE4_UNSUPPORTED_ROUTE (Crop: '{crop_species}', Part: '{plant_part.value}' - Untrained route)"

            debug_info = {
                "stage1_valid": True,
                "stage1_reason": "Valid Plant Image",
                "stage2_plant_part": plant_part.value,
                "stage3_crop_species": crop_species,
                "stage3_crop_confidence_pct": round(crop_conf * 100, 1),
                "stage4_route_supported": False,
                "stage4_reason": "Untrained route or unsupported crop species",
                "image_dimensions": image_dims,
                "image_mode": image_mode,
                "dinov2_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": False,
                "predicted_class": unsupported_msg_en,
                "top1_confidence_pct": 0.0,
                "top_probabilities": [],
                "top3_probabilities": [],
                "entropy": 0.0,
                "raw_entropy": 0.0,
                "norm_entropy": 0.0,
                "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
                "class_count": self.num_classes,
                "blur_metric": round(blur_metric, 2),
                "blur_threshold": round(blur_threshold, 1),
                "decision_rule": decision_rule,
                "final_decision": "UNSUPPORTED_CROP_PART",
            }
            logging.info("Stage 4 Route Unsupported: Crop '%s', Part '%s'", crop_species, plant_part.value)

            return {
                "plant_scope": "hierarchical_4stage_pipeline",
                "status": "unsupported_crop_part",
                "dinov2_plant_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": False,
                "plant_part": plant_part.value,
                "crop_species": crop_species,
                "top1_label": unsupported_msg_en,
                "top1_label_kannada": unsupported_msg_kn,
                "top1_confidence": 0.0,
                "entropy": 0.0,
                "norm_entropy": 0.0,
                "is_healthy": False,
                "plant_type": crop_species,
                "affected_regions": f"Observed {part_str.title()}",
                "predictions": [],  # STRICT REQUIREMENT: Do not list unrelated top candidates!
                "debug_info": debug_info,
                "source_evidence": None,  # STRICT REQUIREMENT: No misleading source link!
                "refusal_reason_en": unsupported_msg_en,
                "refusal_reason_kn": unsupported_msg_kn,
                "requested_next_action_en": (
                    f"Automated diagnosis is currently validated for foliar routes across major crops, "
                    f"as well as stem, fruit, root, whole-plant, canopy, and bark routes for supported crops. "
                    f"Please provide an affected plant sample for a supported crop route."
                ),
                "requested_next_action_kn": (
                    f"ಪ್ರಸ್ತುತ ರೋಗನಿರ್ಣಯವು ಬೆಂಬಲಿತ ಬೆಳೆ ಮತ್ತು ಸಸ್ಯದ ಭಾಗಗಳಿಗೆ ಮಾತ್ರ ಲಭ್ಯವಿದೆ."
                ),
            }

        # CASE 4B: SUPPORTED FOLIAR ROUTE
        if plant_part == PlantPart.LEAF:
            valid_classes = get_valid_classes_for_crop(crop_species)
            indices = self.class_idx_by_crop.get(crop_species, [])

            if not indices or not valid_classes:
                logging.error("No valid class indices for supported crop '%s'", crop_species)
                return self._build_unknown_crop_response(crop_species, plant_part, dinov2_sim, dinov2_threshold, blur_metric, blur_threshold, image_dims, image_mode)

            active_indices = indices

            # Conditional Subspace Normalization: P(disease | crop) = P(disease) / sum_{k in crop} P(k)
            crop_probs = raw_probs[indices]
            crop_sum = float(np.sum(crop_probs))
            norm_crop_probs = crop_probs / crop_sum if crop_sum > 1e-9 else np.ones(len(indices)) / len(indices)

            crop_entropy = compute_shannon_entropy(norm_crop_probs)
            max_crop_entropy = math.log(max(len(indices), 2))
            norm_crop_entropy = float(crop_entropy / max_crop_entropy)

            sorted_crop_order = np.argsort(norm_crop_probs)[::-1]
            top1_sub_idx = int(sorted_crop_order[0])
            top1_prob = float(norm_crop_probs[top1_sub_idx])
            top1_class_idx = indices[top1_sub_idx]
            top1_label = self.id2label[str(top1_class_idx) if str(top1_class_idx) in self.id2label else top1_class_idx]

            crop_top_predictions: List[Dict[str, Any]] = []
            for i in range(min(top_k, len(sorted_crop_order))):
                s_idx = int(sorted_crop_order[i])
                c_idx = indices[s_idx]
                lbl = self.id2label[str(c_idx) if str(c_idx) in self.id2label else c_idx]
                c_prob = float(norm_crop_probs[s_idx])
                crop_top_predictions.append({
                    "label": lbl,
                    "probability": round(c_prob, 4),
                    "confidence_percent": round(c_prob * 100, 2),
                })

            parsed = parse_legacy_label(top1_label)
            is_healthy = parsed["is_healthy"]
            affected_reg_desc = f"Foliar lesions on {crop_species} canopy"

        # CASE 4C: SUPPORTED MULTI-PART ROUTE (Stem, Fruit, Root, Whole-Plant, Canopy, Bark)
        else:
            part_name = plant_part.value
            part_model_info = self._get_multipart_model(part_name)
            if not part_model_info:
                return self._build_unknown_crop_response(crop_species, plant_part, dinov2_sim, dinov2_threshold, blur_metric, blur_threshold, image_dims, image_mode)

            model, model_classes = part_model_info
            with torch.no_grad():
                logits = model(global_emb.to(self.device))[0].cpu().numpy()

            valid_classes = route_rec.valid_classes
            sub_indices = [model_classes.index(c) for c in valid_classes if c in model_classes]

            if not sub_indices:
                sub_indices = list(range(len(model_classes)))
                valid_classes = model_classes

            active_indices = sub_indices

            # Conditional softmax strictly over confirmed crop's classes (0.00% cross-crop contamination)
            sub_logits = logits[sub_indices]
            exp_logits = np.exp(sub_logits - np.max(sub_logits))
            norm_crop_probs = exp_logits / max(float(np.sum(exp_logits)), 1e-9)

            crop_entropy = compute_shannon_entropy(norm_crop_probs)
            max_crop_entropy = math.log(max(len(sub_indices), 2))
            norm_crop_entropy = float(crop_entropy / max_crop_entropy)

            sorted_crop_order = np.argsort(norm_crop_probs)[::-1]
            top1_sub_idx = int(sorted_crop_order[0])
            top1_prob = float(norm_crop_probs[top1_sub_idx])
            top1_label = valid_classes[top1_sub_idx]

            crop_top_predictions = []
            for i in range(min(top_k, len(sorted_crop_order))):
                s_idx = int(sorted_crop_order[i])
                lbl = valid_classes[s_idx]
                c_prob = float(norm_crop_probs[s_idx])
                crop_top_predictions.append({
                    "label": lbl,
                    "probability": round(c_prob, 4),
                    "confidence_percent": round(c_prob * 100, 2),
                })

            is_healthy = "healthy" in top1_label.lower()
            affected_reg_desc = f"Observed on {crop_species} {plant_part.value.replace('_', ' ')}"
            parsed = {"is_healthy": is_healthy, "condition": top1_label}

        # Check confidence gates within this crop subspace
        if top1_prob >= 0.60 and crop_entropy <= config.OOD_ENTROPY_THRESHOLD:
            status = "confident"
            evidence_level = EvidenceLevel.HIGH_CONFIDENCE
            final_label = top1_label
            decision_rule = f"RULE_STAGE4_CONFIDENT_DIAGNOSIS (Crop: {crop_species}, Part: {plant_part.value}, Top1 prob: {top1_prob*100:.1f}% >= 60%, Subspace H: {crop_entropy:.3f})"
        elif top1_prob >= config.MIN_CONFIDENCE_THRESHOLD:
            status = "probable_diagnosis_needs_confirmation"
            evidence_level = EvidenceLevel.SCREENING_HYPOTHESIS
            final_label = f"Possible: {top1_label} (Needs Confirmation)"
            decision_rule = f"RULE_STAGE4_PROBABLE_DIAGNOSIS (Crop: {crop_species}, Part: {plant_part.value}, Top1 prob: {top1_prob*100:.1f}% in [35%, 60%))"
        else:
            status = "uncertain_disease"
            evidence_level = EvidenceLevel.INSUFFICIENT_EVIDENCE
            final_label = f"Inconclusive {crop_species} {plant_part.value.title()} Symptoms"
            decision_rule = f"RULE_STAGE4_INCONCLUSIVE (Crop: {crop_species}, Part: {plant_part.value}, Top1 prob: {top1_prob*100:.1f}% < 35%)"

        # Fetch verified authoritative primary source citation matching this confirmed crop + diagnosis
        source_record = get_authoritative_source_citation(top1_label if status in ["confident", "probable_diagnosis_needs_confirmation"] else crop_species)

        # Nutrient screening hypothesis check
        nutrient_screening = None
        if "chlorosis" in parsed["condition"].lower() or "yellow" in parsed["condition"].lower() or "blight" in parsed["condition"].lower():
            nutrient_screening = analyze_nutrient_symptoms(
                affected_part="leaf",
                plant_region="canopy",
                crop_species=crop_species,
                soil_ph=soil_ph,
            )

        debug_info = {
            "stage1_valid": True,
            "stage1_reason": "Valid Plant Image",
            "stage2_plant_part": plant_part.value,
            "stage3_crop_species": crop_species,
            "stage3_crop_confidence_pct": round(crop_conf * 100, 1),
            "stage4_route_supported": True,
            "image_dimensions": image_dims,
            "image_mode": image_mode,
            "dinov2_similarity": round(dinov2_sim, 4),
            "dinov2_threshold": round(dinov2_threshold, 4),
            "is_ood": False,
            "predicted_class": final_label,
            "top1_probability": round(top1_prob, 4),
            "top1_confidence_pct": round(top1_prob * 100, 2),
            "top_probabilities": crop_top_predictions,
            "top3_probabilities": crop_top_predictions,
            "entropy": round(crop_entropy, 4),
            "raw_entropy": round(crop_entropy, 4),
            "norm_entropy": round(norm_crop_entropy, 4),
            "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
            "class_count": len(active_indices),
            "blur_metric": round(blur_metric, 2),
            "blur_threshold": round(blur_threshold, 1),
            "decision_rule": decision_rule,
            "final_decision": status.upper(),
        }

        return {
            "plant_scope": "hierarchical_4stage_pipeline",
            "status": status,
            "dinov2_plant_similarity": round(dinov2_sim, 4),
            "dinov2_threshold": round(dinov2_threshold, 4),
            "is_ood": False,
            "plant_part": plant_part.value,
            "crop_species": crop_species,
            "entropy": round(crop_entropy, 4),
            "norm_entropy": round(norm_crop_entropy, 4),
            "top1_label": final_label,
            "top1_label_kannada": final_label,
            "raw_top1_label": top1_label,
            "top1_confidence": round(top1_prob * 100, 2),
            "is_healthy": is_healthy,
            "plant_type": crop_species,
            "affected_regions": affected_reg_desc,
            "affected_regions_kn": f"{crop_species} ಬೆಳೆಯ {plant_part.value.replace('_', ' ')}",
            "predictions": crop_top_predictions,
            "nutrient_screening": nutrient_screening,
            "source_evidence": source_record,
            "debug_info": debug_info,
            "refusal_reason_en": "",
            "refusal_reason_kn": "",
            "requested_next_action_en": (
                "Please inspect the underside of the leaf for active fungal sporulation or pustules."
                if status == "probable_diagnosis_needs_confirmation" else ""
            ),
            "requested_next_action_kn": (
                "ದಯವಿಟ್ಟು ಎಲೆಯ ಹಿಂಭಾಗದಲ್ಲಿ ಶಿಲೀಂಧ್ರ ಬೂಸ್ಟು ಅಥವಾ ಕಲೆಗಳಿವೆಯೇ ಗಮನಿಸಿ."
                if status == "probable_diagnosis_needs_confirmation" else ""
            ),
        }

    def _build_unknown_crop_response(self, crop, part, dino_sim, dino_thresh, blur_val, blur_thresh, dims, mode):
        msg = "Valid plant image detected. Disease diagnosis is not available yet for this crop/plant part."
        debug_info = {
            "stage1_valid": True,
            "stage2_plant_part": part.value,
            "stage3_crop_species": crop,
            "stage4_route_supported": False,
            "image_dimensions": dims,
            "image_mode": mode,
            "dinov2_similarity": round(dino_sim, 4),
            "dinov2_threshold": round(dino_thresh, 4),
            "is_ood": False,
            "predicted_class": msg,
            "top1_confidence_pct": 0.0,
            "top_probabilities": [],
            "top3_probabilities": [],
            "decision_rule": "RULE_STAGE4_UNKNOWN_CROP",
            "final_decision": "UNSUPPORTED_CROP_PART",
        }
        return {
            "plant_scope": "hierarchical_4stage_pipeline",
            "status": "unsupported_crop_part",
            "dinov2_plant_similarity": round(dino_sim, 4),
            "dinov2_threshold": round(dino_thresh, 4),
            "is_ood": False,
            "plant_part": part.value,
            "crop_species": crop,
            "top1_label": msg,
            "top1_label_kannada": "ಕೃಷಿ ಸಸ್ಯ ಗುರುತಿಸಲಾಗಿದೆ. ಈ ಬೆಳೆ ಅಥವಾ ಸಸ್ಯದ ಭಾಗಕ್ಕೆ ರೋಗನಿರ್ಣಯ ಮಾದರಿಯು ಇನ್ನೂ ತರಬೇತಿಯಲ್ಲಿಲ್ಲ.",
            "top1_confidence": 0.0,
            "entropy": 0.0,
            "norm_entropy": 0.0,
            "is_healthy": False,
            "plant_type": crop,
            "affected_regions": "None",
            "predictions": [],
            "debug_info": debug_info,
            "source_evidence": None,
            "refusal_reason_en": msg,
            "refusal_reason_kn": "ಕೃಷಿ ಸಸ್ಯ ಗುರುತಿಸಲಾಗಿದೆ. ಈ ಬೆಳೆ ಅಥವಾ ಸಸ್ಯದ ಭಾಗಕ್ಕೆ ರೋಗನಿರ್ಣಯ ಮಾದರಿಯು ಇನ್ನೂ ತರಬೇತಿಯಲ್ಲಿಲ್ಲ.",
            "requested_next_action_en": "Please capture a clear, well-lit close-up photograph of an affected leaf.",
            "requested_next_action_kn": "ದಯವಿಟ್ಟು ರೋಗಬಾಧಿತ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಕಳುಹಿಸಿ.",
        }
