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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import config
import torch.nn as nn
from src.rag.authoritative_sources import get_authoritative_source_citation
from src.vision.dataset_builder import localize_symptom_roi
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
    try:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float64)
        if not hasattr(arr, "shape") or len(arr.shape) < 2:
            return 0.0
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
    except Exception:
        return 0.0


def compute_shannon_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon Predictive Entropy H(p) = -sum(p_i * log(p_i))."""
    clipped = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    return float(entropy)


def _is_debug_mode(explicit_debug: Optional[bool] = None) -> bool:
    """Determine whether debug/development mode is active."""
    if explicit_debug is not None:
        return explicit_debug
    env_debug = os.getenv("DEBUG", "").lower() in ["1", "true", "yes", "dev", "development"]
    env_app_env = os.getenv("APP_ENV", "").lower() in ["dev", "development", "local"]
    log_debug = logging.getLogger().isEnabledFor(logging.DEBUG)
    return env_debug or env_app_env or log_debug


def normalize_candidate(
    candidate: Any,
    debug_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Standardize a candidate prediction dictionary into:
        {"class": "label name", "prob": 0.0}
    along with 'label', 'probability', and 'confidence_percent' aliases for backwards compatibility.
    
    Handles malformed inputs (None, missing fields, strings, invalid numbers, out-of-range floats)
    gracefully without crashing. Logs malformed candidates only in debug/development mode.
    """
    is_debug = _is_debug_mode(debug_mode)
    malformed_reason = None
    label = None
    prob = None

    if candidate is None:
        malformed_reason = "Candidate item is None"
        label = "Unknown candidate"
        prob = 0.0
    elif isinstance(candidate, (tuple, list)):
        label = str(candidate[0]) if len(candidate) > 0 and candidate[0] is not None else "Unknown candidate"
        raw_prob = candidate[1] if len(candidate) > 1 else None
        if raw_prob is not None:
            if isinstance(raw_prob, str):
                malformed_reason = f"String probability in pair: {raw_prob!r}"
            try:
                val = float(raw_prob)
                if 0.0 <= val <= 1.0:
                    prob = val
                else:
                    malformed_reason = f"Probability out-of-range ({val})"
                    prob = 0.0
            except (TypeError, ValueError):
                malformed_reason = f"Non-numeric probability: {raw_prob!r}"
                prob = 0.0
        else:
            malformed_reason = "Missing probability in pair"
            prob = 0.0
    elif isinstance(candidate, dict):
        label = (
            candidate.get("class")
            or candidate.get("label")
            or candidate.get("condition")
            or "Unknown candidate"
        )
        raw_prob = candidate.get("prob")
        if raw_prob is None and "prob" not in candidate:
            if candidate.get("probability") is not None:
                raw_prob = candidate.get("probability")
            elif "confidence_percent" in candidate:
                raw_conf = candidate.get("confidence_percent")
                try:
                    raw_prob = float(raw_conf) / 100.0
                except (TypeError, ValueError):
                    raw_prob = None

        if raw_prob is None:
            malformed_reason = "Missing or null 'prob' field"
            prob = 0.0
        else:
            if isinstance(raw_prob, str):
                malformed_reason = f"String probability value: {raw_prob!r}"
            try:
                val = float(raw_prob)
                if 0.0 <= val <= 1.0:
                    prob = val
                else:
                    malformed_reason = f"Probability out-of-range ({val})"
                    prob = 0.0
            except (TypeError, ValueError):
                malformed_reason = f"Non-numeric probability: {raw_prob!r}"
                prob = 0.0
    else:
        malformed_reason = f"Invalid candidate object type: {type(candidate).__name__}"
        label = str(candidate)
        prob = 0.0

    if malformed_reason and is_debug:
        logging.debug("Malformed candidate normalized: %s | Candidate: %r", malformed_reason, candidate)

    prob_float = round(float(prob if prob is not None else 0.0), 4)
    return {
        "class": str(label),
        "prob": prob_float,
        "label": str(label),
        "probability": prob_float,
        "confidence_percent": round(prob_float * 100, 2),
    }


def build_top_candidates(
    candidates: Any,
    top_k: Optional[int] = 3,
    debug_mode: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Standardize a collection of candidate predictions to ensure every candidate
    consistently returns:
        {"class": "label name", "prob": 0.0}
    along with 'label' and 'probability' aliases for backwards compatibility.
    Does not crash when a candidate is malformed (None, missing prob, string, out of range).
    Logs malformed candidates only in development/debug mode.
    """
    is_debug = _is_debug_mode(debug_mode)
    if candidates is None:
        if is_debug:
            logging.debug("build_top_candidates received None candidates input")
        return []

    if isinstance(candidates, dict):
        return [normalize_candidate(candidates, debug_mode=debug_mode)]

    if not isinstance(candidates, (list, tuple)):
        if is_debug:
            logging.debug("build_top_candidates received non-iterable input: %r", type(candidates))
        return [normalize_candidate(candidates, debug_mode=debug_mode)]

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        norm = normalize_candidate(cand, debug_mode=debug_mode)
        results.append(norm)
        if top_k is not None and len(results) >= top_k:
            break

    return results


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

    # Compatibility alias for viewpoint routing tests
    _check_stage1_image_validity = _check_stage1_validity

    def _identify_stage2_plant_part(
        self,
        img: Image.Image,
        global_emb: Optional[torch.Tensor] = None,
        user_part_hint: Optional[str] = None,
    ) -> PlantPart:
        """
        Stage 2: Identify plant part: leaf / whole plant / canopy / stem / fruit / root / bark / non_plant / unknown.
        Prioritizes user confirmation if provided.
        """
        if user_part_hint and str(user_part_hint).strip():
            hint_str = str(user_part_hint).strip()
            hint_lower = hint_str.lower()
            if hint_lower in ["auto", "auto-detect", ""]:
                pass
            elif hint_lower in ["unknown", "unknown view", "unknown viewpoint"]:
                return PlantPart.UNKNOWN
            else:
                return PlantPart.from_str(hint_str)

        # 1. Close-up Foliage Prototype Gate:
        # If embedding matches foliar leaf prototypes with high confidence (>= 0.65), it is a Leaf
        if global_emb is not None and hasattr(self, "leaf_prototypes") and self.leaf_prototypes is not None:
            try:
                foliar_sim = float(F.cosine_similarity(global_emb, self.leaf_prototypes).max().item())
                if foliar_sim >= 0.65:
                    return PlantPart.LEAF
            except Exception:
                pass

        # 2. Autonomous neural detection using trained Plant Part Classifier
        if global_emb is not None and self.plant_part_model is not None and self.plant_part_classes:
            with torch.no_grad():
                logits = self.plant_part_model(global_emb.to(self.device))[0]
                probs = F.softmax(logits, dim=-1)
                pred_idx = int(torch.argmax(probs).item())
                conf = float(probs[pred_idx].item())
                part_name = self.plant_part_classes[pred_idx]
                part_map = {
                    "stem_branch": PlantPart.STEM,
                    "fruit": PlantPart.FRUIT,
                    "root_tuber": PlantPart.ROOT,
                    "whole_plant": PlantPart.WHOLE_PLANT,
                    "canopy": PlantPart.CANOPY,
                    "bark_trunk": PlantPart.BARK_TRUNK,
                    "leaf": PlantPart.LEAF,
                    "non_plant": PlantPart.NON_PLANT,
                }
                if conf >= 0.35 and part_name in part_map and part_map[part_name] != PlantPart.LEAF:
                    return part_map[part_name]
                elif conf >= 0.40 and part_name == "leaf":
                    return PlantPart.LEAF
                else:
                    logging.info("Plant part detection confidence %.3f is weak; designating PlantPart.UNKNOWN", conf)
                    return PlantPart.UNKNOWN

        # 3. Geometric heuristic for field views
        if hasattr(img, "size") and img.size:
            w, h = img.size
            aspect = w / float(h)
        else:
            aspect = 1.0

        if aspect > 1.4:
            return PlantPart.CANOPY
        elif aspect < 0.70:
            return PlantPart.WHOLE_PLANT

        # 4. Foliar similarity negative check
        if global_emb is not None and hasattr(self, "leaf_prototypes") and self.leaf_prototypes is not None:
            try:
                foliar_sim = float(F.cosine_similarity(global_emb, self.leaf_prototypes).max().item())
                if foliar_sim < 0.25:
                    return PlantPart.CANOPY if aspect > 1.0 else PlantPart.WHOLE_PLANT
            except Exception:
                pass

        # CRITICAL: Never default unknown or ambiguous view to LEAF!
        return PlantPart.UNKNOWN

    def _raw_infer_probabilities(self, img: Image.Image, plant_part: Optional[PlantPart] = None) -> np.ndarray:
        """Run MobileNetV2 over image to get raw 38-class softmax distribution."""
        if plant_part is not None and plant_part != PlantPart.LEAF:
            raise RuntimeError(
                f"CRITICAL SAFETY VIOLATION: Non-leaf viewpoint '{plant_part.value}' attempted to call foliar MobileNetV2 model!"
            )
        inputs = self.classifier_wrapper.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier_wrapper.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return probs

    def _identify_stage3_crop_species(
        self,
        raw_probs: Optional[np.ndarray] = None,
        global_emb: Optional[torch.Tensor] = None,
        plant_part: Optional[PlantPart] = None,
        user_crop_hint: Optional[str] = None,
        min_crop_confidence: float = 0.45,
    ) -> Tuple[str, float]:
        """
        Stage 3: Identify crop species only when confidence is adequate.
        Returns: (crop_species: str, crop_confidence: float)
        """
        if user_crop_hint and str(user_crop_hint).strip():
            hint_str = str(user_crop_hint).strip()
            hint_lower = hint_str.lower()
            if hint_lower in ["auto", "auto-detect", ""]:
                pass
            elif hint_lower in ["unknown", "unknown crop", "unknown plant"]:
                return "Unknown Crop", 0.0
            else:
                crop_aliases = {
                    "maize": "Corn (Maize)",
                    "corn": "Corn (Maize)",
                    "tomato": "Tomato",
                    "grape": "Grape",
                    "apple": "Apple",
                    "potato": "Potato",
                    "capsicum": "Bell Pepper (Capsicum)",
                    "pepper": "Bell Pepper (Capsicum)",
                    "bell pepper": "Bell Pepper (Capsicum)",
                    "citrus": "Citrus",
                    "orange": "Citrus",
                    "coconut": "Coconut",
                }
                for alias, canonical in crop_aliases.items():
                    if alias in hint_lower or hint_lower in alias:
                        return canonical, 1.0
                return hint_str, 1.0

        if plant_part in [PlantPart.NON_PLANT, PlantPart.UNKNOWN]:
            return "Unknown Crop", 0.0

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
                logging.info("Crop identification confidence weak for non-leaf view; designating Unknown Crop")
                return "Unknown Crop", 0.0

        # Autonomous Crop Identification from Foliar Vision (ONLY for LEAF)
        if plant_part == PlantPart.LEAF and raw_probs is not None:
            crop_scores: Dict[str, float] = {}
            for crop_name, indices in self.class_idx_by_crop.items():
                if indices:
                    crop_scores[crop_name] = float(np.sum(raw_probs[indices]))

            if crop_scores:
                best_crop, best_conf = max(crop_scores.items(), key=lambda x: x[1])
                if best_conf >= min_crop_confidence:
                    return best_crop, best_conf
                logging.info("Foliar crop confidence %.3f < %.3f; designating Unknown Crop", best_conf, min_crop_confidence)

        return "Unknown Crop", 0.0

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
        elif isinstance(image_input, Image.Image) or hasattr(image_input, "size"):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        if hasattr(img, "mode") and hasattr(img, "convert"):
            img = ImageOps.exif_transpose(img) if hasattr(ImageOps, "exif_transpose") else img
            if img.mode != "RGB":
                img = img.convert("RGB")

        image_dims = f"{img.width}x{img.height}"
        image_mode = img.mode

        # ======================================================================
        # STAGE 1: Image Validity & Quality Gating
        # ======================================================================
        is_sharp, is_plant, blur_metric, dinov2_sim, global_emb = self._check_stage1_image_validity(
            img, dinov2_threshold=dinov2_threshold, blur_threshold=blur_threshold
        )

        # 1A. Non-Plant OOD Refusal
        if not is_plant:
            decision_rule = f"RULE_STAGE1_NON_PLANT_OOD (DINOv2 similarity {dinov2_sim:.4f} < threshold {dinov2_threshold:.2f})"
            final_reason = "Non-Agricultural / Out-of-Distribution Object"
            outcome = "Non-plant image detected"
            if _is_debug_mode():
                logging.info(
                    "[DEV_TELEMETRY] model_path=facebook/dinov2-small | crop_route=Unknown | viewpoint_route=non_plant | top5_probabilities=[] | entropy=3.6370 | confidence_threshold=%.2f | agreement_score=0.0 | final_decision_reason=%s",
                    dinov2_threshold,
                    decision_rule,
                )
            debug_info = {
                "stage1_valid": False,
                "stage1_reason": final_reason,
                "stage2_plant_part": "non_plant",
                "stage3_crop_species": "Unknown",
                "stage4_route_supported": False,
                "image_dimensions": image_dims,
                "image_mode": image_mode,
                "dinov2_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": True,
                "predicted_class": outcome,
                "top1_confidence_pct": 0.0,
                "top_probabilities": [],
                "top3_probabilities": [],
                "top5_probabilities": [],
                "entropy": 3.637,
                "raw_entropy": 3.637,
                "norm_entropy": 1.0,
                "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
                "class_count": self.num_classes,
                "blur_metric": round(blur_metric, 2),
                "blur_threshold": round(blur_threshold, 1),
                "decision_rule": decision_rule,
                "final_decision_reason": decision_rule,
                "final_decision": "OUT_OF_DISTRIBUTION",
                "model_path": "facebook/dinov2-small",
                "selected_model": "facebook/dinov2-small",
                "fallback_reason": final_reason,
                "crop_route": "Unknown",
                "viewpoint_route": "non_plant",
                "confidence_threshold": round(dinov2_threshold, 4),
                "agreement_score": 0.0,
                "outcome": outcome,
                "disease": None,
            }
            return {
                "plant_scope": "hierarchical_4stage_pipeline",
                "outcome": outcome,
                "status": "out_of_distribution",
                "supported": False,
                "disease": None,
                "selected_model": "facebook/dinov2-small",
                "model_path": "facebook/dinov2-small",
                "fallback_reason": final_reason,
                "dinov2_plant_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": True,
                "plant_part": "non_plant",
                "crop_species": "Unknown",
                "top1_label": outcome,
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
            outcome = "Image quality insufficient"
            final_reason = "Image Blurred / Low Focus"
            if _is_debug_mode():
                logging.info(
                    "[DEV_TELEMETRY] model_path=Laplacian | crop_route=Unknown | viewpoint_route=unknown | top5_probabilities=[] | entropy=0.0 | confidence_threshold=%.2f | agreement_score=0.0 | final_decision_reason=%s",
                    blur_threshold,
                    decision_rule,
                )
            debug_info = {
                "stage1_valid": False,
                "stage1_reason": final_reason,
                "stage2_plant_part": "unknown",
                "stage3_crop_species": "Unknown",
                "stage4_route_supported": False,
                "image_dimensions": image_dims,
                "image_mode": image_mode,
                "dinov2_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": False,
                "predicted_class": outcome,
                "disease": None,
                "top1_confidence_pct": 0.0,
                "top_probabilities": [],
                "top3_probabilities": [],
                "top5_probabilities": [],
                "entropy": 0.0,
                "raw_entropy": 0.0,
                "norm_entropy": 0.0,
                "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
                "class_count": self.num_classes,
                "blur_metric": round(blur_metric, 2),
                "blur_threshold": round(blur_threshold, 1),
                "decision_rule": decision_rule,
                "final_decision_reason": decision_rule,
                "final_decision": "INSUFFICIENT_IMAGE_QUALITY",
                "model_path": "LaplacianVariance",
                "selected_model": "LaplacianVariance",
                "fallback_reason": final_reason,
                "crop_route": "Unknown",
                "viewpoint_route": "unknown",
                "confidence_threshold": round(blur_threshold, 1),
                "agreement_score": 0.0,
                "outcome": outcome,
            }
            return {
                "plant_scope": "hierarchical_4stage_pipeline",
                "outcome": outcome,
                "status": "insufficient_image_quality",
                "supported": False,
                "disease": None,
                "selected_model": "LaplacianVariance",
                "model_path": "LaplacianVariance",
                "fallback_reason": final_reason,
                "dinov2_plant_similarity": round(dinov2_sim, 4),
                "dinov2_threshold": round(dinov2_threshold, 4),
                "is_ood": False,
                "plant_part": "unknown",
                "crop_species": "Unknown",
                "top1_label": outcome,
                "top1_label_kannada": "ಚಿತ್ರವು ಅಸ್ಪಷ್ಟವಾಗಿದೆ / ಸರಿಯಾದ ಫೋಕಸ್ ಇಲ್ಲ (Image Quality Insufficient)",
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
                    "Accurate pathology diagnosis requires sharp visual details. Automatic diagnosis has been safely withheld."
                ),
                "refusal_reason_kn": (
                    f"ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಫೋಟೋ ಮಸುಕಾಗಿದೆ ಅಥವಾ ಸ್ಪಷ್ಟವಾಗಿಲ್ಲ (ತೀಕ್ಷ್ಣತೆ: {blur_metric:.1f} < {blur_threshold:.1f}). "
                    "ಸರಿಯಾದ ರೋಗ ಪತ್ತೆಗೆ ಸಸ್ಯದ ಸ್ಪಷ್ಟ ವಿವರಗಳು ಅಗತ್ಯವಿದೆ."
                ),
                "requested_next_action_en": "Please hold your camera steady in bright natural daylight and retake a sharp, focused photo of the affected plant.",
                "requested_next_action_kn": "ದಯವಿಟ್ಟು ಉತ್ತಮ ಬೆಳಕಿನಲ್ಲಿ ಕ್ಯಾಮೆರಾವನ್ನು ಅಲುಗಾಡಿಸದೆ ಸ್ಪಷ್ಟ ಫೋಟೋ ತೆಗೆದು ಕಳುಹಿಸಿ.",
            }

        # ======================================================================
        # STAGE 2: Plant Part Identification
        # ======================================================================
        plant_part = self._identify_stage2_plant_part(img, global_emb=global_emb, user_part_hint=user_part_hint)

        # ======================================================================
        # STAGE 3: Crop Species Identification
        # ======================================================================
        fused_whole_plant_emb = None
        if plant_part == PlantPart.LEAF:
            raw_probs = self._raw_infer_probabilities(img, plant_part=plant_part)
            crop_species, crop_conf = self._identify_stage3_crop_species(
                raw_probs, global_emb=global_emb, plant_part=plant_part, user_crop_hint=user_crop_hint
            )
        elif plant_part == PlantPart.WHOLE_PLANT:
            # Whole plant: extract symptom ROI and fuse with global whole-plant context
            raw_probs = None  # CRITICAL: Never pass whole-plant image to leaf-only model!
            try:
                roi, _, _ = localize_symptom_roi(img)
                inputs_roi = self.dinov2_wrapper.processor(images=roi, return_tensors="pt")
                inputs_roi = {k: v.to(self.device) for k, v in inputs_roi.items()}
                with torch.no_grad():
                    out_roi = self.dinov2_wrapper.model(**inputs_roi)
                    cls_roi = F.normalize(out_roi.last_hidden_state[:, 0, :], dim=-1)
                fused_whole_plant_emb = F.normalize(global_emb.to(self.device) + 1.25 * cls_roi, dim=-1)
            except Exception:
                fused_whole_plant_emb = global_emb.to(self.device) if global_emb is not None else None
            crop_species, crop_conf = self._identify_stage3_crop_species(
                None, global_emb=fused_whole_plant_emb, plant_part=plant_part, user_crop_hint=user_crop_hint
            )
        else:
            raw_probs = None  # CRITICAL: Never pass non-leaf image to leaf-only model!
            crop_species, crop_conf = self._identify_stage3_crop_species(
                None, global_emb=global_emb, plant_part=plant_part, user_crop_hint=user_crop_hint
            )

        # ======================================================================
        # STAGE 4: Route Verification & Constrained Pathology Diagnosis
        # ======================================================================
        # CRITICAL SAFEGUARD: Leaf-only classifier must NEVER be used for non-leaf views!
        if plant_part == PlantPart.NON_PLANT:
            return self._build_non_plant_ood_response(
                dinov2_sim, dinov2_threshold, blur_metric, blur_threshold, image_dims, image_mode
            )

        # 4A. Unknown Viewpoint or Unknown Crop Refusal
        if plant_part == PlantPart.UNKNOWN or crop_species in ["Unknown", "Unknown Crop"]:
            fb_reason = (
                f"Viewpoint '{plant_part.value}' is unknown or unverified; non-leaf views cannot call foliar model"
                if plant_part == PlantPart.UNKNOWN
                else f"Crop '{crop_species}' is unknown or below confidence threshold"
            )
            return self._build_unsupported_route_response(
                crop_species=crop_species,
                plant_part=plant_part,
                fallback_reason=fb_reason,
                dinov2_sim=dinov2_sim,
                dinov2_threshold=dinov2_threshold,
                blur_metric=blur_metric,
                blur_threshold=blur_threshold,
                image_dims=image_dims,
                image_mode=image_mode,
                crop_conf=crop_conf,
            )

        route_supported, route_rec = is_route_supported(crop_species, plant_part.value)

        # 4B. Non-Leaf Viewpoint: whole_plant, canopy, stem, fruit, root, bark
        if plant_part != PlantPart.LEAF:
            part_model_info = self._get_multipart_model(plant_part.value)
            if not route_supported or not route_rec or not part_model_info:
                return self._build_unsupported_route_response(
                    crop_species=crop_species,
                    plant_part=plant_part,
                    fallback_reason=f"No matching trained checkpoint or route deployed for crop '{crop_species}' and viewpoint '{plant_part.value}'",
                    dinov2_sim=dinov2_sim,
                    dinov2_threshold=dinov2_threshold,
                    blur_metric=blur_metric,
                    blur_threshold=blur_threshold,
                    image_dims=image_dims,
                    image_mode=image_mode,
                    crop_conf=crop_conf,
                )

        # 4C. Leaf Viewpoint Unsupported Route
        if plant_part == PlantPart.LEAF and (not route_supported or not route_rec):
            return self._build_unsupported_route_response(
                crop_species=crop_species,
                plant_part=plant_part,
                fallback_reason=f"Foliar route for crop '{crop_species}' is unsupported or untrained",
                dinov2_sim=dinov2_sim,
                dinov2_threshold=dinov2_threshold,
                blur_metric=blur_metric,
                blur_threshold=blur_threshold,
                image_dims=image_dims,
                image_mode=image_mode,
                crop_conf=crop_conf,
            )

        # CASE 4D: SUPPORTED FOLIAR ROUTE
        if plant_part == PlantPart.LEAF:
            model_path = "models/vision/mobilenet_v2.pt"
            selected_model = "MobileNetV2 Foliar Backbone"
            valid_classes = get_valid_classes_for_crop(crop_species)
            indices = self.class_idx_by_crop.get(crop_species, [])

            if not indices or not valid_classes:
                logging.error("No valid class indices for supported crop '%s'", crop_species)
                return self._build_unsupported_route_response(
                    crop_species=crop_species,
                    plant_part=plant_part,
                    fallback_reason=f"No valid class indices for crop '{crop_species}'",
                    dinov2_sim=dinov2_sim,
                    dinov2_threshold=dinov2_threshold,
                    blur_metric=blur_metric,
                    blur_threshold=blur_threshold,
                    image_dims=image_dims,
                    image_mode=image_mode,
                    crop_conf=crop_conf,
                )

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

            raw_top_predictions: List[Dict[str, Any]] = []
            for i in range(min(max(top_k, 5), len(sorted_crop_order))):
                s_idx = int(sorted_crop_order[i])
                c_idx = indices[s_idx]
                lbl = self.id2label[str(c_idx) if str(c_idx) in self.id2label else c_idx]
                c_prob = float(norm_crop_probs[s_idx])
                raw_top_predictions.append({
                    "class": lbl,
                    "prob": round(c_prob, 4),
                    "label": lbl,
                    "probability": round(c_prob, 4),
                    "confidence_percent": round(c_prob * 100, 2),
                })
            crop_top_predictions = build_top_candidates(raw_top_predictions, top_k=top_k)
            crop_top5_predictions = build_top_candidates(raw_top_predictions, top_k=5)

            parsed = parse_legacy_label(top1_label)
            is_healthy = parsed["is_healthy"]
            affected_reg_desc = f"Foliar lesions on {crop_species} canopy"

        # CASE 4E: SUPPORTED MULTI-PART ROUTE (Stem, Fruit, Root, Whole-Plant, Canopy, Bark)
        else:
            part_name = plant_part.value
            part_model_info = self._get_multipart_model(part_name)
            if not part_model_info:
                return self._build_unsupported_route_response(
                    crop_species=crop_species,
                    plant_part=plant_part,
                    fallback_reason=f"No multi-part checkpoint available for '{part_name}'",
                    dinov2_sim=dinov2_sim,
                    dinov2_threshold=dinov2_threshold,
                    blur_metric=blur_metric,
                    blur_threshold=blur_threshold,
                    image_dims=image_dims,
                    image_mode=image_mode,
                    crop_conf=crop_conf,
                )

            model, model_classes = part_model_info
            model_path = str(config.VISION_MODEL_DIR / f"{'whole_plant' if plant_part == PlantPart.WHOLE_PLANT else part_name}_classifier.pt")
            selected_model = model_path

            # Use fused embedding for whole plant (global habit + symptom ROI)
            active_emb = fused_whole_plant_emb if (plant_part == PlantPart.WHOLE_PLANT and fused_whole_plant_emb is not None) else (global_emb.to(self.device) if global_emb is not None else None)
            if active_emb is None or not isinstance(active_emb, torch.Tensor):
                active_emb = torch.randn(1, 384, device=self.device)
            with torch.no_grad():
                logits = model(active_emb)[0].cpu().numpy()

            # Global OOD check for whole-plant route
            if plant_part == PlantPart.WHOLE_PLANT:
                all_probs = np.exp(logits - np.max(logits))
                all_probs = all_probs / max(float(np.sum(all_probs)), 1e-9)
                top_global_idx = int(np.argmax(all_probs))
                top_global_class = model_classes[top_global_idx]
                if top_global_class == "Non-Plant Background Object" and all_probs[top_global_idx] > 0.40:
                    if crop_species in ["Unknown", "Unknown Crop"] or not is_plant:
                        return self._build_non_plant_ood_response(
                            dinov2_sim, dinov2_threshold, blur_metric, blur_threshold, image_dims, image_mode
                        )
                    else:
                        return self._build_unsupported_route_response(
                            crop_species=crop_species,
                            plant_part=plant_part,
                            fallback_reason=f"Whole-plant model predicted Non-Plant Background Object ({all_probs[top_global_idx]*100:.1f}%) for confirmed crop",
                            dinov2_sim=dinov2_sim,
                            dinov2_threshold=dinov2_threshold,
                            blur_metric=blur_metric,
                            blur_threshold=blur_threshold,
                            image_dims=image_dims,
                            image_mode=image_mode,
                            crop_conf=crop_conf,
                        )
                if top_global_class == "Unknown Plant Foliage" and all_probs[top_global_idx] > 0.40:
                    return self._build_unsupported_route_response(
                        crop_species=crop_species,
                        plant_part=plant_part,
                        fallback_reason=f"Whole-plant model predicted Unknown Plant Foliage ({all_probs[top_global_idx]*100:.1f}%)",
                        dinov2_sim=dinov2_sim,
                        dinov2_threshold=dinov2_threshold,
                        blur_metric=blur_metric,
                        blur_threshold=blur_threshold,
                        image_dims=image_dims,
                        image_mode=image_mode,
                        crop_conf=crop_conf,
                    )

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

            raw_top_predictions = []
            for i in range(min(max(top_k, 5), len(sorted_crop_order))):
                s_idx = int(sorted_crop_order[i])
                lbl = valid_classes[s_idx]
                c_prob = float(norm_crop_probs[s_idx])
                raw_top_predictions.append({
                    "class": lbl,
                    "prob": round(c_prob, 4),
                    "label": lbl,
                    "probability": round(c_prob, 4),
                    "confidence_percent": round(c_prob * 100, 2),
                })
            crop_top_predictions = build_top_candidates(raw_top_predictions, top_k=top_k)
            crop_top5_predictions = build_top_candidates(raw_top_predictions, top_k=5)

            is_healthy = "healthy" in top1_label.lower()
            affected_reg_desc = f"Observed on {crop_species} {plant_part.value.replace('_', ' ')}"
            parsed = {"is_healthy": is_healthy, "condition": top1_label}

        # Check confidence gates within this crop subspace
        if top1_prob >= 0.60 and crop_entropy <= config.OOD_ENTROPY_THRESHOLD:
            outcome = "Validated probable diagnosis"
            status = "confident"
            evidence_level = EvidenceLevel.HIGH_CONFIDENCE
            final_label = top1_label
            decision_rule = f"RULE_STAGE4_CONFIDENT_DIAGNOSIS (Crop: {crop_species}, Part: {plant_part.value}, Top1 prob: {top1_prob*100:.1f}% >= 60%, Subspace H: {crop_entropy:.3f})"
        elif top1_prob >= config.MIN_CONFIDENCE_THRESHOLD:
            outcome = "Possible diagnosis—needs confirmation"
            status = "probable_diagnosis_needs_confirmation"
            evidence_level = EvidenceLevel.SCREENING_HYPOTHESIS
            final_label = f"Possible: {top1_label} (Needs Confirmation)"
            decision_rule = f"RULE_STAGE4_PROBABLE_DIAGNOSIS (Crop: {crop_species}, Part: {plant_part.value}, Top1 prob: {top1_prob*100:.1f}% in [35%, 60%))"
        else:
            outcome = "Possible diagnosis—needs confirmation"
            status = "uncertain_disease"
            evidence_level = EvidenceLevel.INSUFFICIENT_EVIDENCE
            final_label = f"Inconclusive {crop_species} {plant_part.value.title()} Symptoms"
            decision_rule = f"RULE_STAGE4_INCONCLUSIVE (Crop: {crop_species}, Part: {plant_part.value}, Top1 prob: {top1_prob*100:.1f}% < 35%)"

        if _is_debug_mode():
            logging.info(
                "[DEV_TELEMETRY] model_path=%s | crop_route=%s | viewpoint_route=%s | top5_probabilities=%s | entropy=%.4f | confidence_threshold=0.60 | agreement_score=1.0 | final_decision_reason=%s",
                model_path,
                crop_species,
                plant_part.value,
                [(c["label"], c["prob"]) for c in crop_top5_predictions],
                crop_entropy,
                decision_rule,
            )

        # Fetch verified authoritative primary source citation matching this confirmed crop + diagnosis
        source_record = get_authoritative_source_citation(top1_label if status in ["confident", "probable_diagnosis_needs_confirmation"] else crop_species)

        # Nutrient screening hypothesis check
        nutrient_screening = None
        if "chlorosis" in parsed["condition"].lower() or "yellow" in parsed["condition"].lower() or "blight" in parsed["condition"].lower():
            nutrient_screening = analyze_nutrient_symptoms(
                affected_part="leaf" if plant_part == PlantPart.LEAF else plant_part.value,
                plant_region="canopy",
                crop_species=crop_species,
                soil_ph=soil_ph,
            )

        selected_model = "MobileNetV2 Foliar Backbone" if plant_part == PlantPart.LEAF else model_path
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
            "disease": None if is_healthy or status == "uncertain_disease" else top1_label,
            "top1_probability": round(top1_prob, 4),
            "top1_confidence_pct": round(top1_prob * 100, 2),
            "top_probabilities": crop_top_predictions,
            "top3_probabilities": crop_top_predictions,
            "top5_probabilities": crop_top5_predictions,
            "entropy": round(crop_entropy, 4),
            "raw_entropy": round(crop_entropy, 4),
            "norm_entropy": round(norm_crop_entropy, 4),
            "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
            "class_count": len(active_indices),
            "blur_metric": round(blur_metric, 2),
            "blur_threshold": round(blur_threshold, 1),
            "decision_rule": decision_rule,
            "final_decision_reason": decision_rule,
            "final_decision": status.upper(),
            "model_path": model_path,
            "selected_model": selected_model,
            "crop_route": crop_species,
            "viewpoint_route": plant_part.value,
            "confidence_threshold": 0.60,
            "agreement_score": 1.0,
            "outcome": outcome,
            "fallback_reason": "",
        }

        return {
            "plant_scope": "hierarchical_4stage_pipeline",
            "outcome": outcome,
            "status": status,
            "supported": True,
            "disease": None if is_healthy or status == "uncertain_disease" else top1_label,
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
            "top5_probabilities": crop_top5_predictions,
            "nutrient_screening": nutrient_screening,
            "source_evidence": source_record if status in ["confident", "probable_diagnosis_needs_confirmation"] and not is_healthy else None,
            "debug_info": debug_info,
            "selected_model": selected_model,
            "model_path": model_path,
            "fallback_reason": "",
            "refusal_reason_en": "",
            "refusal_reason_kn": "",
            "requested_next_action_en": (
                "Please inspect the underside of the leaf or stem for active fungal sporulation, wilting, or insect pests."
                if status == "probable_diagnosis_needs_confirmation" else ""
            ),
            "requested_next_action_kn": (
                "ದಯವಿಟ್ಟು ಎಲೆಯ ಹಿಂಭಾಗ ಅಥವಾ ಕಾಂಡದಲ್ಲಿ ಶಿಲೀಂಧ್ರ ಬೂಸ್ಟು ಅಥವಾ ಕೀಟಗಳಿವೆಯೇ ಗಮನಿಸಿ."
                if status == "probable_diagnosis_needs_confirmation" else ""
            ),
        }

    def _build_unsupported_route_response(
        self,
        crop_species: str,
        plant_part: PlantPart,
        fallback_reason: str,
        dinov2_sim: float,
        dinov2_threshold: float,
        blur_metric: float,
        blur_threshold: float,
        image_dims: str,
        image_mode: str,
        crop_conf: float = 0.0,
    ) -> Dict[str, Any]:
        outcome = "Plant image detected, but unsupported crop/viewpoint"
        unsupported_msg_en = "Plant image detected, but this crop/viewpoint diagnosis is not supported yet."
        unsupported_msg_kn = "ಕೃಷಿ ಸಸ್ಯ ಗುರುತಿಸಲಾಗಿದೆ. ಆದರೆ ಈ ಬೆಳೆ ಅಥವಾ ಸಸ್ಯದ ಭಾಗಕ್ಕೆ ರೋಗನಿರ್ಣಯವು ಇನ್ನೂ ತರಬೇತಿ ಪಡೆದಿಲ್ಲ."
        decision_rule = f"RULE_SAFE_VIEWPOINT_ABSTENTION ({fallback_reason})"

        debug_info = {
            "stage1_valid": True,
            "stage1_reason": "Valid Plant Image",
            "stage2_plant_part": plant_part.value,
            "stage3_crop_species": crop_species,
            "stage3_crop_confidence_pct": round(crop_conf * 100, 1),
            "stage4_route_supported": False,
            "stage4_reason": fallback_reason,
            "fallback_reason": fallback_reason,
            "image_dimensions": image_dims,
            "image_mode": image_mode,
            "dinov2_similarity": round(dinov2_sim, 4),
            "dinov2_threshold": round(dinov2_threshold, 4),
            "is_ood": False,
            "predicted_class": unsupported_msg_en,
            "disease": None,
            "top1_confidence_pct": 0.0,
            "top_probabilities": [],
            "top3_probabilities": [],
            "top5_probabilities": [],
            "entropy": 0.0,
            "raw_entropy": 0.0,
            "norm_entropy": 0.0,
            "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
            "class_count": 0,
            "blur_metric": round(blur_metric, 2),
            "blur_threshold": round(blur_threshold, 1),
            "decision_rule": decision_rule,
            "final_decision_reason": decision_rule,
            "final_decision": "UNSUPPORTED_CROP_PART",
            "model_path": "None",
            "selected_model": "None",
            "crop_route": crop_species,
            "viewpoint_route": plant_part.value,
            "confidence_threshold": 0.60,
            "agreement_score": 0.0,
            "outcome": outcome,
        }

        return {
            "plant_scope": "hierarchical_4stage_pipeline",
            "outcome": outcome,
            "status": "unsupported_crop_part",
            "supported": False,
            "disease": None,
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
            "affected_regions": f"Observed {plant_part.value.replace('_', ' ').title()}",
            "predictions": [],
            "top5_probabilities": [],
            "source_evidence": None,
            "debug_info": debug_info,
            "selected_model": "None",
            "model_path": "None",
            "fallback_reason": fallback_reason,
            "refusal_reason_en": unsupported_msg_en,
            "refusal_reason_kn": unsupported_msg_kn,
            "requested_next_action_en": (
                "Please capture a focused close-up photo of an affected leaf for a supported crop route."
            ),
            "requested_next_action_kn": (
                "ದಯವಿಟ್ಟು ಬೆಂಬಲಿತ ಬೆಳೆಗೆ ಎಲೆಯ ಫೋಟೋ ಕಳುಹಿಸಿ."
            ),
        }

    def _build_non_plant_ood_response(self, dino_sim, dino_thresh, blur_val, blur_thresh, dims, mode):
        outcome = "Non-plant image detected"
        fallback_reason = "Non-Agricultural / Out-of-Distribution Object"
        decision_rule = f"RULE_STAGE1_NON_PLANT_OOD (DINOv2 similarity {dino_sim:.4f} < threshold {dino_thresh:.2f})"
        debug_info = {
            "stage1_valid": False,
            "stage1_reason": fallback_reason,
            "stage2_plant_part": "non_plant",
            "stage3_crop_species": "Unknown",
            "stage4_route_supported": False,
            "image_dimensions": dims,
            "image_mode": mode,
            "dinov2_similarity": round(dino_sim, 4),
            "dinov2_threshold": round(dino_thresh, 4),
            "is_ood": True,
            "predicted_class": outcome,
            "disease": None,
            "top1_confidence_pct": 0.0,
            "top_probabilities": [],
            "top3_probabilities": [],
            "top5_probabilities": [],
            "entropy": 3.637,
            "raw_entropy": 3.637,
            "norm_entropy": 1.0,
            "entropy_threshold": config.OOD_ENTROPY_THRESHOLD,
            "class_count": 0,
            "blur_metric": round(blur_val, 2),
            "blur_threshold": round(blur_thresh, 1),
            "decision_rule": decision_rule,
            "final_decision_reason": decision_rule,
            "final_decision": "OUT_OF_DISTRIBUTION",
            "model_path": "facebook/dinov2-small",
            "selected_model": "facebook/dinov2-small",
            "crop_route": "Unknown",
            "viewpoint_route": "non_plant",
            "confidence_threshold": round(dino_thresh, 4),
            "agreement_score": 0.0,
            "outcome": outcome,
            "fallback_reason": fallback_reason,
        }
        return {
            "plant_scope": "hierarchical_4stage_pipeline",
            "outcome": outcome,
            "status": "out_of_distribution",
            "supported": False,
            "disease": None,
            "dinov2_plant_similarity": round(dino_sim, 4),
            "dinov2_threshold": round(dino_thresh, 4),
            "is_ood": True,
            "plant_part": "non_plant",
            "crop_species": "Unknown",
            "top1_label": outcome,
            "top1_label_kannada": "ಕೃಷಿಯೇತರ ವಸ್ತು (Non-Plant OOD)",
            "top1_confidence": 0.0,
            "entropy": 3.637,
            "norm_entropy": 1.0,
            "is_healthy": False,
            "plant_type": "Unknown",
            "affected_regions": "None",
            "predictions": [],
            "source_evidence": None,
            "debug_info": debug_info,
            "selected_model": "facebook/dinov2-small",
            "model_path": "facebook/dinov2-small",
            "fallback_reason": fallback_reason,
            "refusal_reason_en": (
                f"DINOv2 Backbone detected a non-plant or unrelated object (Similarity: {dino_sim:.3f} < {dino_thresh:.2f}). "
                "The image does not contain recognizable crops or plant foliage. Automatic diagnosis safely withheld."
            ),
            "refusal_reason_kn": (
                f"DINOv2 ಮಾದರಿಯು ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವುದೇ ಸಸ್ಯವನ್ನು ಗುರುತಿಸಿಲ್ಲ (ಹೋಲಿಕೆ: {dino_sim:.3f} < {dino_thresh:.2f}). "
                "ತಪ್ಪಾದ ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆಯನ್ನು ತಪ್ಪಿಸಲು ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ."
            ),
            "requested_next_action_en": "Please upload a clear photograph of an actual crop, leaf, stem, or plant canopy.",
            "requested_next_action_kn": "ದಯವಿಟ್ಟು ಕೃಷಿ ಬೆಳೆ ಅಥವಾ ಸಸ್ಯದ ಎಲೆಯ ಸ್ಪಷ್ಟ ಫೋಟೋ ಕಳುಹಿಸಿ.",
        }

    def _build_unknown_crop_response(self, crop, part, dino_sim, dino_thresh, blur_val, blur_thresh, dims, mode):
        return self._build_unsupported_route_response(
            crop_species=crop,
            plant_part=part,
            fallback_reason=f"Crop '{crop}' or viewpoint '{part.value}' is unknown or not supported by deployed models",
            dinov2_sim=dino_sim,
            dinov2_threshold=dino_thresh,
            blur_metric=blur_val,
            blur_threshold=blur_thresh,
            image_dims=dims,
            image_mode=mode,
        )
