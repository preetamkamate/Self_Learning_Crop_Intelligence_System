"""
Whole-Plant Field Understanding Vision Predictor.
Combines:
1. DINOv2 Backbone (facebook/dinov2-small) for global structural analysis and non-plant OOD rejection.
2. Multi-Scale Whole-Plant Tiling (Full view + 4 overlapping quadrant crops).
3. Fine-tuned Plant Pathology Head (MobileNetV2) with Shannon entropy uncertainty estimation.
4. Spatial Agronomic Diagnosis identifying affected plant regions (canopy, lower foliage, stem).
"""
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

import config
from src.vision.model import get_vision_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DISEASE_TRANSLATIONS_KN = {
    "Apple Scab": "ಸೇಬಿನ ತುಪ್ಪಳ ರೋಗ (Apple Scab)",
    "Apple with Black Rot": "ಸೇಬಿನ ಕಪ್ಪು ಕೊಳೆ ರೋಗ (Apple Black Rot)",
    "Cedar Apple Rust": "ಸೇಬಿನ ತುಕ್ಕು ರೋಗ (Cedar Apple Rust)",
    "Healthy Apple": "ಆರೋಗ್ಯಕರ ಸೇಬಿನ ಗಿಡ (Healthy Apple)",
    "Healthy Blueberry Plant": "ಆರೋಗ್ಯಕರ ಬ್ಲೂಬೆರ್ರಿ ಗಿಡ (Healthy Blueberry)",
    "Cherry with Powdery Mildew": "ಚೆರ್ರಿ ಬೂದಿ ರೋಗ (Cherry Powdery Mildew)",
    "Healthy Cherry Plant": "ಆರೋಗ್ಯಕರ ಚೆರ್ರಿ ಗಿಡ (Healthy Cherry)",
    "Corn (Maize) with Cercospora and Gray Leaf Spot": "ಮೆಕ್ಕೆಜೋಳದ ಬೂದು ಎಲೆ ಚುಕ್ಕೆ ರೋಗ (Gray Leaf Spot)",
    "Corn (Maize) with Common Rust": "ಮೆಕ್ಕೆಜೋಳದ ಸಾಮಾನ್ಯ ತುಕ್ಕು ರೋಗ (Common Rust)",
    "Corn (Maize) with Northern Leaf Blight": "ಮೆಕ್ಕೆಜೋಳದ ಎಲೆ ಸುಟ್ಟ ರೋಗ (Northern Leaf Blight)",
    "Healthy Corn (Maize) Plant": "ಆರೋಗ್ಯಕರ ಮೆಕ್ಕೆಜೋಳದ ಬೆಳೆ (Healthy Corn/Maize)",
    "Grape with Black Rot": "ದ್ರಾಕ್ಷಿಯ ಕಪ್ಪು ಕೊಳೆ ರೋಗ (Grape Black Rot)",
    "Grape with Esca (Black Measles)": "ದ್ರಾಕ್ಷಿ ಎಸ್ಕ ರೋಗ (Black Measles)",
    "Grape with Isariopsis Leaf Spot": "ದ್ರಾಕ್ಷಿ ಎಲೆ ಚುಕ್ಕೆ ರೋಗ (Leaf Spot)",
    "Healthy Grape Plant": "ಆರೋಗ್ಯಕರ ದ್ರಾಕ್ಷಿ ಗಿಡ (Healthy Grape)",
    "Orange with Citrus Greening": "ಕಿತ್ತಳೆ ಹಸಿರು ಗ್ರೀನಿಂಗ್ ರೋಗ (Citrus Greening)",
    "Peach with Bacterial Spot": "ಪೀಚ್ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಚುಕ್ಕೆ ರೋಗ (Bacterial Spot)",
    "Healthy Peach Plant": "ಆರೋಗ್ಯಕರ ಪೀಚ್ ಗಿಡ (Healthy Peach)",
    "Bell Pepper with Bacterial Spot": "ದಪ್ಪಮೆಣಸಿನಕಾಯಿ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಚುಕ್ಕೆ (Bacterial Spot)",
    "Healthy Bell Pepper Plant": "ಆರೋಗ್ಯಕರ ದಪ್ಪಮೆಣಸಿನಕಾಯಿ ಗಿಡ (Healthy Bell Pepper)",
    "Potato with Early Blight": "ಆಲೂಗಡ್ಡೆ ಮುಂಚಿನ ಅಂಗಮಾರಿ ರೋಗ (Early Blight)",
    "Potato with Late Blight": "ಆಲೂಗಡ್ಡೆ ತಡವಾದ ಅಂಗಮಾರಿ ರೋಗ (Late Blight)",
    "Healthy Potato Plant": "ಆರೋಗ್ಯಕರ ಆಲೂಗಡ್ಡೆ ಗಿಡ (Healthy Potato)",
    "Healthy Raspberry Plant": "ಆರೋಗ್ಯಕರ ರಾಸ್ಪ್ಬೆರಿ ಗಿಡ (Healthy Raspberry)",
    "Healthy Soybean Plant": "ಆರೋಗ್ಯಕರ ಸೋಯಾಬೀನ್ ಗಿಡ (Healthy Soybean)",
    "Squash with Powdery Mildew": "ಕುಂಬಳ ಜಾತಿಯ ಬೂದಿ ರೋಗ (Powdery Mildew)",
    "Strawberry with Leaf Scorch": "ಸ್ಟ್ರಾಬೆರಿ ಎಲೆ ಸುಟ್ಟ ರೋಗ (Leaf Scorch)",
    "Healthy Strawberry Plant": "ಆರೋಗ್ಯಕರ ಸ್ಟ್ರಾಬೆರಿ ಗಿಡ (Healthy Strawberry)",
    "Tomato with Bacterial Spot": "ಟೊಮೇಟೊ ಬ್ಯಾಕ್ಟೀರಿಯಲ್ ಚುಕ್ಕೆ ರೋಗ (Bacterial Spot)",
    "Tomato with Early Blight": "ಟೊಮೇಟೊ ಮುಂಚಿನ ಅಂಗಮಾರಿ ರೋಗ (Early Blight)",
    "Tomato with Late Blight": "ಟೊಮೇಟೊ ತಡವಾದ ಅಂಗಮಾರಿ ರೋಗ (Late Blight)",
    "Tomato with Leaf Mold": "ಟೊಮೇಟೊ ಎಲೆ ಬೂಸ್ಟು ರೋಗ (Leaf Mold)",
    "Tomato with Septoria Leaf Spot": "ಟೊಮೇಟೊ ಸೆಪ್ಟೋರಿಯಾ ಎಲೆ ಚುಕ್ಕೆ ರೋಗ (Septoria Leaf Spot)",
    "Tomato with Spider Mites or Two-spotted Spider Mite": "ಟೊಮೇಟೊ ಕೆಂಪು ಜೇಡ ನುಸಿ ಬಾಧೆ (Spider Mites)",
    "Tomato with Target Spot": "ಟೊಮೇಟೊ ಟಾರ್ಗೆಟ್ ಚುಕ್ಕೆ ರೋಗ (Target Spot)",
    "Tomato Yellow Leaf Curl Virus": "ಟೊಮೇಟೊ ಎಲೆ ಮುದುರು ನಂಜು ರೋಗ (Yellow Leaf Curl Virus)",
    "Tomato Mosaic Virus": "ಟೊಮೇಟೊ ಮೊಸಾಯಿಕ್ ವೈರಸ್ (Mosaic Virus)",
    "Healthy Tomato Plant": "ಆರೋಗ್ಯಕರ ಟೊಮೇಟೊ ಗಿಡ (Healthy Tomato)",
}


def compute_shannon_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon Predictive Entropy H(p) = -sum(p_i * log(p_i))."""
    clipped = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    return float(entropy)


class PlantDiseasePredictor:
    """Whole-Plant Multi-Scale Inference Engine with DINOv2 OOD Gate."""

    def __init__(self):
        self.dinov2_wrapper, self.classifier_wrapper = get_vision_models()
        self.device = self.dinov2_wrapper.device
        self.id2label = self.classifier_wrapper.id2label
        self.label2id = self.classifier_wrapper.label2id

        # Initialize reference agricultural embeddings for DINOv2 OOD gate
        self.ref_embeddings = self._build_reference_embeddings()

    def _build_reference_embeddings(self) -> torch.Tensor:
        """Cache DINOv2 [CLS] embeddings of known agricultural and plant imagery."""
        ref_files = [
            config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg",
            config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg",
            config.RAW_DATA_DIR / "sample_apple_scab.jpg",
            config.RAW_DATA_DIR / "sample_corn_common_rust.jpg",
            config.RAW_DATA_DIR / "sample_tomato_healthy.jpg",
        ]

        embs = []
        for p in ref_files:
            if p.exists():
                try:
                    img = Image.open(p).convert("RGB")
                    inputs = self.dinov2_wrapper.processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        out = self.dinov2_wrapper.model(**inputs)
                        cls_tok = out.last_hidden_state[:, 0, :]
                        embs.append(F.normalize(cls_tok, dim=-1))
                except Exception as e:
                    logging.warning("Could not embed reference image %s: %s", p, e)

        if embs:
            return torch.cat(embs, dim=0)
        else:
            # Fallback zero-reference if images missing
            return torch.zeros((1, 384), device=self.device)

    def _check_dinov2_ood(self, img: Image.Image) -> Tuple[bool, float]:
        """
        Extract DINOv2 global embedding and test cosine similarity against agricultural references.
        Returns: (is_ood: bool, max_similarity: float)
        """
        inputs = self.dinov2_wrapper.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.dinov2_wrapper.model(**inputs)
            target_emb = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)

        sims = F.cosine_similarity(target_emb, self.ref_embeddings)
        max_sim = float(sims.max().item())

        is_ood = max_sim < config.DINOV2_PLANT_OOD_THRESHOLD
        return is_ood, max_sim

    def _generate_multi_scale_crops(self, img: Image.Image) -> Dict[str, Image.Image]:
        """
        Generate 5 views for whole-plant field inspection:
        1. Global Full-Plant view
        2. Top-Left quadrant (Upper canopy foliage left)
        3. Top-Right quadrant (Upper canopy foliage right)
        4. Bottom-Left quadrant (Lower foliage & stem base left)
        5. Bottom-Right quadrant (Lower foliage & stem base right)
        """
        w, h = img.size

        # 60% box dimensions with 20% central overlap
        box_w = int(w * 0.60)
        box_h = int(h * 0.60)

        views = {
            "Whole Canopy (Global)": img,
            "Upper Foliage (Top-Left)": img.crop((0, 0, box_w, box_h)),
            "Upper Foliage (Top-Right)": img.crop((w - box_w, 0, w, box_h)),
            "Lower Foliage/Stem (Bottom-Left)": img.crop((0, h - box_h, box_w, h)),
            "Lower Foliage/Stem (Bottom-Right)": img.crop((w - box_w, h - box_h, w, h)),
        }
        return views

    def _infer_view(self, view_img: Image.Image) -> np.ndarray:
        """Run MobileNetV2 classifier on a single region view."""
        inputs = self.classifier_wrapper.processor(images=view_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.classifier_wrapper.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return probs

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, bytes, io.BytesIO],
        top_k: int = config.VISION_TOP_K,
        ood_entropy_threshold: float = config.OOD_ENTROPY_THRESHOLD,
        min_confidence_threshold: float = config.MIN_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Whole-Plant Multimodal Field Understanding with DINOv2 OOD gate & Multi-Scale Inspection.
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Step 1: Whole-Plant Global Context & DINOv2 OOD Check
        is_ood, dinov2_sim = self._check_dinov2_ood(img)
        if is_ood:
            logging.info("DINOv2 OOD Gate Triggered: Similarity %.4f < %.2f", dinov2_sim, config.DINOV2_PLANT_OOD_THRESHOLD)
            return {
                "plant_scope": "whole_plant_field_analysis",
                "status": "out_of_distribution",
                "dinov2_plant_similarity": round(dinov2_sim, 4),
                "is_ood": True,
                "top1_label": "Non-Agricultural / Out-of-Distribution Object",
                "top1_label_kannada": "ಕೃಷಿಯೇತರ / ಅಸ್ಪಷ್ಟ ವಸ್ತು (Non-Plant OOD)",
                "top1_confidence": 0.0,
                "entropy": 3.0,
                "is_healthy": False,
                "plant_type": "Unknown",
                "affected_regions": "None (Non-Plant Object)",
                "predictions": [],
                "region_breakdown": [],
                "refusal_reason_en": (
                    f"DINOv2 Backbone detected a non-plant or unrelated object (Plant similarity: {dinov2_sim:.3f} < {config.DINOV2_PLANT_OOD_THRESHOLD}). "
                    "The image does not contain agricultural crops, canopy, or field foliage. "
                    "Automatic disease classification has been safely withheld. "
                    "Please upload a clear photograph of a field plant, crop canopy, or affected plant part."
                ),
                "refusal_reason_kn": (
                    f"DINOv2 ಮಾದರಿಯು ಈ ಚಿತ್ರದಲ್ಲಿ ಯಾವುದೇ ಬೆಳೆ ಅಥವಾ ಸಸ್ಯವನ್ನು ಗುರುತಿಸಿಲ್ಲ (ಹೋಲಿಕೆ: {dinov2_sim:.3f}). "
                    "ತಪ್ಪಾದ ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಯಲು ವಿಶ್ಲೇಷಣೆಯನ್ನು ನಿರಾಕರಿಸಲಾಗಿದೆ. "
                    "ದಯವಿಟ್ಟು ಕೃಷಿ ಬೆಳೆ, ಗಿಡದ ಕಾಂಡ, ಎಲೆ ಅಥವಾ ಹಣ್ಣಿನ ಸ್ಪಷ್ಟ ಫೋಟೋವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ."
                ),
            }

        # Step 2: Multi-Region Whole-Plant Inspection (5 Views)
        views = self._generate_multi_scale_crops(img)
        region_breakdown: List[Dict[str, Any]] = []
        all_probs = []

        upper_disease_signal = 0.0
        lower_disease_signal = 0.0
        max_region_name = "Whole Canopy"
        max_region_prob = 0.0

        for r_name, r_img in views.items():
            probs = self._infer_view(r_img)
            all_probs.append(probs)

            r_entropy = compute_shannon_entropy(probs)
            r_top_idx = np.argmax(probs)
            r_top_prob = float(probs[r_top_idx])
            r_top_label = self.id2label[str(r_top_idx) if str(r_top_idx) in self.id2label else r_top_idx]

            region_breakdown.append({
                "region": r_name,
                "top_label": r_top_label,
                "confidence_percent": round(r_top_prob * 100, 2),
                "entropy": round(r_entropy, 3),
            })

            # Track spatial disease signal (excluding healthy classes)
            is_r_disease = "healthy" not in r_top_label.lower()
            if is_r_disease:
                if "Upper" in r_name:
                    upper_disease_signal += r_top_prob
                elif "Lower" in r_name:
                    lower_disease_signal += r_top_prob

                if r_top_prob > max_region_prob:
                    max_region_prob = r_top_prob
                    max_region_name = r_name

        # Step 3: Aggregation & Spatial Reasoning
        probs_matrix = np.array(all_probs)  # shape (5, num_classes)
        mean_probs = np.mean(probs_matrix, axis=0)
        max_probs = np.max(probs_matrix, axis=0)

        # Weighted combination of regional max and consensus mean
        agg_probs = 0.55 * max_probs + 0.45 * mean_probs
        agg_probs = agg_probs / np.sum(agg_probs)

        # Shannon Predictive Entropy
        entropy = compute_shannon_entropy(agg_probs)

        # Top predictions
        top_indices = np.argsort(agg_probs)[::-1]
        top1_idx = top_indices[0]
        top1_prob = float(agg_probs[top1_idx])
        top1_label = self.id2label[str(top1_idx) if str(top1_idx) in self.id2label else top1_idx]
        is_healthy = "healthy" in top1_label.lower()
        plant_type = top1_label.split(" ")[0].replace("(", "").replace(")", "")

        # Spatial Agronomic Affected Region Determination
        if is_healthy:
            affected_regions = "Whole Plant Canopy (Healthy physiological state, no foliar lesions detected)"
            affected_regions_kn = "ಸಂಪೂರ್ಣ ಗಿಡದ ಮೇಲ್ಭಾಗ (ಆರೋಗ್ಯಕರ ಸಸಿ, ಯಾವುದೇ ಕಲೆಗಳಿಲ್ಲ)"
        else:
            if lower_disease_signal > (upper_disease_signal * 1.5):
                affected_regions = "Lower Foliage & Stem Base (Ground-adjacent leaves showing early stage infection)"
                affected_regions_kn = "ಕೆಳಭಾಗದ ಎಲೆಗಳು ಮತ್ತು ಕಾಂಡದ ಬುಡ (ನೆಲಕ್ಕೆ ಸಮೀಪವಿರುವ ಹಳೆಯ ಎಲೆಗಳು)"
            elif upper_disease_signal > (lower_disease_signal * 1.5):
                affected_regions = "Upper Canopy & New Foliage (Young shoots exhibiting stress or viral puckering)"
                affected_regions_kn = "ಮೇಲ್ಭಾಗದ ಚಿಗುರು ಎಲೆಗಳು ಮತ್ತು ತುದಿಯ ರೆಂಬೆಗಳು"
            elif max_region_prob > 0.85:
                affected_regions = f"Prominent localized infection ({max_region_name})"
                affected_regions_kn = f"ಸ್ಥಳೀಯ ತೀವ್ರ ಬಾಧೆ ({max_region_name})"
            else:
                affected_regions = "Whole Canopy (Generalized symptoms across multiple plant quadrants)"
                affected_regions_kn = "ಸಂಪೂರ್ಣ ಗಿಡದ ಮೇಲ್ಭಾಗ (ಹಲವು ಕಡೆಗಳಲ್ಲಿ ಹರಡಿರುವ ಲಕ್ಷಣಗಳು)"

        # Uncertainty check
        is_uncertain = (entropy > ood_entropy_threshold) or (top1_prob < min_confidence_threshold)
        status = "uncertain" if is_uncertain else "confident"

        # Format top-k predictions
        top_predictions: List[Dict[str, Any]] = []
        for i in range(min(top_k, len(top_indices))):
            idx = top_indices[i]
            label_str = self.id2label[str(idx) if str(idx) in self.id2label else idx]
            confidence = float(agg_probs[idx])
            top_predictions.append({
                "label": label_str,
                "label_kannada": DISEASE_TRANSLATIONS_KN.get(label_str, label_str),
                "probability": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 2),
            })

        refusal_en = ""
        refusal_kn = ""
        if is_uncertain:
            refusal_en = (
                f"Multi-region uncertainty detected (Entropy: {entropy:.3f} > {ood_entropy_threshold} or "
                f"Top-1 Confidence: {top1_prob*100:.1f}% < {min_confidence_threshold*100:.0f}%). "
                "Conflicting signals observed across plant quadrants. Refusing blind prescription."
            )
            refusal_kn = (
                f"ಗಿಡದ ವಿವಿಧ ಭಾಗಗಳಲ್ಲಿ ಅನಿಶ್ಚಿತತೆ ಕಂಡುಬಂದಿದೆ (Entropy: {entropy:.3f}, ವಿಶ್ವಾಸ: {top1_prob*100:.1f}%). "
                "ತಪ್ಪಾದ ಕೀಟನಾಶಕ ಶಿಫಾರಸನ್ನು ತಪ್ಪಿಸಲು ರೋಗನಿರ್ಣಯವನ್ನು ತಡೆಹಿಡಿಯಲಾಗಿದೆ."
            )

        return {
            "plant_scope": "whole_plant_field_analysis",
            "status": status,
            "dinov2_plant_similarity": round(dinov2_sim, 4),
            "entropy": round(entropy, 4),
            "entropy_threshold": ood_entropy_threshold,
            "uncertainty_flag": is_uncertain,
            "top1_label": top1_label,
            "top1_label_kannada": DISEASE_TRANSLATIONS_KN.get(top1_label, top1_label),
            "top1_confidence": round(top1_prob * 100, 2),
            "is_healthy": is_healthy,
            "plant_type": plant_type,
            "affected_regions": affected_regions,
            "affected_regions_kn": affected_regions_kn,
            "predictions": top_predictions,
            "region_breakdown": region_breakdown,
            "refusal_reason_en": refusal_en,
            "refusal_reason_kn": refusal_kn,
        }


if __name__ == "__main__":
    predictor = PlantDiseasePredictor()
    print("\n--- Testing Full-Plant Field Image ---")
    res1 = predictor.predict(config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg")
    print("Scope:", res1["plant_scope"])
    print("Status:", res1["status"])
    print("DINOv2 Sim:", res1["dinov2_plant_similarity"])
    print("Top1:", res1["top1_label"], f"({res1['top1_confidence']}%)")
    print("Affected Regions:", res1["affected_regions"])
    print("Entropy:", res1["entropy"])

    print("\n--- Testing Non-Plant Cat Image (DINOv2 OOD) ---")
    res2 = predictor.predict(config.RAW_DATA_DIR / "sample_non_plant_cat.png")
    print("Status:", res2["status"])
    print("DINOv2 Sim:", res2["dinov2_plant_similarity"])
    print("Top1:", res2["top1_label"])
    print("Refusal Reason:", res2["refusal_reason_en"])
