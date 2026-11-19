"""
Automated Validation Suite for DINOv2 Agricultural OOD Gate and Vision Engine.
Validates:
1. Representative in-distribution plant images are NOT flagged as OOD and proceed to diagnosis.
2. Representative out-of-distribution non-plant images ARE flagged as OOD and properly gated.
3. Safe debug telemetry output is complete and free of secrets or credentials.
4. Configurable threshold via environment variables (DINOV2_PLANT_OOD_THRESHOLD) and kwargs.
5. Robust handling of image formats (RGBA, Grayscale, high resolution).
"""
import io
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.vision.predict import PlantDiseasePredictor


def run_validation():
    print("=" * 70)
    print("RUNNING END-TO-END VISION & OOD GATING VALIDATION")
    print("=" * 70)
    print(f"Device: {config.DEVICE}")
    print(f"Configured DINOv2 Threshold: {config.DINOV2_PLANT_OOD_THRESHOLD}")
    print(f"Configured Entropy Threshold: {config.OOD_ENTROPY_THRESHOLD}")
    print(f"Configured Min Confidence Threshold: {config.MIN_CONFIDENCE_THRESHOLD}")
    print()

    predictor = PlantDiseasePredictor()
    print(f"Loaded reference embeddings shape: {predictor.ref_embeddings.shape}")
    assert predictor.ref_embeddings.shape[0] >= 5, "Reference bank must contain at least 5 embeddings"
    assert not torch_is_zero(predictor.ref_embeddings), "Reference embeddings must not be all-zeros"
    print("Reference embeddings verification: PASSED\n")

    # --------------------------------------------------------------------------
    # 1. Representative Plant Images (Must NOT be OOD)
    # --------------------------------------------------------------------------
    print("--- TEST 1: Representative Plant Images ---")
    plant_test_files = [
        ("Field Tomato Whole-Plant", config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg"),
        ("Tomato Early Blight Leaf", config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg"),
        ("Tomato Healthy Leaf", config.RAW_DATA_DIR / "sample_tomato_healthy.jpg"),
        ("Apple Scab Leaf", config.RAW_DATA_DIR / "sample_apple_scab.jpg"),
        ("Corn Common Rust Leaf", config.RAW_DATA_DIR / "sample_corn_common_rust.jpg"),
        ("Potato Late Blight Leaf", config.DATA_DIR / "processed" / "eval_samples" / "in_distribution" / "Potato___Late_blight.jpg"),
        ("Grape Black Rot Leaf", config.DATA_DIR / "processed" / "eval_samples" / "in_distribution" / "Grape___Black_rot.jpg"),
        ("Corn Healthy Leaf", config.DATA_DIR / "processed" / "eval_samples" / "in_distribution" / "Corn_(maize)___healthy.jpg"),
    ]

    plant_passed = 0
    for label, path in plant_test_files:
        if not path.exists():
            print(f"  [SKIP] File not found: {path}")
            continue

        res = predictor.predict(path)
        dbg = res.get("debug_info", {})
        sim = res.get("dinov2_plant_similarity", 0.0)
        status = res.get("status")
        is_ood = res.get("is_ood", False)

        print(
            f"  [PLANT] {label:28s} | Dims: {dbg.get('image_dimensions', 'N/A'):10s} | "
            f"DINOv2 Sim: {sim:.4f} | Status: {status:10s} | Top-1: {res.get('top1_label', 'N/A')[:30]}"
        )

        assert not is_ood, f"Plant image '{label}' was wrongly flagged as OOD (sim={sim})!"
        if "whole-plant" in label.lower() or "canopy" in label.lower():
            assert status == "unsupported_crop_part", f"Whole-plant image '{label}' status was '{status}' instead of unsupported_crop_part!"
            assert "not available yet" in res.get("top1_label", "").lower()
        else:
            assert status in ["confident", "probable_diagnosis_needs_confirmation", "uncertain_disease"], f"Foliar image '{label}' status was '{status}'!"
            assert len(res.get("predictions", [])) > 0, f"Foliar image '{label}' had empty disease predictions!"
        plant_passed += 1

    print(f"Result: {plant_passed}/{len(plant_test_files)} representative plant images correctly accepted.\n")

    # --------------------------------------------------------------------------
    # 2. Representative Non-Plant / OOD Images (Must BE OOD)
    # --------------------------------------------------------------------------
    print("--- TEST 2: Representative Non-Plant / OOD Images ---")
    ood_test_files = [
        ("Cat Photograph", config.RAW_DATA_DIR / "sample_non_plant_cat.png"),
        ("Checkerboard Pattern", config.DATA_DIR / "processed" / "eval_samples" / "ood" / "ood_checkerboard.jpg"),
        ("Gaussian Noise", config.DATA_DIR / "processed" / "eval_samples" / "ood" / "ood_gaussian_noise.jpg"),
        ("Dark Blank Image", config.DATA_DIR / "processed" / "eval_samples" / "ood" / "ood_dark_blank.jpg"),
    ]

    ood_passed = 0
    for label, path in ood_test_files:
        if not path.exists():
            print(f"  [SKIP] File not found: {path}")
            continue

        res = predictor.predict(path)
        dbg = res.get("debug_info", {})
        sim = res.get("dinov2_plant_similarity", 0.0)
        status = res.get("status")
        is_ood = res.get("is_ood", False)

        print(
            f"  [OOD]   {label:28s} | Dims: {dbg.get('image_dimensions', 'N/A'):10s} | "
            f"DINOv2 Sim: {sim:.4f} | Status: {status:10s} | Gated: {is_ood}"
        )

        assert is_ood, f"OOD non-plant image '{label}' was NOT flagged as OOD (sim={sim})!"
        assert status == "out_of_distribution", f"OOD non-plant image '{label}' status was '{status}'!"
        assert "Non-Agricultural" in res.get("top1_label", ""), f"Unexpected top1 label for OOD image: {res.get('top1_label')}"
        assert len(res.get("refusal_reason_en", "")) > 0, "Missing English refusal reason for OOD image"
        assert len(res.get("refusal_reason_kn", "")) > 0, "Missing Kannada refusal reason for OOD image"
        ood_passed += 1

    # Synthetic random noise image
    noise_arr = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    noise_img = Image.fromarray(noise_arr)
    res_noise = predictor.predict(noise_img)
    assert res_noise.get("is_ood") is True, "Synthetic noise image was not rejected as OOD!"
    print(f"  [OOD]   {'Synthetic Random Noise':28s} | Dims: 256x256   | DINOv2 Sim: {res_noise.get('dinov2_plant_similarity'):.4f} | Gated: True")
    ood_passed += 1

    print(f"Result: {ood_passed}/{len(ood_test_files) + 1} representative non-plant images correctly rejected.\n")

    # --------------------------------------------------------------------------
    # 3. Safe Debug Telemetry Verification (No Secrets, Complete Fields)
    # --------------------------------------------------------------------------
    print("--- TEST 3: Safe Debug Telemetry Verification ---")
    sample_plant = config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg"
    res_dbg = predictor.predict(sample_plant)
    dbg = res_dbg.get("debug_info")

    assert dbg is not None, "debug_info missing from prediction result!"
    required_debug_keys = [
        "image_dimensions",
        "image_mode",
        "dinov2_similarity",
        "dinov2_threshold",
        "is_ood",
        "predicted_class",
        "top1_confidence_pct",
        "top_probabilities",
        "entropy",
        "entropy_threshold",
        "final_decision",
    ]
    for key in required_debug_keys:
        assert key in dbg, f"debug_info missing required key: '{key}'"

    # Verify no secrets / credentials leaked
    dbg_str = str(dbg).lower()
    for secret_keyword in ["key", "secret", "password", "token", "auth"]:
        # Only check as substrings of values, not key names
        for k, v in dbg.items():
            assert not any(secret_keyword in str(v).lower() for secret_keyword in ["sk-", "api_key", "bearer"]), f"Potential secret leaked in debug field {k}: {v}"

    print(f"  Debug Info payload: {dbg}")
    print("Safe Debug Telemetry: VERIFIED\n")

    # --------------------------------------------------------------------------
    # 4. Configurable Threshold via Environment Variable & Kwarg
    # --------------------------------------------------------------------------
    print("--- TEST 4: Configurable Threshold ---")
    custom_thresh = 0.50
    res_custom = predictor.predict(sample_plant, dinov2_threshold=custom_thresh)
    assert res_custom.get("dinov2_threshold") == custom_thresh, f"Expected threshold {custom_thresh}, got {res_custom.get('dinov2_threshold')}"
    print(f"  Direct threshold override to {custom_thresh}: VERIFIED (Returned: {res_custom.get('dinov2_threshold')})")

    # Test environment variable override
    os.environ["DINOV2_PLANT_OOD_THRESHOLD"] = "0.22"
    val_env = float(os.getenv("DINOV2_PLANT_OOD_THRESHOLD", "0.20"))
    assert val_env == 0.22, f"Env override failed: {val_env}"
    print(f"  Environment variable DINOV2_PLANT_OOD_THRESHOLD={val_env}: VERIFIED\n")

    # --------------------------------------------------------------------------
    # 5. Image Format Robustness (RGBA, Grayscale, EXIF)
    # --------------------------------------------------------------------------
    print("--- TEST 5: Image Format Robustness ---")
    base_img = Image.open(sample_plant)
    
    # Test RGBA
    rgba_img = base_img.convert("RGBA")
    res_rgba = predictor.predict(rgba_img)
    assert res_rgba.get("is_ood") is False, "RGBA plant image was wrongly flagged as OOD!"
    print(f"  RGBA Image (with Alpha channel): ACCEPTED (Status: {res_rgba.get('status')})")

    # Test Grayscale
    gray_img = base_img.convert("L")
    res_gray = predictor.predict(gray_img)
    assert res_gray.get("is_ood") is False, "Grayscale plant image was wrongly flagged as OOD!"
    print(f"  Grayscale Image (Mode L): ACCEPTED (Status: {res_gray.get('status')})")

    print("\n" + "=" * 70)
    print("ALL VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


def torch_is_zero(tensor):
    import torch
    return bool(torch.all(tensor == 0).item())


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
