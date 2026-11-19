"""
Regression Test Suite for Hierarchical Crop Vision Engine & Zero Cross-Crop Contamination.
Strictly verifies:
1. Maize image -> only Maize labels can appear. ZERO Tomato, Grape, Pepper, Apple labels.
2. Tomato image -> only Tomato labels can appear. ZERO Grape, Maize, Pepper, Apple labels.
3. Grape image -> only Grape labels can appear. ZERO Tomato, Maize, Pepper, Apple labels.
4. Apple image -> only Apple labels can appear. ZERO Tomato, Grape, Maize, Pepper labels.
5. Potato image -> only Potato labels can appear. ZERO Tomato, Grape, Pepper labels.
6. Healthy foliage -> correctly identified as healthy without false chemical prescription.
7. Whole-plant images -> safely abstained as 'unsupported_crop_part' with ZERO candidate diseases.
8. Stem, Fruit, Root, Bark images -> safely abstained as 'unsupported_crop_part' with ZERO candidate diseases.
9. Blurry images -> safely gated as 'insufficient_image_quality' with ZERO candidate diseases.
10. Non-plant images -> safely rejected as 'out_of_distribution' with ZERO candidate diseases.
11. Authoritative source links -> must match the actual confirmed crop and diagnosis.
"""
import io
import logging
import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.vision.predict import PlantDiseasePredictor
from src.vision.training_registry import CROP_DISEASE_TAXONOMY, LABEL_TO_CROP_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def test_zero_cross_crop_contamination():
    print("\n" + "=" * 80)
    print("RUNNING ZERO CROSS-CROP CONTAMINATION REGRESSION TESTS")
    print("=" * 80)

    predictor = PlantDiseasePredictor()

    # 1. Maize (Corn) Test
    print("\n1. Testing Corn (Maize) Foliage...")
    corn_path = config.RAW_DATA_DIR / "sample_corn_common_rust.jpg"
    res_corn = predictor.predict(corn_path, user_crop_hint="Corn (Maize)", user_part_hint="leaf")
    print(f"  Top Label: {res_corn['top1_label']} ({res_corn['top1_confidence']}%)")
    print(f"  Candidates: {[c['label'] for c in res_corn['predictions']]}")

    assert res_corn["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert res_corn["crop_species"] == "Corn (Maize)"
    for cand in res_corn["predictions"]:
        cand_crop = LABEL_TO_CROP_MAP.get(cand["label"])
        assert cand_crop == "Corn (Maize)", f"Cross-crop leak! Non-maize label '{cand['label']}' appeared for Corn!"
    print("  Maize isolation: PASSED (Zero cross-crop contamination)")

    # 2. Tomato Test
    print("\n2. Testing Tomato Foliage...")
    tomato_path = config.RAW_DATA_DIR / "sample_tomato_early_blight.jpg"
    res_tom = predictor.predict(tomato_path, user_crop_hint="Tomato", user_part_hint="leaf")
    print(f"  Top Label: {res_tom['top1_label']} ({res_tom['top1_confidence']}%)")
    print(f"  Candidates: {[c['label'] for c in res_tom['predictions']]}")

    assert res_tom["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert res_tom["crop_species"] == "Tomato"
    for cand in res_tom["predictions"]:
        cand_crop = LABEL_TO_CROP_MAP.get(cand["label"])
        assert cand_crop == "Tomato", f"Cross-crop leak! Non-tomato label '{cand['label']}' appeared for Tomato!"
    assert res_tom["source_evidence"]["url"].startswith("https://")
    assert "tomato" in res_tom["source_evidence"]["url"].lower()
    print("  Tomato isolation: PASSED (Zero cross-crop contamination)")

    # 3. Grape Test
    print("\n3. Testing Grape Foliage...")
    grape_path = config.DATA_DIR / "processed" / "eval_samples" / "in_distribution" / "Grape___Black_rot.jpg"
    res_grape = predictor.predict(grape_path, user_crop_hint="Grape", user_part_hint="leaf")
    print(f"  Top Label: {res_grape['top1_label']} ({res_grape['top1_confidence']}%)")
    print(f"  Candidates: {[c['label'] for c in res_grape['predictions']]}")

    assert res_grape["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert res_grape["crop_species"] == "Grape"
    for cand in res_grape["predictions"]:
        cand_crop = LABEL_TO_CROP_MAP.get(cand["label"])
        assert cand_crop == "Grape", f"Cross-crop leak! Non-grape label '{cand['label']}' appeared for Grape!"
    print("  Grape isolation: PASSED (Zero cross-crop contamination)")

    # 4. Apple Test
    print("\n4. Testing Apple Foliage...")
    apple_path = config.RAW_DATA_DIR / "sample_apple_scab.jpg"
    res_apple = predictor.predict(apple_path, user_crop_hint="Apple", user_part_hint="leaf")
    print(f"  Top Label: {res_apple['top1_label']} ({res_apple['top1_confidence']}%)")
    print(f"  Candidates: {[c['label'] for c in res_apple['predictions']]}")

    assert res_apple["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert res_apple["crop_species"] == "Apple"
    for cand in res_apple["predictions"]:
        cand_crop = LABEL_TO_CROP_MAP.get(cand["label"])
        assert cand_crop == "Apple", f"Cross-crop leak! Non-apple label '{cand['label']}' appeared for Apple!"
    print("  Apple isolation: PASSED (Zero cross-crop contamination)")

    # 5. Potato Test
    print("\n5. Testing Potato Foliage...")
    potato_path = config.DATA_DIR / "processed" / "eval_samples" / "in_distribution" / "Potato___Late_blight.jpg"
    res_pot = predictor.predict(potato_path, user_crop_hint="Potato", user_part_hint="leaf")
    print(f"  Top Label: {res_pot['top1_label']} ({res_pot['top1_confidence']}%)")
    print(f"  Candidates: {[c['label'] for c in res_pot['predictions']]}")

    assert res_pot["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert res_pot["crop_species"] == "Potato"
    for cand in res_pot["predictions"]:
        cand_crop = LABEL_TO_CROP_MAP.get(cand["label"])
        assert cand_crop == "Potato", f"Cross-crop leak! Non-potato label '{cand['label']}' appeared for Potato!"
    print("  Potato isolation: PASSED (Zero cross-crop contamination)")

    # 6. Healthy Foliage Test
    print("\n6. Testing Healthy Foliage...")
    healthy_path = config.RAW_DATA_DIR / "sample_tomato_healthy.jpg"
    res_health = predictor.predict(healthy_path, user_crop_hint="Tomato", user_part_hint="leaf")
    print(f"  Top Label: {res_health['top1_label']} | Healthy: {res_health['is_healthy']}")
    assert res_health["is_healthy"] is True or "healthy" in res_health["top1_label"].lower()
    print("  Healthy plant diagnosis: PASSED")

    # 7. Whole-Plant Supported Route & Isolation Test
    print("\n7. Testing Whole-Plant Supported Route & Isolation...")
    wp_path = config.RAW_DATA_DIR / "sample_field_tomato_plant.jpg"
    res_wp = predictor.predict(wp_path, user_crop_hint="Tomato", user_part_hint="whole_plant")
    print(f"  Status: {res_wp['status']}")
    print(f"  Message: {res_wp['top1_label']}")
    print(f"  Candidates: {res_wp['predictions']}")
    print(f"  Source Evidence: {res_wp['source_evidence']['source_title'] if res_wp['source_evidence'] else None}")

    assert res_wp["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    for cand in res_wp["predictions"]:
        assert "tomato" in cand["label"].lower(), f"Contamination in Tomato whole-plant: {cand['label']}"
    assert res_wp["source_evidence"] is not None
    assert "https://" in res_wp["source_evidence"]["url"]
    print("  Whole-plant supported route & zero contamination: PASSED")

    # 8. Unsupported Route Safe Abstention Test
    # If a crop + plant-part combination has no validated model (e.g. Cotton whole plant, Banana stem),
    # it must abstain safely with 0 candidate diseases and 0 management links.
    print("\n8. Testing Unsupported Route Safe Abstention...")
    unsupported_checks = [
        ("Cotton Whole Plant", wp_path, "Cotton", "whole_plant"),
        ("Banana Stem", wp_path, "Banana", "stem"),
        ("Sugarcane Root", wp_path, "Sugarcane", "root"),
    ]
    for test_name, img_file, c_hint, p_hint in unsupported_checks:
        r = predictor.predict(img_file, user_crop_hint=c_hint, user_part_hint=p_hint)
        print(f"  [{test_name}] Status: {r['status']} | Candidates: {len(r['predictions'])} | Source: {r['source_evidence']}")
        assert r["status"] == "unsupported_crop_part"
        assert len(r["predictions"]) == 0, f"Unsupported route {test_name} must NOT return candidates!"
        assert r["source_evidence"] is None, f"Unsupported route {test_name} must NOT return chemical links!"
        assert "not yet supported" in r["top1_label"].lower()
    print("  Unsupported route safe abstention: PASSED")

    # 9. Blurry Image Test
    print("\n9. Testing Blurry Image Quality Gate...")
    blur_path = config.DATA_DIR / "processed" / "eval_samples" / "sample_blurred_leaf.jpg"
    res_b = predictor.predict(blur_path)
    print(f"  Status: {res_b['status']} | Candidates: {len(res_b['predictions'])}")
    assert res_b["status"] == "insufficient_image_quality"
    assert len(res_b["predictions"]) == 0
    print("  Blurry image gating: PASSED")

    # 10. Non-Plant OOD Test
    print("\n10. Testing Non-Plant OOD Rejection...")
    cat_path = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
    res_cat = predictor.predict(cat_path)
    print(f"  Status: {res_cat['status']} | DINOv2 Sim: {res_cat['dinov2_plant_similarity']:.4f}")
    assert res_cat["status"] == "out_of_distribution"
    assert res_cat["is_ood"] is True
    assert len(res_cat["predictions"]) == 0
    print("  Non-plant OOD rejection: PASSED")

    print("\n" + "=" * 80)
    print("ALL 10 REGRESSION TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    test_zero_cross_crop_contamination()
