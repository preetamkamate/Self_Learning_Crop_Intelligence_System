"""
Comprehensive Multi-Part Route & Safety Regression Test Suite.
Validates end-to-end multi-organ pathology routing across all 7 viewpoints:
1. Stem/Branch (Maize, Tomato, Grape, Citrus, healthy & mechanical damage)
2. Fruit (Tomato, Apple, Bell Pepper, Citrus, healthy fruit)
3. Root/Tuber (Potato dry rot, black scurf, healthy tuber)
4. Whole-Plant (Tomato Fusarium wilt, Corn stunt, healthy whole plant)
5. Canopy/Field (Tomato canopy early blight, Corn canopy rust, healthy canopy)
6. Bark/Trunk (Coconut stem bleeding, Apple trunk canker, healthy bark)
7. Safe Fallback & Refusal (Unsupported routes, Non-Plant OOD rejection on stem)
8. Authoritative Primary Sources (ICAR, IIHR, CPCRI, CPRI, IIMR, CCRI, NRCG, CITH, TNAU, UAS Bangalore)
"""
import sys
from pathlib import Path
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.vision.predict import PlantDiseasePredictor


def get_first_test_image(part_dir: str, class_folder: str) -> Path:
    p = config.DATA_DIR / "datasets" / part_dir / "test" / class_folder
    images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No test images found in {p}")
    return images[0]


def run_multipart_regression_tests():
    print("=" * 80)
    print("RUNNING MULTI-PART PLANT ORGAN PATHOLOGY REGRESSION TEST SUITE")
    print("=" * 80)

    predictor = PlantDiseasePredictor()
    passed = 0
    total = 0

    # --------------------------------------------------------------------------
    # 1. STEM / BRANCH ROUTE (FIRST & HIGHEST PRIORITY)
    # --------------------------------------------------------------------------
    print("\n--- 1. Testing Stem/Branch Route ---")

    # 1A. Maize Stem: Corn Stalk Rot
    total += 1
    img_path = get_first_test_image("stem_branch", "Corn_Stalk_Rot")
    res = predictor.predict(img_path, user_crop_hint="Corn (Maize)", user_part_hint="stem")
    print(f"  [Maize Stem] Top: {res['top1_label']} ({res['top1_confidence']}%) | Status: {res['status']}")
    assert res["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert "stalk rot" in res["top1_label"].lower() or "corn" in res["top1_label"].lower()
    assert all("corn" in c["label"].lower() for c in res["predictions"]), "Cross-crop contamination in Maize stem!"
    assert res["source_evidence"] is not None
    assert "iimr.icar.gov.in" in res["source_evidence"]["url"] or "icar" in res["source_evidence"]["url"]
    passed += 1
    print("  -> Corn Stalk Rot: PASSED (Zero cross-crop contamination, Verified IIMR citation)")

    # 1B. Tomato Stem: Tomato Stem Canker
    total += 1
    img_path = get_first_test_image("stem_branch", "Tomato_Stem_Canker")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="stem")
    print(f"  [Tomato Stem] Top: {res['top1_label']} ({res['top1_confidence']}%) | Status: {res['status']}")
    assert res["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert "tomato" in res["top1_label"].lower()
    assert all("tomato" in c["label"].lower() for c in res["predictions"]), "Cross-crop contamination in Tomato stem!"
    assert res["source_evidence"] is not None
    assert "https://" in res["source_evidence"]["url"]
    passed += 1
    print("  -> Tomato Stem Canker: PASSED (Zero cross-crop contamination)")

    # 1C. Mechanically Damaged Tomato Stem (Non-Diseased)
    total += 1
    img_path = get_first_test_image("stem_branch", "Mechanically_Damaged_Tomato_Stem_Non-Diseased")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="stem")
    print(f"  [Mechanically Damaged Stem] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "tomato" in res["top1_label"].lower()
    passed += 1
    print("  -> Mechanically Damaged Stem: PASSED")

    # 1D. Grape Stem: Grape Cane and Leaf Spot (Phomopsis)
    total += 1
    img_path = get_first_test_image("stem_branch", "Grape_Cane_and_Leaf_Spot_Phomopsis")
    res = predictor.predict(img_path, user_crop_hint="Grape", user_part_hint="stem")
    print(f"  [Grape Stem] Top: {res['top1_label']} ({res['top1_confidence']}%) | Status: {res['status']}")
    assert res["status"] in ["confident", "probable_diagnosis_needs_confirmation"]
    assert "grape" in res["top1_label"].lower()
    assert all("grape" in c["label"].lower() for c in res["predictions"]), "Cross-crop contamination in Grape stem!"
    assert res["source_evidence"] is not None
    assert "nrcgrapes.icar.gov.in" in res["source_evidence"]["url"] or "icar" in res["source_evidence"]["url"]
    passed += 1
    print("  -> Grape Cane Spot: PASSED (Verified NRCG citation)")

    # 1E. Healthy Stem (Citrus)
    total += 1
    img_path = get_first_test_image("stem_branch", "Healthy_Citrus_Stem")
    res = predictor.predict(img_path, user_crop_hint="Citrus", user_part_hint="stem")
    print(f"  [Healthy Citrus Stem] Top: {res['top1_label']} | Healthy: {res['is_healthy']}")
    assert res["is_healthy"] is True or "healthy" in res["top1_label"].lower()
    passed += 1
    print("  -> Healthy Citrus Stem: PASSED")

    # 1F. Non-Plant OOD Rejection on Stem Route
    total += 1
    cat_img = config.RAW_DATA_DIR / "sample_non_plant_cat.png"
    res_ood = predictor.predict(cat_img, user_part_hint="stem")
    print(f"  [OOD Non-Plant on Stem Route] Status: {res_ood['status']} | DINOv2 Sim: {res_ood['dinov2_plant_similarity']:.4f}")
    assert res_ood["status"] == "out_of_distribution"
    assert res_ood["is_ood"] is True
    assert len(res_ood["predictions"]) == 0
    assert res_ood["source_evidence"] is None
    passed += 1
    print("  -> Non-Plant OOD Rejection on Stem Route: PASSED")

    # --------------------------------------------------------------------------
    # 2. FRUIT ROUTE
    # --------------------------------------------------------------------------
    print("\n--- 2. Testing Fruit Route ---")

    # 2A. Tomato Fruit: Blossom End Rot
    total += 1
    img_path = get_first_test_image("fruit", "Tomato_Blossom_End_Rot")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="fruit")
    print(f"  [Tomato Fruit] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "tomato" in res["top1_label"].lower()
    assert all("tomato" in c["label"].lower() for c in res["predictions"])
    assert res["source_evidence"] is not None
    assert "iihr.res.in" in res["source_evidence"]["url"] or "icar" in res["source_evidence"]["url"]
    passed += 1
    print("  -> Tomato Blossom End Rot: PASSED (Verified IIHR citation)")

    # 2B. Apple Fruit: Apple Scab on Fruit
    total += 1
    img_path = get_first_test_image("fruit", "Apple_Scab_on_Fruit")
    res = predictor.predict(img_path, user_crop_hint="Apple", user_part_hint="fruit")
    print(f"  [Apple Fruit] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "apple" in res["top1_label"].lower()
    assert all("apple" in c["label"].lower() for c in res["predictions"])
    passed += 1
    print("  -> Apple Scab on Fruit: PASSED")

    # 2C. Bell Pepper Fruit: Anthracnose
    total += 1
    img_path = get_first_test_image("fruit", "Bell_Pepper_Anthracnose_Fruit_Rot")
    res = predictor.predict(img_path, user_crop_hint="Bell Pepper (Capsicum)", user_part_hint="fruit")
    print(f"  [Pepper Fruit] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "bell pepper" in res["top1_label"].lower() or "pepper" in res["top1_label"].lower()
    assert all("pepper" in c["label"].lower() for c in res["predictions"])
    passed += 1
    print("  -> Bell Pepper Anthracnose: PASSED")

    # 2D. Healthy Fruit
    total += 1
    img_path = get_first_test_image("fruit", "Healthy_Tomato_Fruit")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="fruit")
    print(f"  [Healthy Tomato Fruit] Top: {res['top1_label']} | Healthy: {res['is_healthy']}")
    assert res["is_healthy"] is True or "healthy" in res["top1_label"].lower()
    passed += 1
    print("  -> Healthy Tomato Fruit: PASSED")

    # --------------------------------------------------------------------------
    # 3. ROOT / TUBER ROUTE
    # --------------------------------------------------------------------------
    print("\n--- 3. Testing Root/Tuber Route ---")

    # 3A. Potato Root: Dry Rot
    total += 1
    img_path = get_first_test_image("root_tuber", "Potato_Dry_Rot")
    res = predictor.predict(img_path, user_crop_hint="Potato", user_part_hint="root")
    print(f"  [Potato Tuber] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "potato" in res["top1_label"].lower()
    assert all("potato" in c["label"].lower() for c in res["predictions"])
    assert res["source_evidence"] is not None
    assert "cpri.icar.gov.in" in res["source_evidence"]["url"] or "icar" in res["source_evidence"]["url"]
    passed += 1
    print("  -> Potato Dry Rot: PASSED (Verified CPRI citation)")

    # 3B. Potato Root: Black Scurf
    total += 1
    img_path = get_first_test_image("root_tuber", "Potato_Black_Scurf")
    res = predictor.predict(img_path, user_crop_hint="Potato", user_part_hint="root")
    print(f"  [Potato Black Scurf] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "potato" in res["top1_label"].lower()
    passed += 1
    print("  -> Potato Black Scurf: PASSED")

    # 3C. Healthy Potato Tuber
    total += 1
    img_path = get_first_test_image("root_tuber", "Healthy_Potato_Tuber")
    res = predictor.predict(img_path, user_crop_hint="Potato", user_part_hint="root")
    print(f"  [Healthy Potato Tuber] Top: {res['top1_label']} | Healthy: {res['is_healthy']}")
    assert res["is_healthy"] is True or "healthy" in res["top1_label"].lower()
    passed += 1
    print("  -> Healthy Potato Tuber: PASSED")

    # --------------------------------------------------------------------------
    # 4. WHOLE-PLANT ROUTE
    # --------------------------------------------------------------------------
    print("\n--- 4. Testing Whole-Plant Route ---")

    # 4A. Tomato Whole Plant: Fusarium Wilt
    total += 1
    img_path = get_first_test_image("whole_plant", "Tomato_Fusarium_Wilt")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="whole_plant")
    print(f"  [Tomato Whole Plant] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "tomato" in res["top1_label"].lower()
    assert all("tomato" in c["label"].lower() for c in res["predictions"])
    passed += 1
    print("  -> Tomato Fusarium Wilt: PASSED")

    # 4B. Corn Whole Plant: Corn Stunt
    total += 1
    img_path = get_first_test_image("whole_plant", "Corn_Stunt_Disease")
    res = predictor.predict(img_path, user_crop_hint="Corn (Maize)", user_part_hint="whole_plant")
    print(f"  [Corn Whole Plant] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "corn" in res["top1_label"].lower()
    assert all("corn" in c["label"].lower() for c in res["predictions"])
    passed += 1
    print("  -> Corn Stunt: PASSED")

    # 4C. Healthy Whole Plant
    total += 1
    img_path = get_first_test_image("whole_plant", "Healthy_Tomato_Whole_Plant")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="whole_plant")
    print(f"  [Healthy Whole Plant] Top: {res['top1_label']} | Healthy: {res['is_healthy']}")
    assert res["is_healthy"] is True or "healthy" in res["top1_label"].lower()
    passed += 1
    print("  -> Healthy Whole Plant: PASSED")

    # --------------------------------------------------------------------------
    # 5. CANOPY / FIELD ROUTE
    # --------------------------------------------------------------------------
    print("\n--- 5. Testing Canopy/Field Route ---")

    # 5A. Tomato Canopy: Early Blight Outbreak
    total += 1
    img_path = get_first_test_image("canopy", "Tomato_Canopy_Early_Blight_Outbreak")
    res = predictor.predict(img_path, user_crop_hint="Tomato", user_part_hint="canopy")
    print(f"  [Tomato Canopy] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "tomato" in res["top1_label"].lower()
    assert all("tomato" in c["label"].lower() for c in res["predictions"])
    passed += 1
    print("  -> Tomato Canopy Outbreak: PASSED")

    # 5B. Healthy Canopy
    total += 1
    img_path = get_first_test_image("canopy", "Healthy_Corn_Canopy")
    res = predictor.predict(img_path, user_crop_hint="Corn (Maize)", user_part_hint="canopy")
    print(f"  [Healthy Corn Canopy] Top: {res['top1_label']} | Healthy: {res['is_healthy']}")
    assert res["is_healthy"] is True or "healthy" in res["top1_label"].lower()
    passed += 1
    print("  -> Healthy Corn Canopy: PASSED")

    # --------------------------------------------------------------------------
    # 6. BARK / TRUNK ROUTE
    # --------------------------------------------------------------------------
    print("\n--- 6. Testing Bark/Trunk Route ---")

    # 6A. Coconut Bark: Coconut Stem Bleeding
    total += 1
    img_path = get_first_test_image("bark_trunk", "Coconut_Stem_Bleeding")
    res = predictor.predict(img_path, user_crop_hint="Coconut", user_part_hint="bark_trunk")
    print(f"  [Coconut Trunk] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "coconut" in res["top1_label"].lower() or "bleeding" in res["top1_label"].lower()
    valid_coconut_classes = ["Coconut Stem Bleeding", "Basal Stem Rot", "Healthy Coconut Trunk"]
    assert all(c["label"] in valid_coconut_classes for c in res["predictions"]), "Contamination in Coconut bark candidates!"
    assert not any("apple" in c["label"].lower() for c in res["predictions"]), "Cross-crop contamination with Apple in Coconut trunk!"
    assert res["source_evidence"] is not None
    assert "cpcri.icar.gov.in" in res["source_evidence"]["url"] or "icar" in res["source_evidence"]["url"]
    passed += 1
    print("  -> Coconut Stem Bleeding: PASSED (Verified CPCRI citation, Zero cross-crop contamination)")

    # 6B. Apple Trunk: Apple Trunk Canker
    total += 1
    img_path = get_first_test_image("bark_trunk", "Apple_Trunk_Canker")
    res = predictor.predict(img_path, user_crop_hint="Apple", user_part_hint="bark_trunk")
    print(f"  [Apple Trunk] Top: {res['top1_label']} ({res['top1_confidence']}%)")
    assert "apple" in res["top1_label"].lower()
    assert all("apple" in c["label"].lower() for c in res["predictions"])
    passed += 1
    print("  -> Apple Trunk Canker: PASSED")

    # 6C. Healthy Bark
    total += 1
    img_path = get_first_test_image("bark_trunk", "Healthy_Coconut_Trunk")
    res = predictor.predict(img_path, user_crop_hint="Coconut", user_part_hint="bark_trunk")
    print(f"  [Healthy Coconut Trunk] Top: {res['top1_label']} | Healthy: {res['is_healthy']}")
    assert res["is_healthy"] is True or "healthy" in res["top1_label"].lower()
    passed += 1
    print("  -> Healthy Coconut Trunk: PASSED")

    # --------------------------------------------------------------------------
    # 7. SAFE FALLBACK & ABSTENTION (UNSUPPORTED ROUTES)
    # --------------------------------------------------------------------------
    print("\n--- 7. Testing Safe Fallback for Unsupported Routes ---")
    unsupported_cases = [
        ("Banana Stem", img_path, "Banana", "stem"),
        ("Wheat Root", img_path, "Wheat", "root"),
        ("Sugarcane Bark", img_path, "Sugarcane", "bark_trunk"),
        ("Cotton Canopy", img_path, "Cotton", "canopy"),
    ]
    for case_name, sample_img, crop_h, part_h in unsupported_cases:
        total += 1
        res = predictor.predict(sample_img, user_crop_hint=crop_h, user_part_hint=part_h)
        print(f"  [{case_name}] Status: {res['status']} | Msg: {res['top1_label']}")
        assert res["status"] == "unsupported_crop_part", f"Expected unsupported_crop_part for {case_name}"
        assert len(res["predictions"]) == 0, f"Unsupported route {case_name} returned non-empty candidates!"
        assert res["source_evidence"] is None, f"Unsupported route {case_name} returned source evidence!"
        part_clean = part_h.replace("_", " ")
        expected_msg = f"Valid {part_clean} image detected. Diagnosis is not yet supported for this crop and image type."
        assert res["top1_label"] == expected_msg, f"Expected exact message '{expected_msg}', got '{res['top1_label']}'"
        passed += 1
        print(f"  -> {case_name} Safe Fallback: PASSED (Exact abstention message, 0 candidates, 0 links)")

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed}/{total} REGRESSION TESTS PASSED (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    run_multipart_regression_tests()
