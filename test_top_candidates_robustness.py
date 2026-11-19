"""
Automated Test Suite for Top Candidate Prediction Robustness and Streamlit UI Safety.
Tests:
1. Candidate is None (single item and list).
2. Missing 'prob' key or null probability value.
3. String probability ('0.75', 'high', empty string).
4. Out-of-range probability (> 1.0, < 0.0, NaN, Inf).
5. Valid float probability (0.0 to 1.0).
6. Streamlit UI rendering logic verification (never crashes, proper formatting).
7. Development/debug mode logging vs production mode suppression.
"""
import logging
import math
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vision.hierarchical_classifier import build_top_candidates, normalize_candidate


def simulate_streamlit_caption(t: dict) -> str:
    """Simulates the exact rendering logic in interface/app.py lines 457-468."""
    label = t.get("class") or t.get("label") or "Unknown candidate"
    raw_prob = t.get("prob") if t.get("prob") is not None else t.get("probability")

    try:
        prob = float(raw_prob)
    except (TypeError, ValueError):
        prob = None

    if prob is not None and not math.isnan(prob) and 0.0 <= prob <= 1.0:
        return f"- `{label}`: **{prob * 100:.2f}%**"
    else:
        return f"- `{label}`: confidence unavailable"


def test_none_candidate():
    print("--- 1. Testing None Candidate ---")
    # Single None candidate
    cand_none = normalize_candidate(None, debug_mode=False)
    assert cand_none["class"] == "Unknown candidate"
    assert cand_none["prob"] == 0.0
    assert "class" in cand_none and "prob" in cand_none

    # None passed to build_top_candidates
    empty_res = build_top_candidates(None, debug_mode=False)
    assert empty_res == []

    # List containing None
    list_with_none = build_top_candidates([None], debug_mode=False)
    assert len(list_with_none) == 1
    assert list_with_none[0]["class"] == "Unknown candidate"
    assert list_with_none[0]["prob"] == 0.0

    # UI caption simulation
    cap = simulate_streamlit_caption(cand_none)
    assert "Unknown candidate" in cap
    print(f"  None candidate normalized: {cand_none} -> UI: {cap}")
    print("  -> None Candidate: PASSED\n")


def test_missing_prob():
    print("--- 2. Testing Missing 'prob' Key ---")
    # Dict with class only
    cand_no_prob = normalize_candidate({"class": "Tomato with Early Blight"}, debug_mode=False)
    assert cand_no_prob["class"] == "Tomato with Early Blight"
    assert cand_no_prob["prob"] == 0.0
    assert isinstance(cand_no_prob["prob"], float)

    # Dict with label only (legacy)
    cand_legacy = normalize_candidate({"label": "Apple Scab"}, debug_mode=False)
    assert cand_legacy["class"] == "Apple Scab"
    assert cand_legacy["prob"] == 0.0

    # Dict with explicit None prob
    cand_null_prob = normalize_candidate({"class": "Corn Common Rust", "prob": None}, debug_mode=False)
    assert cand_null_prob["class"] == "Corn Common Rust"
    assert cand_null_prob["prob"] == 0.0

    # UI caption simulation
    cap1 = simulate_streamlit_caption({"class": "Tomato with Early Blight"})
    assert "confidence unavailable" in cap1
    cap2 = simulate_streamlit_caption(cand_no_prob)
    assert "**0.00%**" in cap2  # Normalized candidate has valid float 0.0

    print(f"  Missing prob normalized: {cand_no_prob}")
    print("  -> Missing 'prob': PASSED\n")


def test_string_probability():
    print("--- 3. Testing String Probability ---")
    # Valid numeric string
    cand_num_str = normalize_candidate({"class": "Tomato Early Blight", "prob": "0.8542"}, debug_mode=False)
    assert cand_num_str["class"] == "Tomato Early Blight"
    assert cand_num_str["prob"] == 0.8542
    cap_num = simulate_streamlit_caption(cand_num_str)
    assert "**85.42%**" in cap_num

    # String with whitespace
    cand_ws = normalize_candidate({"class": "Apple Scab", "prob": " 0.60 "}, debug_mode=False)
    assert cand_ws["prob"] == 0.60

    # Non-numeric string (e.g. "high")
    cand_invalid_str = normalize_candidate({"class": "Corn Rust", "prob": "high_confidence"}, debug_mode=False)
    assert cand_invalid_str["class"] == "Corn Rust"
    assert cand_invalid_str["prob"] == 0.0

    # Empty string
    cand_empty_str = normalize_candidate({"class": "Potato Late Blight", "prob": ""}, debug_mode=False)
    assert cand_empty_str["prob"] == 0.0

    # Raw UI test with non-numeric string directly
    cap_raw_invalid = simulate_streamlit_caption({"class": "Corn Rust", "prob": "high_confidence"})
    assert "confidence unavailable" in cap_raw_invalid

    print(f"  Valid string '0.8542' -> {cand_num_str['prob']} | Invalid string 'high' -> {cand_invalid_str['prob']}")
    print("  -> String Probability: PASSED\n")


def test_out_of_range_probability():
    print("--- 4. Testing Out-of-Range Probability ---")
    # Greater than 1.0 (e.g. 1.5, 999.0)
    cand_high = normalize_candidate({"class": "Tomato Wilt", "prob": 1.5}, debug_mode=False)
    assert cand_high["class"] == "Tomato Wilt"
    assert cand_high["prob"] == 0.0  # Clamped/reset to 0.0

    # Negative probability (e.g. -0.25)
    cand_neg = normalize_candidate({"class": "Apple Scab", "prob": -0.25}, debug_mode=False)
    assert cand_neg["class"] == "Apple Scab"
    assert cand_neg["prob"] == 0.0

    # Float Inf and NaN
    cand_nan = normalize_candidate({"class": "Corn Rust", "prob": float("nan")}, debug_mode=False)
    assert cand_nan["prob"] == 0.0

    cand_inf = normalize_candidate({"class": "Grape Rot", "prob": float("inf")}, debug_mode=False)
    assert cand_inf["prob"] == 0.0

    # Raw UI tests with out of range directly
    cap_high = simulate_streamlit_caption({"class": "Tomato Wilt", "prob": 1.5})
    assert "confidence unavailable" in cap_high
    cap_neg = simulate_streamlit_caption({"class": "Apple Scab", "prob": -0.25})
    assert "confidence unavailable" in cap_neg

    print(f"  prob=1.5 normalized to: {cand_high['prob']} | prob=-0.25 normalized to: {cand_neg['prob']}")
    print("  -> Out-of-Range Probability: PASSED\n")


def test_valid_float_probability():
    print("--- 5. Testing Valid Float Probability ---")
    # Standard valid float
    cand_valid = normalize_candidate({"class": "Grape with Black Rot", "prob": 0.9543}, debug_mode=False)
    assert cand_valid["class"] == "Grape with Black Rot"
    assert cand_valid["prob"] == 0.9543
    assert cand_valid["confidence_percent"] == 95.43
    cap_valid = simulate_streamlit_caption(cand_valid)
    assert "**95.43%**" in cap_valid

    # Exact boundary 0.0
    cand_zero = normalize_candidate({"class": "Healthy Plant", "prob": 0.0}, debug_mode=False)
    assert cand_zero["prob"] == 0.0
    cap_zero = simulate_streamlit_caption(cand_zero)
    assert "**0.00%**" in cap_zero

    # Exact boundary 1.0
    cand_one = normalize_candidate({"class": "Healthy Plant", "prob": 1.0}, debug_mode=False)
    assert cand_one["prob"] == 1.0
    cap_one = simulate_streamlit_caption(cand_one)
    assert "**100.00%**" in cap_one

    print(f"  Valid float 0.9543 -> {cand_valid} -> UI: {cap_valid}")
    print("  -> Valid Float Probability: PASSED\n")


def test_build_top_candidates_batch():
    print("--- 6. Testing Batch build_top_candidates ---")
    raw_batch = [
        {"class": "Tomato Early Blight", "prob": 0.85},
        None,
        {"class": "Tomato Septoria", "prob": "0.10"},
        {"class": "Tomato Late Blight", "prob": 2.5},  # Out of range
        {"class": "Malformed"},                        # Missing prob
    ]

    clean_batch = build_top_candidates(raw_batch, top_k=5, debug_mode=False)
    assert len(clean_batch) == 5
    for item in clean_batch:
        assert isinstance(item, dict)
        assert "class" in item and isinstance(item["class"], str)
        assert "prob" in item and isinstance(item["prob"], float)
        assert 0.0 <= item["prob"] <= 1.0

        # Verify UI simulation never crashes on any item in batch
        cap = simulate_streamlit_caption(item)
        assert isinstance(cap, str) and len(cap) > 0

    print(f"  Processed {len(clean_batch)} mixed candidates without error.")
    print("  -> Batch Normalization: PASSED\n")


def test_debug_logging(caplog=None):
    print("--- 7. Testing Debug vs Production Logging ---")
    log_capture = []
    
    class TestLogHandler(logging.Handler):
        def emit(self, record):
            log_capture.append(record.getMessage())

    handler = TestLogHandler()
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    try:
        # Production mode: debug_mode=False (must NOT log)
        log_capture.clear()
        normalize_candidate(None, debug_mode=False)
        normalize_candidate({"class": "A", "prob": "invalid"}, debug_mode=False)
        assert len(log_capture) == 0, f"Expected 0 logs in production mode, got {len(log_capture)}"

        # Debug mode: debug_mode=True (MUST log malformed candidates)
        log_capture.clear()
        normalize_candidate(None, debug_mode=True)
        assert len(log_capture) == 1
        assert "Malformed candidate" in log_capture[0]

        log_capture.clear()
        normalize_candidate({"class": "A", "prob": "invalid"}, debug_mode=True)
        assert len(log_capture) == 1
        assert "Non-numeric probability" in log_capture[0]

        log_capture.clear()
        normalize_candidate({"class": "A", "prob": 5.0}, debug_mode=True)
        assert len(log_capture) == 1
        assert "out-of-range" in log_capture[0]

        # Valid candidate in debug mode (must NOT log)
        log_capture.clear()
        normalize_candidate({"class": "A", "prob": 0.8}, debug_mode=True)
        assert len(log_capture) == 0, "Valid candidate should not log warnings"

        print("  Production mode silence and Debug mode logging verified.")
        print("  -> Debug Logging Behavior: PASSED\n")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def run_all_tests():
    print("=" * 70)
    print("RUNNING TOP CANDIDATES ROBUSTNESS & UI RESILIENCE TEST SUITE")
    print("=" * 70)
    test_none_candidate()
    test_missing_prob()
    test_string_probability()
    test_out_of_range_probability()
    test_valid_float_probability()
    test_build_top_candidates_batch()
    test_debug_logging()
    print("=" * 70)
    print("ALL 7 TOP CANDIDATE ROBUSTNESS TESTS PASSED SUCCESSFULLY (100% SUCCESS)!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
