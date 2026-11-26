"""
Unit tests for Startup Artifact Checks and Graceful Degradation.
Verifies:
1. Complete artifact verification on production repository.
2. Graceful degradation when artifacts are missing (zero uncaught exceptions).
3. Correct Model-Release Policy metadata (58.33% held-out accuracy, experimental tier).
"""
from pathlib import Path
import pytest

from src.common.startup_checks import run_startup_artifact_checks


def test_startup_checks_on_active_repository():
    """Verify that startup checks successfully inspect active models and manifests."""
    report = run_startup_artifact_checks()

    assert report.timestamp is not None
    assert report.overall_status in ["production_ready_experimental", "degraded_safe_mode"]
    assert report.disease_model_ready is True
    assert report.pest_model_ready is True
    assert report.manifests_ready is True
    assert report.location_registry_ready is True

    # Model release policy assertions
    policy = report.model_release_policy
    assert policy["disease_model_tier"] == "EXPERIMENTAL_RESEARCH_PROTOTYPE"
    assert policy["held_out_accuracy"] >= 0.50
    assert policy["held_out_macro_f1"] >= 0.10
    assert policy["dependable_diagnosis_claimed"] is False
    assert policy["validation_gates"]["withhold_chemical_below_threshold"] is True
    assert "Tomato" in policy["supported_crops"] or "Apple" in policy["supported_crops"]
    assert "stem_branch" in policy["blocked_viewpoints"]


def test_startup_checks_graceful_missing_artifacts(tmp_path):
    """Verify graceful handling and clean abstention when root directory lacks artifacts."""
    empty_root = tmp_path / "empty_system"
    empty_root.mkdir()

    # Must complete safely without raising FileNotFoundError
    report = run_startup_artifact_checks(root_dir=empty_root)

    assert report.overall_status == "critical_missing"
    assert report.disease_model_ready is False
    assert report.pest_model_ready is False
    assert report.manifests_ready is False
    assert any("missing" in msg.lower() or "degraded" in msg.lower() for msg in report.summary_messages)

    # Artifacts should be marked as exists=False
    for name, artifact in report.artifacts.items():
        assert artifact.exists is False
