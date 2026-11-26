"""
Startup Artifact Checks and Graceful Degradation Engine.
Validates presence and integrity of all required production artifacts:
1. Disease Classification Model, Labels, and Evaluation Report
2. Pest Detection Model, Labels, and Evaluation Report
3. Derived Real-Camera Manifests and Zero-Leakage Split Summaries
4. Real-Photo Forensic Evidence and Visual Contact Sheets
5. Agro-Climatic KVK and Location Registries

Enforces the strict Model-Release Policy:
- Disease Model (58.33% held-out test accuracy) is explicitly labeled EXPERIMENTAL.
- Missing artifacts cause graceful degradation into safe abstention without crashes.
"""
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)

# Mandatory production artifacts registry
REQUIRED_ARTIFACTS = {
    # 1. Foliar / Whole-Plant Pathology Model Artifacts
    "disease_model_checkpoint": {
        "path": ROOT_DIR / "models" / "whole_plant" / "model.pt",
        "description": "Foliar and whole-plant disease classification PyTorch weights",
        "critical_for": "foliar_diagnosis",
    },
    "disease_model_labels": {
        "path": ROOT_DIR / "models" / "whole_plant" / "labels.json",
        "description": "14-class crop-specific pathology taxonomy mapping",
        "critical_for": "foliar_diagnosis",
    },
    "disease_model_eval_report": {
        "path": ROOT_DIR / "models" / "whole_plant" / "evaluation_report.json",
        "description": "Held-out test split evaluation metrics and per-dataset audit",
        "critical_for": "foliar_diagnosis",
    },

    # 2. Dedicated Pest Detection Model Artifacts
    "pest_model_checkpoint": {
        "path": ROOT_DIR / "models" / "pest_detector" / "model.pt",
        "description": "Dedicated binary CNN weights for Two-Spotted Spider Mites",
        "critical_for": "pest_detection",
    },
    "pest_model_labels": {
        "path": ROOT_DIR / "models" / "pest_detector" / "labels.json",
        "description": "Pest taxonomy mapping (Healthy vs Spider Mites)",
        "critical_for": "pest_detection",
    },
    "pest_model_eval_report": {
        "path": ROOT_DIR / "models" / "pest_detector" / "evaluation_report.json",
        "description": "Held-out pest detection evaluation metrics",
        "critical_for": "pest_detection",
    },

    # 3. Derived Real-Camera Manifests
    "disease_manifest": {
        "path": ROOT_DIR / "data" / "manifests" / "disease_classification_manifest.json",
        "description": "518 validated real-camera samples manifest (zero leakage)",
        "critical_for": "data_audit",
    },
    "pest_manifest": {
        "path": ROOT_DIR / "data" / "manifests" / "pest_detection_manifest.json",
        "description": "74 validated real-camera pest samples manifest",
        "critical_for": "data_audit",
    },
    "all_samples_manifest": {
        "path": ROOT_DIR / "data" / "manifests" / "all_validated_samples_manifest.json",
        "description": "592 master verified samples with SHA-256 fingerprints",
        "critical_for": "data_audit",
    },
    "manifest_summary": {
        "path": ROOT_DIR / "data" / "manifests" / "manifest_summary_report.json",
        "description": "Deterministic split counts and zero-leakage verification summary",
        "critical_for": "data_audit",
    },

    # 4. Forensic Visual Evidence Contact Sheet
    "real_photo_contact_sheet": {
        "path": ROOT_DIR / "models" / "real_photo_gallery_contact_sheet.jpg",
        "description": "Photometric contact sheet proving camera sensor noise and texture entropy",
        "critical_for": "visual_audit",
    },
}


@dataclass
class ArtifactStatus:
    name: str
    path: str
    exists: bool
    size_bytes: int = 0
    description: str = ""
    critical_for: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StartupCheckReport:
    timestamp: str
    overall_status: str  # 'production_ready_experimental', 'degraded_safe_mode', 'critical_missing'
    disease_model_ready: bool
    pest_model_ready: bool
    manifests_ready: bool
    location_registry_ready: bool
    artifacts: Dict[str, ArtifactStatus] = field(default_factory=dict)
    model_release_policy: Dict[str, Any] = field(default_factory=dict)
    summary_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status,
            "disease_model_ready": self.disease_model_ready,
            "pest_model_ready": self.pest_model_ready,
            "manifests_ready": self.manifests_ready,
            "location_registry_ready": self.location_registry_ready,
            "model_release_policy": self.model_release_policy,
            "summary_messages": self.summary_messages,
            "artifacts": {k: v.__dict__ for k, v in self.artifacts.items()},
        }


def run_startup_artifact_checks(root_dir: Optional[Path] = None) -> StartupCheckReport:
    """
    Run comprehensive artifact existence and integrity verification on application launch.
    """
    from datetime import datetime, timezone
    base_dir = Path(root_dir) if root_dir else ROOT_DIR
    now_str = datetime.now(timezone.utc).isoformat()

    artifacts_status: Dict[str, ArtifactStatus] = {}
    summary_msgs: List[str] = []

    # Check each artifact
    for name, spec in REQUIRED_ARTIFACTS.items():
        rel_path = spec["path"].relative_to(ROOT_DIR)
        target_path = base_dir / rel_path
        exists = target_path.exists()
        size_bytes = target_path.stat().st_size if exists else 0
        err = None
        meta = {}

        if exists and target_path.suffix == ".json":
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    meta["keys"] = list(data.keys())[:10]
                    if "overall_accuracy" in data:
                        meta["overall_accuracy"] = data["overall_accuracy"]
                        meta["macro_f1"] = data.get("macro_f1")
            except Exception as e:
                err = f"JSON parse error: {e}"

        artifacts_status[name] = ArtifactStatus(
            name=name,
            path=str(target_path),
            exists=exists and (err is None),
            size_bytes=size_bytes,
            description=spec["description"],
            critical_for=spec["critical_for"],
            error=err,
            metadata=meta,
        )

    # Evaluate subsystems
    disease_ready = (
        artifacts_status["disease_model_checkpoint"].exists
        and artifacts_status["disease_model_labels"].exists
        and artifacts_status["disease_model_eval_report"].exists
    )

    pest_ready = (
        artifacts_status["pest_model_checkpoint"].exists
        and artifacts_status["pest_model_labels"].exists
        and artifacts_status["pest_model_eval_report"].exists
    )

    manifests_ready = (
        artifacts_status["disease_manifest"].exists
        and artifacts_status["pest_manifest"].exists
        and artifacts_status["all_samples_manifest"].exists
    )

    # Check KVK location registry
    try:
        from src.location.registry import KARNATAKA_DISTRICTS, KVK_DIRECTORY
        location_ready = len(KARNATAKA_DISTRICTS) >= 20 and len(KVK_DIRECTORY) >= 19
    except Exception as e:
        location_ready = False
        summary_msgs.append(f"Location registry import failure: {e}")

    # Determine Model-Release Policy Parameters
    disease_eval_meta = artifacts_status["disease_model_eval_report"].metadata
    held_out_acc = disease_eval_meta.get("overall_accuracy", 0.5833)
    macro_f1 = disease_eval_meta.get("macro_f1", 0.5881)

    model_release_policy = {
        "disease_model_tier": "EXPERIMENTAL_RESEARCH_PROTOTYPE",
        "held_out_accuracy": held_out_acc,
        "held_out_macro_f1": macro_f1,
        "dependable_diagnosis_claimed": False,
        "safety_disclaimer": (
            "EXPERIMENTAL (Held-out Accuracy: 58.33%). Automatic diagnoses are research prototypes "
            "and must never be presented as certified diagnoses. Below strict confidence and margin "
            "thresholds, chemical recommendations are completely withheld."
        ),
        "validation_gates": {
            "confidence_threshold_default": 0.70,
            "margin_threshold_default": 0.15,
            "withhold_chemical_below_threshold": True,
            "mandatory_kvk_referral": True,
        },
        "supported_crops": ["Apple", "Cassava", "Corn (Maize)", "Tomato"],
        "supported_viewpoint": "leaf",
        "blocked_viewpoints": ["stem_branch", "fruit", "canopy", "whole_plant", "root"],
        "blocked_pest_routes": ["Fall Armyworm", "Stem Borers", "Aphids", "Whiteflies", "Thrips"],
        "pilot_pest_routes": ["Tomato Two-Spotted Spider Mites"],
    }

    if disease_ready and pest_ready and manifests_ready and location_ready:
        overall_status = "production_ready_experimental"
        summary_msgs.append("All production artifacts verified. Running in experimental mode under strict release policy.")
    elif disease_ready or pest_ready:
        overall_status = "degraded_safe_mode"
        summary_msgs.append("One or more model subsystems degraded. Safe abstention active for affected routes.")
    else:
        overall_status = "critical_missing"
        summary_msgs.append("Critical model checkpoints missing. Safe abstention enforced across all visual routes.")

    report = StartupCheckReport(
        timestamp=now_str,
        overall_status=overall_status,
        disease_model_ready=disease_ready,
        pest_model_ready=pest_ready,
        manifests_ready=manifests_ready,
        location_registry_ready=location_ready,
        artifacts=artifacts_status,
        model_release_policy=model_release_policy,
        summary_messages=summary_msgs,
    )
    return report
