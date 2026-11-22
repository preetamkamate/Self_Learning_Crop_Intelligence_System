"""
Unified Hierarchical Plant Pathology Taxonomy.
Structures diagnoses into a 5-tier agronomic hierarchy:
1. Crop / Tree Species (e.g., Tomato, Maize, Citrus, Coconut, Apple)
2. Plant Part / Image View (Leaf, Whole Plant, Canopy, Stem, Fruit, Root, Bark/Trunk, Flower)
3. Issue Category (Disease, Pest, Nutrient Deficiency, Abiotic Stress, Physiological Disorder, Healthy, Unknown)
4. Specific Diagnosis Label (e.g., Early Blight, Nitrogen Deficiency, Stem Bleeding, Citrus Canker)
5. Evidence Level (High Confidence, Moderate Confidence, Screening Hypothesis, Insufficient Evidence, Out-of-Distribution)
"""
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PlantPart(str, Enum):
    LEAF = "leaf"
    WHOLE_PLANT = "whole_plant"
    CANOPY = "canopy"
    STEM = "stem"
    FRUIT = "fruit"
    ROOT = "root"
    BARK = "bark"
    BARK_TRUNK = "bark"
    FLOWER = "flower"
    NON_PLANT = "non_plant"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, val: Optional[str]) -> "PlantPart":
        if not val:
            return cls.UNKNOWN
        v = val.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")
        for member in cls:
            if member.value == v or member.name.lower() == v:
                return member
        if "non_plant" in v or "nonplant" in v or "not_plant" in v:
            return cls.NON_PLANT
        if "leaf" in v or "foliage" in v:
            return cls.LEAF
        if "whole" in v or "plant" in v:
            return cls.WHOLE_PLANT
        if "canopy" in v or "field" in v:
            return cls.CANOPY
        if "bark" in v or "trunk" in v or "wood" in v:
            return cls.BARK
        if "stem" in v or "twig" in v or "branch" in v:
            return cls.STEM
        if "fruit" in v or "pod" in v or "berry" in v or "grain" in v:
            return cls.FRUIT
        if "root" in v or "tuber" in v:
            return cls.ROOT
        if "flower" in v or "blossom" in v:
            return cls.FLOWER
        return cls.UNKNOWN


class IssueCategory(str, Enum):
    DISEASE = "disease"
    PEST = "pest"
    NUTRIENT_DEFICIENCY = "nutrient_deficiency"
    ABIOTIC_STRESS = "abiotic_stress"
    PHYSIOLOGICAL_DISORDER = "physiological_disorder"
    HEALTHY = "healthy"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, val: Optional[str]) -> "IssueCategory":
        if not val:
            return cls.UNKNOWN
        v = val.lower().strip()
        if "nutrient" in v or "deficiency" in v:
            return cls.NUTRIENT_DEFICIENCY
        if "pest" in v or "mite" in v or "borer" in v or "hopper" in v or "aphid" in v:
            return cls.PEST
        if "healthy" in v:
            return cls.HEALTHY
        if "abiotic" in v or "drought" in v or "scorch" in v or "sunburn" in v or "frost" in v:
            return cls.ABIOTIC_STRESS
        if "blight" in v or "rot" in v or "spot" in v or "canker" in v or "rust" in v or "mildew" in v or "virus" in v or "disease" in v:
            return cls.DISEASE
        return cls.UNKNOWN


class EvidenceLevel(str, Enum):
    HIGH_CONFIDENCE = "high_confidence"
    MODERATE_CONFIDENCE = "moderate_confidence"
    SCREENING_HYPOTHESIS = "screening_hypothesis"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


@dataclass
class HierarchicalDiagnosis:
    """Comprehensive agricultural diagnostic record."""

    crop_species: str = "Unknown Crop"
    plant_part: PlantPart = PlantPart.UNKNOWN
    issue_category: IssueCategory = IssueCategory.UNKNOWN
    diagnosis_label: str = "Insufficient Visual Evidence"
    diagnosis_label_kannada: str = "ಅಸ್ಪಷ್ಟ ರೋಗಲಕ್ಷಣ (Insufficient Evidence)"
    confidence: float = 0.0  # 0.0 to 1.0
    evidence_level: EvidenceLevel = EvidenceLevel.INSUFFICIENT_EVIDENCE
    affected_region: str = "Canopy"
    affected_region_kannada: str = "ಸಸ್ಯದ ಮೇಲ್ಭಾಗ"
    pathogen_or_cause: str = "Unknown"
    is_healthy: bool = False
    is_ood: bool = False
    screening_disclaimer: Optional[str] = None
    recommended_lab_tests: List[str] = field(default_factory=list)
    differential_diagnoses: List[Dict[str, Any]] = field(default_factory=list)
    source_citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d["plant_part"] = self.plant_part.value
        d["issue_category"] = self.issue_category.value
        d["evidence_level"] = self.evidence_level.value
        return d


def parse_legacy_label(raw_label: str) -> Dict[str, Any]:
    """
    Parse flat legacy PlantVillage or classifier labels into structured components.
    Example: 'Tomato with Early Blight' -> {
        'crop': 'Tomato', 'part': 'leaf', 'category': 'disease', 'condition': 'Early Blight'
    }
    """
    clean = raw_label.replace("___", " ").replace("__", " ").strip()
    if " with " in clean:
        parts = clean.split(" with ")
        crop = parts[0].strip()
        condition = parts[1].strip()
    elif " - " in clean:
        parts = clean.split(" - ")
        crop = parts[0].strip()
        condition = parts[1].strip()
    elif clean.lower().startswith("healthy "):
        crop = clean[8:].replace(" Plant", "").strip()
        condition = "Healthy"
    else:
        tokens = clean.split(" ", 1)
        crop = tokens[0] if tokens else "Unknown Crop"
        condition = tokens[1] if len(tokens) > 1 else clean

    crop = crop.replace("(", "").replace(")", "").strip()
    if crop.lower() == "corn":
        crop = "Corn (Maize)"
    elif crop.lower() == "bell pepper":
        crop = "Bell Pepper (Capsicum)"

    is_healthy = "healthy" in condition.lower()
    if is_healthy:
        category = IssueCategory.HEALTHY
    elif "deficiency" in condition.lower() or "chlorosis" in condition.lower():
        category = IssueCategory.NUTRIENT_DEFICIENCY
    elif any(p in condition.lower() for p in ["mite", "caterpillar", "borer", "bug", "aphid"]):
        category = IssueCategory.PEST
    else:
        category = IssueCategory.DISEASE

    return {
        "crop": crop,
        "condition": condition,
        "category": category,
        "is_healthy": is_healthy,
    }
