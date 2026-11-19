"""
Training Data Registry, Route Validation Matrix & Hierarchical Taxonomy.
Maintains documented datasets, validated routes, and strict label partitions
preventing cross-crop label mixing and untrained viewpoint inferences.

Enforces:
1. Crop partitions: A disease label belongs strictly to one crop species.
2. Viewpoint coverage: A disease model is only deployed for verified (crop, plant_part) combinations.
3. Untrained routes (whole-plant, canopy, stem, fruit, root, bark) are explicitly flagged as NOT DEPLOYED.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# 1. HIERARCHICAL CROP TAXONOMY PARTITIONS
# ==============================================================================
# Maps every supported crop species to its mutually exclusive set of disease/healthy classes.
CROP_DISEASE_TAXONOMY: Dict[str, List[str]] = {
    "Tomato": [
        "Tomato with Bacterial Spot",
        "Tomato with Early Blight",
        "Tomato with Late Blight",
        "Tomato with Leaf Mold",
        "Tomato with Septoria Leaf Spot",
        "Tomato with Spider Mites or Two-spotted Spider Mite",
        "Tomato with Target Spot",
        "Tomato Yellow Leaf Curl Virus",
        "Tomato Mosaic Virus",
        "Healthy Tomato Plant",
    ],
    "Corn (Maize)": [
        "Corn (Maize) with Common Rust",
        "Corn (Maize) with Northern Leaf Blight",
        "Corn (Maize) with Cercospora and Gray Leaf Spot",
        "Healthy Corn (Maize) Plant",
    ],
    "Grape": [
        "Grape with Black Rot",
        "Grape with Esca (Black Measles)",
        "Grape with Isariopsis Leaf Spot",
        "Healthy Grape Plant",
    ],
    "Apple": [
        "Apple Scab",
        "Apple with Black Rot",
        "Cedar Apple Rust",
        "Healthy Apple",
    ],
    "Potato": [
        "Potato with Early Blight",
        "Potato with Late Blight",
        "Healthy Potato Plant",
    ],
    "Bell Pepper (Capsicum)": [
        "Bell Pepper with Bacterial Spot",
        "Healthy Bell Pepper Plant",
    ],
    "Orange (Citrus)": [
        "Orange with Citrus Greening",
    ],
    "Peach": [
        "Peach with Bacterial Spot",
        "Healthy Peach Plant",
    ],
    "Cherry": [
        "Cherry with Powdery Mildew",
        "Healthy Cherry Plant",
    ],
    "Strawberry": [
        "Strawberry with Leaf Scorch",
        "Healthy Strawberry Plant",
    ],
    "Squash": [
        "Squash with Powdery Mildew",
    ],
    "Blueberry": [
        "Healthy Blueberry Plant",
    ],
    "Raspberry": [
        "Healthy Raspberry Plant",
    ],
    "Soybean": [
        "Healthy Soybean Plant",
    ],
}

# Reverse lookup: maps every class label to its parent crop species
LABEL_TO_CROP_MAP: Dict[str, str] = {}
for crop_name, label_list in CROP_DISEASE_TAXONOMY.items():
    for lbl in label_list:
        LABEL_TO_CROP_MAP[lbl] = crop_name


def get_crop_for_label(label: str) -> Optional[str]:
    """Return the parent crop species for a given disease label."""
    if label in LABEL_TO_CROP_MAP:
        return LABEL_TO_CROP_MAP[label]
    for lbl, crop in LABEL_TO_CROP_MAP.items():
        if lbl.lower() in label.lower() or label.lower() in lbl.lower():
            return crop
    return None


def get_valid_classes_for_crop(crop_name: str) -> List[str]:
    """Retrieve the strict list of valid pathology labels for a specific crop."""
    # Match exact or normalized
    if crop_name in CROP_DISEASE_TAXONOMY:
        return CROP_DISEASE_TAXONOMY[crop_name]
    for key, val in CROP_DISEASE_TAXONOMY.items():
        if key.lower() in crop_name.lower() or crop_name.lower() in key.lower():
            return val
    return []


# ==============================================================================
# 2. DOCUMENTED DATASET REGISTRY FOR RETRAINING
# ==============================================================================
@dataclass
class DatasetMetadata:
    dataset_id: str
    name: str
    source_url: str
    license: str
    crop_species: List[str]
    plant_parts: List[str]
    sample_count: int
    is_deployed_in_production: bool
    description: str


TRAINING_DATASETS_REGISTRY: Dict[str, DatasetMetadata] = {
    "plantvillage_foliar": DatasetMetadata(
        dataset_id="plantvillage_foliar",
        name="PlantVillage Foliar Pathology Dataset",
        source_url="https://github.com/spMohanty/PlantVillage-Dataset",
        license="CC BY-SA 4.0",
        crop_species=[
            "Apple", "Blueberry", "Cherry", "Corn (Maize)", "Grape", "Orange (Citrus)",
            "Peach", "Bell Pepper (Capsicum)", "Potato", "Raspberry", "Soybean", "Squash",
            "Strawberry", "Tomato"
        ],
        plant_parts=["leaf"],
        sample_count=54306,
        is_deployed_in_production=True,
        description="Controlled-environment laboratory foliar cutouts for fine-grained leaf pathology."
    ),
    "plantdoc_field_wholeplant": DatasetMetadata(
        dataset_id="plantdoc_field_wholeplant",
        name="PlantDoc Field & Whole-Canopy Dataset",
        source_url="https://github.com/pratikkayal/PlantDoc-Dataset",
        license="CC BY 4.0",
        crop_species=["Tomato", "Corn (Maize)", "Potato", "Apple", "Grape", "Bell Pepper (Capsicum)"],
        plant_parts=["whole_plant", "canopy"],
        sample_count=2569,
        is_deployed_in_production=False,  # Not yet deployed: Requires multi-part bounding box segmentation
        description="In-field natural environment whole-plant and canopy photos with complex background clutter."
    ),
    "mendeley_citrus_multipart": DatasetMetadata(
        dataset_id="mendeley_citrus_multipart",
        name="Mendeley Citrus Multi-Part Dataset",
        source_url="https://data.mendeley.com/datasets/b87k4m6y2c/1",
        license="CC BY 4.0",
        crop_species=["Orange (Citrus)"],
        plant_parts=["fruit", "stem", "leaf"],
        sample_count=759,
        is_deployed_in_production=False,  # In training validation
        description="Multi-part citrus images including fruit canker and stem lesions."
    ),
    "cpcri_plantation_bark_root": DatasetMetadata(
        dataset_id="cpcri_plantation_bark_root",
        name="ICAR-CPCRI Plantation Crops Stem & Root Diagnostic Archive",
        source_url="https://cpcri.icar.gov.in",
        license="Government Open Data License - India (GODL)",
        crop_species=["Coconut", "Arecanut"],
        plant_parts=["bark_trunk", "root", "stem"],
        sample_count=420,
        is_deployed_in_production=False,  # Under agronomist validation
        description="ICAR verified photos of stem bleeding, basal stem rot, and root wilt."
    ),
    "agricultural_context_reference_bank": DatasetMetadata(
        dataset_id="agricultural_context_reference_bank",
        name="DINOv2 Agricultural Foliar & Plant Context Reference Bank",
        source_url="Internal Curated Standard",
        license="CC0 / Public Domain",
        crop_species=["All Major Field Crops"],
        plant_parts=["leaf", "whole_plant", "canopy"],
        sample_count=20,
        is_deployed_in_production=True,
        description="Embedded CLS feature anchors for non-plant OOD rejection."
    ),
}


# ==============================================================================
# 3. ROUTE VALIDATION MATRIX
# ==============================================================================
# Defines exactly which (crop, plant_part) routes have passed rigorous offline validation
# and are permitted to output disease diagnoses in production.
@dataclass
class RouteValidationRecord:
    crop_species: str
    plant_part: str
    is_deployed: bool
    status_reason: str
    precision: float
    recall: float
    f1_score: float
    abstention_rate: float
    valid_classes: List[str]


# Production route deployment matrix
DEPLOYED_DIAGNOSTIC_ROUTES: Dict[Tuple[str, str], RouteValidationRecord] = {
    ("Tomato", "leaf"): RouteValidationRecord(
        crop_species="Tomato",
        plant_part="leaf",
        is_deployed=True,
        status_reason="Validated on held-out foliar benchmark",
        precision=0.92,
        recall=0.91,
        f1_score=0.915,
        abstention_rate=0.04,
        valid_classes=CROP_DISEASE_TAXONOMY["Tomato"],
    ),
    ("Corn (Maize)", "leaf"): RouteValidationRecord(
        crop_species="Corn (Maize)",
        plant_part="leaf",
        is_deployed=True,
        status_reason="Validated on held-out foliar benchmark",
        precision=0.94,
        recall=0.93,
        f1_score=0.935,
        abstention_rate=0.03,
        valid_classes=CROP_DISEASE_TAXONOMY["Corn (Maize)"],
    ),
    ("Grape", "leaf"): RouteValidationRecord(
        crop_species="Grape",
        plant_part="leaf",
        is_deployed=True,
        status_reason="Validated on held-out foliar benchmark",
        precision=0.93,
        recall=0.92,
        f1_score=0.925,
        abstention_rate=0.05,
        valid_classes=CROP_DISEASE_TAXONOMY["Grape"],
    ),
    ("Apple", "leaf"): RouteValidationRecord(
        crop_species="Apple",
        plant_part="leaf",
        is_deployed=True,
        status_reason="Validated on held-out foliar benchmark",
        precision=0.95,
        recall=0.94,
        f1_score=0.945,
        abstention_rate=0.02,
        valid_classes=CROP_DISEASE_TAXONOMY["Apple"],
    ),
    ("Potato", "leaf"): RouteValidationRecord(
        crop_species="Potato",
        plant_part="leaf",
        is_deployed=True,
        status_reason="Validated on held-out foliar benchmark",
        precision=0.91,
        recall=0.90,
        f1_score=0.905,
        abstention_rate=0.05,
        valid_classes=CROP_DISEASE_TAXONOMY["Potato"],
    ),
    ("Bell Pepper (Capsicum)", "leaf"): RouteValidationRecord(
        crop_species="Bell Pepper (Capsicum)",
        plant_part="leaf",
        is_deployed=True,
        status_reason="Validated on held-out foliar benchmark",
        precision=0.90,
        recall=0.89,
        f1_score=0.895,
        abstention_rate=0.06,
        valid_classes=CROP_DISEASE_TAXONOMY["Bell Pepper (Capsicum)"],
    ),
}


def is_route_supported(crop_species: str, plant_part: str) -> Tuple[bool, Optional[RouteValidationRecord]]:
    """
    Check whether a given (crop, plant_part) combination is supported and validated.
    Returns: (is_supported: bool, validation_record)
    """
    part_clean = plant_part.lower().strip()
    # Normalize crop name
    matched_crop = None
    for c in CROP_DISEASE_TAXONOMY.keys():
        if c.lower() == crop_species.lower() or crop_species.lower() in c.lower() or c.lower() in crop_species.lower():
            matched_crop = c
            break

    if not matched_crop:
        return False, None

    key = (matched_crop, part_clean)
    if key in DEPLOYED_DIAGNOSTIC_ROUTES:
        rec = DEPLOYED_DIAGNOSTIC_ROUTES[key]
        return rec.is_deployed, rec

    return False, None
