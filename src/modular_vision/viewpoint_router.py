"""
Modular Viewpoint & Plant Organ Router.
Routes image inputs across the 6 canonical anatomical viewpoints:
1. leaf (foliar)
2. stem_branch (stem, branch, cane, stalk, twig)
3. fruit (fruit, pod, berry, grain, cob)
4. canopy (field canopy, plot, row overview)
5. whole_plant (entire plant from ground to apex)
6. root (root, tuber, rhizome, root crown)
Enforces verified data gate: diagnoses only if an exact tested route exists;
otherwise routes to honest photo guidance with zero disease guesses.
"""
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

CANONICAL_VIEWPOINTS = [
    "leaf",
    "stem_branch",
    "fruit",
    "canopy",
    "whole_plant",
    "root",
    "bark_trunk",
    "non_plant",
    "unknown",
]

VIEWPOINT_ALIASES: Dict[str, str] = {
    # Leaf
    "leaf": "leaf",
    "leaves": "leaf",
    "foliar": "leaf",
    "foliage": "leaf",
    "single_leaf": "leaf",
    "close_up_leaf": "leaf",
    # Stem / Branch
    "stem": "stem_branch",
    "branch": "stem_branch",
    "stem_branch": "stem_branch",
    "stem / branch": "stem_branch",
    "stem/branch": "stem_branch",
    "stalk": "stem_branch",
    "cane": "stem_branch",
    "twig": "stem_branch",
    # Bark / Trunk
    "bark": "bark_trunk",
    "trunk": "bark_trunk",
    "bark_trunk": "bark_trunk",
    "bark / trunk": "bark_trunk",
    "tree_bark": "bark_trunk",
    # Fruit
    "fruit": "fruit",
    "pod": "fruit",
    "berry": "fruit",
    "grain": "fruit",
    "cob": "fruit",
    "ear": "fruit",
    # Canopy
    "canopy": "canopy",
    "canopy / field": "canopy",
    "canopy/field": "canopy",
    "field": "canopy",
    "plot": "canopy",
    "row": "canopy",
    "aerial": "canopy",
    # Whole Plant
    "whole_plant": "whole_plant",
    "whole plant": "whole_plant",
    "full plant": "whole_plant",
    "entire plant": "whole_plant",
    "full_plant": "whole_plant",
    # Root
    "root": "root",
    "tuber": "root",
    "root / tuber": "root",
    "root/tuber": "root",
    "rhizome": "root",
    "corm": "root",
    "bulb": "root",
    # Non plant
    "non_plant": "non_plant",
    "not_plant": "non_plant",
    "nonplant": "non_plant",
    "object": "non_plant",
    # Unknown
    "unknown": "unknown",
    "auto": "unknown",
    "auto-detect": "unknown",
}


@dataclass
class ViewpointRouteResult:
    """Standardized viewpoint routing decision."""
    raw_viewpoint: str
    canonical_viewpoint: str
    is_supported_route: bool
    is_plant: bool
    status_reason: str
    requires_photo_guidance: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_viewpoint": self.raw_viewpoint,
            "canonical_viewpoint": self.canonical_viewpoint,
            "is_supported_route": self.is_supported_route,
            "is_plant": self.is_plant,
            "status_reason": self.status_reason,
            "requires_photo_guidance": self.requires_photo_guidance,
            "metadata": self.metadata,
        }


class ViewpointRouter:
    """Modular Viewpoint Router for the 6 agricultural viewpoints."""

    def __init__(self, verified_crop_viewpoints: Optional[Dict[str, List[str]]] = None):
        # Maps crop -> list of verified canonical viewpoints across all unlocked routes
        self.verified_crop_viewpoints = verified_crop_viewpoints or {
            "Tomato": ["leaf", "whole_plant", "canopy", "stem_branch", "fruit"],
            "Corn (Maize)": ["leaf", "whole_plant", "canopy", "stem_branch"],
            "Apple": ["leaf", "fruit", "bark_trunk"],
            "Potato": ["leaf", "root"],
            "Grape": ["leaf", "stem_branch"],
            "Citrus": ["stem_branch", "fruit"],
            "Bell Pepper (Capsicum)": ["leaf", "fruit"],
            "Coconut": ["bark_trunk"],
            "Cassava": ["leaf"],
        }

    def normalize_viewpoint(self, raw_viewpoint: Optional[str]) -> str:
        """Map any viewpoint string to one of the canonical viewpoints."""
        if not raw_viewpoint or str(raw_viewpoint).strip() in ["", "None", "Auto-detect", "auto"]:
            return "unknown"

        clean = str(raw_viewpoint).lower().strip().replace("-", "_").replace(" ", "_")

        if clean in VIEWPOINT_ALIASES:
            return VIEWPOINT_ALIASES[clean]

        for alias, canonical in VIEWPOINT_ALIASES.items():
            alias_clean = alias.replace("-", "_").replace(" ", "_")
            if alias_clean in clean or clean in alias_clean:
                return canonical

        if "entire" in clean and "plant" in clean:
            return "whole_plant"
        elif "full" in clean and "plant" in clean:
            return "whole_plant"
        elif "whole" in clean and "plant" in clean:
            return "whole_plant"
        elif "canopy" in clean or "field" in clean or "plot" in clean:
            return "canopy"
        elif "stem" in clean or "branch" in clean or "stalk" in clean or "bark" in clean or "trunk" in clean:
            return "stem_branch"
        elif "fruit" in clean or "pod" in clean or "grain" in clean or "cob" in clean or "berry" in clean:
            return "fruit"
        elif "root" in clean or "tuber" in clean or "rhizome" in clean:
            return "root"
        elif "leaf" in clean or "foliar" in clean or "foliage" in clean:
            return "leaf"
        elif "non" in clean or "not_plant" in clean:
            return "non_plant"

        return "unknown"

    def route_viewpoint(
        self,
        crop: str,
        viewpoint_hint: Optional[str] = None,
    ) -> ViewpointRouteResult:
        """
        Validate and route a crop and viewpoint combination.
        """
        raw = viewpoint_hint or "unknown"
        canonical = self.normalize_viewpoint(raw)

        # 1. Non-plant rejection
        if canonical == "non_plant":
            return ViewpointRouteResult(
                raw_viewpoint=raw,
                canonical_viewpoint="non_plant",
                is_supported_route=False,
                is_plant=False,
                status_reason="Non-plant input detected.",
                requires_photo_guidance=True,
            )

        # 2. Check if crop is supported
        supported_vps = self.verified_crop_viewpoints.get(crop, [])
        if not supported_vps:
            return ViewpointRouteResult(
                raw_viewpoint=raw,
                canonical_viewpoint=canonical,
                is_supported_route=False,
                is_plant=True,
                status_reason=f"Crop '{crop}' does not have verified trained diagnosis models.",
                requires_photo_guidance=True,
            )

        # 3. Check if this exact viewpoint is verified for this crop
        # Note: 'leaf' is verified for the 4 crops; non-leaf viewpoints are safely gated to photo guidance
        if canonical in supported_vps:
            return ViewpointRouteResult(
                raw_viewpoint=raw,
                canonical_viewpoint=canonical,
                is_supported_route=True,
                is_plant=True,
                status_reason=f"Exact verified route deployed: {crop} ({canonical}).",
                requires_photo_guidance=False,
            )

        # Non-leaf or unsupported viewpoint
        part_name = canonical.replace("_", " ").title()
        reason = f"Viewpoint '{part_name}' diagnosis is not trained yet for crop '{crop}'."
        return ViewpointRouteResult(
            raw_viewpoint=raw,
            canonical_viewpoint=canonical,
            is_supported_route=False,
            is_plant=True,
            status_reason=reason,
            requires_photo_guidance=True,
        )
