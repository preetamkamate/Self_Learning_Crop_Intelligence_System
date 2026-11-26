"""
Location Service Package.
Provides consent-based location resolution, nearest KVK directory,
and keyless offline weather abstraction.
"""
from src.location.registry import (
    DistrictInfo,
    KARNATAKA_DISTRICTS,
    KVK_DIRECTORY,
    KVKInfo,
)
from src.location.service import (
    AgroLocationService,
    DistrictResolution,
    haversine_distance,
)

__all__ = [
    "DistrictInfo",
    "KARNATAKA_DISTRICTS",
    "KVK_DIRECTORY",
    "KVKInfo",
    "AgroLocationService",
    "DistrictResolution",
    "haversine_distance",
]
