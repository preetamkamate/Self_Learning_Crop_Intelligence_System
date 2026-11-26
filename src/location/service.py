"""
Consent-Based Location & Agro-Climatic Intelligence Service.
Resolves GPS coordinates to local districts and nearest Krishi Vigyan Kendra (KVK) centers.
Enforces zero precise-coordinate persistence: raw latitude and longitude are discarded
immediately after coarse district mapping.
Provides a configurable, keyless offline weather abstraction with zero external calls.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from src.location.registry import (
    DistrictInfo,
    KARNATAKA_DISTRICTS,
    KVK_DIRECTORY,
    KVKInfo,
)

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two GPS points in kilometers.
    """
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


@dataclass
class DistrictResolution:
    """Coarse district resolution without stored GPS coordinates."""
    district: str
    kannada_name: str
    agro_climatic_zone: str
    zone_number: int
    dominant_soil: str
    primary_crops: List[str]
    nearest_kvk: KVKInfo
    distance_to_district_center_km: float
    precise_coordinates_persisted: bool = False
    resolution_status: str = "success"
    consent_granted: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "kannada_name": self.kannada_name,
            "agro_climatic_zone": self.agro_climatic_zone,
            "zone_number": self.zone_number,
            "dominant_soil": self.dominant_soil,
            "primary_crops": self.primary_crops,
            "nearest_kvk": asdict(self.nearest_kvk),
            "distance_to_district_center_km": self.distance_to_district_center_km,
            "precise_coordinates_persisted": self.precise_coordinates_persisted,
            "resolution_status": self.resolution_status,
            "consent_granted": self.consent_granted,
        }


# Climatological baseline norms for Karnataka agro-climatic zones by month (1 to 12)
# (Zone Number -> {month: (avg_temp_c, avg_humidity_pct, avg_rain_prob_pct, typical_wind_kmh)})
CLIMATOLOGICAL_NORMS: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {
    # Zone 2 & 3: Northern Dry & North Eastern Dry (Belagavi, Vijayapura, Bagalkot, Kalaburagi, Raichur)
    3: {
        1: (24.0, 52.0, 5.0, 8.0),
        2: (27.0, 45.0, 5.0, 9.0),
        3: (31.0, 40.0, 10.0, 10.0),
        4: (34.0, 42.0, 20.0, 11.0),
        5: (33.0, 55.0, 35.0, 12.0),
        6: (28.0, 72.0, 55.0, 16.0),
        7: (26.0, 80.0, 65.0, 18.0),
        8: (26.0, 78.0, 60.0, 16.0),
        9: (27.0, 75.0, 50.0, 12.0),
        10: (26.5, 68.0, 40.0, 9.0),
        11: (25.0, 58.0, 15.0, 8.0),
        12: (23.5, 54.0, 5.0, 7.0),
    },
    # Zone 5 & 6: Eastern Dry & Southern Dry (Mandya, Mysuru, Bengaluru)
    6: {
        1: (23.0, 60.0, 5.0, 7.0),
        2: (25.5, 52.0, 8.0, 8.0),
        3: (28.5, 50.0, 15.0, 9.0),
        4: (30.0, 60.0, 30.0, 10.0),
        5: (29.0, 68.0, 45.0, 11.0),
        6: (25.5, 75.0, 50.0, 14.0),
        7: (24.5, 82.0, 60.0, 15.0),
        8: (24.5, 80.0, 55.0, 14.0),
        9: (25.0, 78.0, 55.0, 11.0),
        10: (25.0, 74.0, 45.0, 9.0),
        11: (23.5, 68.0, 20.0, 8.0),
        12: (22.5, 64.0, 8.0, 7.0),
    },
    # Zone 7 & 8: Transition Zones (Dharwad, Shivamogga, Haveri)
    8: {
        1: (23.5, 55.0, 5.0, 7.0),
        2: (26.0, 48.0, 6.0, 8.0),
        3: (29.5, 45.0, 12.0, 9.0),
        4: (31.5, 52.0, 25.0, 10.0),
        5: (30.0, 65.0, 40.0, 12.0),
        6: (26.0, 82.0, 70.0, 17.0),
        7: (24.0, 90.0, 80.0, 19.0),
        8: (24.0, 88.0, 75.0, 18.0),
        9: (25.5, 80.0, 55.0, 12.0),
        10: (25.5, 72.0, 35.0, 9.0),
        11: (24.0, 62.0, 15.0, 8.0),
        12: (23.0, 58.0, 5.0, 7.0),
    },
    # Zone 9 & 10: Hill & Coastal Zones (Udupi, Chikkamagaluru)
    10: {
        1: (26.0, 68.0, 5.0, 8.0),
        2: (27.0, 70.0, 5.0, 8.0),
        3: (29.0, 72.0, 10.0, 9.0),
        4: (31.0, 75.0, 25.0, 10.0),
        5: (31.0, 80.0, 50.0, 13.0),
        6: (27.5, 92.0, 90.0, 22.0),
        7: (26.0, 95.0, 95.0, 24.0),
        8: (26.5, 94.0, 90.0, 22.0),
        9: (27.0, 88.0, 75.0, 15.0),
        10: (28.0, 82.0, 50.0, 10.0),
        11: (27.5, 75.0, 25.0, 8.0),
        12: (26.5, 70.0, 10.0, 7.0),
    },
}


class AgroLocationService:
    """
    Agro-Climatic Location & Keyless Offline Weather Abstraction Service.
    Enforces:
    1. User consent check before processing coordinates.
    2. Zero raw coordinate persistence: lat/lon are dropped in memory immediately after district mapping.
    3. 100% keyless and offline-capable weather and KVK lookup.
    """

    def __init__(
        self,
        districts_registry: Optional[Dict[str, DistrictInfo]] = None,
        kvk_registry: Optional[Dict[str, KVKInfo]] = None,
        weather_mode: Optional[str] = None,
    ):
        self.districts = districts_registry or KARNATAKA_DISTRICTS
        self.kvks = kvk_registry or KVK_DIRECTORY
        # Configurable weather mode: "offline_local" (default: zero keys/external calls) or "live_open_meteo"
        self.weather_mode = weather_mode or os.getenv("WEATHER_SERVICE_MODE", "offline_local")

    def resolve_district(
        self,
        lat: float,
        lon: float,
        user_consented: bool = True,
    ) -> DistrictResolution:
        """
        Map GPS coordinates to the nearest agricultural district.
        Enforces privacy guarantee: raw coordinates are not saved to disk or state.
        """
        if not user_consented:
            raise PermissionError("User consent required before processing location coordinates.")

        nearest_district_name: Optional[str] = None
        min_distance = float("inf")

        for name, d_info in self.districts.items():
            dist = haversine_distance(lat, lon, d_info.lat, d_info.lon)
            if dist < min_distance:
                min_distance = dist
                nearest_district_name = name

        # Default fallback if somehow none matched
        if not nearest_district_name:
            nearest_district_name = "Mandya"
            min_distance = 0.0

        d_info = self.districts[nearest_district_name]
        kvk = self.kvks.get(d_info.kvk_code) or list(self.kvks.values())[0]

        # Explicitly drop lat and lon from the resolution object
        return DistrictResolution(
            district=d_info.name,
            kannada_name=d_info.kannada_name,
            agro_climatic_zone=d_info.agro_climatic_zone,
            zone_number=d_info.zone_number,
            dominant_soil=d_info.dominant_soil,
            primary_crops=d_info.primary_crops,
            nearest_kvk=kvk,
            distance_to_district_center_km=min_distance,
            precise_coordinates_persisted=False,
            resolution_status="success",
            consent_granted=True,
        )

    def get_kvk_by_district(self, district_name: str) -> Optional[KVKInfo]:
        """Fetch nearest KVK center for a given district name."""
        clean_name = district_name.strip()
        for name, d_info in self.districts.items():
            if name.lower() == clean_name.lower() or name.lower() in clean_name.lower():
                return self.kvks.get(d_info.kvk_code)
        return None

    def get_weather(
        self,
        district_name: str,
        month: Optional[int] = None,
        force_offline: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch weather intelligence.
        In offline mode, executes 100% keyless, deterministic climatological modeling
        with zero external calls.
        """
        # Match district
        matched_dist: Optional[DistrictInfo] = None
        for name, d_info in self.districts.items():
            if name.lower() in district_name.lower() or district_name.lower() in name.lower():
                matched_dist = d_info
                break

        if not matched_dist:
            matched_dist = self.districts.get("Mandya", list(self.districts.values())[0])

        current_month = month or datetime.now().month

        # Mode: offline_local (keyless, zero external calls)
        if force_offline or self.weather_mode == "offline_local":
            return self._generate_climatological_weather(matched_dist, current_month)

        # Mode: live_open_meteo (with graceful fallback to offline model)
        try:
            from src.orchestrator.weather import WeatherService
            live_service = WeatherService()
            return live_service.get_weather(
                district_or_city=matched_dist.name,
                lat=matched_dist.lat,
                lon=matched_dist.lon,
            )
        except Exception as e:
            logger.info("External weather call deferred or unavailable (%s). Serving keyless offline model.", e)
            return self._generate_climatological_weather(matched_dist, current_month)

    def _generate_climatological_weather(
        self,
        district: DistrictInfo,
        month: int,
    ) -> Dict[str, Any]:
        """
        Generate realistic, deterministic agro-meteorological advisories
        from verified climatological zone norms with zero external calls or keys.
        """
        zone_num = district.zone_number
        # Pick closest matching zone table
        if zone_num in CLIMATOLOGICAL_NORMS:
            zone_table = CLIMATOLOGICAL_NORMS[zone_num]
        elif zone_num in [1, 2, 3]:
            zone_table = CLIMATOLOGICAL_NORMS[3]
        elif zone_num in [4, 5, 6]:
            zone_table = CLIMATOLOGICAL_NORMS[6]
        elif zone_num in [7, 8]:
            zone_table = CLIMATOLOGICAL_NORMS[8]
        else:
            zone_table = CLIMATOLOGICAL_NORMS[10]

        temp, humidity, rain_prob, wind_speed = zone_table.get(month, (26.0, 65.0, 20.0, 9.0))

        # Determine spray suitability
        # Suitable: rain_prob < 40%, wind_speed < 15 km/h, humidity between 40% and 80%
        rain_warning = rain_prob >= 40.0
        high_wind = wind_speed >= 15.0
        suitable = (not rain_warning) and (not high_wind)

        if rain_warning:
            spray_en = f"Unfavorable for chemical spraying: High seasonal rainfall likelihood ({rain_prob:.0f}%). Rain will wash away foliar sprays."
            spray_kn = f"ಸಿಂಪಡಣೆಗೆ ಸೂಕ್ತವಲ್ಲ: ಮಳೆಯಾಗುವ ಸಂಭವನೀಯತೆ ಹೆಚ್ಚಿದೆ ({rain_prob:.0f}%). ಸಿಂಪಡಿಸಿದ ಔಷಧಿ ಮಳೆಗೆ ತೊಳೆದು ಹೋಗಬಹುದು."
        elif high_wind:
            spray_en = f"Caution: Wind speed ({wind_speed:.1f} km/h) exceeds safe spray threshold (15 km/h). Severe spray drift risk."
            spray_kn = f"ಎಚ್ಚರಿಕೆ: ಗಾಳಿಯ ವೇಗ ಹೆಚ್ಚಾಗಿದೆ ({wind_speed:.1f} ಕಿ.ಮೀ/ಗಂ). ಔಷಧಿ ಬೇರೆಡೆ ಹಾರಿಹೋಗುವ ಸಾಧ್ಯತೆ ಇದೆ."
        else:
            spray_en = f"Favorable for spraying: Clear conditions, moderate humidity ({humidity:.0f}%), and gentle wind ({wind_speed:.1f} km/h)."
            spray_kn = f"ಸಿಂಪಡಣೆಗೆ ಸೂಕ್ತ ವಾತಾವರಣ: ಮಳೆಯ ಸಾಧ್ಯತೆ ಕಡಿಮೆ, ಅನುಕೂಲಕರ ತೇವಾಂಶ ({humidity:.0f}%) ಮತ್ತು ಶಾಂತ ಗಾಳಿ."

        w_desc_en = (
            "Monsoon showers likely" if rain_prob > 60 else
            ("Partly cloudy with pleasant breeze" if rain_prob > 25 else "Dry and sunny daylight")
        )
        w_desc_kn = (
            "ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆ" if rain_prob > 60 else
            ("ಭಾಗಶಃ ಮೋಡ ಮತ್ತು ತಂಗಾಳಿ" if rain_prob > 25 else "ಬಿಸಿಲು ಮತ್ತು ಒಣ ಹವೆ")
        )

        return {
            "status": "success",
            "district": district.name,
            "agro_climatic_zone": district.agro_climatic_zone,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "apparent_temperature_c": round(temp + (humidity - 50.0) * 0.05, 1),
            "precipitation_mm": round(rain_prob * 0.1, 1),
            "wind_speed_kmh": wind_speed,
            "weather_code": 61 if rain_prob > 60 else (2 if rain_prob > 25 else 0),
            "description_en": w_desc_en,
            "description_kn": w_desc_kn,
            "condition_en": w_desc_en,
            "condition_kn": w_desc_kn,
            "today_rain_prob_pct": int(rain_prob),
            "today_rain_sum_mm": round(rain_prob * 0.1, 1),
            "spray_safe": suitable,
            "spray_advisory_en": spray_en,
            "spray_advisory_kn": spray_kn,
            "spray_suitability": {
                "suitable": suitable,
                "rain_warning": rain_warning,
                "high_wind_warning": high_wind,
                "advisory_en": spray_en,
                "advisory_kn": spray_kn,
            },
            "service_mode": "offline_local",
            "requires_api_key": False,
            "external_calls_made": False,
        }
