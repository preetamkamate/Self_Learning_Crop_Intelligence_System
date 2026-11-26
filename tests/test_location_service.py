"""
Comprehensive Test Suite for Consent-Based Live Location, District Resolution,
Nearest KVK Matching, and Keyless Offline Weather Abstraction.
Verifies:
1. User-controlled action and explicit consent gating.
2. No precise coordinate persistence: raw lat/lon are dropped in memory.
3. Accurate spatial mapping to Karnataka districts and agro-climatic zones.
4. Nearest KVK center lookup with authentic ICAR/SAU affiliation.
5. 100% keyless, offline weather abstraction with zero external network calls.
"""
from unittest.mock import MagicMock, patch
import pytest

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
from src.orchestrator.state import FarmerProfile, SessionState
from src.orchestrator.weather import WeatherService


# ==============================================================================
# 1. User Consent & Permission Gating Tests
# ==============================================================================
def test_location_service_requires_user_consent():
    """Verify that location service refuses to process coordinates without explicit consent."""
    service = AgroLocationService()

    # When user_consented is False, must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        service.resolve_district(lat=15.8497, lon=74.4977, user_consented=False)

    assert "User consent required" in str(exc_info.value)


def test_location_service_succeeds_when_consent_granted():
    """Verify that location service proceeds when user explicitly consents."""
    service = AgroLocationService()
    resolution = service.resolve_district(lat=15.8497, lon=74.4977, user_consented=True)

    assert resolution.resolution_status == "success"
    assert resolution.consent_granted is True
    assert resolution.district == "Belagavi"


# ==============================================================================
# 2. No Precise-Location Persistence Guarantee Tests
# ==============================================================================
def test_no_raw_coordinates_persisted_in_resolution_or_state():
    """
    CRITICAL PRIVACY TEST:
    Assert that raw latitude and longitude are discarded in memory
    and NEVER stored on the DistrictResolution or FarmerProfile.
    """
    service = AgroLocationService()
    state = SessionState()

    raw_lat = 15.84972345
    raw_lon = 74.49771234

    resolution = service.resolve_district(lat=raw_lat, lon=raw_lon, user_consented=True)
    res_dict = resolution.to_dict()

    # 1. DistrictResolution must not have raw coordinates
    assert "lat" not in res_dict
    assert "latitude" not in res_dict
    assert "lon" not in res_dict
    assert "longitude" not in res_dict
    assert resolution.precise_coordinates_persisted is False

    # 2. State profile must update coarse district name only
    state.update_profile(district=resolution.district)
    profile_dict = state.profile.__dict__

    assert profile_dict["district"] == "Belagavi"
    assert "lat" not in profile_dict
    assert "lon" not in profile_dict
    assert "raw_coordinates" not in profile_dict


# ==============================================================================
# 3. Spatial District Resolution Tests
# ==============================================================================
def test_spatial_resolution_across_karnataka_districts():
    """Verify accurate distance-based nearest district resolution across regions."""
    service = AgroLocationService()

    test_coordinates = [
        # (lat, lon, expected_district, expected_zone_number)
        (15.85, 74.50, "Belagavi", 3),
        (15.46, 75.01, "Dharwad", 8),
        (12.52, 76.90, "Mandya", 6),
        (12.30, 76.64, "Mysuru", 6),
        (12.97, 77.59, "Bengaluru Urban", 5),
        (13.93, 75.57, "Shivamogga", 7),
        (13.01, 76.10, "Hassan", 7),
        (16.83, 75.71, "Vijayapura", 3),
        (16.19, 75.70, "Bagalkot", 3),
        (17.33, 76.83, "Kalaburagi", 2),
        (16.21, 77.34, "Raichur", 2),
        (13.34, 74.74, "Udupi", 10),
        (13.32, 75.77, "Chikkamagaluru", 9),
    ]

    for lat, lon, expected_dist, expected_zone in test_coordinates:
        res = service.resolve_district(lat=lat, lon=lon, user_consented=True)
        assert res.district == expected_dist, f"Failed for ({lat}, {lon}): got {res.district}, expected {expected_dist}"
        assert res.zone_number == expected_zone
        assert res.distance_to_district_center_km < 30.0  # Within 30 km of district centroid


def test_haversine_distance_calculation():
    """Verify Great-Circle Haversine formula distance."""
    # Distance between Belagavi (15.8497, 74.4977) and Dharwad (15.4589, 75.0078) is ~70-75 km
    dist = haversine_distance(15.8497, 74.4977, 15.4589, 75.0078)
    assert 65.0 <= dist <= 80.0


# ==============================================================================
# 4. Nearest KVK Center Registry & Lookup Tests
# ==============================================================================
def test_nearest_kvk_matching_with_authentic_institution_data():
    """Verify that district resolution automatically links the authentic KVK center."""
    service = AgroLocationService()

    # 1. Belagavi -> KVK Mattikopp
    res_belagavi = service.resolve_district(lat=15.85, lon=74.50, user_consented=True)
    kvk_b = res_belagavi.nearest_kvk
    assert "Mattikopp" in kvk_b.name
    assert "KLE" in kvk_b.name or "ICAR" in kvk_b.name
    assert "+91" in kvk_b.contact_phone
    assert len(kvk_b.services_offered) >= 3

    # 2. Dharwad -> Saidapur Farm, UAS Dharwad
    res_dharwad = service.resolve_district(lat=15.46, lon=75.01, user_consented=True)
    kvk_d = res_dharwad.nearest_kvk
    assert "Saidapur" in kvk_d.name or "Dharwad" in kvk_d.name
    assert "UAS" in kvk_d.nodal_institution

    # 3. Mandya -> V.C. Farm, UAS Bangalore
    res_mandya = service.resolve_district(lat=12.52, lon=76.90, user_consented=True)
    kvk_m = res_mandya.nearest_kvk
    assert "V.C. Farm" in kvk_m.name or "Mandya" in kvk_m.name

    # 4. Direct lookup by district name
    kvk_direct = service.get_kvk_by_district("Tumakuru")
    assert kvk_direct is not None
    assert "Hirehalli" in kvk_direct.name
    assert "IIHR" in kvk_direct.nodal_institution


# ==============================================================================
# 5. Keyless Offline Weather Abstraction Tests (No Keys, No External Calls)
# ==============================================================================
def test_keyless_offline_weather_generation_requires_no_keys_or_network():
    """
    CRITICAL ZERO-EXTERNAL-CALL TEST:
    Assert that the weather abstraction operates completely offline
    with zero API keys and zero network calls.
    """
    service = AgroLocationService(weather_mode="offline_local")

    # Patch requests to ensure NO external HTTP calls can be made
    with patch("requests.get", side_effect=RuntimeError("Illegal external network call attempt!")):
        weather = service.get_weather(district_name="Belagavi", month=7)

        assert weather["status"] == "success"
        assert weather["district"] == "Belagavi"
        assert weather["requires_api_key"] is False
        assert weather["external_calls_made"] is False
        assert weather["service_mode"] == "offline_local"

        # Check climatological values for monsoon month (July) in Zone 3
        assert 20.0 <= weather["temperature_c"] <= 35.0
        assert weather["humidity_pct"] >= 70.0
        assert weather["today_rain_prob_pct"] >= 50
        assert weather["spray_safe"] is False  # Monsoon rain warning prevents spraying
        assert "ಮಳೆ" in weather["spray_advisory_kn"]


def test_offline_weather_seasonal_variation():
    """Verify seasonal climatological behavior across winter, summer, and monsoon."""
    service = AgroLocationService(weather_mode="offline_local")

    # Winter (January) in Mandya: Dry, favorable for spraying
    w_jan = service.get_weather("Mandya", month=1)
    assert w_jan["today_rain_prob_pct"] <= 15
    assert w_jan["spray_safe"] is True
    assert "Favorable" in w_jan["spray_advisory_en"]

    # Summer (April) in Vijayapura: Hot & dry
    w_apr = service.get_weather("Vijayapura", month=4)
    assert w_apr["temperature_c"] >= 30.0

    # Monsoon (July) in Udupi (Coastal Zone): Heavy rain likelihood
    w_jul = service.get_weather("Udupi", month=7)
    assert w_jul["today_rain_prob_pct"] >= 80
    assert w_jul["spray_safe"] is False


def test_weather_service_integration_with_offline_local_mode():
    """Verify WeatherService seamlessly uses AgroLocationService when in offline_local mode."""
    ws = WeatherService(service_mode="offline_local")

    with patch("requests.get", side_effect=RuntimeError("Illegal external call!")):
        w = ws.get_weather("Dharwad")
        assert w["district"] == "Dharwad"
        assert "temperature_c" in w
        assert "humidity_pct" in w
        assert "spray_safe" in w
        assert "spray_advisory_en" in w


def test_weather_service_graceful_fallback_when_network_fails():
    """Verify WeatherService automatically falls back to keyless offline model if live API throws."""
    ws = WeatherService(service_mode="live_open_meteo")

    # Mock requests.get raising ConnectionError
    with patch("requests.get", side_effect=Exception("Connection timed out / DNS failure")):
        w = ws.get_weather("Belagavi")
        assert w["district"] == "Belagavi"
        assert w["requires_api_key"] is False
        assert w["service_mode"] == "offline_local"
        assert 15.0 <= w["temperature_c"] <= 40.0
