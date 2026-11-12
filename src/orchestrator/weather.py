"""
Live Weather Intelligence Service using Open-Meteo API (No Key Required).
Fetches real-time temperature, humidity, and rainfall forecast with agronomic spray advisories.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# WMO Weather interpretation codes
WMO_DESCRIPTIONS = {
    0: {"en": "Clear sky", "kn": "ಸ್ವಚ್ಛ ಆಕಾಶ (Clear Sky)"},
    1: {"en": "Mainly clear", "kn": "ಹೆಚ್ಚಾಗಿ ಸ್ವಚ್ಛ ಆಕಾಶ"},
    2: {"en": "Partly cloudy", "kn": "ಭಾಗಶಃ ಮೋಡ ಕವಿದ ವಾತಾವರಣ"},
    3: {"en": "Overcast", "kn": "ಸಂಪೂರ್ಣ ಮೋಡ ಕವಿದ ವಾತಾವರಣ"},
    45: {"en": "Foggy", "kn": "ಮಂಜು ಮುಸುಕಿದ ವಾತಾವರಣ"},
    51: {"en": "Light drizzle", "kn": "ಹಗುರ ತುಂತುರು ಮಳೆ"},
    53: {"en": "Moderate drizzle", "kn": "ಮಧ್ಯಮ ತುಂತುರು ಮಳೆ"},
    61: {"en": "Slight rain", "kn": "ಸಣ್ಣ ಪ್ರಮಾಣದ ಮಳೆ"},
    63: {"en": "Moderate rain", "kn": "ಮಧ್ಯಮ ಮಳೆ"},
    65: {"en": "Heavy rain", "kn": "ಭಾರಿ ಮಳೆ"},
    80: {"en": "Rain showers", "kn": "ಮಳೆಯ ಸುರಿಮಳೆ"},
    95: {"en": "Thunderstorm", "kn": "ಗುಡುಗು ಸಹಿತ ಮಳೆ"},
}


class WeatherService:
    """Fetches real-time weather and generates spray suitability advisories."""

    def __init__(self, base_url: str = config.OPEN_METEO_BASE_URL):
        self.base_url = base_url

    def get_weather(
        self,
        district_or_city: str = "Mandya (Cauvery Basin)",
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Fetch real-time weather and 3-day forecast from Open-Meteo.
        """
        # Determine coordinates
        if lat is None or lon is None:
            coords = config.DISTRICT_COORDINATES.get(
                district_or_city,
                config.DISTRICT_COORDINATES["Mandya (Cauvery Basin)"],
            )
            lat, lon = coords["lat"], coords["lon"]

        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ],
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_probability_max",
            ],
            "timezone": "Asia/Kolkata",
            "forecast_days": 3,
        }

        try:
            resp = requests.get(self.base_url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            current = data.get("current", {})
            daily = data.get("daily", {})

            w_code = current.get("weather_code", 0)
            w_desc = WMO_DESCRIPTIONS.get(
                w_code,
                {"en": "Scattered Clouds", "kn": "ಚದುರಿದ ಮೋಡಗಳು"}
            )

            temp = current.get("temperature_2m", 25.0)
            humidity = current.get("relative_humidity_2m", 65.0)
            precip = current.get("precipitation", 0.0)
            wind_speed = current.get("wind_speed_10m", 5.0)

            # Daily forecast summaries
            rain_sum_today = daily.get("precipitation_sum", [0.0])[0] if daily.get("precipitation_sum") else 0.0
            rain_prob_today = daily.get("precipitation_probability_max", [0])[0] if daily.get("precipitation_probability_max") else 0

            # Agronomic spraying & fungal risk assessment
            spray_safe = True
            spray_warning_en = "Conditions favorable for field spraying."
            spray_warning_kn = "ಕೀಟನಾಶಕ ಸಿಂಪಡಣೆಗೆ ಅನುಕೂಲಕರ ವಾತಾವರಣವಿದೆ."

            if rain_prob_today > 40 or precip > 0.5 or rain_sum_today > 2.0:
                spray_safe = False
                spray_warning_en = (
                    f"RAIN ALERT: {rain_prob_today}% rain probability ({rain_sum_today} mm forecast). "
                    "DO NOT spray chemical insecticides/fungicides today as rain will wash away deposits."
                )
                spray_warning_kn = (
                    f"ಮಳೆಯ ಎಚ್ಚರಿಕೆ: ಇಂದು {rain_prob_today}% ಮಳೆಯ ಸಂಭವನೀಯತೆಯಿದೆ ({rain_sum_today} mm ಮುನ್ಸೂಚನೆ). "
                    "ಇಂದು ಔಷಧ ಸಿಂಪಡಿಸಬೇಡಿ; ಮಳೆಗೆ ಔಷಧಿ ತೊಳೆದು ವ್ಯರ್ಥವಾಗುತ್ತದೆ."
                )
            elif wind_speed > 15.0:
                spray_safe = False
                spray_warning_en = f"HIGH WIND: Wind speed is {wind_speed:.1f} km/h. Avoid foliar spray to prevent pesticide drift."
                spray_warning_kn = f"ವೇಗದ ಗಾಳಿ: ಗಾಳಿಯ ವೇಗ {wind_speed:.1f} km/h ಇದೆ. ಸಿಂಪಡಣೆ ದ್ರವ್ಯ ಪಕ್ಕದ ಜಮೀನಿಗೆ ಹರಡುವುದರಿಂದ ಸಿಂಪಡಣೆ ಮುಂದೂಡಿ."
            elif humidity > 85.0:
                spray_warning_en += " High humidity alert: Increased risk for foliar fungal blights."
                spray_warning_kn += " ಹೆಚ್ಚು ತೇವಾಂಶವಿದೆ: ಶಿಲೀಂಧ್ರ ರೋಗಗಳ ಬಾಧೆ ಹೆಚ್ಚಾಗುವ ಸಾಧ್ಯತೆಯಿದೆ."

            return {
                "status": "success",
                "district": district_or_city,
                "latitude": lat,
                "longitude": lon,
                "timestamp": current.get("time", datetime.now().strftime("%Y-%m-%dT%H:%M")),
                "temperature_c": temp,
                "humidity_pct": humidity,
                "apparent_temperature_c": current.get("apparent_temperature", temp),
                "precipitation_mm": precip,
                "wind_speed_kmh": wind_speed,
                "condition_en": w_desc["en"],
                "condition_kn": w_desc["kn"],
                "today_rain_prob_pct": rain_prob_today,
                "today_rain_sum_mm": rain_sum_today,
                "spray_safe": spray_safe,
                "spray_advisory_en": spray_warning_en,
                "spray_advisory_kn": spray_warning_kn,
                "daily_forecast": {
                    "dates": daily.get("time", []),
                    "temp_max": daily.get("temperature_2m_max", []),
                    "temp_min": daily.get("temperature_2m_min", []),
                    "rain_sum": daily.get("precipitation_sum", []),
                    "rain_prob": daily.get("precipitation_probability_max", []),
                },
            }

        except Exception as e:
            logging.warning("Weather fetch failed: %s. Using climatological fallback.", e)
            return {
                "status": "fallback",
                "district": district_or_city,
                "latitude": lat,
                "longitude": lon,
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M"),
                "temperature_c": 26.5,
                "humidity_pct": 70.0,
                "apparent_temperature_c": 27.0,
                "precipitation_mm": 0.0,
                "wind_speed_kmh": 6.5,
                "condition_en": "Partly Cloudy (Fallback)",
                "condition_kn": "ಭಾಗಶಃ ಮೋಡ (ಹಿನ್ನೆಲೆ)",
                "today_rain_prob_pct": 20,
                "today_rain_sum_mm": 0.0,
                "spray_safe": True,
                "spray_advisory_en": "Normal spraying conditions (Offline mode). Check skies before spraying.",
                "spray_advisory_kn": "ಸಾಮಾನ್ಯ ಸಿಂಪಡಣಾ ವಾತಾವರಣ (ಆಫ್‌ಲೈನ್ ಮೋಡ್).",
                "daily_forecast": {},
            }


if __name__ == "__main__":
    ws = WeatherService()
    weather = ws.get_weather("Bengaluru (Urban/Rural)")
    print("Fetched weather:", weather["temperature_c"], "C,", weather["condition_en"])
    print("Spray Safe:", weather["spray_safe"])
    print("Spray Advisory:", weather["spray_advisory_en"])

