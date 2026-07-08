"""Weather route — uses Open-Meteo (free, no API key) + geocoding."""
from __future__ import annotations

import requests
from flask import Blueprint, jsonify, request

weather_bp = Blueprint("weather", __name__)

GEOCODE_URL = "https://nominatim.openstreetmap.org/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
}

FARM_ALERTS = {
    "heavy_rain": {"threshold": 20, "message": "Heavy rain forecast — delay pesticide/fertilizer application, check field drainage."},
    "high_temp": {"threshold": 40, "message": "Extreme heat alert — irrigate early morning, provide shade for nurseries."},
    "low_temp": {"threshold": 5, "message": "Cold wave alert — cover seedlings, delay early-season sowing."},
    "strong_wind": {"threshold": 40, "message": "Strong winds — avoid spraying operations, secure crop covers and poly-houses."},
}


def _geocode(location: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(
            GEOCODE_URL,
            params={"q": location + ", India", "format": "json", "limit": 1},
            headers={"User-Agent": "SmartFarmingAI/1.0"},
            timeout=5,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _generate_farm_alerts(forecast: dict) -> list[str]:
    alerts = []
    for day in forecast.get("daily", {}).get("precipitation_sum", [])[:3]:
        if day and day >= FARM_ALERTS["heavy_rain"]["threshold"]:
            alerts.append(FARM_ALERTS["heavy_rain"]["message"])
            break
    for temp in forecast.get("daily", {}).get("temperature_2m_max", [])[:3]:
        if temp and temp >= FARM_ALERTS["high_temp"]["threshold"]:
            alerts.append(FARM_ALERTS["high_temp"]["message"])
            break
    for temp in forecast.get("daily", {}).get("temperature_2m_min", [])[:3]:
        if temp and temp <= FARM_ALERTS["low_temp"]["threshold"]:
            alerts.append(FARM_ALERTS["low_temp"]["message"])
            break
    for speed in forecast.get("daily", {}).get("windspeed_10m_max", [])[:3]:
        if speed and speed >= FARM_ALERTS["strong_wind"]["threshold"]:
            alerts.append(FARM_ALERTS["strong_wind"]["message"])
            break
    return alerts


@weather_bp.route("/", methods=["GET"])
def get_weather():
    location = request.args.get("location", "Pune, Maharashtra")
    coords = _geocode(location)

    if not coords:
        # Default to Pune
        coords = (18.5204, 73.8567)

    lat, lon = coords
    try:
        resp = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": [
                    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "precipitation", "weather_code", "windspeed_10m", "winddirection_10m",
                    "uv_index",
                ],
                "daily": [
                    "weather_code", "temperature_2m_max", "temperature_2m_min",
                    "precipitation_sum", "windspeed_10m_max", "et0_fao_evapotranspiration",
                    "sunshine_duration",
                ],
                "timezone": "Asia/Kolkata",
                "forecast_days": 7,
            },
            timeout=10,
        )
        data = resp.json()
    except Exception as e:
        return jsonify({"error": f"Weather service unavailable: {str(e)}"}), 503

    current = data.get("current", {})
    daily = data.get("daily", {})

    # Build 7-day forecast list
    forecast_days = []
    dates = daily.get("time", [])
    for i, date in enumerate(dates):
        forecast_days.append({
            "date": date,
            "weather_code": daily.get("weather_code", [None]*7)[i],
            "description": WMO_CODES.get(daily.get("weather_code", [0]*7)[i], "Unknown"),
            "temp_max": daily.get("temperature_2m_max", [None]*7)[i],
            "temp_min": daily.get("temperature_2m_min", [None]*7)[i],
            "precipitation_mm": daily.get("precipitation_sum", [0]*7)[i],
            "wind_max_kmh": daily.get("windspeed_10m_max", [None]*7)[i],
            "evapotranspiration_mm": daily.get("et0_fao_evapotranspiration", [None]*7)[i],
        })

    farm_alerts = _generate_farm_alerts(daily)

    return jsonify({
        "location": location,
        "coordinates": {"lat": lat, "lon": lon},
        "current": {
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("windspeed_10m"),
            "wind_direction": current.get("winddirection_10m"),
            "uv_index": current.get("uv_index"),
            "description": WMO_CODES.get(current.get("weather_code", 0), "Unknown"),
        },
        "forecast": forecast_days,
        "farm_alerts": farm_alerts,
    })
