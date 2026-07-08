"""Market route — Mandi prices using Agmarknet-compatible public data."""
from __future__ import annotations

import datetime
from flask import Blueprint, jsonify, request

from app.prompts import build_system_prompt
from app.rag_engine import get_rag_engine
from app.watsonx_client import generate_response

market_bp = Blueprint("market", __name__)

# Static indicative mandi prices (2024-25 season, ₹/quintal)
# Updated periodically; in production integrate with Agmarknet/e-NAM API
MANDI_PRICES = {
    "wheat": {
        "name": "Wheat / गेहूं / गहू",
        "msp": 2275,
        "prices": [
            {"mandi": "Azadpur, Delhi", "price": 2450, "trend": "stable"},
            {"mandi": "Khanna, Punjab", "price": 2380, "trend": "up"},
            {"mandi": "Indore, MP", "price": 2320, "trend": "stable"},
            {"mandi": "Jaipur, Rajasthan", "price": 2290, "trend": "down"},
        ],
        "unit": "₹/quintal",
    },
    "rice": {
        "name": "Rice / चावल / तांदूळ",
        "msp": 2300,
        "prices": [
            {"mandi": "Navi Mumbai, MH", "price": 3200, "trend": "up"},
            {"mandi": "Khammam, TS", "price": 2950, "trend": "stable"},
            {"mandi": "Kolkata, WB", "price": 3100, "trend": "up"},
            {"mandi": "Cuttack, Odisha", "price": 2800, "trend": "stable"},
        ],
        "unit": "₹/quintal",
    },
    "cotton": {
        "name": "Cotton / कपास / कापूस",
        "msp": 7121,
        "prices": [
            {"mandi": "Rajkot, Gujarat", "price": 7850, "trend": "up"},
            {"mandi": "Yavatmal, MH", "price": 7500, "trend": "stable"},
            {"mandi": "Adilabad, TS", "price": 7300, "trend": "down"},
            {"mandi": "Sirsa, Haryana", "price": 7600, "trend": "up"},
        ],
        "unit": "₹/quintal",
    },
    "soybean": {
        "name": "Soybean / सोयाबीन / सोयाबीन",
        "msp": 4892,
        "prices": [
            {"mandi": "Indore, MP", "price": 5200, "trend": "up"},
            {"mandi": "Akola, MH", "price": 5050, "trend": "stable"},
            {"mandi": "Kota, Rajasthan", "price": 4980, "trend": "up"},
            {"mandi": "Latur, MH", "price": 5100, "trend": "stable"},
        ],
        "unit": "₹/quintal",
    },
    "maize": {
        "name": "Maize / मक्का / मका",
        "msp": 2225,
        "prices": [
            {"mandi": "Davangere, KA", "price": 2450, "trend": "up"},
            {"mandi": "Gulbarga, KA", "price": 2380, "trend": "stable"},
            {"mandi": "Nizamabad, TS", "price": 2350, "trend": "up"},
            {"mandi": "Kurnool, AP", "price": 2400, "trend": "stable"},
        ],
        "unit": "₹/quintal",
    },
    "tomato": {
        "name": "Tomato / टमाटर / टोमॅटो",
        "msp": None,
        "prices": [
            {"mandi": "Nashik, MH", "price": 1800, "trend": "up"},
            {"mandi": "Kolar, KA", "price": 1500, "trend": "stable"},
            {"mandi": "Madanapalle, AP", "price": 1600, "trend": "down"},
            {"mandi": "Azadpur, Delhi", "price": 2200, "trend": "up"},
        ],
        "unit": "₹/quintal",
    },
    "sugarcane": {
        "name": "Sugarcane / गन्ना / ऊस",
        "msp": 340,
        "prices": [
            {"mandi": "Kolhapur, MH", "price": 360, "trend": "stable"},
            {"mandi": "Muzaffarnagar, UP", "price": 350, "trend": "up"},
            {"mandi": "Belgaum, KA", "price": 345, "trend": "stable"},
            {"mandi": "Coimbatore, TN", "price": 355, "trend": "stable"},
        ],
        "unit": "₹/quintal",
    },
    "mustard": {
        "name": "Mustard / सरसों / मोहरी",
        "msp": 5650,
        "prices": [
            {"mandi": "Alwar, Rajasthan", "price": 5900, "trend": "up"},
            {"mandi": "Agra, UP", "price": 5750, "trend": "stable"},
            {"mandi": "Bharatpur, RJ", "price": 5820, "trend": "up"},
            {"mandi": "Morena, MP", "price": 5700, "trend": "stable"},
        ],
        "unit": "₹/quintal",
    },
    "groundnut": {
        "name": "Groundnut / मूंगफली / भुईमूग",
        "msp": 6783,
        "prices": [
            {"mandi": "Rajkot, Gujarat", "price": 7200, "trend": "up"},
            {"mandi": "Junagadh, Gujarat", "price": 7100, "trend": "stable"},
            {"mandi": "Kurnool, AP", "price": 6900, "trend": "up"},
            {"mandi": "Bidar, KA", "price": 6800, "trend": "stable"},
        ],
        "unit": "₹/quintal",
    },
}


@market_bp.route("/prices", methods=["GET"])
def get_prices():
    crop = request.args.get("crop", "").lower()
    if crop and crop in MANDI_PRICES:
        return jsonify({
            "crop": crop,
            "data": MANDI_PRICES[crop],
            "last_updated": datetime.date.today().isoformat(),
            "source": "Indicative prices based on Agmarknet/e-NAM data",
        })
    # Return all crops summary
    summary = {}
    for key, val in MANDI_PRICES.items():
        prices = [p["price"] for p in val["prices"]]
        summary[key] = {
            "name": val["name"],
            "msp": val["msp"],
            "avg_price": round(sum(prices) / len(prices)),
            "max_price": max(prices),
            "min_price": min(prices),
            "unit": val["unit"],
        }
    return jsonify({
        "summary": summary,
        "last_updated": datetime.date.today().isoformat(),
        "source": "Indicative prices based on Agmarknet/e-NAM data",
        "disclaimer": "Prices are indicative. Check agmarknet.gov.in or enam.gov.in for live rates.",
    })


@market_bp.route("/advice", methods=["POST"])
def market_advice():
    data = request.get_json(force=True)
    crop: str = data.get("crop", "")
    location: str = data.get("location", "")
    language: str = data.get("language", "en")
    quantity: str = data.get("quantity", "")

    price_context = ""
    if crop.lower() in MANDI_PRICES:
        p = MANDI_PRICES[crop.lower()]
        prices_str = ", ".join(
            f"{item['mandi']}: ₹{item['price']}" for item in p["prices"]
        )
        price_context = f"\nCurrent mandi prices for {crop}: {prices_str} (per quintal)"
        if p["msp"]:
            price_context += f"\nMSP: ₹{p['msp']}/quintal"

    query = f"market selling advice for {crop} crop farmer"
    rag = get_rag_engine()
    rag_context = rag.build_rag_context(query, top_k=2)
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    prompt = (
        f"Provide market selling advice for a farmer with {crop} crop:\n"
        f"- Location: {location}\n"
        f"- Quantity to sell: {quantity} quintals\n"
        f"{price_context}\n\n"
        f"Advise on:\n"
        f"1. Best time and mandi to sell for maximum profit\n"
        f"2. Whether to sell now or store (price trend analysis)\n"
        f"3. Storage options and costs\n"
        f"4. Value addition opportunities\n"
        f"5. Direct buyer/FPO connections available\n"
        f"6. Government procurement options"
    )

    advice = generate_response(prompt, system_prompt)

    return jsonify({
        "crop": crop,
        "advice": advice,
        "language": language,
        "prices": MANDI_PRICES.get(crop.lower(), {}),
    })
