"""Advisory route — irrigation, schemes, and seasonal advisories."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.prompts import build_system_prompt
from app.rag_engine import get_rag_engine
from app.watsonx_client import generate_response

advisory_bp = Blueprint("advisory", __name__)

GOVERNMENT_SCHEMES = [
    {
        "name": "PM-KISAN",
        "full_name": "Pradhan Mantri Kisan Samman Nidhi",
        "benefit": "₹6,000/year in 3 instalments",
        "eligibility": "All landholding farmer families",
        "portal": "pmkisan.gov.in",
        "category": "income_support",
    },
    {
        "name": "PMFBY",
        "full_name": "Pradhan Mantri Fasal Bima Yojana",
        "benefit": "Crop insurance at 2% (Kharif), 1.5% (Rabi) premium",
        "eligibility": "All farmers with insurable crops",
        "portal": "pmfby.gov.in",
        "category": "insurance",
    },
    {
        "name": "PMKSY",
        "full_name": "Pradhan Mantri Krishi Sinchayee Yojana",
        "benefit": "55% subsidy on micro-irrigation (drip/sprinkler) for small farmers",
        "eligibility": "Small & marginal farmers (< 2 ha)",
        "portal": "pmksy.gov.in",
        "category": "irrigation",
    },
    {
        "name": "KCC",
        "full_name": "Kisan Credit Card",
        "benefit": "Crop loans up to ₹3 lakh at 4% interest with timely repayment",
        "eligibility": "All farmers, sharecroppers, tenant farmers",
        "portal": "Bank branch / CSC",
        "category": "credit",
    },
    {
        "name": "e-NAM",
        "full_name": "National Agriculture Market",
        "benefit": "Online trading platform for competitive mandi prices",
        "eligibility": "All farmers with registered land/produce",
        "portal": "enam.gov.in",
        "category": "market",
    },
    {
        "name": "Soil Health Card",
        "full_name": "Soil Health Card Scheme",
        "benefit": "Free soil testing and nutrient management recommendations",
        "eligibility": "All farmers",
        "portal": "soilhealth.dac.gov.in",
        "category": "soil",
    },
]


@advisory_bp.route("/irrigation", methods=["POST"])
def irrigation_advice():
    data = request.get_json(force=True)
    crop: str = data.get("crop", "")
    soil_type: str = data.get("soil_type", "")
    area: str = data.get("area", "1")
    water_source: str = data.get("water_source", "canal")
    current_stage: str = data.get("current_stage", "")
    language: str = data.get("language", "en")

    query = f"irrigation schedule for {crop} crop in {soil_type} soil"
    rag = get_rag_engine()
    rag_context = rag.build_rag_context(query, top_k=3)
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    prompt = (
        f"Create a detailed irrigation schedule for:\n"
        f"- Crop: {crop}\n"
        f"- Soil type: {soil_type}\n"
        f"- Land area: {area} acres\n"
        f"- Water source: {water_source}\n"
        f"- Current crop stage: {current_stage}\n\n"
        f"Include:\n"
        f"1. Number of irrigations and timing\n"
        f"2. Water quantity per irrigation\n"
        f"3. Critical growth stages not to miss\n"
        f"4. Recommended irrigation method\n"
        f"5. Water saving tips\n"
        f"6. Signs of water stress and excess water"
    )

    advice = generate_response(prompt, system_prompt)
    return jsonify({"crop": crop, "advice": advice, "language": language})


@advisory_bp.route("/seasonal", methods=["POST"])
def seasonal_advisory():
    data = request.get_json(force=True)
    season: str = data.get("season", "kharif")
    location: str = data.get("location", "Maharashtra")
    crops: list = data.get("crops", [])
    language: str = data.get("language", "en")

    query = f"seasonal farming advisory {season} season {location} {' '.join(crops)}"
    rag = get_rag_engine()
    rag_context = rag.build_rag_context(query, top_k=4)
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    crops_str = ", ".join(crops) if crops else "general crops"
    prompt = (
        f"Provide a comprehensive seasonal farming advisory for:\n"
        f"- Season: {season}\n"
        f"- Location: {location}\n"
        f"- Crops: {crops_str}\n\n"
        f"Cover:\n"
        f"1. Key activities for this month\n"
        f"2. Pest and disease watch-list for the season\n"
        f"3. Nutrient management reminders\n"
        f"4. Irrigation planning\n"
        f"5. Market preparation advice\n"
        f"6. Government scheme deadlines to be aware of"
    )

    advisory = generate_response(prompt, system_prompt)
    return jsonify({"season": season, "advisory": advisory, "language": language})


@advisory_bp.route("/schemes", methods=["GET"])
def list_schemes():
    category = request.args.get("category", "")
    if category:
        filtered = [s for s in GOVERNMENT_SCHEMES if s["category"] == category]
        return jsonify({"schemes": filtered})
    return jsonify({"schemes": GOVERNMENT_SCHEMES})


@advisory_bp.route("/schemes/advice", methods=["POST"])
def scheme_advice():
    data = request.get_json(force=True)
    farmer_profile: dict = data.get("farmer_profile", {})
    language: str = data.get("language", "en")

    system_prompt = build_system_prompt(language=language)
    land = farmer_profile.get("land_area", "unknown")
    crops = ", ".join(farmer_profile.get("crops", ["general"]))
    state = farmer_profile.get("state", "India")

    prompt = (
        f"Recommend the most relevant government schemes for this farmer:\n"
        f"- Land area: {land} acres\n"
        f"- State: {state}\n"
        f"- Crops: {crops}\n\n"
        f"For each recommended scheme:\n"
        f"1. Scheme name and benefit\n"
        f"2. How to apply (steps)\n"
        f"3. Documents required\n"
        f"4. Important deadlines\n"
        f"5. Contact: portal or helpline number"
    )

    advice = generate_response(prompt, system_prompt)
    return jsonify({"advice": advice, "language": language})
