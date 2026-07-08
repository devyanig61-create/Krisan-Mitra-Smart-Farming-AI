"""Crops route — AI-powered crop recommendations using RAG + watsonx."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.prompts import build_system_prompt
from app.rag_engine import get_rag_engine
from app.watsonx_client import generate_response

crops_bp = Blueprint("crops", __name__)

# Static crop data for quick UI display (top crops)
CROPS_DATA = {
    "wheat":     {"name": "Wheat",    "hindi": "गेहूं",   "marathi": "गहू",    "season": "Rabi",   "soil": "Loamy", "icon": "🌾"},
    "rice":      {"name": "Rice",     "hindi": "चावल",   "marathi": "तांदूळ", "season": "Kharif", "soil": "Clay",  "icon": "🌾"},
    "cotton":    {"name": "Cotton",   "hindi": "कपास",   "marathi": "कापूस",  "season": "Kharif", "soil": "Black", "icon": "🌿"},
    "sugarcane": {"name": "Sugarcane","hindi": "गन्ना",  "marathi": "ऊस",     "season": "Annual", "soil": "Loamy", "icon": "🎋"},
    "soybean":   {"name": "Soybean",  "hindi": "सोयाबीन","marathi": "सोयाबीन","season": "Kharif", "soil": "Loamy", "icon": "🫛"},
    "maize":     {"name": "Maize",    "hindi": "मक्का",  "marathi": "मका",    "season": "Kharif", "soil": "Sandy", "icon": "🌽"},
    "tomato":    {"name": "Tomato",   "hindi": "टमाटर",  "marathi": "टोमॅटो", "season": "Annual", "soil": "Sandy", "icon": "🍅"},
    "mustard":   {"name": "Mustard",  "hindi": "सरसों",  "marathi": "मोहरी",  "season": "Rabi",   "soil": "Loamy", "icon": "🌻"},
}


@crops_bp.route("/recommend", methods=["POST"])
def recommend_crops():
    data = request.get_json(force=True)
    soil_type: str = data.get("soil_type", "")
    season: str = data.get("season", "")
    location: str = data.get("location", "")
    language: str = data.get("language", "en")
    water_availability: str = data.get("water_availability", "medium")
    land_area: str = data.get("land_area", "")

    query = (
        f"Best crops for {soil_type} soil in {season} season at {location}. "
        f"Water availability: {water_availability}. "
        f"Provide top 3 crop recommendations with expected yield and care tips."
    )

    rag = get_rag_engine()
    rag_context = rag.build_rag_context(query, top_k=5)
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    prompt = (
        f"Recommend the top 3 most suitable crops for the following conditions:\n"
        f"- Soil type: {soil_type}\n"
        f"- Season: {season}\n"
        f"- Location: {location}\n"
        f"- Water availability: {water_availability}\n"
        f"- Land area: {land_area} acres\n\n"
        f"For each crop provide: suitability reason, expected yield, key care tips, "
        f"and estimated profit potential."
    )

    recommendation = generate_response(prompt, system_prompt)

    return jsonify({
        "recommendation": recommendation,
        "soil_type": soil_type,
        "season": season,
        "location": location,
        "language": language,
    })


@crops_bp.route("/list", methods=["GET"])
def list_crops():
    return jsonify({"crops": CROPS_DATA})


@crops_bp.route("/disease", methods=["POST"])
def identify_disease():
    data = request.get_json(force=True)
    crop: str = data.get("crop", "")
    symptoms: str = data.get("symptoms", "")
    language: str = data.get("language", "en")

    query = f"{crop} crop disease symptoms: {symptoms}"
    rag = get_rag_engine()
    rag_context = rag.build_rag_context(query)
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    prompt = (
        f"Identify the disease or pest affecting {crop} crop with these symptoms: {symptoms}\n\n"
        f"Provide:\n"
        f"1. Most likely disease/pest name\n"
        f"2. Cause and spread mechanism\n"
        f"3. Preventive measures\n"
        f"4. Organic treatment options\n"
        f"5. Chemical control with product name, dosage, and application method\n"
        f"6. Expected recovery time"
    )

    diagnosis = generate_response(prompt, system_prompt)

    return jsonify({
        "crop": crop,
        "symptoms": symptoms,
        "diagnosis": diagnosis,
        "language": language,
    })


@crops_bp.route("/fertilizer", methods=["POST"])
def fertilizer_schedule():
    data = request.get_json(force=True)
    crop: str = data.get("crop", "")
    soil_type: str = data.get("soil_type", "")
    area: str = data.get("area", "1")
    language: str = data.get("language", "en")

    query = f"fertilizer schedule for {crop} in {soil_type} soil"
    rag = get_rag_engine()
    rag_context = rag.build_rag_context(query, top_k=3)
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    prompt = (
        f"Create a complete fertilizer schedule for {crop} cultivation:\n"
        f"- Soil type: {soil_type}\n"
        f"- Land area: {area} acres\n\n"
        f"Include:\n"
        f"1. Basal dose (at sowing/transplanting)\n"
        f"2. Top dressing schedule with timing and quantity\n"
        f"3. Micronutrient recommendations\n"
        f"4. Organic alternatives\n"
        f"5. Estimated cost of fertilizers\n"
        f"6. Signs of nutrient deficiency to watch for"
    )

    schedule = generate_response(prompt, system_prompt)

    return jsonify({
        "crop": crop,
        "soil_type": soil_type,
        "area": area,
        "schedule": schedule,
        "language": language,
    })
