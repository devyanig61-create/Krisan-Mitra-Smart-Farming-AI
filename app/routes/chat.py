"""Chat route — handles multilingual RAG + watsonx.ai conversations."""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.prompts import build_system_prompt
from app.rag_engine import get_rag_engine
from app.watsonx_client import generate_response

chat_bp = Blueprint("chat", __name__)

# Conversation history per session (in-memory, keyed by session_id)
_sessions: dict[str, list[dict]] = {}


@chat_bp.route("/", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_message: str = data.get("message", "").strip()
    language: str = data.get("language", "en")
    session_id: str = data.get("session_id", "default")
    farmer_context: dict = data.get("farmer_context", {})

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Build enriched query from farmer context
    enriched_query = _enrich_query(user_message, farmer_context)

    # RAG retrieval
    rag = get_rag_engine()
    rag_context = rag.build_rag_context(enriched_query)

    # Build system prompt
    system_prompt = build_system_prompt(language=language, rag_context=rag_context)

    # Maintain conversation history
    history = _sessions.get(session_id, [])

    # Build full prompt with history
    full_prompt = _build_conversation_prompt(history, user_message, farmer_context)

    # Generate response
    response_text = generate_response(full_prompt, system_prompt)

    # Update history (keep last 6 turns)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response_text})
    _sessions[session_id] = history[-12:]

    return jsonify({
        "response": response_text,
        "language": language,
        "session_id": session_id,
        "rag_used": bool(rag_context),
    })


@chat_bp.route("/history/<session_id>", methods=["GET"])
def get_history(session_id: str):
    history = _sessions.get(session_id, [])
    return jsonify({"history": history})


@chat_bp.route("/clear/<session_id>", methods=["DELETE"])
def clear_history(session_id: str):
    _sessions.pop(session_id, None)
    return jsonify({"message": "Session cleared"})


def _enrich_query(message: str, ctx: dict) -> str:
    parts = [message]
    if ctx.get("soil_type"):
        parts.append(f"soil type: {ctx['soil_type']}")
    if ctx.get("location"):
        parts.append(f"location: {ctx['location']}")
    if ctx.get("season"):
        parts.append(f"season: {ctx['season']}")
    if ctx.get("crop"):
        parts.append(f"crop: {ctx['crop']}")
    return " ".join(parts)


def _build_conversation_prompt(history: list, current_message: str, ctx: dict) -> str:
    context_str = ""
    if ctx:
        ctx_parts = []
        if ctx.get("name"):
            ctx_parts.append(f"Farmer name: {ctx['name']}")
        if ctx.get("location"):
            ctx_parts.append(f"Location: {ctx['location']}")
        if ctx.get("soil_type"):
            ctx_parts.append(f"Soil type: {ctx['soil_type']}")
        if ctx.get("season"):
            ctx_parts.append(f"Season: {ctx['season']}")
        if ctx.get("crop"):
            ctx_parts.append(f"Current crop: {ctx['crop']}")
        if ctx.get("land_area"):
            ctx_parts.append(f"Land area: {ctx['land_area']} acres")
        if ctx_parts:
            context_str = "Farmer Profile:\n" + "\n".join(ctx_parts) + "\n\n"

    history_str = ""
    for turn in history[-6:]:
        role = "Farmer" if turn["role"] == "user" else "KisanMitra"
        history_str += f"{role}: {turn['content']}\n"

    return f"{context_str}{history_str}Farmer: {current_message}\nKisanMitra:"
