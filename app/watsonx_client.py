"""IBM watsonx.ai client — ibm-watsonx-ai 1.1.x compatible."""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

_client: Optional[ModelInference] = None


def _build_client() -> ModelInference:
    api_key    = os.getenv("WATSONX_API_KEY", "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    url        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()

    if not api_key or api_key == "your_ibm_cloud_api_key_here":
        raise ValueError(
            "WATSONX_API_KEY is not set. "
            "Open .env and add your IBM Cloud API key."
        )
    if not project_id or project_id == "your_watsonx_project_id_here":
        raise ValueError(
            "WATSONX_PROJECT_ID is not set. "
            "Open .env and add your watsonx.ai Project ID."
        )

    credentials = Credentials(url=url, api_key=api_key)

    return ModelInference(
        model_id="ibm/granite-4-h-small",
        credentials=credentials,
        project_id=project_id,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 800,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.REPETITION_PENALTY: 1.1,
        },
    )


def get_watsonx_client() -> ModelInference:
    """Return a singleton ModelInference instance."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for HTTP 429 / consumption_limit_reached errors."""
    msg = str(exc).lower()
    return "429" in msg or "consumption_limit_reached" in msg or "rate limit" in msg


def _build_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Format a prompt string compatible with Granite instruct models.
    Uses the <|system|> / <|user|> / <|assistant|> chat template.
    """
    parts = []
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt.strip()}")
    parts.append(f"<|user|>\n{user_prompt.strip()}")
    parts.append("<|assistant|>")
    return "\n".join(parts)


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=3, min=10, max=90),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_generate(client: ModelInference, full_prompt: str) -> str:
    """Inner call with retry logic for 429 rate-limit responses."""
    return client.generate_text(prompt=full_prompt)


def generate_response(prompt: str, system_prompt: str = "") -> str:
    """Generate a response from IBM Granite via generate_text()."""
    try:
        client = get_watsonx_client()
        full_prompt = _build_prompt(system_prompt, prompt)
        result = _call_generate(client, full_prompt)
        return result.strip()
    except ValueError as e:
        return f"⚙️ Configuration error: {e}"
    except Exception as e:
        msg = str(e)
        logger.warning("watsonx.ai error: %s", msg)
        if "429" in msg or "consumption_limit_reached" in msg:
            return (
                "⏳ IBM watsonx.ai free-tier limit reached. "
                "Please wait 30–60 seconds and try again. "
                "(IBM Cloud Lite allows 10 concurrent requests.)"
            )
        return f"Error communicating with IBM watsonx.ai: {e}"
