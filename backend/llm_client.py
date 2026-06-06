"""
llm_client.py
Shared Groq client with 4-key fallback.

Use this everywhere instead of directly calling Groq().
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY"),  # backward-compatible fallback
]

GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    raise RuntimeError(
        "No Groq API key found. Set GROQ_API_KEY_1, GROQ_API_KEY_2, "
        "GROQ_API_KEY_3, GROQ_API_KEY_4, or GROQ_API_KEY."
    )


def call_groq_chat(
    messages: list[dict],
    model: str | None = None,
    fallback_model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> str:
    """
    Calls Groq with 4-key fallback.

    Try order:
    - all keys with primary model
    - all keys with fallback model

    Returns assistant text.
    Raises RuntimeError only if all attempts fail.
    """

    primary_model = model or GROQ_MODEL
    fallback = fallback_model or GROQ_FALLBACK_MODEL

    models_to_try = [primary_model]
    if fallback and fallback != primary_model:
        models_to_try.append(fallback)

    last_error = None

    for current_model in models_to_try:
        for index, api_key in enumerate(GROQ_API_KEYS, start=1):
            try:
                client = Groq(api_key=api_key)

                response = client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = response.choices[0].message.content

                if content and content.strip():
                    print(f"[llm] success key={index} model={current_model}")
                    return content.strip()

                last_error = RuntimeError("Empty Groq response")
                print(f"[llm] empty response key={index} model={current_model}")

            except Exception as error:
                last_error = error
                print(f"[llm] failed key={index} model={current_model} error={error}")
                continue

    raise RuntimeError(f"All Groq attempts failed. Last error: {last_error}")


def call_groq_json(
    messages: list[dict],
    model: str | None = None,
    fallback_model: str | None = None,
    temperature: float = 0,
    max_tokens: int = 200,
) -> dict:
    """
    Calls Groq and parses a JSON object safely.
    Returns {} if parsing fails.
    """

    text = call_groq_chat(
        messages=messages,
        model=model,
        fallback_model=fallback_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return {}

    return {}
