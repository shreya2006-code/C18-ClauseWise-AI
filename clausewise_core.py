"""Shared core utilities for ClauseWise AI prompt loading, Gemini API calls, and evaluation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEST_CASES_DIR = BASE_DIR / "test_cases"


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def list_test_cases() -> list[Path]:
    """Return sorted list of .txt files in test_cases/."""
    if not TEST_CASES_DIR.exists():
        return []
    return sorted(TEST_CASES_DIR.glob("*.txt"))


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Create a Google GenAI client using the provided key or GEMINI_API_KEY env var."""
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if not key or key == "your_key_here":
        raise ValueError(
            "GEMINI_API_KEY is missing or still set to placeholder. "
            "Set a valid key in your .env file."
        )
    return genai.Client(api_key=key)


def call_gemini(
    client: genai.Client,
    prompt_text: str,
    json_mode: bool = False,
    temperature: float = 0.1,
    max_retries: int = 4,
) -> tuple[str, float]:
    """Call Gemini model with transient-error backoff and return (response_text, latency_seconds)."""
    config_kwargs: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    for attempt in range(max_retries + 1):
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt_text,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            latency = round(time.perf_counter() - start, 3)
            text = (response.text or "").strip()
            return text, latency
        except genai_errors.APIError as exc:
            is_transient = getattr(exc, "code", None) in (429, 500, 502, 503, 504)
            if not is_transient or attempt == max_retries:
                raise
            wait_sec = 3 * (2 ** attempt)
            print(f"    [WARN] Transient API error ({exc.code}); retrying in {wait_sec}s...")
            time.sleep(wait_sec)


def strip_markdown_fences(text: str) -> str:
    """Remove surrounding ```json ... ``` or ``` ... ``` fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_safe(text: str) -> tuple[bool, Any]:
    """Attempt to parse JSON from model output. Returns (is_valid, parsed_obj)."""
    if not text:
        return False, None
    cleaned = strip_markdown_fences(text)
    try:
        return True, json.loads(cleaned)
    except json.JSONDecodeError:
        return False, None


def extract_risk_items(parsed: Any) -> list[dict[str, Any]]:
    """Extract the list of risk/clause dictionaries across v2 and v3 schemas."""
    if not isinstance(parsed, dict):
        return []
    for key in ("risk_table", "risks", "extracted_clauses"):
        items = parsed.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def verify_quotes_against_source(parsed: Any, source_text: str) -> tuple[bool, list[dict[str, Any]]]:
    """
    Verify that every non-'NOT FOUND' exact_quote appears verbatim in source_text.
    Returns (all_verified, per_item_results).
    """
    items = extract_risk_items(parsed)
    if not items:
        return False, []

    results: list[dict[str, Any]] = []
    all_ok = True

    for item in items:
        quote = str(item.get("exact_quote", "")).strip()
        if not quote:
            verified = False
            all_ok = False
        elif quote.upper() == "NOT FOUND":
            verified = True
        else:
            verified = quote in source_text
            if not verified:
                all_ok = False

        results.append(
            {
                "category": item.get("category", "Unknown"),
                "severity": item.get("severity", "Unknown"),
                "exact_quote": quote,
                "verified": verified,
            }
        )

    return all_ok, results


def has_legal_disclaimer(raw_text: str, parsed: Any = None) -> bool:
    """Check if the output includes a clear non-legal-advice disclaimer."""
    targets = [raw_text]
    if isinstance(parsed, dict) and parsed.get("disclaimer"):
        targets.append(str(parsed["disclaimer"]))

    combined = " ".join(targets).lower()
    markers = (
        "not legal advice",
        "not a substitute for legal advice",
        "not constitute legal advice",
        "no legal advice",
    )
    return any(marker in combined for marker in markers)
