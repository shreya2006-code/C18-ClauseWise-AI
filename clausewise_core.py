"""Shared core utilities for ClauseWise AI prompt loading, Gemini API calls, and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_INPUT_CHARS = 9000
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
TEST_CASES_DIR = BASE_DIR / "test_cases"
DISK_CACHE_DIR = BASE_DIR / ".cache"

LEGAL_SIGNAL_TERMS = (
    "privacy",
    "data",
    "fee",
    "renew",
    "arbitration",
    "dispute",
    "license",
    "content",
    "terminate",
    "termination",
    "cancel",
    "liability",
    "indemnif",
    "refund",
    "charge",
    "waive",
    "track",
    "biometric",
    "sell",
    "share",
    "train",
)

DEFAULT_DISCLAIMER = (
    "Disclaimer: This summary provides plain-language information for educational "
    "purposes only and is NOT legal advice."
)


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def list_test_cases() -> list[Path]:
    """Return sorted list of .txt files in test_cases/."""
    if not TEST_CASES_DIR.exists():
        return []
    return sorted(TEST_CASES_DIR.glob("*.txt"))


def resolve_api_key(api_key: str | None = None) -> str:
    """Resolve GEMINI_API_KEY from explicit arg, environment, or Streamlit secrets."""
    key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
    if key and key != "your_key_here":
        return key

    try:
        import streamlit as st

        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if secret_key and secret_key != "your_key_here":
            return secret_key
    except Exception:
        pass

    return ""


def has_configured_api_key() -> bool:
    """Return True if a non-placeholder GEMINI_API_KEY is available."""
    return bool(resolve_api_key())


@lru_cache(maxsize=4)
def _cached_client(api_key: str) -> genai.Client:
    """Cache and reuse a single genai.Client instance per API key."""
    return genai.Client(api_key=api_key)


def get_genai_client(api_key: str | None = None) -> genai.Client:
    """Return a reused Google GenAI client using env or Streamlit secrets."""
    key = resolve_api_key(api_key)
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing or still set to placeholder. "
            "Set a valid key in .env or Streamlit Cloud secrets."
        )
    return _cached_client(key)


def truncate_tc_for_llm(tc_text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """
    Prioritize high-signal legal paragraphs and cap input at max_chars (9000)
    while preserving original paragraph order.
    """
    if len(tc_text) <= max_chars:
        return tc_text

    paragraphs = [p.strip() for p in re.split(r"\r?\n\s*\r?\n", tc_text) if p.strip()]
    if len(paragraphs) <= 1:
        return tc_text[:max_chars]

    scored = []
    for idx, para in enumerate(paragraphs):
        lower = para.lower()
        hits = sum(1 for term in LEGAL_SIGNAL_TERMS if term in lower)
        # Always prioritize title/intro paragraph (idx == 0), then highest keyword density
        priority = (1000 if idx == 0 else hits, -idx)
        scored.append((priority, idx, para))

    scored.sort(reverse=True)
    selected_indices: set[int] = set()
    used_chars = 0

    for _, idx, para in scored:
        addition = len(para) + (2 if selected_indices else 0)
        if used_chars + addition <= max_chars:
            selected_indices.add(idx)
            used_chars += addition

    if not selected_indices:
        return tc_text[:max_chars]

    ordered = [paragraphs[i] for i in sorted(selected_indices)]
    return "\n\n".join(ordered)


def _disk_cache_path(prompt_key: str, tc_text: str) -> Path:
    digest = hashlib.sha256(f"{prompt_key}\0{tc_text}".encode("utf-8")).hexdigest()
    return DISK_CACHE_DIR / f"{digest}.json"


def get_disk_cached_result(prompt_key: str, tc_text: str) -> dict | None:
    """Load cached analysis result from .cache/ if present."""
    path = _disk_cache_path(prompt_key, tc_text)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_disk_cached_result(prompt_key: str, tc_text: str, result: dict) -> None:
    """Persist analysis result to .cache/ for instant cross-restart retrieval."""
    try:
        DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _disk_cache_path(prompt_key, tc_text)
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _build_thinking_config() -> types.ThinkingConfig:
    """Build minimal-latency ThinkingConfig using ThinkingLevel.MINIMAL."""
    return types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)


def call_gemini(
    client: genai.Client,
    prompt_text: str,
    json_mode: bool = False,
    temperature: float = 0.1,
    max_output_tokens: int = 1024,
    max_retries: int = 4,
) -> tuple[str, float]:
    """Call Gemini model with minimal thinking latency and transient-error backoff."""
    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "thinking_config": _build_thinking_config(),
    }
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
            code = getattr(exc, "code", None)
            is_transient = code in (429, 500, 502, 503, 504)
            if not is_transient or attempt == max_retries:
                raise
            wait_sec = 3 * (2 ** attempt)
            print(f"    [WARN] Transient API error ({code}); retrying in {wait_sec}s...")
            time.sleep(wait_sec)

    raise RuntimeError("Exceeded maximum retries calling Gemini API.")


def _is_not_found_clause(clause: dict[str, Any]) -> bool:
    severity = str(clause.get("severity", "")).strip().lower()
    quote = str(clause.get("exact_quote", "")).strip().upper()
    status = str(clause.get("status", "")).strip().upper()
    return severity == "none" or quote == "NOT FOUND" or status == "NOT FOUND"


def _build_clean_row(category: str) -> dict[str, Any]:
    return {
        "category": category,
        "severity": "None",
        "exact_quote": "NOT FOUND",
        "plain_english_summary": "No concerning clause found in this section.",
        "actionable_tip": "No action needed; keep a copy of these terms for your records.",
    }


def run_v3_chained(client: genai.Client, tc_text: str) -> dict[str, Any]:
    """
    Execute v3 2-step pipeline with smart clause windowing and Step 2 skipping
    for 'NOT FOUND' / 'None' clauses.
    """
    llm_input = truncate_tc_for_llm(tc_text)

    s1_prompt = load_prompt("v3_step1_extractor.txt").format(tc_text=llm_input)
    s1_out, s1_lat = call_gemini(
        client, s1_prompt, json_mode=True, temperature=0.1, max_output_tokens=750
    )

    s1_valid, s1_parsed = parse_json_safe(s1_out)
    if not s1_valid or not isinstance(s1_parsed, dict):
        # Fallback to full Step 2 if Step 1 JSON failed unexpectedly
        s2_prompt = load_prompt("v3_step2_simplifier.txt").format(step1_json=s1_out)
        s2_out, s2_lat = call_gemini(
            client, s2_prompt, json_mode=True, temperature=0.1, max_output_tokens=1100
        )
        return {
            "prompt_key": "v3_chained",
            "raw_output": s2_out,
            "step1_output": s1_out,
            "latency": round(s1_lat + s2_lat, 3),
            "step1_latency": s1_lat,
            "step2_latency": s2_lat,
        }

    clauses = s1_parsed.get("extracted_clauses", [])
    if not isinstance(clauses, list):
        clauses = []

    found_clauses = [c for c in clauses if isinstance(c, dict) and not _is_not_found_clause(c)]

    # Zero risks found: skip Step 2 LLM call completely!
    if not found_clauses:
        clean_table = [
            _build_clean_row(str(c.get("category", "Category")))
            for c in clauses
            if isinstance(c, dict)
        ]
        merged = {
            "tldr_summary": (
                "No concerning consumer risks were found across the five audited categories. "
                "These terms appear fair and transparent."
            ),
            "risk_table": clean_table,
            "disclaimer": DEFAULT_DISCLAIMER,
        }
        return {
            "prompt_key": "v3_chained",
            "raw_output": json.dumps(merged, ensure_ascii=False),
            "step1_output": s1_out,
            "latency": s1_lat,
            "step1_latency": s1_lat,
            "step2_latency": 0.0,
        }

    # Send ONLY active risky clauses to Step 2
    filtered_s1 = {
        "document_tone": s1_parsed.get("document_tone", "Restrictive"),
        "extracted_clauses": found_clauses,
    }
    s2_prompt = load_prompt("v3_step2_simplifier.txt").format(
        step1_json=json.dumps(filtered_s1, ensure_ascii=False)
    )
    s2_out, s2_lat = call_gemini(
        client, s2_prompt, json_mode=True, temperature=0.1, max_output_tokens=1100
    )

    s2_valid, s2_parsed = parse_json_safe(s2_out)
    if s2_valid and isinstance(s2_parsed, dict):
        s2_rows = {
            str(r.get("category", "")).strip(): r
            for r in s2_parsed.get("risk_table", [])
            if isinstance(r, dict)
        }
        merged_table = []
        for c in clauses:
            if not isinstance(c, dict):
                continue
            cat = str(c.get("category", "")).strip()
            if _is_not_found_clause(c):
                merged_table.append(_build_clean_row(cat))
            elif cat in s2_rows:
                merged_table.append(s2_rows[cat])

        # Include any extra rows Step 2 returned under slightly altered category names
        seen_cats = {str(r.get("category", "")).strip() for r in merged_table}
        for r in s2_parsed.get("risk_table", []):
            if isinstance(r, dict) and str(r.get("category", "")).strip() not in seen_cats:
                merged_table.append(r)

        s2_parsed["risk_table"] = merged_table
        if not s2_parsed.get("disclaimer"):
            s2_parsed["disclaimer"] = DEFAULT_DISCLAIMER
        final_raw = json.dumps(s2_parsed, ensure_ascii=False)
    else:
        final_raw = s2_out

    return {
        "prompt_key": "v3_chained",
        "raw_output": final_raw,
        "step1_output": s1_out,
        "latency": round(s1_lat + s2_lat, 3),
        "step1_latency": s1_lat,
        "step2_latency": s2_lat,
    }


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


def _quote_matches_source(quote: str, source_text: str, norm_source: str, compact_source: str) -> bool:
    """Check exact, whitespace-normalized, and space-stripped lowercase substring match."""
    if quote in source_text:
        return True
    norm_quote = re.sub(r"\s+", " ", quote).strip()
    if norm_quote and norm_quote in norm_source:
        return True
    compact_quote = re.sub(r"\s+", "", quote).lower()
    return bool(compact_quote) and compact_quote in compact_source


def verify_quotes_against_source(parsed: Any, source_text: str) -> tuple[bool, list[dict[str, Any]]]:
    """
    Verify that every non-'NOT FOUND' exact_quote appears in source_text
    (exact, whitespace-normalized, or space-stripped fallback).
    Returns (all_verified, per_item_results).
    """
    items = extract_risk_items(parsed)
    if not items:
        return False, []

    norm_source = re.sub(r"\s+", " ", source_text).strip()
    compact_source = re.sub(r"\s+", "", source_text).lower()

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
            verified = _quote_matches_source(quote, source_text, norm_source, compact_source)
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
