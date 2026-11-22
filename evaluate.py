"""Automated prompt engineering evaluator for ClauseWise AI (v1, v2, v3)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from clausewise_core import (
    BASE_DIR,
    MODEL_NAME,
    call_gemini,
    get_genai_client,
    has_legal_disclaimer,
    list_test_cases,
    load_prompt,
    parse_json_safe,
    verify_quotes_against_source,
)

LOGS_DIR = BASE_DIR / "logs"
RATE_LIMIT_SLEEP_SECONDS = 2


def evaluate_single_run(
    prompt_version: str,
    tc_name: str,
    tc_text: str,
    raw_output: str,
    latency: float,
    step1_output: str = "",
) -> dict:
    """Compute automated evaluation metrics for a single prompt run."""
    word_count = len(raw_output.split())
    valid_json, parsed = parse_json_safe(raw_output)
    quote_verified, _ = verify_quotes_against_source(parsed, tc_text) if valid_json else (False, [])
    disclaimer_present = has_legal_disclaimer(raw_output, parsed)

    return {
        "test_case": tc_name,
        "prompt_version": prompt_version,
        "model": MODEL_NAME,
        "latency_sec": latency,
        "word_count": word_count,
        "valid_json": valid_json,
        "verbatim_quote_verified": quote_verified,
        "disclaimer_present": disclaimer_present,
        "source_excerpt": tc_text[:250].replace("\n", " ") + "...",
        "step1_output": step1_output,
        "full_output": raw_output,
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without extra dependencies."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        cells = [str(val).replace("|", "\\|").replace("\n", " ") for val in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_markdown_report(df: pd.DataFrame, report_path: Path) -> None:
    """Write a formatted Markdown summary report to logs/experiment_report.md."""
    summary_cols = [
        "test_case",
        "prompt_version",
        "latency_sec",
        "word_count",
        "valid_json",
        "verbatim_quote_verified",
        "disclaimer_present",
    ]
    display_df = df[summary_cols]

    agg = (
        df.groupby("prompt_version", sort=False)
        .agg(
            avg_latency_sec=("latency_sec", "mean"),
            avg_word_count=("word_count", "mean"),
            json_compliance_rate=("valid_json", "mean"),
            quote_verification_rate=("verbatim_quote_verified", "mean"),
            disclaimer_rate=("disclaimer_present", "mean"),
        )
        .reset_index()
    )
    agg["avg_latency_sec"] = agg["avg_latency_sec"].round(2)
    agg["avg_word_count"] = agg["avg_word_count"].round(1)
    for pct_col in ("json_compliance_rate", "quote_verification_rate", "disclaimer_rate"):
        agg[pct_col] = (agg[pct_col] * 100).round(1).astype(str) + "%"

    lines = [
        "# ClauseWise AI — Prompt Engineering Experiment Report",
        "",
        f"- **Model:** `{MODEL_NAME}`",
        f"- **Test Cases Evaluated:** {df['test_case'].nunique()}",
        f"- **Total Runs:** {len(df)}",
        "",
        "## Aggregate Comparison by Prompt Version",
        "",
        df_to_markdown(agg),
        "",
        "## Per-Run Evaluation Results",
        "",
        df_to_markdown(display_df),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    load_dotenv()
    try:
        client = get_genai_client()
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    test_cases = list_test_cases()
    if not test_cases:
        print("[ERROR] No .txt files found in test_cases/.", file=sys.stderr)
        return 1

    v1_template = load_prompt("v1_baseline.txt")
    v2_template = load_prompt("v2_structured_risk.txt")
    v3_step1_template = load_prompt("v3_step1_extractor.txt")
    v3_step2_template = load_prompt("v3_step2_simplifier.txt")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for tc_path in test_cases:
        tc_name = tc_path.name
        tc_text = tc_path.read_text(encoding="utf-8")
        print(f"\n=== Evaluating {tc_name} ===")

        # 1. v1_baseline
        print("  -> Running v1_baseline...")
        v1_prompt = v1_template.format(tc_text=tc_text)
        v1_out, v1_lat = call_gemini(client, v1_prompt, json_mode=False, temperature=0.3)
        records.append(evaluate_single_run("v1_baseline", tc_name, tc_text, v1_out, v1_lat))
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        # 2. v2_structured_risk
        print("  -> Running v2_structured_risk...")
        v2_prompt = v2_template.format(tc_text=tc_text)
        v2_out, v2_lat = call_gemini(client, v2_prompt, json_mode=True, temperature=0.1)
        records.append(evaluate_single_run("v2_structured_risk", tc_name, tc_text, v2_out, v2_lat))
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        # 3. v3_chained (Step 1: Extractor -> Step 2: Simplifier)
        print("  -> Running v3_chained (Step 1: Extractor)...")
        s1_prompt = v3_step1_template.format(tc_text=tc_text)
        s1_out, s1_lat = call_gemini(client, s1_prompt, json_mode=True, temperature=0.0)
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        print("  -> Running v3_chained (Step 2: Simplifier)...")
        s2_prompt = v3_step2_template.format(step1_json=s1_out)
        s2_out, s2_lat = call_gemini(client, s2_prompt, json_mode=True, temperature=0.1)
        total_v3_lat = round(s1_lat + s2_lat, 3)

        # Verify Step 2 quotes against source text, and keep Step 1 output for auditability
        records.append(
            evaluate_single_run(
                "v3_chained",
                tc_name,
                tc_text,
                s2_out,
                total_v3_lat,
                step1_output=json.dumps({"step1_latency_sec": s1_lat, "step1_json": s1_out}),
            )
        )
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

    df = pd.DataFrame(records)
    csv_path = LOGS_DIR / "experiment_log.csv"
    report_path = LOGS_DIR / "experiment_report.md"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    generate_markdown_report(df, report_path)

    print(f"\n[DONE] Saved CSV log to {csv_path}")
    print(f"[DONE] Saved Markdown report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
