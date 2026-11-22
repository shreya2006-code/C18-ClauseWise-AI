"""Streamlit UI for ClauseWise AI — Terms & Conditions Plain-Language Analyzer (C18)."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from clausewise_core import (
    MODEL_NAME,
    call_gemini,
    extract_risk_items,
    get_genai_client,
    has_legal_disclaimer,
    list_test_cases,
    load_prompt,
    parse_json_safe,
    verify_quotes_against_source,
)

load_dotenv()

st.set_page_config(
    page_title="ClauseWise AI — T&C Analyzer (C18)",
    page_icon="⚖️",
    layout="wide",
)

PROMPT_OPTIONS = {
    "v1 (Baseline)": "v1_baseline",
    "v2 (Structured Single-Prompt)": "v2_structured_risk",
    "v3 (2-Step Chained + Verified)": "v3_chained",
}


def render_disclaimer_banner() -> None:
    """Display the prominent Responsible-AI non-legal-advice disclaimer."""
    st.warning(
        "**Responsible AI & Legal Disclaimer:** ClauseWise AI provides plain-language "
        "summaries for educational and consumer-literacy purposes only. This output is **NOT "
        "legal advice** and does not create an attorney-client relationship. Always consult a "
        "qualified legal professional for binding legal guidance."
    )


def render_footer() -> None:
    """Display clean project title in the app footer."""
    st.divider()
    st.caption("ClauseWise AI — Terms & Conditions Plain-Language Analyzer (Problem C18)")


def extract_uploaded_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF or TXT file."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    raw_bytes = uploaded_file.getvalue()
    return raw_bytes.decode("utf-8", errors="replace")


@st.cache_data(show_spinner=False)
def run_analysis(prompt_key: str, tc_text: str) -> dict:
    """Execute the selected prompt version against Gemini (cached per prompt + input)."""
    client = get_genai_client()

    if prompt_key == "v1_baseline":
        prompt = load_prompt("v1_baseline.txt").format(tc_text=tc_text)
        raw_out, latency = call_gemini(client, prompt, json_mode=False, temperature=0.3)
        return {
            "prompt_key": prompt_key,
            "raw_output": raw_out,
            "step1_output": None,
            "latency": latency,
        }

    if prompt_key == "v2_structured_risk":
        prompt = load_prompt("v2_structured_risk.txt").format(tc_text=tc_text)
        raw_out, latency = call_gemini(client, prompt, json_mode=True, temperature=0.1)
        return {
            "prompt_key": prompt_key,
            "raw_output": raw_out,
            "step1_output": None,
            "latency": latency,
        }

    # v3_chained: Step 1 (Extractor) -> Step 2 (Simplifier)
    s1_prompt = load_prompt("v3_step1_extractor.txt").format(tc_text=tc_text)
    s1_out, s1_lat = call_gemini(client, s1_prompt, json_mode=True, temperature=0.1)

    s2_prompt = load_prompt("v3_step2_simplifier.txt").format(step1_json=s1_out)
    s2_out, s2_lat = call_gemini(client, s2_prompt, json_mode=True, temperature=0.1)

    return {
        "prompt_key": prompt_key,
        "raw_output": s2_out,
        "step1_output": s1_out,
        "latency": round(s1_lat + s2_lat, 3),
        "step1_latency": s1_lat,
        "step2_latency": s2_lat,
    }


def main() -> None:
    st.title("ClauseWise AI — Terms & Conditions Plain-Language Analyzer (C18)")
    render_disclaimer_banner()

    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        selected_label = st.radio(
            "Prompt Version",
            options=list(PROMPT_OPTIONS.keys()),
            index=2,
            help="Compare v1 zero-shot, v2 structured single-prompt, and v3 2-step verified chaining.",
        )
        prompt_key = PROMPT_OPTIONS[selected_label]

        test_files = list_test_cases()
        sample_names = [f.name for f in test_files] + ["Custom Input (Paste Below)"]
        selected_sample = st.selectbox("Load Sample T&C", options=sample_names, index=0)

        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key and env_key != "your_key_here":
            st.success("API Connected (.env)")
        else:
            st.error("GEMINI_API_KEY missing in .env")

        st.caption(f"Model: `{MODEL_NAME}`")

    # File uploader above text area
    uploaded_file = st.file_uploader(
        "Upload Terms & Conditions (PDF or TXT)",
        type=["pdf", "txt"],
    )

    default_text = ""
    if uploaded_file is not None:
        try:
            default_text = extract_uploaded_text(uploaded_file)
            if not default_text.strip():
                st.warning("Uploaded file appears to contain no extractable text.")
        except Exception as exc:
            st.error(f"Could not read uploaded file: {exc}")
    elif selected_sample != "Custom Input (Paste Below)":
        matched = next((f for f in test_files if f.name == selected_sample), None)
        if matched:
            default_text = matched.read_text(encoding="utf-8")

    tc_text = st.text_area(
        "Terms & Conditions Input Text",
        value=default_text,
        height=240,
        placeholder="Paste Terms & Conditions text here, upload a PDF/TXT above, or select a sample from the sidebar...",
    )

    analyze_clicked = st.button("Analyze Terms & Conditions", type="primary")

    if analyze_clicked:
        if not tc_text.strip():
            st.error("Please provide Terms & Conditions text to analyze.")
        else:
            with st.spinner(f"Running {selected_label} with `{MODEL_NAME}`..."):
                try:
                    st.session_state["last_result"] = run_analysis(prompt_key, tc_text)
                    st.session_state["last_source"] = tc_text
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("last_result")
    source_text = st.session_state.get("last_source", tc_text)

    if result:
        raw_output = result["raw_output"]
        valid_json, parsed = parse_json_safe(raw_output)
        disclaimer_ok = has_legal_disclaimer(raw_output, parsed)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Prompt Version", result["prompt_key"])
        col2.metric("Latency (s)", f"{result['latency']}s")
        col3.metric("Valid JSON", "Yes" if valid_json else "No")
        col4.metric("Disclaimer Guardrail", "Present" if disclaimer_ok else "Missing")

        tab_summary, tab_audit = st.tabs(
            ["Plain-English Summary & Risk Table", "Raw JSON / Step-1 Audit"]
        )

        with tab_summary:
            st.subheader("(a) Plain-English TL;DR Summary")
            if valid_json and isinstance(parsed, dict):
                tldr = parsed.get("tldr_summary", "No summary provided.")
                st.info(tldr)

                st.subheader("(b) Risk Severity Table & Verbatim Quote Verification")
                all_verified, quote_checks = verify_quotes_against_source(parsed, source_text)
                items = extract_risk_items(parsed)

                if items:
                    if all_verified:
                        st.success("All extracted quotes verified verbatim against the source T&C.")
                    else:
                        st.error("One or more quotes could not be verified verbatim in the source text.")

                    rows = []
                    for item, check in zip(items, quote_checks):
                        explanation = (
                            item.get("plain_english_summary")
                            or item.get("plain_english_explanation")
                            or ""
                        )
                        row = {
                            "Category": item.get("category", ""),
                            "Severity": item.get("severity", ""),
                            "Exact Quote": item.get("exact_quote", ""),
                            "Verbatim Match": "✅ Verified" if check["verified"] else "❌ Mismatch",
                            "Plain-English Explanation": explanation,
                        }
                        if "actionable_tip" in item:
                            row["Actionable Tip"] = item["actionable_tip"]
                        rows.append(row)

                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("JSON parsed, but no risk items were found in the response.")
            else:
                st.write(raw_output)

        with tab_audit:
            if result.get("step1_output"):
                st.subheader("Step 1 — Verbatim Clause Extractor Output (Audit)")
                s1_valid, s1_parsed = parse_json_safe(result["step1_output"])
                if s1_valid:
                    st.json(s1_parsed)
                else:
                    st.code(result["step1_output"], language="json")

            st.subheader("Final Output / Raw JSON")
            if valid_json:
                st.json(parsed)
            else:
                st.code(raw_output, language="text")

    render_footer()


if __name__ == "__main__":
    main()
