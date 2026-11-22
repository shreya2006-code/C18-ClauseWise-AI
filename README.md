# ClauseWise AI — Terms & Conditions Plain-Language Analyzer (Problem C18)

A Prompt Engineering mini-project for **Bennett University B.Tech 5th Semester** (Problem **C18**: *"Terms and Conditions in Plain Language"*).

ClauseWise AI demonstrates iterative prompt engineering—from a zero-shot baseline (`v1`), to a role-based structured JSON prompt (`v2`), to a hallucination-resistant **2-step Prompt Chaining** pipeline (`v3`)—powered by the Google GenAI Python SDK (`gemini-3.5-flash-lite`) and an interactive Streamlit interface.

---

## Team Members

| Name | Enrollment ID | Role |
| :--- | :--- | :--- |
| **Shreya Ghosh** | `S24CSEU0519` | Prompt Architect & Chaining Design |
| **Saneha Singh** | `S24CSEU0462` | Dataset Curation & Evaluation Rubric |
| **Shreyash Mishra** | `S24CSEU0523` | Streamlit UI & API Integration |
| **Prithivi Alamyan** | `S24CSEU1155` | Error Analysis & Hallucination Testing |
| **Shree Bhattacharya** | `S24CSEU0516` | Responsible AI Guardrails & Documentation |

---

## Prompt Engineering Progression (`prompts/`)

1. **`prompts/v1_baseline.txt`** — Basic zero-shot prompt asking the model to summarize a Terms & Conditions document and list risks. Produces unstructured prose with no quote grounding or disclaimer guarantee.
2. **`prompts/v2_structured_risk.txt`** — Role-based prompt (*"Consumer Legal Literacy Assistant"*) enforcing:
   - 5 fixed risk categories: `Data Privacy`, `Hidden Fees/Auto-Renewal`, `Forced Arbitration`, `Content Ownership`, `Account Termination`
   - Severity levels (`High`, `Medium`, `Low`, `None`)
   - Strict JSON schema output
   - Mandatory non-legal-advice disclaimer guardrail
3. **`prompts/v3_step1_extractor.txt` & `prompts/v3_step2_simplifier.txt`** — **2-Step Prompt Chaining**:
   - **Step 1 (Extractor):** Audits the raw T&C text, extracts character-for-character verbatim quotes into JSON, and explicitly outputs `"NOT FOUND"` when a risk category is absent (preventing hallucinated clauses).
   - **Step 2 (Simplifier):** Receives **only** Step 1's verified JSON (never the raw T&C) and translates it into an 8th-grade reading level plain-English summary table with actionable user tips and the non-legal-advice disclaimer.

---

## Test Suite (`test_cases/`)

Five realistic consumer Terms & Conditions documents, including a negative control test:
- `01_social_media_app.txt` — Broad IP license over user photos, third-party ad data sharing, binding arbitration.
- `02_subscription_streaming.txt` — Silent auto-renewal, non-refundable price hikes, unilateral termination.
- `03_food_delivery_app.txt` — Dynamic surge/platform fees, zero liability for delays, continuous background GPS tracking.
- `04_ai_productivity_tool.txt` — AI model training on user inputs, indefinite retention/human review, user indemnification.
- `05_clean_privacy_first_app.txt` — **Negative / Hallucination Control:** Clean, fair T&C with zero hidden traps to verify the model outputs `"NOT FOUND"` instead of inventing risks.

---

## Setup & Usage

### 1. Configure Environment
```bash
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Add your Gemini API key
copy .env.example .env
# Edit .env and set GEMINI_API_KEY=your_actual_key
```

### 2. Run Automated Evaluation (`evaluate.py`)
Runs `v1_baseline`, `v2_structured_risk`, and `v3_chained` across all 5 test cases with a 2-second rate-limit pause between API calls, and writes:
- `logs/experiment_log.csv` (full inputs, outputs, and metrics)
- `logs/experiment_report.md` (formatted comparison table)

```bash
python evaluate.py
```

### 3. Launch Streamlit UI (`app.py`)
```bash
streamlit run app.py
```
Use the sidebar to switch live between `v1`, `v2`, and `v3`, load any of the 5 sample T&Cs or paste custom text, and inspect the TL;DR summary, verbatim-verified Risk Severity Table, and Raw JSON / Step-1 Audit tab.

---

## Responsible AI Disclaimer
ClauseWise AI provides plain-language summaries for educational and informational purposes only. **It is NOT legal advice.** Always consult a qualified attorney for legal guidance.
