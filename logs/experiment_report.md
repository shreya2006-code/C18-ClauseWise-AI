# ClauseWise AI — Prompt Engineering Experiment Report

- **Model:** `gemini-3.5-flash-lite`
- **Test Cases Evaluated:** 5
- **Total Runs:** 15

## Aggregate Comparison by Prompt Version

| prompt_version | avg_latency_sec | avg_word_count | json_compliance_rate | quote_verification_rate | disclaimer_rate |
| --- | --- | --- | --- | --- | --- |
| v1_baseline | 5.67 | 427.6 | 0.0% | 0.0% | 0.0% |
| v2_structured_risk | 3.65 | 389.2 | 100.0% | 80.0% | 100.0% |
| v3_chained | 6.73 | 435.2 | 100.0% | 100.0% | 100.0% |

## Per-Run Evaluation Results

| test_case | prompt_version | latency_sec | word_count | valid_json | verbatim_quote_verified | disclaimer_present |
| --- | --- | --- | --- | --- | --- | --- |
| 01_social_media_app.txt | v1_baseline | 11.582 | 435 | False | False | False |
| 01_social_media_app.txt | v2_structured_risk | 5.447 | 410 | True | True | True |
| 01_social_media_app.txt | v3_chained | 8.699 | 494 | True | True | True |
| 02_subscription_streaming.txt | v1_baseline | 6.616 | 518 | False | False | False |
| 02_subscription_streaming.txt | v2_structured_risk | 1.725 | 353 | True | True | True |
| 02_subscription_streaming.txt | v3_chained | 4.269 | 522 | True | True | True |
| 03_food_delivery_app.txt | v1_baseline | 2.387 | 404 | False | False | False |
| 03_food_delivery_app.txt | v2_structured_risk | 6.584 | 373 | True | False | True |
| 03_food_delivery_app.txt | v3_chained | 11.493 | 464 | True | True | True |
| 04_ai_productivity_tool.txt | v1_baseline | 2.485 | 391 | False | False | False |
| 04_ai_productivity_tool.txt | v2_structured_risk | 2.274 | 383 | True | True | True |
| 04_ai_productivity_tool.txt | v3_chained | 4.02 | 437 | True | True | True |
| 05_clean_privacy_first_app.txt | v1_baseline | 5.259 | 390 | False | False | False |
| 05_clean_privacy_first_app.txt | v2_structured_risk | 2.222 | 427 | True | True | True |
| 05_clean_privacy_first_app.txt | v3_chained | 5.188 | 259 | True | True | True |
