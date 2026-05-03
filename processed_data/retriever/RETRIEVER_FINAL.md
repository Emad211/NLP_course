# TF‑IDF Doctor Retriever — Finalization (v1)

This file defines "Definition of Done" for the TF‑IDF doctor-level retriever.

## What it is
- Each doctor becomes one document: concatenation of all `text_for_tfidf` comments for that doctor.
- Retrieval: query -> preprocess (same logic as tfidf_retriever_query.py) -> TF‑IDF -> cosine similarity -> top doctors.

## Inputs / Outputs
Inputs:
- processed_data/final/comments_for_tfidf_retriever.csv

Outputs (Build):
- processed_data/retriever/doctor_documents.csv
- processed_data/retriever/doctor_keywords.csv
- processed_data/retriever/tfidf_retriever_build_report.txt
- models/tfidf_doctor_retriever.joblib

Diagnostics:
- processed_data/retriever/tfidf_test_suite_report.txt
- processed_data/retriever/df_audit_report.txt
- processed_data/retriever/tfidf_holdout_eval_report.txt

## Definition of Done (DoD)
1) Preprocess QC
- `rows_text_for_tfidf_empty` should be near 0 (<= 1 is acceptable).
- Placeholder negative rows are correctly flagged (`is_placeholder_negative`).

2) Build sanity
- `doctors_in_corpus` > 500 (current dataset).
- `tfidf_matrix_shape` looks reasonable (sparse, vocab ~ 20k+).

3) Test suite
- `queries_with_nnz_0 == 0`
- `queries_with_top1_sim_0 == 0`
- NOTE: OOV reporting must be normalized with vectorizer lowercase behavior.

4) Holdout eval
- IMPORTANT: Holdout eval vectorizer params must match build params.
- Record recall@k as baseline (not necessarily high, because task is hard: single comment -> same doctor).

## Known limitations (not bugs)
- Service queries like "حوصله" are very frequent across doctors (high document frequency), so results can be noisy.
- Dataset contains many duplicated/generic comments; TF‑IDF downweights common terms but cannot create information that isn't in data.

## Next micro-step if quality needs improvement (optional)
- Tune `max_df` and/or add domain stopwords for extremely common service words.
- Consider BM25 (still lexical, often stronger than TF‑IDF) as a drop-in next baseline.