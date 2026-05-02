obat NLP Project (Preprocess Pipeline + TF‑IDF Retriever + MLP later)

Overview
This project contains:
Scraped doctor profiles + comments (raw JSON files)
Canonical raw dataset builder (deduplicates doctor_id)
Step-by-step preprocessing pipeline (deterministic and auditable)
TF-IDF doctor-level retriever (query → top doctors)
Test suite and diagnostics
Next phases:
Candidate mining from Neutral → labeling_pool.csv
Binary sentiment MLP (Positive vs Non-Positive)
Then 3-class sentiment
Project Structure (current)
data/
Raw scraped JSON files

src/
All scripts

processed_data/
Generated CSVs and reports

processed_data/preprocess_steps/
Step-by-step preprocessing outputs

processed_data/final/
Final preprocessed outputs for retriever and later ML

processed_data/retriever/
Doctor documents + keywords + retriever reports

models/
Saved retriever model(s) as .joblib

Python Requirements
Recommended: Python 3.10+
Install:
pip install pandas numpy scikit-learn joblib hazm

Run Pipeline From Scratch

Step A) Canonicalize raw JSON dataset
Command:
python -u src/raw_dataset_builder.py

Outputs:

processed_data/doctors_raw_canonical.csv
processed_data/comments_raw_canonical.csv
processed_data/raw_dataset_builder_report.txt
processed_data/duplicate_doctor_ids.csv
Step B) EDA on canonical raw (no preprocessing)
Commands:
python -u src/eda_canonical.py
python -u src/text_quality_audit.py

Outputs:

processed_data/eda_report_canonical.txt
processed_data/text_quality_audit.txt
Step C) Preprocessing pipeline (run in order)
Commands:
python -u src/preproc_step01_char_normalize.py
python -u src/preproc_step02_pre_hazm_clean.py
python -u src/preproc_step03_hazm_normalize.py
python -u src/preproc_step04_domain_deglue.py
python -u src/preproc_step05_tokenize_hazm.py
python -u src/preproc_step06_finalize_and_qc.py

Final outputs:

processed_data/final/comments_preprocessed_full.csv
processed_data/final/comments_for_tfidf_retriever.csv
processed_data/final/preprocess_qc_report.txt
Step D) Build TF‑IDF Doctor Retriever (doctor-level corpus)
Command:
python -u src/tfidf_retriever_build.py

Outputs:

processed_data/retriever/doctor_documents.csv
processed_data/retriever/doctor_keywords.csv
processed_data/retriever/tfidf_retriever_build_report.txt
models/tfidf_doctor_retriever.joblib
Step E) Run interactive retriever queries (CLI)
Command:
python -u src/tfidf_retriever_query.py

Example queries:

سنگ کلیه
دیسک کمر
پارگی رباط صلیبی
ریزش مو
مراقبت پس از زایمان
بد برخورد
با حوصله
Step F) Diagnostics / Tests (recommended)
Command:
python -u src/tfidf_retriever_test_suite.py

Optional DF audit (why some tokens go OOV with strict max_df):
python -u src/tfidf_retriever_df_audit.py

Optional holdout evaluation:
python -u src/tfidf_retriever_eval_holdout.py

Notes About Sentiment Labels
Raw Negative class is a placeholder text (“عدم رضایت”), so it is not usable for true text-based negative modeling.
Planned approach:
Finish preprocessing + retriever
Mine negative candidates from Neutral (labeling_pool.csv)
Label 500–1500 samples (manual or LLM-assisted + audit)
Train MLP first as Binary (Positive vs Non-Positive), then expand to 3-class
