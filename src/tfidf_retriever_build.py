import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "final" / "comments_for_tfidf_retriever.csv"

DOCTORS_CSV = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"

OUT_DIR = BASE_DIR / "processed_data" / "retriever"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOCTOR_DOCS_CSV = OUT_DIR / "doctor_documents.csv"
DOCTOR_KEYWORDS_CSV = OUT_DIR / "doctor_keywords.csv"
REPORT_TXT = OUT_DIR / "tfidf_retriever_build_report.txt"
INDEX_JOBLIB = MODELS_DIR / "tfidf_doctor_retriever.joblib"


def top_terms_for_row(row, feature_names, top_n=20):
    v = row.toarray().ravel()
    if v.size == 0:
        return []
    idx = np.argpartition(v, -top_n)[-top_n:]
    idx = idx[np.argsort(v[idx])[::-1]]
    return [(feature_names[i], float(v[i])) for i in idx if v[i] > 0]


def main():
    t0 = time.time()

    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    df = pd.read_csv(IN_CSV, keep_default_na=False)
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_for_tfidf"] = df["text_for_tfidf"].astype(str)

    total_rows = len(df)
    placeholder_rows = int(df.get("is_placeholder_negative", pd.Series([False] * total_rows)).sum())

    use = df[df["text_for_tfidf"].str.strip().ne("")].copy()
    use = use[~use.get("is_placeholder_negative", False)].copy()

    used_rows = len(use)

    doc_stats = use.groupby("doctor_id").size().rename("comments_used").reset_index()
    doctor_docs = (
        use.groupby("doctor_id")["text_for_tfidf"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index()
        .rename(columns={"text_for_tfidf": "document"})
    )

    doctor_docs = doctor_docs.merge(doc_stats, on="doctor_id", how="left")
    doctor_docs["doc_char_len"] = doctor_docs["document"].str.len()
    doctor_docs["doc_token_len_est"] = doctor_docs["document"].str.split().str.len()

    doctor_docs.to_csv(DOCTOR_DOCS_CSV, index=False, encoding="utf-8-sig")

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=True,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        max_features=60000,
        sublinear_tf=True,
        norm="l2",
    )

    X = vectorizer.fit_transform(doctor_docs["document"].astype(str).tolist())
    feature_names = np.array(vectorizer.get_feature_names_out())

    kw_rows = []
    for i in range(X.shape[0]):
        terms = top_terms_for_row(X[i], feature_names, top_n=25)
        kw_rows.append(
            {
                "doctor_id": doctor_docs.loc[i, "doctor_id"],
                "comments_used": int(doctor_docs.loc[i, "comments_used"]),
                "doc_token_len_est": int(doctor_docs.loc[i, "doc_token_len_est"]),
                "top_terms": " | ".join([t for t, _ in terms[:15]]),
                "top_terms_scored": " | ".join([f"{t}:{s:.4f}" for t, s in terms[:15]]),
            }
        )

    keywords = pd.DataFrame(kw_rows)
    keywords.to_csv(DOCTOR_KEYWORDS_CSV, index=False, encoding="utf-8-sig")

    doctors = pd.read_csv(DOCTORS_CSV, keep_default_na=False) if DOCTORS_CSV.exists() else pd.DataFrame()
    name_map = {}
    if not doctors.empty and "doctor_id" in doctors.columns and "name" in doctors.columns:
        tmp = doctors.copy()
        tmp["doctor_id"] = tmp["doctor_id"].astype(str)
        name_map = dict(zip(tmp["doctor_id"], tmp["name"]))

    index_obj = {
        "vectorizer": vectorizer,
        "X": X,
        "doctor_ids": doctor_docs["doctor_id"].tolist(),
        "doctor_docs": doctor_docs[["doctor_id", "document", "comments_used", "doc_token_len_est"]].copy(),
        "name_map": name_map,
    }
    joblib.dump(index_obj, INDEX_JOBLIB)

    lines = []
    lines.append("TF-IDF DOCTOR RETRIEVER — BUILD REPORT")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_comments_csv={IN_CSV.resolve()}")
    lines.append(f"doctor_docs_csv={DOCTOR_DOCS_CSV.resolve()}")
    lines.append(f"keywords_csv={DOCTOR_KEYWORDS_CSV.resolve()}")
    lines.append(f"index_joblib={INDEX_JOBLIB.resolve()}")

    lines.append("")
    lines.append(f"rows_total_in_input={total_rows:,}")
    lines.append(f"placeholder_rows_in_input={placeholder_rows:,}")
    lines.append(f"rows_used_for_doctor_corpus={used_rows:,}")
    lines.append(f"doctors_in_corpus={len(doctor_docs):,}")

    lines.append("")
    lines.append(f"tfidf_matrix_shape={X.shape[0]} x {X.shape[1]}")

    top_docs = doctor_docs.sort_values("comments_used", ascending=False).head(8)
    lines.append("")
    lines.append("SAMPLE TOP KEYWORDS (top 8 doctors by comments_used)")
    for r in top_docs.itertuples(index=False):
        did = str(r.doctor_id)
        nm = name_map.get(did, "")
        kw = keywords[keywords["doctor_id"].astype(str) == did]["top_terms"].values
        kw = kw[0] if len(kw) else ""
        header = f"doctor_id={did}"
        if nm:
            header += f"  name={nm}"
        header += f"  comments_used={int(r.comments_used)}  doc_tokens_est={int(r.doc_token_len_est)}"
        lines.append(header)
        lines.append(f"  top_terms={kw}")
        lines.append("")

    dt = time.time() - t0
    lines.append(f"seconds={dt:.2f}")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()