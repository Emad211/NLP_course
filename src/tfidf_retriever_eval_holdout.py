from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "final" / "comments_for_tfidf_retriever.csv"
OUT_TXT = BASE_DIR / "processed_data" / "retriever" / "tfidf_holdout_eval_report.txt"


def main():
    df = pd.read_csv(IN_CSV, keep_default_na=False)
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_for_tfidf"] = df["text_for_tfidf"].astype(str)

    df = df[df["text_for_tfidf"].str.strip().ne("")].copy()
    if "is_placeholder_negative" in df.columns:
        df = df[~df["is_placeholder_negative"]].copy()

    rng = np.random.default_rng(42)

    train_parts = []
    test_parts = []

    for did, sub in df.groupby("doctor_id"):
        n = len(sub)
        if n < 5:
            continue
        test_n = max(2, int(round(n * 0.2)))
        idx = rng.permutation(n)
        test_idx = idx[:test_n]
        train_idx = idx[test_n:]
        test_parts.append(sub.iloc[test_idx])
        train_parts.append(sub.iloc[train_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    doc_stats = train_df.groupby("doctor_id").size().rename("train_comments").reset_index()
    doctor_docs = (
        train_df.groupby("doctor_id")["text_for_tfidf"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index()
        .rename(columns={"text_for_tfidf": "document"})
        .merge(doc_stats, on="doctor_id", how="left")
    )

    vec = TfidfVectorizer(
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

    X = vec.fit_transform(doctor_docs["document"].tolist())
    doctor_ids = doctor_docs["doctor_id"].tolist()

    doctor_index = {did: i for i, did in enumerate(doctor_ids)}

    ks = [1, 3, 5, 10]
    hits = {k: 0 for k in ks}
    total_q = 0

    for r in test_df.itertuples(index=False):
        q = str(r.text_for_tfidf)
        did_true = str(r.doctor_id)
        if did_true not in doctor_index:
            continue
        qv = vec.transform([q])
        sims = cosine_similarity(qv, X).ravel()
        order = np.argsort(-sims)
        total_q += 1
        for k in ks:
            topk = set(doctor_ids[i] for i in order[:k])
            if did_true in topk:
                hits[k] += 1

    lines = []
    lines.append("TF-IDF DOCTOR RETRIEVER — HOLDOUT EVAL REPORT")
    lines.append(f"input={IN_CSV.resolve()}")
    lines.append(f"doctors_train_corpus={len(doctor_docs):,}")
    lines.append(f"train_rows={len(train_df):,}")
    lines.append(f"test_rows={len(test_df):,}")
    lines.append(f"queries_evaluated={total_q:,}")
    lines.append("")
    for k in ks:
        rec = hits[k] / total_q if total_q else 0.0
        lines.append(f"recall@{k}={rec:.4f}  ({hits[k]:,}/{total_q:,})")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved_report={OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()