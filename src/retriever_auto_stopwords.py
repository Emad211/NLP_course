# src/retriever_auto_stopwords.py
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retriever_medical_stopwords import MEDICAL_ALLOWLIST_LATIN, norm_tok


BASE_DIR = Path(__file__).resolve().parent.parent
RETR_DIR = BASE_DIR / "processed_data" / "retriever"
FINAL_DIR = BASE_DIR / "processed_data" / "final"


def parse_args():
    p = argparse.ArgumentParser(description="Auto-stopword candidate extractor for retriever (baseline/medical).")
    p.add_argument("--variant", type=str, default="medical", choices=["baseline", "medical"])

    # thresholds (start conservative)
    p.add_argument("--min-doc-ratio", type=float, default=0.60)
    p.add_argument("--min-doc-freq", type=int, default=30)
    p.add_argument("--min-token-len", type=int, default=2)

    # report size
    p.add_argument("--top-n", type=int, default=200)
    return p.parse_args()


def entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution (natural log)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def resolve_inputs(variant: str):
    if variant == "baseline":
        doc_csv = RETR_DIR / "doctor_documents.csv"
        out_candidates = FINAL_DIR / "retriever_stopwords_auto_candidates_baseline.csv"
        out_report = FINAL_DIR / "retriever_stopwords_auto_report_baseline.txt"
        out_suggested = FINAL_DIR / "retriever_stopwords_auto_suggested_baseline.txt"
    else:
        doc_csv = RETR_DIR / "doctor_documents_medical.csv"
        out_candidates = FINAL_DIR / "retriever_stopwords_auto_candidates_medical.csv"
        out_report = FINAL_DIR / "retriever_stopwords_auto_report_medical.txt"
        out_suggested = FINAL_DIR / "retriever_stopwords_auto_suggested_medical.txt"

    doctors_csv = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"
    return doc_csv, doctors_csv, out_candidates, out_report, out_suggested


def main():
    args = parse_args()
    t0 = time.time()

    doc_csv, doctors_csv, out_candidates, out_report, out_suggested = resolve_inputs(args.variant)

    if not doc_csv.exists():
        raise FileNotFoundError(f"Missing doctor documents CSV. Build retriever first: {doc_csv.resolve()}")

    docs = pd.read_csv(doc_csv, keep_default_na=False)
    if "doctor_id" not in docs.columns or "document" not in docs.columns:
        raise ValueError("doctor_documents csv must have columns: doctor_id, document")

    docs["doctor_id"] = docs["doctor_id"].astype(str)
    n_docs = len(docs)

    # doctor_id -> specialty (optional but very useful)
    did_to_spec = {}
    if doctors_csv.exists():
        ddf = pd.read_csv(doctors_csv, keep_default_na=False)
        if "doctor_id" in ddf.columns and "specialty" in ddf.columns:
            ddf["doctor_id"] = ddf["doctor_id"].astype(str)
            did_to_spec = dict(zip(ddf["doctor_id"], ddf["specialty"]))

    # Build token stats across doctor documents
    # We count per-token doc_freq (in how many doctors)
    # and corpus_tf (total appearances across all doctors)
    doc_freq = {}
    corpus_tf = {}

    # also track specialty distribution among doctors containing token
    token_specs = {}

    for r in docs.itertuples(index=False):
        did = str(getattr(r, "doctor_id"))
        text = str(getattr(r, "document"))
        toks = [norm_tok(t) for t in text.split() if norm_tok(t)]
        toks = [t for t in toks if len(t) >= args.min_token_len]

        # ignore allowlisted latin tokens from being stopword candidates
        toks = [t for t in toks if t not in MEDICAL_ALLOWLIST_LATIN]

        # doc-level unique set for doc_freq
        uniq = set(toks)

        for t in uniq:
            doc_freq[t] = doc_freq.get(t, 0) + 1

            if did_to_spec:
                sp = did_to_spec.get(did, "")
                if sp:
                    token_specs.setdefault(t, []).append(sp)

        # term freq (not unique)
        for t in toks:
            corpus_tf[t] = corpus_tf.get(t, 0) + 1

    # Build candidates dataframe
    rows = []
    for t, dfreq in doc_freq.items():
        ratio = dfreq / n_docs if n_docs else 0.0

        specs = token_specs.get(t, [])
        ent = 0.0
        ent_norm = 0.0
        if specs:
            vc = pd.Series(specs).value_counts().values.astype(float)
            ent = entropy(vc)
            ent_max = math.log(len(vc)) if len(vc) > 1 else 0.0
            ent_norm = float(ent / ent_max) if ent_max > 0 else 0.0

        rows.append(
            {
                "token": t,
                "doc_freq": int(dfreq),
                "doc_ratio": float(ratio),
                "corpus_tf": int(corpus_tf.get(t, 0)),
                "specialty_entropy_norm": float(ent_norm),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out_candidates.write_text("EMPTY\n", encoding="utf-8")
        out_report.write_text("EMPTY\n", encoding="utf-8")
        out_suggested.write_text("", encoding="utf-8")
        print("No tokens found.")
        return

    # Suggest stopwords: very common across doctors (high doc_ratio)
    # and spread across specialties (high entropy) => general / non-discriminative
    out["suggested"] = (
        (out["doc_freq"] >= args.min_doc_freq)
        & (out["doc_ratio"] >= args.min_doc_ratio)
        & (out["specialty_entropy_norm"] >= 0.60)  # conservative start
    )

    out = out.sort_values(["suggested", "doc_ratio", "doc_freq", "corpus_tf"], ascending=[False, False, False, False])
    out.to_csv(out_candidates, index=False, encoding="utf-8-sig")

    # Write report
    lines = []
    lines.append(f"AUTO STOPWORD REPORT ({args.variant})")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"doctor_documents_csv={doc_csv.resolve()}")
    lines.append(f"doctors_csv={doctors_csv.resolve()}")
    lines.append("")
    lines.append(f"n_doctors={n_docs:,}")
    lines.append(f"min_doc_freq={args.min_doc_freq}")
    lines.append(f"min_doc_ratio={args.min_doc_ratio}")
    lines.append(f"min_token_len={args.min_token_len}")
    lines.append("rule: suggested if doc_freq>=min_doc_freq AND doc_ratio>=min_doc_ratio AND entropy_norm>=0.60")
    lines.append("")
    lines.append(f"candidates_total={len(out):,}")
    lines.append(f"suggested_count={int(out['suggested'].sum()):,}")
    lines.append("")

    top_sug = out[out["suggested"]].head(args.top_n)
    lines.append(f"TOP SUGGESTED (n={len(top_sug)})")
    for r in top_sug.itertuples(index=False):
        lines.append(
            f"{r.token}\tdoc_ratio={r.doc_ratio:.3f}\tdoc_freq={r.doc_freq}\ttf={r.corpus_tf}\tentropy_norm={r.specialty_entropy_norm:.3f}"
        )

    dt = time.time() - t0
    lines.append("")
    lines.append(f"seconds={dt:.2f}")

    out_report.write_text("\n".join(lines), encoding="utf-8")

    # Suggested stopword list file (tokens only)
    out_suggested.write_text("\n".join(top_sug["token"].tolist()), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nsaved_candidates={out_candidates.resolve()}")
    print(f"saved_report={out_report.resolve()}")
    print(f"saved_suggested_list={out_suggested.resolve()}")


if __name__ == "__main__":
    main()