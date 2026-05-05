# src/tfidf_retriever_eval_holdout.py
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "processed_data" / "final"
RETR_DIR = BASE_DIR / "processed_data" / "retriever"
RETR_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class EvalConfig:
    variant: str
    holdout_frac: float
    seed: int
    min_comments_per_doctor: int
    max_queries_per_doctor: int
    topk_list: List[int]


def resolve_inputs_outputs(variant: str) -> Tuple[Path, Path, Path, Path]:
    """
    Returns:
      in_comments_csv,
      out_report_txt,
      out_details_csv,
      out_by_doctor_csv
    """
    if variant not in {"baseline", "medical"}:
        raise ValueError("variant must be one of: baseline, medical")

    if variant == "baseline":
        in_csv = FINAL_DIR / "comments_for_tfidf_retriever.csv"
        out_report = RETR_DIR / "tfidf_holdout_eval_report.txt"
        out_details = RETR_DIR / "tfidf_holdout_eval_details.csv"
        out_by_doctor = RETR_DIR / "tfidf_holdout_eval_by_doctor.csv"
    else:
        in_csv = FINAL_DIR / "comments_for_tfidf_retriever_medical.csv"
        out_report = RETR_DIR / "tfidf_holdout_eval_report_medical.txt"
        out_details = RETR_DIR / "tfidf_holdout_eval_details_medical.csv"
        out_by_doctor = RETR_DIR / "tfidf_holdout_eval_by_doctor_medical.csv"

    return in_csv, out_report, out_details, out_by_doctor


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="TF-IDF Retriever Holdout Eval (baseline/medical, coverage-aware).")
    p.add_argument("--variant", type=str, default="baseline", choices=["baseline", "medical"])
    p.add_argument("--holdout-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-comments-per-doctor", type=int, default=5)
    p.add_argument("--max-queries-per-doctor", type=int, default=50)
    p.add_argument("--topk", type=str, default="1,5,10,20")

    a = p.parse_args()

    topk_list = [int(x.strip()) for x in str(a.topk).split(",") if str(x).strip()]
    topk_list = sorted(list(set(topk_list)))
    if not topk_list:
        topk_list = [1, 5, 10, 20]

    return EvalConfig(
        variant=a.variant,
        holdout_frac=float(a.holdout_frac),
        seed=int(a.seed),
        min_comments_per_doctor=int(a.min_comments_per_doctor),
        max_queries_per_doctor=int(a.max_queries_per_doctor),
        topk_list=topk_list,
    )


def build_doctor_corpus(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    From comment-level train_df (doctor_id, text_for_tfidf),
    build doctor-level corpus:
      doctor_id, document, comments_used, doc_token_len_est
    """
    doc_stats = train_df.groupby("doctor_id").size().rename("comments_used").reset_index()
    doctor_docs = (
        train_df.groupby("doctor_id")["text_for_tfidf"]
        .apply(lambda x: " ".join(x.tolist()))
        .reset_index()
        .rename(columns={"text_for_tfidf": "document"})
    )
    doctor_docs = doctor_docs.merge(doc_stats, on="doctor_id", how="left")
    doctor_docs["doc_token_len_est"] = doctor_docs["document"].str.split().str.len()
    return doctor_docs


def rank_of_true_doctor(sims_row: np.ndarray, true_index: int) -> int:
    """
    Rank = 1 means best match.
    Ties: if multiple docs have same sim, this rank behaves like "1 + count(sim > sim_true)".
    """
    sim_true = float(sims_row[true_index])
    better = int(np.sum(sims_row > sim_true))
    return better + 1


def main() -> None:
    cfg = parse_args()
    t0 = time.time()

    IN_CSV, OUT_REPORT, OUT_DETAILS, OUT_BY_DOCTOR = resolve_inputs_outputs(cfg.variant)
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    df = pd.read_csv(IN_CSV, keep_default_na=False)
    if "doctor_id" not in df.columns or "text_for_tfidf" not in df.columns:
        raise ValueError("Input CSV must contain columns: doctor_id, text_for_tfidf")

    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_for_tfidf"] = df["text_for_tfidf"].astype(str)

    # Safety: remove empty queries/doc text
    df = df[df["text_for_tfidf"].str.strip().ne("")].copy()

    # Keep placeholders out if present
    if "is_placeholder_negative" in df.columns:
        df = df[~df["is_placeholder_negative"].astype(bool)].copy()

    rows_input = len(df)
    rng = np.random.default_rng(cfg.seed)

    # Split holdout per doctor
    test_parts = []
    train_parts = []
    dropped_doctors_too_few = 0

    for did, g in df.groupby("doctor_id"):
        n = len(g)
        if n < cfg.min_comments_per_doctor:
            dropped_doctors_too_few += 1
            continue

        # choose test size, ensure at least 1 train remains
        n_test = int(round(n * cfg.holdout_frac))
        n_test = max(1, n_test)
        n_test = min(n_test, n - 1)

        idx = np.arange(n)
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        g = g.reset_index(drop=True)
        test_g = g.loc[test_idx].copy()
        train_g = g.loc[train_idx].copy()

        # cap queries per doctor to keep runtime stable
        if len(test_g) > cfg.max_queries_per_doctor:
            keep = rng.choice(np.arange(len(test_g)), size=cfg.max_queries_per_doctor, replace=False)
            test_g = test_g.iloc[keep].copy()

        test_parts.append(test_g)
        train_parts.append(train_g)

    if not train_parts or not test_parts:
        raise RuntimeError("No train/test splits produced. Check thresholds / input data.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    # Build doctor documents from train
    doctor_docs = build_doctor_corpus(train_df)
    doctors_in_corpus = len(doctor_docs)

    # Fit TF-IDF (same params as build.py)
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

    X = vec.fit_transform(doctor_docs["document"].astype(str).tolist())

    # Map doctor_id -> row index in X
    did_to_idx = {str(d): i for i, d in enumerate(doctor_docs["doctor_id"].astype(str).tolist())}

    # Prepare queries (test)
    test_df["doctor_idx"] = test_df["doctor_id"].map(did_to_idx)
    missing_in_corpus = int(test_df["doctor_idx"].isna().sum())
    test_eval = test_df.dropna(subset=["doctor_idx"]).copy()
    test_eval["doctor_idx"] = test_eval["doctor_idx"].astype(int)

    queries_total = len(test_df)
    queries_evaluable = len(test_eval)

    # Vectorize queries
    Q = vec.transform(test_eval["text_for_tfidf"].astype(str).tolist())
    nnz = np.asarray(Q.getnnz(axis=1)).astype(int)

    # Similarities
    # cosine_similarity returns dense (n_queries x n_doctors) - size ok here (<= ~ (doctors * queries))
    sims = cosine_similarity(Q, X)

    # Evaluate per query
    topk_list = cfg.topk_list
    max_k = max(topk_list)

    ranks = []
    topk_hits = {k: 0 for k in topk_list}
    nnz0 = 0

    detail_rows = []

    doctor_ids_order = doctor_docs["doctor_id"].astype(str).tolist()

    for i in range(queries_evaluable):
        q_text = str(test_eval.iloc[i]["text_for_tfidf"])
        true_did = str(test_eval.iloc[i]["doctor_id"])
        true_idx = int(test_eval.iloc[i]["doctor_idx"])

        q_nnz = int(nnz[i])
        if q_nnz == 0:
            nnz0 += 1
            # still record as evaluated-but-dead query
            detail_rows.append(
                {
                    "doctor_id_true": true_did,
                    "query_text": q_text,
                    "nnz": 0,
                    "rank": None,
                    "hit_at_1": False,
                    "hit_at_5": False,
                    "hit_at_10": False,
                    "hit_at_20": False,
                    "top_ids": "",
                    "top_sims": "",
                }
            )
            continue

        row = sims[i]
        rnk = rank_of_true_doctor(row, true_idx)
        ranks.append(rnk)

        # topK
        top_idx = np.argsort(-row)[:max_k]
        top_ids = [doctor_ids_order[int(j)] for j in top_idx]
        top_sims = [float(row[int(j)]) for j in top_idx]

        for k in topk_list:
            if rnk <= k:
                topk_hits[k] += 1

        detail = {
            "doctor_id_true": true_did,
            "query_text": q_text,
            "nnz": q_nnz,
            "rank": int(rnk),
            "top_ids": " | ".join(top_ids[:max_k]),
            "top_sims": " | ".join([f"{s:.4f}" for s in top_sims[:max_k]]),
        }
        # convenience boolean columns for common ks
        for k in [1, 5, 10, 20]:
            detail[f"hit_at_{k}"] = bool(rnk <= k) if k in topk_list else bool(rnk <= k)
        detail_rows.append(detail)

    # Metrics
    ranks_arr = np.array([r for r in ranks if r is not None], dtype=float)
    mrr = float(np.mean(1.0 / ranks_arr)) if ranks_arr.size else 0.0
    mean_rank = float(np.mean(ranks_arr)) if ranks_arr.size else 0.0
    median_rank = float(np.median(ranks_arr)) if ranks_arr.size else 0.0

    # Recall@k computed on evaluable queries with nnz>0 (ranks list)
    eval_effective = len(ranks)
    recall_at = {k: (topk_hits[k] / eval_effective) if eval_effective else 0.0 for k in topk_list}

    # Write details CSV
    details_df = pd.DataFrame(detail_rows)
    OUT_DETAILS.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(OUT_DETAILS, index=False, encoding="utf-8-sig")

    # By-doctor summary
    by_doc = (
        details_df.dropna(subset=["rank"])
        .groupby("doctor_id_true", dropna=False)
        .agg(
            queries=("query_text", "size"),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            hit_at_10=("hit_at_10", "mean"),
        )
        .reset_index()
        .rename(columns={"doctor_id_true": "doctor_id"})
        .sort_values(["queries", "hit_at_10"], ascending=[False, False])
    )
    by_doc.to_csv(OUT_BY_DOCTOR, index=False, encoding="utf-8-sig")

    # Report
    lines: List[str] = []
    lines.append(f"TF-IDF RETRIEVER — HOLDOUT EVAL REPORT ({cfg.variant})")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input_comments_csv={IN_CSV.resolve()}")
    lines.append(f"out_details_csv={OUT_DETAILS.resolve()}")
    lines.append(f"out_by_doctor_csv={OUT_BY_DOCTOR.resolve()}")
    lines.append("")
    lines.append("CONFIG")
    lines.append(f"holdout_frac={cfg.holdout_frac}")
    lines.append(f"seed={cfg.seed}")
    lines.append(f"min_comments_per_doctor={cfg.min_comments_per_doctor}")
    lines.append(f"max_queries_per_doctor={cfg.max_queries_per_doctor}")
    lines.append(f"topk_list={cfg.topk_list}")
    lines.append("")
    lines.append("COVERAGE")
    lines.append(f"rows_input_after_basic_filters={rows_input:,}")
    lines.append(f"doctors_dropped_too_few_comments={dropped_doctors_too_few:,}")
    lines.append(f"doctors_in_corpus={doctors_in_corpus:,}")
    lines.append(f"queries_total(sampled)={queries_total:,}")
    lines.append(f"queries_missing_doctor_in_corpus={missing_in_corpus:,}")
    lines.append(f"queries_evaluable={queries_evaluable:,}")
    lines.append(f"queries_with_nnz_0={nnz0:,} ({(nnz0 / queries_evaluable) * 100:.1f}%)" if queries_evaluable else "queries_with_nnz_0=0 (N/A)")
    lines.append("")
    lines.append("QUALITY (on effective queries: doctor in corpus AND nnz>0)")
    lines.append(f"effective_queries={eval_effective:,}")
    for k in topk_list:
        lines.append(f"recall@{k}={recall_at[k]:.4f}")
    lines.append(f"MRR={mrr:.4f}")
    lines.append(f"mean_rank={mean_rank:.2f}")
    lines.append(f"median_rank={median_rank:.2f}")
    lines.append("")
    lines.append(f"tfidf_matrix_shape={X.shape[0]} x {X.shape[1]}")
    lines.append(f"seconds={(time.time() - t0):.2f}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved_report={OUT_REPORT.resolve()}")


if __name__ == "__main__":
    main()