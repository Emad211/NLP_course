# src/tfidf_retriever_test_suite.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from retriever_medical_stopwords import (
    MEDICAL_ALLOWLIST_LATIN,
    MEDICAL_STOPWORD_BIGRAMS_V01,
    filter_text_tokens_space_split,
    norm_tok,
)
from tfidf_retriever_query import preprocess_query_to_text_for_tfidf


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INDEX_BASELINE = BASE_DIR / "models" / "tfidf_doctor_retriever.joblib"
DEFAULT_INDEX_MEDICAL = BASE_DIR / "models" / "tfidf_doctor_retriever_medical.joblib"

DEFAULT_OUT_BASELINE = BASE_DIR / "processed_data" / "retriever" / "tfidf_test_suite_report.txt"
DEFAULT_OUT_MEDICAL = BASE_DIR / "processed_data" / "retriever" / "tfidf_test_suite_report_medical.txt"

# For medical query preprocessing (must match Step06 medical stopwording)
DEFAULT_MED_STOPWORDS_FILE = BASE_DIR / "processed_data" / "final" / "retriever_medical_stopwords_final_v03.txt"


# ----------------------------
# Queries
# ----------------------------
QUERY_SETS: Dict[str, List[str]] = {
    "medical": [
        "سنگ کلیه",
        "دیسک کمر",
        "پارگی رباط صلیبی",
        "ریزش مو",
        "میگرن",
        "اگزما",
        "جوش صورت",
        "تست پاپ اسمیر",
        "HPV",
        "MRI گردن",
        "PRP زانو",
        "تنگی کانال نخاعی",
        "آندوسکوپی بینی",
    ],
    "service": [
        "با حوصله",
        "وقت شناس",
        "بد برخورد",
        "معطلی زیاد",
        "نوبت دهی افتضاح",
        "پاسخگو نبودن مطب",
        "هزینه بالا",
        "تشخیص اشتباه",
    ],
    "procedures": [
        "سزارین",
        "زایمان طبیعی",
        "سرکلاژ",
        "لاپاراسکوپی",
        "عمل لوزه",
        "لیزر پوست",
        "تعویض مفصل زانو",
        "جراحی دیسک کمر",
        "سنگ شکن کلیه",
    ],
    "two_word_variants": [
        "زایمان",
        "مراقبت زایمان",
        "مراقبت پس از زایمان",
        "درد زانو",
        "درد زانو راه رفتن",
        "سنگ کلیه",
        "درد سنگ کلیه",
        "پارگی رباط",
        "پارگی رباط صلیبی",
        "با حوصله",
        "خیلی با حوصله",
    ],
}


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class QueryStat:
    set_name: str
    query_raw: str
    query_preprocessed: str
    query_variant_text: str
    removed_tokens_sample: str
    empty_after_variant: bool
    nnz: int
    top1_sim: float
    oov_terms_count: int
    total_terms_count: int


# ----------------------------
# Helpers
# ----------------------------
def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def load_stopwords_unigrams(path: Path) -> Tuple[set[str], List[str]]:
    """
    Loads a stopword unigram list from a file (one token per line).
    Ignores empty lines and comment lines starting with '#'.
    Returns:
      stopwords_set_normed, raw_loaded_normed_list
    """
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))

    raw = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        t = norm_tok(line)
        if not t:
            continue
        raw.append(t)

    return set(raw), raw


def vectorizer_terms(vec, text: str) -> List[str]:
    """
    Terms exactly as the vectorizer analyzer sees them.
    This respects lowercase + tokenizer + ngram_range.
    """
    if not text or not str(text).strip():
        return []
    analyzer = vec.build_analyzer()
    return unique_preserve_order(list(analyzer(text)))


def top_terms_from_sparse_row(qv, feature_names: np.ndarray, top_n: int = 8) -> List[Tuple[str, float]]:
    """
    Get top terms from a single-row CSR query vector without densifying.
    """
    if qv is None or int(qv.nnz) == 0:
        return []

    data = qv.data
    idxs = qv.indices

    order = np.argsort(-data)
    order = order[: min(top_n, order.size)]

    out: List[Tuple[str, float]] = []
    for j in order:
        fi = int(idxs[j])
        out.append((str(feature_names[fi]), float(data[j])))
    return out


def fmt_vec_params(vec) -> str:
    parts = [
        f"lowercase={getattr(vec, 'lowercase', None)}",
        f"min_df={getattr(vec, 'min_df', None)}",
        f"max_df={getattr(vec, 'max_df', None)}",
        f"ngram_range={getattr(vec, 'ngram_range', None)}",
        f"max_features={getattr(vec, 'max_features', None)}",
        f"sublinear_tf={getattr(vec, 'sublinear_tf', None)}",
        f"norm={getattr(vec, 'norm', None)}",
    ]
    return ", ".join(parts)


def resolve_paths(variant: str) -> Tuple[Path, Path]:
    if variant not in {"baseline", "medical"}:
        raise ValueError("variant must be one of: baseline, medical")
    if variant == "baseline":
        return DEFAULT_INDEX_BASELINE, DEFAULT_OUT_BASELINE
    return DEFAULT_INDEX_MEDICAL, DEFAULT_OUT_MEDICAL


def preprocess_query_variant(
    query_raw: str,
    variant: str,
    medical_stopwords: set[str] | None,
    high_risk_keep: set[str],
) -> Tuple[str, str, str]:
    """
    Returns:
      query_preprocessed (baseline pipeline),
      query_variant_text (baseline or medical),
      removed_tokens_sample (only for medical; else '')
    """
    q_p = preprocess_query_to_text_for_tfidf(query_raw)

    if variant == "baseline":
        return q_p, q_p, ""

    # medical:
    # Apply the same "token-space" stopwording that Step06 applied.
    # Safety guard: never remove high-risk medical core tokens.
    stop = set(medical_stopwords or set())
    stop -= set(high_risk_keep)

    q_med, removed = filter_text_tokens_space_split(
        q_p,
        stopwords=stop,
        allowlist_latin=MEDICAL_ALLOWLIST_LATIN,
        stopword_bigrams=MEDICAL_STOPWORD_BIGRAMS_V01,
    )

    # make readable unique sample
    if removed:
        seen = set()
        uniq = []
        for r in removed:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        removed_sample = " | ".join(uniq[:30])
    else:
        removed_sample = ""

    return q_p, q_med, removed_sample


def parse_args():
    p = argparse.ArgumentParser(description="TF-IDF Retriever Test Suite (baseline/medical, coverage-aware).")
    p.add_argument("--variant", type=str, default="baseline", choices=["baseline", "medical"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--top-terms", type=int, default=8)
    p.add_argument("--out", type=str, default="", help="Override output report path.")
    p.add_argument(
        "--medical-stopwords-file",
        type=str,
        default=str(DEFAULT_MED_STOPWORDS_FILE),
        help="Used only when variant=medical. One token per line.",
    )
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()

    INDEX_JOBLIB, OUT_TXT_DEFAULT = resolve_paths(args.variant)
    OUT_TXT = Path(args.out) if str(args.out).strip() else OUT_TXT_DEFAULT

    if not INDEX_JOBLIB.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_JOBLIB.resolve()}")

    idx = joblib.load(INDEX_JOBLIB)
    vec = idx["vectorizer"]
    X = idx["X"]
    doctor_ids = idx["doctor_ids"]
    name_map = idx.get("name_map", {}) or {}

    vocab = vec.vocabulary_  # dict term -> index
    feature_names = np.array(vec.get_feature_names_out())

    # Load medical stopwords if needed
    medical_stopwords = None
    medical_stopwords_src = ""
    if args.variant == "medical":
        sw_path = Path(args.medical_stopwords_file)
        sw_set, sw_raw = load_stopwords_unigrams(sw_path)
        medical_stopwords = sw_set
        medical_stopwords_src = f"file:{sw_path.name} raw={len(sw_raw)} effective={len(sw_set)}"

    # MUST-NOT-REMOVE tokens (guardrail against over-aggressive stopwording)
    HIGH_RISK_KEEP = {"درد", "دارو", "درمان", "تشخیص", "عمل", "جراحی", "کار"}

    lines: List[str] = []
    lines.append("TF-IDF RETRIEVER — TEST SUITE REPORT (coverage-aware)")
    lines.append(f"variant={args.variant}")
    lines.append(f"index={INDEX_JOBLIB.resolve()}")
    lines.append(f"out_report={OUT_TXT.resolve()}")
    lines.append(f"doctors={len(doctor_ids):,}")
    lines.append(f"vocab_size={len(vocab):,}")
    lines.append(f"vectorizer_params={fmt_vec_params(vec)}")
    if args.variant == "medical":
        lines.append(f"medical_stopwords={medical_stopwords_src}")
        lines.append(f"high_risk_keep={sorted(HIGH_RISK_KEEP)}")
        lines.append(f"medical_stopword_bigrams={sorted([' '.join(x) for x in MEDICAL_STOPWORD_BIGRAMS_V01])}")
    lines.append("")

    all_stats: List[QueryStat] = []

    for set_name, queries in QUERY_SETS.items():
        lines.append("=" * 72)
        lines.append(f"SET={set_name}  size={len(queries)}")
        lines.append("=" * 72)

        for q in queries:
            q_p, q_var, removed_sample = preprocess_query_variant(
                q,
                variant=args.variant,
                medical_stopwords=medical_stopwords,
                high_risk_keep=HIGH_RISK_KEEP,
            )

            empty_after = (not str(q_var).strip())

            # Human-friendly token view (split)
            toks_split = q_var.split() if q_var.strip() else []

            # Vectorizer terms (respect analyzer)
            terms = vectorizer_terms(vec, q_var)
            in_vocab = [t for t in terms if t in vocab]
            oov = [t for t in terms if t not in vocab]

            # Vectorize and retrieve
            qv = vec.transform([q_var])
            nnz = int(qv.nnz)

            if nnz > 0:
                sims = cosine_similarity(qv, X).ravel()
                top_idx = np.argsort(-sims)[: max(1, int(args.topk))]
                topk = [(int(i), str(doctor_ids[int(i)]), float(sims[int(i)])) for i in top_idx]
                top1 = float(topk[0][2]) if topk else 0.0
            else:
                topk = []
                top1 = 0.0

            q_terms = top_terms_from_sparse_row(qv, feature_names, top_n=int(args.top_terms))

            all_stats.append(
                QueryStat(
                    set_name=set_name,
                    query_raw=q,
                    query_preprocessed=q_p,
                    query_variant_text=q_var,
                    removed_tokens_sample=removed_sample,
                    empty_after_variant=bool(empty_after),
                    nnz=int(nnz),
                    top1_sim=float(top1),
                    oov_terms_count=int(len(oov)),
                    total_terms_count=int(len(terms)),
                )
            )

            # Report block
            lines.append(f"query_raw={q}")
            lines.append(f"query_preprocessed={q_p}")
            if args.variant == "medical":
                lines.append(f"query_medical={q_var}")
                lines.append(f"removed_tokens_sample={removed_sample}")

            lines.append(f"tokens_split={toks_split}")
            lines.append(f"vectorizer_terms={terms}")
            lines.append(f"in_vocab={in_vocab}")
            lines.append(f"oov={oov}")
            lines.append(f"query_vector_nnz={nnz}")
            lines.append(f"query_top_terms={q_terms}")

            if topk:
                lines.append(f"top{len(topk)}:")
                for rank_i, did, sim in [(r + 1, did, sim) for r, (_, did, sim) in enumerate(topk)]:
                    nm = name_map.get(str(did), "")
                    if nm:
                        lines.append(f"  rank={rank_i:02d}  doctor_id={did}  sim={sim:.4f}  name={nm}")
                    else:
                        lines.append(f"  rank={rank_i:02d}  doctor_id={did}  sim={sim:.4f}")
            else:
                if empty_after:
                    lines.append("topk: none (query empty after variant preprocessing)")
                else:
                    lines.append("topk: none (nnz=0: likely OOV / filtered too much)")

            lines.append("")

    # ----------------------------
    # Summary (overall + per set)
    # ----------------------------
    lines.append("=" * 72)
    lines.append("SUMMARY (overall)")
    lines.append("=" * 72)

    total = len(all_stats)
    empty_cnt = sum(1 for s in all_stats if s.empty_after_variant)
    nnz0 = sum(1 for s in all_stats if s.nnz == 0)
    sim0 = sum(1 for s in all_stats if s.top1_sim == 0.0)

    # OOV ratios (only when terms exist)
    term_total = sum(s.total_terms_count for s in all_stats)
    oov_total = sum(s.oov_terms_count for s in all_stats)
    oov_ratio = (oov_total / term_total) * 100.0 if term_total else 0.0

    lines.append(f"queries_total={total}")
    lines.append(f"queries_empty_after_variant_preproc={empty_cnt} ({(empty_cnt / total) * 100:.1f}%)")
    lines.append(f"queries_with_nnz_0={nnz0} ({(nnz0 / total) * 100:.1f}%)")
    lines.append(f"queries_with_top1_sim_0={sim0} ({(sim0 / total) * 100:.1f}%)")
    lines.append(f"terms_total={term_total}")
    lines.append(f"terms_oov_total={oov_total} ({oov_ratio:.1f}%)")
    lines.append("")

    # Per-set summary
    lines.append("=" * 72)
    lines.append("SUMMARY (per set)")
    lines.append("=" * 72)

    for set_name in QUERY_SETS.keys():
        ss = [s for s in all_stats if s.set_name == set_name]
        if not ss:
            continue
        t = len(ss)
        e = sum(1 for s in ss if s.empty_after_variant)
        z = sum(1 for s in ss if s.nnz == 0)
        s0 = sum(1 for s in ss if s.top1_sim == 0.0)

        term_t = sum(s.total_terms_count for s in ss)
        oov_t = sum(s.oov_terms_count for s in ss)
        oov_r = (oov_t / term_t) * 100.0 if term_t else 0.0

        lines.append(f"SET={set_name}")
        lines.append(f"  queries={t}")
        lines.append(f"  empty_after_variant_preproc={e} ({(e / t) * 100:.1f}%)")
        lines.append(f"  nnz_0={z} ({(z / t) * 100:.1f}%)")
        lines.append(f"  top1_sim_0={s0} ({(s0 / t) * 100:.1f}%)")
        lines.append(f"  oov_terms_ratio={oov_r:.1f}%")
        lines.append("")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nsaved_report={OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()