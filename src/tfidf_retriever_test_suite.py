from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tfidf_retriever_query import preprocess_query_to_text_for_tfidf


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_JOBLIB = BASE_DIR / "models" / "tfidf_doctor_retriever.joblib"
OUT_TXT = BASE_DIR / "processed_data" / "retriever" / "tfidf_test_suite_report.txt"


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
# Helpers
# ----------------------------
def unique_preserve_order(items: List[str]) -> List[str]:
    # مثل dict.fromkeys ولی واضح‌تر
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def vectorizer_terms(vec, text: str) -> List[str]:
    """
    Terms (unigram/bigram/...) exactly as the vectorizer sees them.
    This respects lowercase + tokenizer + ngram_range.
    """
    if not text or not str(text).strip():
        return []
    analyzer = vec.build_analyzer()
    # analyzer ممکن است تکراری بده؛ برای گزارش یکتا می‌کنیم
    return unique_preserve_order(list(analyzer(text)))


def top_terms_from_sparse_row(qv, feature_names: np.ndarray, top_n: int = 8) -> List[Tuple[str, float]]:
    """
    Get top terms from a single-row CSR query vector without densifying.
    """
    if qv is None or int(qv.nnz) == 0:
        return []

    data = qv.data
    idxs = qv.indices

    # data already only for non-zeros
    order = np.argsort(-data)
    order = order[: min(top_n, order.size)]

    out: List[Tuple[str, float]] = []
    for j in order:
        fi = int(idxs[j])
        out.append((str(feature_names[fi]), float(data[j])))
    return out


def fmt_vec_params(vec) -> str:
    # چند پارامتر مهم برای اینکه report self-contained باشد
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


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not INDEX_JOBLIB.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_JOBLIB}")

    idx = joblib.load(INDEX_JOBLIB)
    vec = idx["vectorizer"]
    X = idx["X"]
    doctor_ids = idx["doctor_ids"]

    vocab = vec.vocabulary_  # dict term -> index
    feature_names = vec.get_feature_names_out()

    lines: List[str] = []
    lines.append("TF-IDF RETRIEVER — TEST SUITE REPORT")
    lines.append(f"index={INDEX_JOBLIB.resolve()}")
    lines.append(f"doctors={len(doctor_ids):,}")
    lines.append(f"vocab_size={len(vocab):,}")
    lines.append(f"vectorizer_params={fmt_vec_params(vec)}")
    lines.append("")

    # global stats: (set_name, query_raw, query_preprocessed, nnz, top1_sim)
    global_stats: List[Tuple[str, str, str, int, float]] = []

    for set_name, queries in QUERY_SETS.items():
        lines.append("=" * 72)
        lines.append(f"SET={set_name}  size={len(queries)}")
        lines.append("=" * 72)

        for q in queries:
            q_p = preprocess_query_to_text_for_tfidf(q)

            # 1) نمایش توکن‌های ساده (برای خوانایی انسانی)
            toks_split = q_p.split() if q_p.strip() else []

            # 2) termهای واقعی مطابق vectorizer (lowercase + ngrams)
            terms = vectorizer_terms(vec, q_p)

            # in-vocab / oov روی termهای واقعی vectorizer
            in_vocab = [t for t in terms if t in vocab]
            oov = [t for t in terms if t not in vocab]

            # vectorize
            qv = vec.transform([q_p])
            nnz = int(qv.nnz)

            # similarity + top5
            if nnz > 0:
                sims = cosine_similarity(qv, X).ravel()
                top_idx = np.argsort(-sims)[:5]
                top5 = [(doctor_ids[i], float(sims[i])) for i in top_idx]
                top1 = float(top5[0][1]) if top5 else 0.0
            else:
                top5 = []
                top1 = 0.0

            q_terms = top_terms_from_sparse_row(qv, feature_names, top_n=8)
            global_stats.append((set_name, q, q_p, nnz, top1))

            # report block
            lines.append(f"query_raw={q}")
            lines.append(f"query_preprocessed={q_p}")
            lines.append(f"tokens_split={toks_split}")
            lines.append(f"vectorizer_terms={terms}")
            lines.append(f"in_vocab={in_vocab}")
            lines.append(f"oov={oov}")
            lines.append(f"query_vector_nnz={nnz}")
            lines.append(f"query_top_terms={q_terms}")

            if top5:
                lines.append("top5:")
                for did, sim in top5:
                    lines.append(f"  doctor_id={did}  sim={sim:.4f}")
            else:
                lines.append("top5: none (nnz=0)")

            lines.append("")

    # Summary
    lines.append("=" * 72)
    lines.append("SUMMARY")
    lines.append("=" * 72)

    total = len(global_stats)
    nnz0 = sum(1 for x in global_stats if x[3] == 0)
    sim0 = sum(1 for x in global_stats if x[4] == 0.0)

    lines.append(f"queries_total={total}")
    lines.append(f"queries_with_nnz_0={nnz0} ({(nnz0 / total) * 100:.1f}%)")
    lines.append(f"queries_with_top1_sim_0={sim0} ({(sim0 / total) * 100:.1f}%)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nsaved_report={OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()