from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from tfidf_retriever_query import preprocess_query_to_text_for_tfidf


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_JOBLIB = BASE_DIR / "models" / "tfidf_doctor_retriever.joblib"
OUT_TXT = BASE_DIR / "processed_data" / "retriever" / "tfidf_test_suite_report.txt"


QUERY_SETS = {
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


def top_terms_from_query_vector(qv, feature_names, top_n=10):
    arr = qv.toarray().ravel()
    if arr.size == 0 or arr.max() == 0:
        return []
    idx = np.argpartition(arr, -top_n)[-top_n:]
    idx = idx[np.argsort(arr[idx])[::-1]]
    return [(feature_names[i], float(arr[i])) for i in idx if arr[i] > 0]


def main():
    idx = joblib.load(INDEX_JOBLIB)
    vec = idx["vectorizer"]
    X = idx["X"]
    doctor_ids = idx["doctor_ids"]
    vocab = vec.vocabulary_
    feature_names = vec.get_feature_names_out()

    lines = []
    lines.append("TF-IDF RETRIEVER — TEST SUITE REPORT")
    lines.append(f"index={INDEX_JOBLIB.resolve()}")
    lines.append(f"doctors={len(doctor_ids):,}")
    lines.append(f"vocab_size={len(vocab):,}")
    lines.append("")

    global_stats = []

    for set_name, queries in QUERY_SETS.items():
        lines.append("=" * 72)
        lines.append(f"SET={set_name}  size={len(queries)}")
        lines.append("=" * 72)

        for q in queries:
            q_p = preprocess_query_to_text_for_tfidf(q)
            toks = q_p.split() if q_p.strip() else []
            in_vocab = [t for t in toks if t in vocab]
            oov = [t for t in toks if t not in vocab]

            qv = vec.transform([q_p])
            nnz = int(qv.nnz)

            if nnz > 0:
                sims = cosine_similarity(qv, X).ravel()
                top_idx = np.argsort(-sims)[:5]
                top5 = [(doctor_ids[i], float(sims[i])) for i in top_idx]
                top1 = top5[0][1]
            else:
                top5 = []
                top1 = 0.0

            q_terms = top_terms_from_query_vector(qv, feature_names, top_n=8)

            global_stats.append((set_name, q, q_p, nnz, top1))

            lines.append(f"query_raw={q}")
            lines.append(f"query_preprocessed={q_p}")
            lines.append(f"tokens={toks}")
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

    lines.append("=" * 72)
    lines.append("SUMMARY")
    lines.append("=" * 72)

    total = len(global_stats)
    nnz0 = sum(1 for x in global_stats if x[3] == 0)
    sim0 = sum(1 for x in global_stats if x[4] == 0.0)

    lines.append(f"queries_total={total}")
    lines.append(f"queries_with_nnz_0={nnz0} ({(nnz0/total)*100:.1f}%)")
    lines.append(f"queries_with_top1_sim_0={sim0} ({(sim0/total)*100:.1f}%)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved_report={OUT_TXT.resolve()}")


if __name__ == "__main__":
    main()