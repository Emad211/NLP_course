import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from hazm import Normalizer, WordTokenizer, stopwords_list
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_JOBLIB = BASE_DIR / "models" / "tfidf_doctor_retriever.joblib"
DOCTOR_KEYWORDS_CSV = BASE_DIR / "processed_data" / "retriever" / "doctor_keywords.csv"
DOCTORS_CSV = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"

DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
CONTROL_CHARS_RE = re.compile(r"[\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u00ad\u2060\u180e]")
KEEP_CHARS_RE = re.compile(r"[^0-9A-Za-z\u0600-\u06FF\s]")
ELONG_RE = re.compile(r"(\S)\1{2,}")
PERS_PUNCT_RE = re.compile(r"[،؛؟!٪«»]+")
DOT_SAFE_RE = re.compile(r"(?<!\d)\.(?!\d)")
MULTI_SPACE = re.compile(r"\s+")

DEGLUE_RULES = [
    (re.compile(r"(?<!\S)درحال(?!\S)"), "در حال"),
    (re.compile(r"(?<!\S)باسلام(?!\S)"), "با سلام"),
    (re.compile(r"(?<!\S)خداروشکر(?!\S)"), "خدا رو شکر"),
    (re.compile(r"(?<!\S)فوقالعاده(?!\S)"), "فوق العاده"),
    (re.compile(r"(?<!\S)اززحماتشون(?!\S)"), "از زحماتشون"),
    (re.compile(r"(?<!\S)ازبچگی(?!\S)"), "از بچگی"),
    (re.compile(r"(?<!\S)مداومبه(?!\S)"), "مداوم به"),
    (re.compile(r"(?<!\S)هردکتری(?!\S)"), "هر دکتری"),
    (re.compile(r"(?<!\S)اماایشان(?!\S)"), "اما ایشان"),
    (re.compile(r"(?<!\S)وسپس(?!\S)"), "و سپس"),
    (re.compile(r"(?<!\S)برطرفشد(?!\S)"), "برطرف شد"),
]

MIN_TOKEN_LEN = 2


def build_normalizer():
    return Normalizer(
        correct_spacing=True,
        remove_diacritics=True,
        remove_specials_chars=False,
        decrease_repeated_chars=False,
        persian_style=True,
        persian_numbers=False,
    )


def build_tokenizer():
    return WordTokenizer(
        separate_emoji=False,
        replace_links=False,
        replace_ids=False,
        replace_emails=False,
        replace_numbers=False,
    )


def build_stopwords():
    sw = set(stopwords_list())
    keep = {"نه", "نخیر", "نیست", "نمی", "نمي", "نبود", "نشد", "بدون"}
    return sw, keep


def preprocess_query_to_text_for_tfidf(query: str) -> str:
    if query is None:
        return ""
    t = str(query)

    t = CONTROL_CHARS_RE.sub(" ", t)
    t = t.translate(DIGIT_MAP)

    t = t.replace("ي", "ی").replace("ك", "ک").replace("أ", "ا").replace("إ", "ا").replace("ؤ", "و").replace("ة", "ه").replace("ٱ", "ا")
    t = KEEP_CHARS_RE.sub(" ", t)
    t = ELONG_RE.sub(r"\1\1", t)

    t = PERS_PUNCT_RE.sub(" ", t)
    t = DOT_SAFE_RE.sub(" ", t)
    t = MULTI_SPACE.sub(" ", t).strip()

    norm = build_normalizer()
    t = norm.normalize(t)

    for pat, repl in DEGLUE_RULES:
        t = pat.sub(repl, t)
    t = MULTI_SPACE.sub(" ", t).strip()

    tokenizer = build_tokenizer()
    sw, keep = build_stopwords()

    toks_all = [x for x in tokenizer.tokenize(t) if len(x) >= MIN_TOKEN_LEN]
    toks_ns = [x for x in toks_all if (x not in sw) or (x in keep)]

    if len(toks_ns) > 0:
        return " ".join(toks_ns)
    return " ".join(toks_all)


def load_metadata():
    name_map = {}
    specialty_map = {}
    profile_url_map = {}

    if DOCTORS_CSV.exists():
        d = pd.read_csv(DOCTORS_CSV, keep_default_na=False)
        d["doctor_id"] = d["doctor_id"].astype(str)
        if "name" in d.columns:
            name_map = dict(zip(d["doctor_id"], d["name"]))
        if "specialty" in d.columns:
            specialty_map = dict(zip(d["doctor_id"], d["specialty"]))
        if "profile_url" in d.columns:
            profile_url_map = dict(zip(d["doctor_id"], d["profile_url"]))

    kw_map = {}
    if DOCTOR_KEYWORDS_CSV.exists():
        k = pd.read_csv(DOCTOR_KEYWORDS_CSV, keep_default_na=False)
        k["doctor_id"] = k["doctor_id"].astype(str)
        kw_map = dict(zip(k["doctor_id"], k["top_terms"]))

    return name_map, specialty_map, profile_url_map, kw_map


def retrieve(query: str, top_k: int = 10):
    idx = joblib.load(INDEX_JOBLIB)
    vectorizer = idx["vectorizer"]
    X = idx["X"]
    doctor_ids = idx["doctor_ids"]

    q = preprocess_query_to_text_for_tfidf(query)
    if not q.strip():
        return q, []

    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, X).ravel()

    top_idx = np.argsort(-sims)[:top_k]
    results = [(doctor_ids[i], float(sims[i])) for i in top_idx]
    return q, results


def main():
    if not INDEX_JOBLIB.exists():
        raise FileNotFoundError(str(INDEX_JOBLIB.resolve()))

    name_map, specialty_map, profile_url_map, kw_map = load_metadata()

    print(f"index={INDEX_JOBLIB.resolve()}")
    print("type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break

        q_clean, results = retrieve(query, top_k=10)
        print(f"\nquery_preprocessed='{q_clean}'")
        if not results:
            print("no_results\n")
            continue

        print("\nTop results:")
        for rank, (did, sim) in enumerate(results, start=1):
            nm = name_map.get(did, "")
            sp = specialty_map.get(did, "")
            pu = profile_url_map.get(did, "")
            kw = kw_map.get(did, "")
            line = f"{rank:>2}. doctor_id={did}  sim={sim:.4f}"
            if nm:
                line += f"  name={nm}"
            if sp:
                line += f"  specialty={sp}"
            print(line)
            if pu:
                print(f"    url={pu}")
            if kw:
                print(f"    keywords={kw}")
        print("")


if __name__ == "__main__":
    main()