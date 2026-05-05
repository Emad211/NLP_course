# src/tfidf_retriever_query.py
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from hazm import Normalizer, WordTokenizer, stopwords_list
from sklearn.metrics.pairwise import cosine_similarity

from retriever_medical_stopwords import (
    MEDICAL_ALLOWLIST_LATIN,
    MEDICAL_STOPWORD_BIGRAMS_V01,
    filter_text_tokens_space_split,
    norm_tok,
)


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_BASELINE = BASE_DIR / "models" / "tfidf_doctor_retriever.joblib"
INDEX_MEDICAL = BASE_DIR / "models" / "tfidf_doctor_retriever_medical.joblib"

KW_BASELINE = BASE_DIR / "processed_data" / "retriever" / "doctor_keywords.csv"
KW_MEDICAL = BASE_DIR / "processed_data" / "retriever" / "doctor_keywords_medical.csv"

DOCTORS_CSV = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"

COMMENTS_BASELINE = BASE_DIR / "processed_data" / "final" / "comments_for_tfidf_retriever.csv"
COMMENTS_MEDICAL = BASE_DIR / "processed_data" / "final" / "comments_for_tfidf_retriever_medical.csv"

MEDICAL_STOPWORDS_FILE_DEFAULT = BASE_DIR / "processed_data" / "final" / "retriever_medical_stopwords_final_v03.txt"


# ----------------------------
# Query preprocess (baseline; same idea as Step05/Step06)
# ----------------------------
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


def _build_normalizer() -> Normalizer:
    return Normalizer(
        correct_spacing=True,
        remove_diacritics=True,
        remove_specials_chars=False,
        decrease_repeated_chars=False,
        persian_style=True,
        persian_numbers=False,
    )


def _build_tokenizer() -> WordTokenizer:
    return WordTokenizer(
        separate_emoji=False,
        replace_links=False,
        replace_ids=False,
        replace_emails=False,
        replace_numbers=False,
    )


# build once (avoid repeated work)
_NORMALIZER = _build_normalizer()
_TOKENIZER = _build_tokenizer()

_STOPWORDS = set(stopwords_list())
_STOPWORDS_KEEP = {"نه", "نخیر", "نیست", "نمی", "نمي", "نبود", "نشد", "بدون"}


def preprocess_query_to_text_for_tfidf(query: str) -> str:
    """
    Baseline query preprocessing: matches your Step05/Step06 baseline idea.
    Returns space-tokenized text suitable for vectorizer(tokenizer=str.split).

    NOTE: This function is used by other scripts (test suite), so keep it stable.
    """
    if query is None:
        return ""
    t = str(query)

    t = CONTROL_CHARS_RE.sub(" ", t)
    t = t.translate(DIGIT_MAP)

    # Arabic to Persian-ish normalization
    t = (
        t.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("أ", "ا")
        .replace("إ", "ا")
        .replace("ؤ", "و")
        .replace("ة", "ه")
        .replace("ٱ", "ا")
    )

    t = KEEP_CHARS_RE.sub(" ", t)
    t = ELONG_RE.sub(r"\1\1", t)

    # pre-hazm clean
    t = PERS_PUNCT_RE.sub(" ", t)
    t = DOT_SAFE_RE.sub(" ", t)
    t = MULTI_SPACE.sub(" ", t).strip()

    # hazm normalize
    t = _NORMALIZER.normalize(t)

    # domain deglue
    for pat, repl in DEGLUE_RULES:
        t = pat.sub(repl, t)
    t = MULTI_SPACE.sub(" ", t).strip()

    # tokenize
    toks_all = [x for x in _TOKENIZER.tokenize(t) if len(x) >= MIN_TOKEN_LEN]
    toks_ns = [x for x in toks_all if (x not in _STOPWORDS) or (x in _STOPWORDS_KEEP)]

    if toks_ns:
        return " ".join(toks_ns)
    return " ".join(toks_all)


# ----------------------------
# Medical query preprocessing (must match Step06 medical stopwording)
# ----------------------------
HIGH_RISK_KEEP = {"درد", "دارو", "درمان", "تشخیص", "عمل", "جراحی", "کار"}


def load_stopwords_unigrams(path: Path) -> set[str]:
    """
    Loads stopwords (unigrams) from file.
    One token per line. Ignores empty lines and comment lines starting with '#'.
    Tokens are normalized using norm_tok (lowercase + remove ZWNJ).
    """
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))

    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        t = norm_tok(line)
        if not t:
            continue
        out.add(t)
    return out


def preprocess_query_to_text_for_tfidf_medical(
    query_raw: str,
    stopwords_unigrams: set[str],
) -> Tuple[str, str, str]:
    """
    Returns:
      q_baseline: output of preprocess_query_to_text_for_tfidf
      q_medical : baseline + medical stopwording (file-based) + bigram stopwords + allowlist + safety guard
      removed_tokens_sample: readable unique removed tokens (normed)
    """
    q_baseline = preprocess_query_to_text_for_tfidf(query_raw)
    if not q_baseline.strip():
        return q_baseline, "", ""

    # safety guard: never remove core medical tokens
    effective_stop = set(stopwords_unigrams) - set(HIGH_RISK_KEEP)

    q_medical, removed = filter_text_tokens_space_split(
        q_baseline,
        stopwords=effective_stop,
        allowlist_latin=MEDICAL_ALLOWLIST_LATIN,
        stopword_bigrams=MEDICAL_STOPWORD_BIGRAMS_V01,
    )

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

    return q_baseline, q_medical, removed_sample


# ----------------------------
# Metadata + Evidence store (for recommendation scoring)
# ----------------------------
def load_metadata(variant: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      name_map, specialty_map, profile_url_map, keywords_map
    """
    name_map: Dict[str, str] = {}
    specialty_map: Dict[str, str] = {}
    profile_url_map: Dict[str, str] = {}

    if DOCTORS_CSV.exists():
        d = pd.read_csv(DOCTORS_CSV, keep_default_na=False)
        d["doctor_id"] = d["doctor_id"].astype(str)
        if "name" in d.columns:
            name_map = dict(zip(d["doctor_id"], d["name"]))
        if "specialty" in d.columns:
            specialty_map = dict(zip(d["doctor_id"], d["specialty"]))
        if "profile_url" in d.columns:
            profile_url_map = dict(zip(d["doctor_id"], d["profile_url"]))

    kw_path = KW_MEDICAL if variant == "medical" else KW_BASELINE
    kw_map: Dict[str, str] = {}
    if kw_path.exists():
        k = pd.read_csv(kw_path, keep_default_na=False)
        k["doctor_id"] = k["doctor_id"].astype(str)
        if "top_terms" in k.columns:
            kw_map = dict(zip(k["doctor_id"], k["top_terms"]))

    return name_map, specialty_map, profile_url_map, kw_map


@dataclass
class DoctorEvidence:
    total_comments: int
    total_recommend: int  # rate >= 4
    texts: List[str]
    is_recommend: List[bool]


def load_evidence(variant: str) -> Dict[str, DoctorEvidence]:
    """
    Loads comment-level evidence for query-specific recommendation scoring.
    We only need this for the *rerank* step on top-K doctors.

    For medical variant: reads processed_data/final/comments_for_tfidf_retriever_medical.csv
    For baseline: reads processed_data/final/comments_for_tfidf_retriever.csv

    Uses:
      - doctor_id
      - text_for_tfidf   (already tokenized)
      - rate
      - is_placeholder_negative (optional)
    """
    in_csv = COMMENTS_MEDICAL if variant == "medical" else COMMENTS_BASELINE
    if not in_csv.exists():
        raise FileNotFoundError(str(in_csv.resolve()))

    df = pd.read_csv(in_csv, keep_default_na=False)
    if "doctor_id" not in df.columns or "text_for_tfidf" not in df.columns:
        raise ValueError(f"Evidence CSV missing required columns: {in_csv.name}")

    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_for_tfidf"] = df["text_for_tfidf"].astype(str)

    # filter empties
    df = df[df["text_for_tfidf"].str.strip().ne("")].copy()

    # filter placeholders if column exists
    if "is_placeholder_negative" in df.columns:
        df = df[~df["is_placeholder_negative"].astype(bool)].copy()

    # rate -> numeric
    if "rate" in df.columns:
        df["rate_num"] = pd.to_numeric(df["rate"], errors="coerce")
    else:
        df["rate_num"] = np.nan

    df["is_recommend"] = df["rate_num"].ge(4.0)  # True if rate >= 4

    evid: Dict[str, DoctorEvidence] = {}
    for did, g in df.groupby("doctor_id", sort=False):
        texts = g["text_for_tfidf"].astype(str).tolist()
        flags = g["is_recommend"].fillna(False).astype(bool).tolist()
        total = len(texts)
        total_rec = int(np.sum(flags))
        evid[str(did)] = DoctorEvidence(
            total_comments=total,
            total_recommend=total_rec,
            texts=texts,
            is_recommend=flags,
        )

    return evid


# ----------------------------
# Retrieval + Rerank
# ----------------------------
def retrieve_similar(
    idx_obj: dict,
    query_text_for_index: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    """
    Returns list of (doctor_id, sim) sorted by similarity desc.
    """
    vec = idx_obj["vectorizer"]
    X = idx_obj["X"]
    doctor_ids = idx_obj["doctor_ids"]

    if not query_text_for_index.strip():
        return []

    qv = vec.transform([query_text_for_index])
    if int(qv.nnz) == 0:
        return []

    sims = cosine_similarity(qv, X).ravel()
    top_idx = np.argsort(-sims)[: int(top_k)]
    return [(str(doctor_ids[int(i)]), float(sims[int(i)])) for i in top_idx]


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_query_match_terms(vectorizer, query_text_for_index: str) -> List[str]:
    """
    Use the SAME analyzer as the retriever vectorizer to build the terms we consider "query evidence".
    Includes unigrams + bigrams if ngram_range=(1,2).
    """
    if not query_text_for_index.strip():
        return []
    analyzer = vectorizer.build_analyzer()
    terms = list(analyzer(query_text_for_index))
    return unique_preserve_order([t for t in terms if str(t).strip()])


def comment_matches_any_term(comment_text: str, terms: List[str]) -> bool:
    """
    Token-boundary-safe match for space-tokenized texts.
    We assume comment_text is space-separated tokens.
    """
    if not comment_text or not terms:
        return False
    padded = f" {comment_text} "
    for t in terms:
        if f" {t} " in padded:
            return True
    return False


@dataclass
class RerankRow:
    doctor_id: str
    sim: float
    final_score: float
    q_support: int
    q_recommend: int
    q_recommend_pct: float
    q_recommend_smooth: float
    overall_support: int
    overall_recommend_pct: float


def rerank_with_recommendation(
    idx_obj: dict,
    evidence: Dict[str, DoctorEvidence],
    top_candidates: List[Tuple[str, float]],
    query_text_for_index: str,
) -> List[RerankRow]:
    """
    Rerank top candidates by combining:
      - similarity (TF-IDF)
      - query-specific recommendation ratio among matched comments (rate>=4)
      - support (how many matched comments)

    final_score = sim * smooth_recommend * log1p(q_support)

    smooth_recommend = (q_recommend + 1) / (q_support + 2)  (Beta(1,1) smoothing)
    """
    vec = idx_obj["vectorizer"]
    terms = build_query_match_terms(vec, query_text_for_index)

    out: List[RerankRow] = []

    for did, sim in top_candidates:
        ev = evidence.get(did)
        if ev is None or ev.total_comments <= 0:
            out.append(
                RerankRow(
                    doctor_id=did,
                    sim=float(sim),
                    final_score=0.0,
                    q_support=0,
                    q_recommend=0,
                    q_recommend_pct=0.0,
                    q_recommend_smooth=0.5,
                    overall_support=0,
                    overall_recommend_pct=0.0,
                )
            )
            continue

        matched = 0
        matched_rec = 0

        for txt, rec in zip(ev.texts, ev.is_recommend):
            if comment_matches_any_term(txt, terms):
                matched += 1
                if rec:
                    matched_rec += 1

        if matched > 0:
            rec_pct = matched_rec / matched
        else:
            rec_pct = 0.0

        rec_smooth = (matched_rec + 1.0) / (matched + 2.0)  # smoothing
        final = float(sim) * float(rec_smooth) * float(np.log1p(matched))

        overall_pct = (ev.total_recommend / ev.total_comments) if ev.total_comments else 0.0

        out.append(
            RerankRow(
                doctor_id=did,
                sim=float(sim),
                final_score=float(final),
                q_support=int(matched),
                q_recommend=int(matched_rec),
                q_recommend_pct=float(rec_pct),
                q_recommend_smooth=float(rec_smooth),
                overall_support=int(ev.total_comments),
                overall_recommend_pct=float(overall_pct),
            )
        )

    out.sort(key=lambda r: (-r.final_score, -r.sim, -r.q_support))
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="TF-IDF Doctor Retriever (medical) + recommendation rerank (rate>=4).")
    p.add_argument("--variant", type=str, default="medical", choices=["baseline", "medical"])
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--show-sim-ranking", action="store_true", help="Also print raw similarity ranking.")
    p.add_argument(
        "--medical-stopwords-file",
        type=str,
        default=str(MEDICAL_STOPWORDS_FILE_DEFAULT),
        help="Used when variant=medical. One token per line.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    variant = str(args.variant)

    index_path = INDEX_MEDICAL if variant == "medical" else INDEX_BASELINE
    if not index_path.exists():
        raise FileNotFoundError(str(index_path.resolve()))

    # Load index
    idx_obj = joblib.load(index_path)

    # Load stopwords for medical query preprocessing (if needed)
    medical_sw = set()
    medical_sw_src = ""
    if variant == "medical":
        sw_path = Path(args.medical_stopwords_file)
        medical_sw = load_stopwords_unigrams(sw_path)
        medical_sw_src = f"file:{sw_path.name} raw={len(medical_sw)} effective={len(medical_sw)}"

    # Load metadata + keywords (variant-specific)
    name_map, specialty_map, profile_url_map, kw_map = load_metadata(variant)

    # Load evidence for reranking
    evidence = load_evidence(variant)

    print(f"variant={variant}")
    print(f"index={index_path.resolve()}")
    if variant == "medical":
        print(f"medical_stopwords={medical_sw_src}")
        print(f"high_risk_keep={sorted(HIGH_RISK_KEEP)}")
        print(f"medical_stopword_bigrams={sorted([' '.join(x) for x in MEDICAL_STOPWORD_BIGRAMS_V01])}")
    print("type 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break

        # Baseline preprocessing always computed (useful to display)
        q_baseline = preprocess_query_to_text_for_tfidf(query)

        # Decide the actual query text used for the index
        if variant == "medical":
            _, q_for_index, removed_sample = preprocess_query_to_text_for_tfidf_medical(query, medical_sw)
        else:
            q_for_index = q_baseline
            removed_sample = ""

        print(f"\nquery_raw='{query}'")
        print(f"query_preprocessed='{q_baseline}'")
        if variant == "medical":
            print(f"query_medical_for_index='{q_for_index}'")
            print(f"removed_tokens_sample='{removed_sample}'")

        if not q_for_index.strip():
            print("no_results (query empty after preprocessing)\n")
            continue

        # Retrieve by TF-IDF similarity
        sim_results = retrieve_similar(idx_obj, q_for_index, top_k=int(args.topk))
        if not sim_results:
            print("no_results (nnz=0 or empty retrieval)\n")
            continue

        # Rerank using recommendation evidence
        reranked = rerank_with_recommendation(
            idx_obj=idx_obj,
            evidence=evidence,
            top_candidates=sim_results,
            query_text_for_index=q_for_index,
        )

        # Print reranked results
        print("\nTop results (reranked by: sim * smooth_recommend * log1p(query_support)):")
        for rank, r in enumerate(reranked, start=1):
            did = r.doctor_id
            nm = name_map.get(did, "")
            sp = specialty_map.get(did, "")
            pu = profile_url_map.get(did, "")
            kw = kw_map.get(did, "")

            q_pct = 100.0 * r.q_recommend_pct if r.q_support > 0 else 0.0
            overall_pct = 100.0 * r.overall_recommend_pct if r.overall_support > 0 else 0.0

            line = (
                f"{rank:>2}. doctor_id={did}"
                f"  final={r.final_score:.4f}"
                f"  sim={r.sim:.4f}"
                f"  q_support={r.q_support}"
                f"  q_rec={r.q_recommend}/{r.q_support} ({q_pct:.1f}%)"
                f"  overall_rec={overall_pct:.1f}%"
            )
            if nm:
                line += f"  name={nm}"
            if sp:
                line += f"  specialty={sp}"
            print(line)

            if pu:
                print(f"    url={pu}")
            if kw:
                print(f"    keywords={kw}")

        # Optional: show raw similarity ranking
        if args.show_sim_ranking:
            print("\nRaw similarity ranking (top-k):")
            for rank, (did, sim) in enumerate(sim_results, start=1):
                nm = name_map.get(did, "")
                sp = specialty_map.get(did, "")
                line = f"{rank:>2}. doctor_id={did}  sim={sim:.4f}"
                if nm:
                    line += f"  name={nm}"
                if sp:
                    line += f"  specialty={sp}"
                print(line)

        print("")


if __name__ == "__main__":
    main()