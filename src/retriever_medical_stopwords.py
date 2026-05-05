# src/retriever_medical_stopwords.py
from __future__ import annotations

from pathlib import Path


def norm_tok(t: str) -> str:
    """
    Normalize token for stopword matching:
      - str + strip
      - lowercase
      - remove ZWNJ
    """
    if t is None:
        return ""
    t = str(t).strip().lower()
    t = t.replace("\u200c", "")  # ZWNJ
    return t


# Seed stopwords for MEDICAL retriever (v0.2)
# Notes:
# - Unigram "بی/فوق/العاده" intentionally NOT included (risk for medical phrases)
# - Praise phrases removed via bigrams
MEDICAL_STOPWORDS_SEED_V02: set[str] = {
    # praise / sentiment
    "عالی",
    "خوب",
    "بهترین",
    "فوقالعاده",   # single-token form in data
    "محشر",
    "بینظیر",      # single-token form in data
    "راضی",
    "رضایت",
    "ناراضی",

    # thanks
    "مرسی",
    "ممنون",
    "تشکر",
    "سپاس",
    "قدردانی",

    # doctor addressing
    "دکتر",
    "پزشک",
    "خانم",
    "آقا",
    "آقای",
    "جناب",
    "ایشون",
    "ایشان",

    # service/clinic ops
    "مطب",
    "منشی",
    "پرسنل",
    "پذیرش",
    "نوبت",
    "نوبتدهی",
    "وقت",
    "معطلی",
    "شلوغ",
    "شلوغی",
    "هزینه",
    "ویزیت",
    "پرداخت",
    "کارت",
    "بیمه",
    "تومان",
    "تومن",
    "خدمات",
    "برخورد",
    "پاسخگو",

    # behavioral adjectives (mostly not symptom-specific)
    "حوصله",
    "اخلاق",
    "حاذق",
    "مودب",
    "محترم",
    "مهربان",
    "دلسوز",
    "صبور",
    "دقیق",
    "خوشبرخورد",

    # generic fillers that are frequent in your corpus
    "سلام",
    "درود",
    "واقعا",
    "فعلا",
    "مراجعه",
    "نتیجه",
    "هستم",
    "هستن",
    "هستند",
    "داشتم",
    "بودم",
    "بودن",
}

# Bigram stopwords (phrase-aware praise)
MEDICAL_STOPWORD_BIGRAMS_V01: set[tuple[str, str]] = {
    ("فوق", "العاده"),
    ("بی", "نظیر"),
}

# Latin allowlist extracted from your audit (always keep)
MEDICAL_ALLOWLIST_LATIN: set[str] = {
    "acl", "mcl", "pcl",
    "mri", "ct",
    "prp", "hpv",
    "pcnl", "rirs", "tul",
    "ivf", "iui",
    "psa", "ms", "avm",
    "co2",
    "l5", "s1", "t12",
}

# HIGH-RISK medical core tokens:
# Even if auto-stopwords suggests them, we do NOT allow removing them (very important for bigrams).
HIGH_RISK_KEEP: set[str] = {
    "درد",
    "دارو",
    "درمان",
    "تشخیص",
    "عمل",
    "جراحی",
    # additionally risky due to common medical phrases
    "بی",       # بی اختیاری / بی حسی / ...
    "فوق",      # فوق تخصص / ...
}


def load_stopwords_file(path: Path) -> set[str]:
    """
    Load stopwords from a text file (one token per line).
    Ignores:
      - blank lines
      - lines starting with '#'
      - lines with spaces (we treat them as not-unigram stopwords)
    """
    if not path.exists():
        raise FileNotFoundError(str(path.resolve()))

    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        # only accept unigram tokens in this loader
        if " " in s:
            continue
        out.add(norm_tok(s))
    return out


def make_effective_stopwords(stopwords: set[str]) -> tuple[set[str], dict]:
    """
    Safety filter so important terms never get removed.
    Returns:
      effective_stopwords, meta
    """
    raw = {norm_tok(x) for x in stopwords if norm_tok(x)}

    excluded_high_risk = sorted([t for t in raw if t in HIGH_RISK_KEEP])
    excluded_latin_allow = sorted([t for t in raw if t in MEDICAL_ALLOWLIST_LATIN])
    excluded_short = sorted([t for t in raw if len(t) < 2])

    effective = set(raw)
    effective -= set(excluded_high_risk)
    effective -= set(excluded_latin_allow)
    effective -= set(excluded_short)

    meta = {
        "raw_count": len(raw),
        "effective_count": len(effective),
        "excluded_high_risk": excluded_high_risk,
        "excluded_latin_allow": excluded_latin_allow,
        "excluded_short": excluded_short,
    }
    return effective, meta


def load_medical_stopwords(
    final_path: Path | None,
    fallback_seed: set[str] | None = None,
) -> tuple[set[str], dict]:
    """
    Loads stopwords with a safe fallback.
    Priority:
      1) final_path if exists
      2) fallback_seed (default: MEDICAL_STOPWORDS_SEED_V02)

    Returns:
      effective_stopwords, meta (includes source + exclusion details)
    """
    if fallback_seed is None:
        fallback_seed = MEDICAL_STOPWORDS_SEED_V02

    if final_path is not None and final_path.exists():
        raw = load_stopwords_file(final_path)
        source = f"file:{final_path.name}"
    else:
        raw = set(fallback_seed)
        source = "fallback_seed_v02"

    effective, meta2 = make_effective_stopwords(raw)
    meta = {"source": source, **meta2}
    return effective, meta


def filter_text_tokens_space_split(
    text: str,
    stopwords: set[str],
    allowlist_latin: set[str] | None = None,
    stopword_bigrams: set[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """
    text: already tokenized (space-separated).
    Removes stopword unigrams and stopword bigrams.
    Returns:
      filtered_text, removed_tokens_normed (may contain duplicates)
    """
    if text is None:
        return "", []

    allowlist_latin = allowlist_latin or set()
    stopword_bigrams = stopword_bigrams or set()

    toks = str(text).split()
    kept: list[str] = []
    removed: list[str] = []

    i = 0
    while i < len(toks):
        t1 = toks[i]
        n1 = norm_tok(t1)

        # Keep allowlisted latin tokens no matter what
        if n1 in allowlist_latin:
            kept.append(t1)
            i += 1
            continue

        # Bigram stopword check
        if i + 1 < len(toks):
            t2 = toks[i + 1]
            n2 = norm_tok(t2)
            if (n1, n2) in stopword_bigrams:
                removed.append(n1)
                removed.append(n2)
                i += 2
                continue

        # Unigram stopword check
        if n1 in stopwords:
            removed.append(n1)
            i += 1
            continue

        kept.append(t1)
        i += 1

    return " ".join(kept), removed