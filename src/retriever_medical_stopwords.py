# src/retriever_medical_stopwords.py
from __future__ import annotations


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
# - We intentionally DO NOT include unigram: "بی", "فوق", "العاده"
#   because they can be part of medical phrases like: بی اختیاری, بی حسی, فوق تخصص
# - Instead, we remove praise phrases using BIGRAM stopwords: ("فوق","العاده"), ("بی","نظیر")
MEDICAL_STOPWORDS_SEED_V02: set[str] = {
    # praise / sentiment (very common noise)
    "عالی",
    "خوب",
    "بهترین",
    "فوقالعاده",   # some users write as a single token
    "محشر",
    "بینظیر",      # some users write without space
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

    # service/clinic ops (noise for medical symptom retrieval)
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

    # generic fillers that your reports show as high-frequency
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

# Bigram stopwords (phrase-aware)
MEDICAL_STOPWORD_BIGRAMS_V01: set[tuple[str, str]] = {
    ("فوق", "العاده"),
    ("بی", "نظیر"),
}

# Latin allowlist extracted from your own text audit inventory (keep these always)
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
      filtered_text, removed_tokens_normed (can contain duplicates)
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