from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

IN_FULL = BASE_DIR / "processed_data" / "final" / "comments_preprocessed_full.csv"
DOCTORS_CANON = BASE_DIR / "processed_data" / "doctors_raw_canonical.csv"

OUT_DIR = BASE_DIR / "processed_data" / "sentiment"
OUT_CSV = OUT_DIR / "labeling_pool.csv"
OUT_JSONL = OUT_DIR / "labeling_pool.jsonl"
OUT_REPORT = OUT_DIR / "labeling_pool_report.txt"


@dataclass
class PoolConfig:
    seed: int = 42
    total_size: int = 8000

    # هدف: منفیِ بیشتری وارد pool کنیم (برای لیبل‌گذاری)
    n_high_neg: int = 1600
    n_service_neg: int = 1400
    n_rate_neg: int = 1400
    n_mid_neg: int = 900
    n_mixed: int = 700
    n_high_pos: int = 700
    n_random: int = 1300  # بقیه برای پوشش خنثی/اطلاعاتی

    max_per_doctor: int = 6
    max_per_specialty: int = 5000

    min_chars: int = 5
    min_words: int = 2

    dedup_col: str = "text_step04"
    keep_longer_on_dup: bool = True

    # thresholds
    high_neg_threshold: float = 2.2
    mid_neg_threshold: float = 1.2

    mixed_neg_threshold: float = 1.2
    mixed_pos_threshold: float = 1.3

    high_pos_threshold: float = 3.0
    high_pos_max_neg: float = 1.0


CFG = PoolConfig()

SPACE_RE = re.compile(r"\s+")
ZWNJ_RE = re.compile(r"\u200c")


def norm_space(s: str) -> str:
    s = str(s or "")
    s = ZWNJ_RE.sub("‌", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_col(df: pd.DataFrame, col: str, default=None) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    return df


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


NEG_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("strong_negative", re.compile(r"(افتضاح|مزخرف|فاجعه|بدترین|خیلی\s*بد|به\s*درد\s*نخورد|به\s*درد\s*نمیخوره|به\s*درد\s*نمی\s*خوره)"), 2.6),
    ("not_recommend", re.compile(r"(پیشنهاد\s*نمی\s*کنم|توصیه\s*نمی\s*کنم|دیگه\s*نمیرم|دیگر\s*نمیرم|پشیمون\s*شدم|پشیمان\s*شدم)"), 2.6),
    ("rude_behavior", re.compile(r"(بد\s*برخورد|بی\s*ادب|بی\s*احترام|توهین|تحقیر|داد\s*زد|اخمو|بی\s*حوصله)"), 2.2),
    ("long_wait", re.compile(r"(معطلی|منتظر|تاخیر|تأخیر|دیر\s*کرد|ساعت\s*ها|نوبت.*عقب|شلوغ)"), 1.8),
    ("no_response", re.compile(r"(پاسخگو\s*نبود|جواب\s*نمی\s*ده|تلفن.*جواب|منشی.*جواب|کسی\s*پاسخگو)"), 1.9),
    ("money", re.compile(r"(هزینه|گران|گرون|پولکی|پول\s*اضافی|ویزیت.*بالا|بیخود\s*پول)"), 1.4),
    ("misdiagnosis", re.compile(r"(تشخیص.*اشتباه|اشتباه\s*تشخیص|غلط\s*تشخیص|دارو.*اشتباه)"), 2.2),
    ("no_improve", re.compile(r"(بهتر\s*نشد|بدتر\s*شدم|اثر\s*نداشت|فایده\s*نداشت|نتیجه\s*نگرفتم|درمان\s*نشد|هیچ\s*تغییری)"), 2.0),
    ("careless", re.compile(r"(سرسری|بی\s*دقت|بدون\s*معاینه|توجه\s*نکرد|گوش\s*نکرد|بی\s*تفاوت)"), 1.6),
    ("disappointed_soft", re.compile(r"(متاسفانه|متأسفانه|کاش|ناراضی|راضی\s*نبودم|اصلا\s*خوب\s*نبود|اصلاً\s*خوب\s*نبود)"), 1.6),
]

POS_PATTERNS: List[Tuple[str, re.Pattern, float]] = [
    ("strong_positive", re.compile(r"(عالی|فوق\s*العاده|بی\s*نظیر|محشر|بهترین)"), 2.5),
    ("thanks", re.compile(r"(ممنون|سپاس|تشکر|مرسی|خدا\s*خیر|دستتون\s*درد\s*نکنه|دستش\s*درد\s*نکنه)"), 1.3),
    ("skilled", re.compile(r"(حاذق|ماهر|کاربلد|متخصص|تشخیص\s*دقیق|تجربه\s*بالا)"), 1.6),
    ("kind", re.compile(r"(مهربان|خوش\s*برخورد|با\s*حوصله|صبور|آرام)"), 1.0),
    ("improved", re.compile(r"(بهتر\s*شدم|خوب\s*شدم|درمان\s*شدم|نتیجه\s*گرفتم)"), 1.8),
]

SERVICE_TERMS = re.compile(r"(نوبت|نوبت\s*دهی|منشی|مطب|تلفن|تماس|ویزیت|هزینه|معطلی|انتظار|صف|ساعت|شلوغ)")
MILD_NEG_CUE = re.compile(r"(بد|ضعیف|اشتباه|ناراضی|راضی\s*نبود|متاسفانه|متأسفانه|پشیمون|پشیمان|بی\s*نظم|بی\s*مسئولیت|بی\s*توجه|سرسری)")


def score_patterns(text: str, patterns: List[Tuple[str, re.Pattern, float]]) -> Tuple[float, List[str]]:
    t = safe_str(text)
    hits = []
    score = 0.0
    for name, pat, w in patterns:
        if pat.search(t):
            hits.append(name)
            score += w
    return score, hits


def detect_placeholder_negative(df: pd.DataFrame, text_col: str) -> pd.Series:
    if "is_placeholder_negative" in df.columns:
        s = df["is_placeholder_negative"].fillna(0).astype(int) == 1
    else:
        s = pd.Series(False, index=df.index)

    t = df[text_col].fillna("").astype(str)
    t_norm = t.str.replace(SPACE_RE, " ", regex=True).str.strip()
    s2 = t_norm.str.fullmatch(r"عدم رضایت") | t_norm.str.contains(r"\bعدم\s*رضایت\b", regex=True)

    return (s | s2)


def build_dedup_key(series: pd.Series) -> pd.Series:
    key = series.fillna("").astype(str)
    key = key.str.replace(SPACE_RE, " ", regex=True).str.strip()
    return key


def sample_with_caps(
    df: pd.DataFrame,
    n: int,
    seed: int,
    doctor_col: str,
    specialty_col: str,
    max_per_doctor: int,
    max_per_specialty: int,
) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.head(0)

    rng = np.random.RandomState(seed)
    df_shuf = df.sample(frac=1.0, random_state=rng)

    doc_cnt: Dict[str, int] = {}
    spec_cnt: Dict[str, int] = {}
    picked = []

    for idx, r in df_shuf.iterrows():
        did = safe_str(r.get(doctor_col, ""))
        spec = safe_str(r.get(specialty_col, ""))

        if did and doc_cnt.get(did, 0) >= max_per_doctor:
            continue
        if spec and spec_cnt.get(spec, 0) >= max_per_specialty:
            continue

        picked.append(idx)

        if did:
            doc_cnt[did] = doc_cnt.get(did, 0) + 1
        if spec:
            spec_cnt[spec] = spec_cnt.get(spec, 0) + 1

        if len(picked) >= n:
            break

    return df.loc[picked].copy()


def load_specialty_map() -> Dict[str, str]:
    if not DOCTORS_CANON.exists():
        return {}
    d = pd.read_csv(DOCTORS_CANON, encoding="utf-8")
    did_col = pick_first_existing_col(d, ["doctor_id", "dr_id", "id"])
    spec_col = pick_first_existing_col(d, ["specialty", "speciality", "field"])
    if did_col is None or spec_col is None:
        return {}

    d[did_col] = d[did_col].astype(str)
    d[spec_col] = d[spec_col].fillna("").astype(str).apply(norm_space)
    return dict(zip(d[did_col], d[spec_col]))


def rate_neg_bonus(rate) -> float:
    try:
        r = float(rate)
    except Exception:
        return 0.0
    if np.isnan(r):
        return 0.0
    if r <= 2.0:
        return 2.6
    if r <= 3.0:
        return 1.6
    if r <= 3.5:
        return 1.0
    return 0.0


def rate_pos_bonus(rate) -> float:
    try:
        r = float(rate)
    except Exception:
        return 0.0
    if np.isnan(r):
        return 0.0
    if r >= 4.8:
        return 0.9
    if r >= 4.5:
        return 0.6
    return 0.0


def main():
    if not IN_FULL.exists():
        raise FileNotFoundError(f"Missing input: {IN_FULL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_FULL, encoding="utf-8")

    text_col = pick_first_existing_col(df, ["text_step04", "text_raw", "text", "comment_text", "text_for_tfidf"])
    if text_col is None:
        raise RuntimeError("No usable text column found.")

    df = ensure_col(df, "doctor_id", "")
    df = ensure_col(df, "specialty", "")
    df = ensure_col(df, "rate", np.nan)
    df = ensure_col(df, "label", np.nan)
    df = ensure_col(df, "text_for_tfidf", "")

    df["doctor_id"] = df["doctor_id"].fillna("").astype(str)
    df[text_col] = df[text_col].apply(norm_space)
    df["text_label"] = df[text_col].fillna("").astype(str).apply(norm_space)

    # Fill specialty (as v2)
    spec_map = load_specialty_map()
    if spec_map:
        df["specialty"] = df["specialty"].fillna("").astype(str).apply(norm_space)
        empty_spec = df["specialty"].eq("")
        df.loc[empty_spec, "specialty"] = df.loc[empty_spec, "doctor_id"].map(spec_map).fillna("")
    df["specialty"] = df["specialty"].fillna("").astype(str).apply(norm_space)
    df.loc[df["specialty"].eq(""), "specialty"] = "UNKNOWN"

    # Remove placeholder negatives
    placeholder_mask = detect_placeholder_negative(df, text_col=text_col)
    df_work = df.loc[~placeholder_mask].copy()

    # Basic sanity
    df_work["n_chars"] = df_work["text_label"].str.len()
    df_work["n_words"] = df_work["text_label"].str.split().apply(len)
    df_work = df_work.loc[(df_work["n_chars"] >= CFG.min_chars) & (df_work["n_words"] >= CFG.min_words)].copy()

    # Dedup
    dedup_col = CFG.dedup_col if CFG.dedup_col in df_work.columns else "text_label"
    df_work[dedup_col] = df_work[dedup_col].fillna("").astype(str).apply(norm_space)
    df_work["dedup_key"] = build_dedup_key(df_work[dedup_col])
    df_work["src_row_id"] = df_work.index.astype(int)

    if CFG.keep_longer_on_dup:
        df_work = df_work.sort_values(["dedup_key", "n_words", "n_chars"], ascending=[True, False, False])
        df_work = df_work.drop_duplicates("dedup_key", keep="first")
    else:
        df_work = df_work.drop_duplicates("dedup_key", keep="first")

    # Scores
    neg_scores, neg_hits = [], []
    pos_scores, pos_hits = [], []
    svc_term, svc_neg = [], []

    for t in df_work["text_label"].tolist():
        ns, nh = score_patterns(t, NEG_PATTERNS)
        ps, ph = score_patterns(t, POS_PATTERNS)
        neg_scores.append(ns)
        neg_hits.append("|".join(nh))
        pos_scores.append(ps)
        pos_hits.append("|".join(ph))

        has_svc = bool(SERVICE_TERMS.search(t))
        has_mild_neg = bool(MILD_NEG_CUE.search(t))
        svc_term.append(int(has_svc))
        svc_neg.append(int(has_svc and has_mild_neg))

    df_work["neg_score_kw"] = neg_scores
    df_work["neg_hits"] = neg_hits
    df_work["pos_score_kw"] = pos_scores
    df_work["pos_hits"] = pos_hits

    df_work["neg_score_rate"] = df_work["rate"].apply(rate_neg_bonus)
    df_work["pos_score_rate"] = df_work["rate"].apply(rate_pos_bonus)

    df_work["neg_score"] = df_work["neg_score_kw"] + df_work["neg_score_rate"]
    df_work["pos_score"] = df_work["pos_score_kw"] + df_work["pos_score_rate"]

    df_work["has_service_term"] = svc_term
    df_work["has_service_neg_cue"] = svc_neg

    # Buckets (priority order, non-overlapping)
    mixed_mask = (df_work["neg_score_kw"] >= CFG.mixed_neg_threshold) & (df_work["pos_score_kw"] >= CFG.mixed_pos_threshold)

    high_neg_mask = (df_work["neg_score"] >= CFG.high_neg_threshold) & (df_work["pos_score"] <= 1.0) & (~mixed_mask)

    service_neg_mask = (
        (df_work["has_service_term"] == 1) &
        (
            (df_work["has_service_neg_cue"] == 1) |
            (df_work["neg_score_rate"] >= 1.0)  # service + low-ish rate
        ) &
        (~mixed_mask) & (~high_neg_mask)
    )

    rate_neg_mask = (
        (df_work["neg_score_rate"] >= 1.0) &
        (~mixed_mask) & (~high_neg_mask) & (~service_neg_mask)
    )

    mid_neg_mask = (
        (df_work["neg_score_kw"] >= CFG.mid_neg_threshold) &
        (df_work["neg_score"] < CFG.high_neg_threshold) &
        (~mixed_mask) & (~high_neg_mask) & (~service_neg_mask) & (~rate_neg_mask)
    )

    high_pos_mask = (df_work["pos_score"] >= CFG.high_pos_threshold) & (df_work["neg_score"] <= CFG.high_pos_max_neg)

    df_high_neg = df_work.loc[high_neg_mask].copy(); df_high_neg["bucket"] = "high_neg"
    df_service_neg = df_work.loc[service_neg_mask].copy(); df_service_neg["bucket"] = "service_neg"
    df_rate_neg = df_work.loc[rate_neg_mask].copy(); df_rate_neg["bucket"] = "rate_neg"
    df_mid_neg = df_work.loc[mid_neg_mask].copy(); df_mid_neg["bucket"] = "mid_neg"
    df_mixed = df_work.loc[mixed_mask].copy(); df_mixed["bucket"] = "mixed"
    df_high_pos = df_work.loc[high_pos_mask].copy(); df_high_pos["bucket"] = "high_pos"

    used = set(df_high_neg["src_row_id"]) | set(df_service_neg["src_row_id"]) | set(df_rate_neg["src_row_id"]) | set(df_mid_neg["src_row_id"]) | set(df_mixed["src_row_id"]) | set(df_high_pos["src_row_id"])
    df_rest = df_work.loc[~df_work["src_row_id"].isin(list(used))].copy()
    df_rest["bucket"] = "random"

    doctor_col = "doctor_id"
    specialty_col = "specialty"

    s_high_neg = sample_with_caps(df_high_neg, CFG.n_high_neg, CFG.seed + 1, doctor_col, specialty_col, CFG.max_per_doctor, CFG.max_per_specialty)
    s_service_neg = sample_with_caps(df_service_neg, CFG.n_service_neg, CFG.seed + 2, doctor_col, specialty_col, CFG.max_per_doctor, CFG.max_per_specialty)
    s_rate_neg = sample_with_caps(df_rate_neg, CFG.n_rate_neg, CFG.seed + 3, doctor_col, specialty_col, CFG.max_per_doctor, CFG.max_per_specialty)
    s_mid_neg = sample_with_caps(df_mid_neg, CFG.n_mid_neg, CFG.seed + 4, doctor_col, specialty_col, CFG.max_per_doctor, CFG.max_per_specialty)
    s_mixed = sample_with_caps(df_mixed, CFG.n_mixed, CFG.seed + 5, doctor_col, specialty_col, CFG.max_per_doctor, CFG.max_per_specialty)
    s_high_pos = sample_with_caps(df_high_pos, CFG.n_high_pos, CFG.seed + 6, doctor_col, specialty_col, CFG.max_per_doctor, CFG.max_per_specialty)
    s_random = sample_with_caps(df_rest, CFG.n_random, CFG.seed + 7, doctor_col, specialty_col, max(2, CFG.max_per_doctor // 2), CFG.max_per_specialty)

    pool = pd.concat([s_high_neg, s_service_neg, s_rate_neg, s_mid_neg, s_mixed, s_random, s_high_pos], axis=0)
    pool = pool.drop_duplicates("src_row_id", keep="first").copy()

    if len(pool) < CFG.total_size:
        need = CFG.total_size - len(pool)
        df_left = df_work.loc[~df_work["src_row_id"].isin(pool["src_row_id"].tolist())].copy()

        # 1) اول: topup با اولویت منفی‌نما
        df_left_pref = df_left.loc[
            (df_left["neg_score_rate"] > 0.0) |
            (df_left["has_service_term"] == 1) |
            (df_left["neg_score_kw"] > 0.0)
        ].copy()
        df_left_pref["bucket"] = "topup_pref"

        s1 = sample_with_caps(
            df_left_pref, need, CFG.seed + 99,
            doctor_col, specialty_col,
            max_per_doctor=20,                 # شُل‌تر برای اینکه کم نیاریم
            max_per_specialty=CFG.max_per_specialty
        )
        pool = pd.concat([pool, s1], axis=0).drop_duplicates("src_row_id", keep="first")
        need = CFG.total_size - len(pool)

        # 2) اگر هنوز کم بود: topup آزادتر از کل باقی‌مانده
        if need > 0:
            df_left2 = df_left.loc[~df_left["src_row_id"].isin(pool["src_row_id"].tolist())].copy()
            df_left2["bucket"] = "topup_random"

            s2 = sample_with_caps(
                df_left2, need, CFG.seed + 100,
                doctor_col, specialty_col,
                max_per_doctor=50,              # باز هم شُل‌تر
                max_per_specialty=CFG.max_per_specialty
            )
            pool = pd.concat([pool, s2], axis=0).drop_duplicates("src_row_id", keep="first")

    if len(pool) < CFG.total_size:
        need2 = CFG.total_size - len(pool)
        df_left2 = df_work.loc[~df_work["src_row_id"].isin(pool["src_row_id"].tolist())].copy()
        if not df_left2.empty:
            s_extra = df_left2.sample(n=min(need2, len(df_left2)), random_state=CFG.seed + 123).copy()
            s_extra["bucket"] = "final_backfill"
            pool = pd.concat([pool, s_extra], axis=0).drop_duplicates("src_row_id", keep="first")

    pool = pool.sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)
    pool["item_id"] = [f"sent_{i:06d}" for i in range(len(pool))]

    # labeling fields
    pool["label_human"] = ""
    pool["label_human_note"] = ""
    pool["rater_id"] = ""

    # future LLM fields
    pool["label_llm"] = ""
    pool["llm_confidence"] = ""
    pool["llm_rationale_short"] = ""

    keep_cols = [
        "item_id",
        "doctor_id",
        "specialty",
        "rate",
        "label",
        "bucket",
        "neg_score", "neg_score_kw", "neg_score_rate", "neg_hits",
        "pos_score", "pos_score_kw", "pos_score_rate", "pos_hits",
        "has_service_term", "has_service_neg_cue",
        "n_words", "n_chars",
        "text_label",
        "text_for_tfidf",
        "label_human", "label_human_note", "rater_id",
        "label_llm", "llm_confidence", "llm_rationale_short",
        "src_row_id",
    ]
    keep_cols = [c for c in keep_cols if c in pool.columns]
    out_df = pool[keep_cols].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for _, r in out_df.iterrows():
            payload = {
                "schema_version": "sent_label_pool_v1",
                "item_id": safe_str(r.get("item_id")),
                "text": safe_str(r.get("text_label")),
                "meta": {
                    "doctor_id": safe_str(r.get("doctor_id")),
                    "specialty": safe_str(r.get("specialty")),
                    "rate": r.get("rate") if not pd.isna(r.get("rate")) else None,
                    "label_from_rate": r.get("label") if not pd.isna(r.get("label")) else None,
                    "bucket": safe_str(r.get("bucket")),
                    "neg_score": float(r.get("neg_score", 0.0)),
                    "pos_score": float(r.get("pos_score", 0.0)),
                    "neg_hits": safe_str(r.get("neg_hits")),
                    "pos_hits": safe_str(r.get("pos_hits")),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    lines = []
    lines.append("SENTIMENT LABELING POOL — BUILD REPORT (v3)")
    lines.append(f"rows_input_full={len(df):,}")
    lines.append(f"rows_placeholder_removed={int(placeholder_mask.sum()):,}")
    lines.append(f"rows_after_sanity_and_dedup={len(df_work):,}")
    lines.append(f"rows_in_pool={len(out_df):,}")
    lines.append("")
    lines.append("Candidate pool sizes (before sampling):")
    lines.append(f"  high_neg_candidates={len(df_high_neg):,}")
    lines.append(f"  service_neg_candidates={len(df_service_neg):,}")
    lines.append(f"  rate_neg_candidates={len(df_rate_neg):,}")
    lines.append(f"  mid_neg_candidates={len(df_mid_neg):,}")
    lines.append(f"  mixed_candidates={len(df_mixed):,}")
    lines.append(f"  high_pos_candidates={len(df_high_pos):,}")
    lines.append(f"  random_candidates={len(df_rest):,}")
    lines.append("")
    lines.append("Bucket counts (final pool):")
    vc = out_df["bucket"].value_counts()
    for k, v in vc.items():
        lines.append(f"  {k}: {int(v):,}")
    lines.append("")
    lines.append(f"unique_doctors_in_pool={out_df['doctor_id'].nunique():,}")
    lines.append("specialty_top15:")
    vc2 = out_df["specialty"].value_counts().head(15)
    for k, v in vc2.items():
        lines.append(f"  {k}: {int(v):,}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()