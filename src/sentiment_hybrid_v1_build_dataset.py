from __future__ import annotations

from pathlib import Path
import os
import re
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix


# ============================================================
# Paths (v1 fixed)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

POOL_CSV = BASE_DIR / "processed_data" / "sentiment" / "labeling_pool.csv"
BATCH1_CSV = BASE_DIR / "processed_data" / "sentiment" / "labeling_batch_01.csv"

OUT_DIR = BASE_DIR / "processed_data" / "sentiment_hybrid_v1"
OUT_ALL = OUT_DIR / "dataset_all.csv"
OUT_TRAIN = OUT_DIR / "train.csv"
OUT_DEV = OUT_DIR / "dev.csv"
OUT_TEST = OUT_DIR / "test.csv"
OUT_REPORT = OUT_DIR / "build_report.txt"

SEED = 42


# ============================================================
# Controls (env override)
# ============================================================
CONF_TH_NEG = float(os.environ.get("HYB_CONF_TH_NEG", "0.78"))
CONF_TH_POS = float(os.environ.get("HYB_CONF_TH_POS", "0.70"))

# Neutral: we now allow a few "mixed" neutrals too (to fix positive drift)
CONF_TH_NEU_INFO = float(os.environ.get("HYB_CONF_TH_NEU_INFO", "0.90"))
CONF_TH_NEU_MIXED = float(os.environ.get("HYB_CONF_TH_NEU_MIXED", "0.82"))

GOLD_REPEAT = int(os.environ.get("HYB_GOLD_REPEAT", "3"))
MAX_WEAK_PER_CLASS_MULT = float(os.environ.get("HYB_MAX_WEAK_PER_CLASS_MULT", "6.0"))


# ============================================================
# Robust label parsing
# ============================================================
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_ARABIC_DIGITS  = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(s: str) -> str:
    return s.translate(_PERSIAN_DIGITS).translate(_ARABIC_DIGITS)

def parse_label(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        xi = int(x)
        return xi if xi in (-1, 0, 1) else None
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        xi = int(round(float(x)))
        return xi if xi in (-1, 0, 1) and abs(float(x) - xi) < 1e-6 else None

    s = normalize_digits(str(x)).strip().lower()
    if s == "":
        return None
    try:
        fv = float(s)
        iv = int(round(fv))
        if iv in (-1, 0, 1) and abs(fv - iv) < 1e-6:
            return iv
    except Exception:
        pass

    if s in {"-1", "neg", "negative", "منفی"}: return -1
    if s in {"0", "neu", "neutral", "خنثی", "mixed"}: return 0
    if s in {"1", "pos", "positive", "مثبت"}: return 1
    return None


# ============================================================
# Text normalization helpers
# ============================================================
SPACE_RE = re.compile(r"\s+")
ZWNJ_RE = re.compile(r"\u200c")
ELONG_CHAR_RE = re.compile(r"(\S)\1{2,}")

def norm_text(t: str) -> str:
    t = str(t or "")
    t = ZWNJ_RE.sub(" ", t)
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = SPACE_RE.sub(" ", t).strip()
    # aaa -> aa (handles خووووب)
    t = ELONG_CHAR_RE.sub(r"\1\1", t)
    return t

def word_count(t: str) -> int:
    t = norm_text(t)
    return 0 if not t else len(t.split())


# ============================================================
# Pattern sets
# ============================================================
NEG_PATTERNS = [
    ("strong_negative", re.compile(r"(افتضاح|مزخرف|فاجعه|بدترین|اصلا\s*خوب\s*نیست|اصلاً\s*خوب\s*نیست)"), 3.0),
    ("not_recommend", re.compile(r"(پیشنهاد\s*نمی\s*کنم|توصیه\s*نمی\s*کنم|دیگه\s*نمیرم|دیگر\s*نمیرم|پشیمون|پشیمان)"), 2.6),
    ("rude_behavior", re.compile(r"(بد\s*برخورد|بی\s*ادب|بی\s*احترام|توهین|تحقیر|داد\s*زد|بی\s*حوصله)"), 2.4),
    ("misdiagnosis", re.compile(r"(تشخیص.*اشتباه|اشتباه\s*تشخیص|غلط\s*تشخیص|دارو.*اشتباه)"), 2.4),
    ("no_improve_hard", re.compile(r"(اثر\s*نداشت|فایده\s*نداشت|بدتر\s*شدم|درمان\s*نشد)"), 2.0),
    ("no_response", re.compile(r"(پاسخگو\s*نبود|جواب\s*نمی\s*ده|تلفن.*جواب|منشی.*جواب)"), 2.0),

    # IMPORTANT FIX: remove plain "منتظر" (was causing false -1 for lab results waiting)
    ("long_wait", re.compile(r"(معطلی|معطل|تاخیر|تأخیر|ساعت\s*ها|چهار\s*ساعت|پنج\s*ساعت|صف|شلوغ)"), 1.6),

    ("money", re.compile(
        r"(هزینه\s*تراش\w*|پولکی|ویزیت.*بالا|"
        r"خیلی\s*گرون|خیلی\s*گران|بسیار\s*گرون|بسیار\s*گران|فوق\s*العاده\s*گرون|فوق\s*العاده\s*گران|"
        r"گرون\s*بود|گران\s*بود|گرون\s*حساب|گران\s*حساب|"
        r"گرانقیمت|گران\s*قیمت|قیمت.*بالا|هزینه.*بالا)"
    ), 1.8),

    ("not_satisfied", re.compile(r"(راضی\s*نبودم|اصلا\s*راضی\s*نیستم|اصلاً\s*راضی\s*نیستم)"), 2.6),

    # NEW: pharmacy/supplement complaint (fixes sent_003406 type)
    ("pharmacy_profit", re.compile(r"(داروخانه|مکمل|کلاژن|قرص.*گران|به\s*نفع\s*داروخانه|مکمل.*زیاد)"), 1.9),

    ("disappointed_soft", re.compile(r"(متاسفانه|متأسفانه|ناراضی)"), 1.2),
]

POS_PATTERNS = [
    ("strong_positive", re.compile(r"(عالی|فوق\s*العاده|بی\s*نظیر|محشر|بهترین)"), 2.5),
    ("improved", re.compile(r"(بهتر\s*شدم|خوب\s*شدم|درمان\s*شدم|نتیجه\s*گرفتم)"), 1.8),
    ("skilled", re.compile(r"(حاذق|ماهر|کاربلد|تشخیص\s*دقیق|تجربه\s*بالا)"), 1.4),
    ("kind", re.compile(r"(مهربان|خوش\s*برخورد|خوش‌برخورد|با\s*حوصله|صبور|آرام|خوش\s*اخلاق|خوش‌اخلاق|محترم|مودب)"), 1.2),
    ("thanks", re.compile(r"(ممنون|سپاس|تشکر|مرسی|خدا\s*خیر|دستتون\s*درد\s*نکنه|دستش\s*درد\s*نکنه)"), 1.0),
    ("good_doctor", re.compile(r"(دکتر|پزشک)\s*خ[و]{1,3}ب"), 1.6),
    ("good_word", re.compile(r"\bخ[و]{1,3}ب\b"), 0.9),
]

SERVICE_TERMS = re.compile(r"(نوبت|نوبت\s*دهی|منشی|مطب|تلفن|تماس|ویزیت|هزینه|معطلی|انتظار|صف|شلوغ|لغو|نقدی)")
SERVICE_NEG_STRONG = re.compile(r"(افتضاح|بی\s*نظم|بی\s*نظمی|بد\s*برخورد|پاسخگو\s*نبود|جواب\s*نمی\s*ده|ساعت\s*ها|خیلی\s*معطل|خیلی\s*شلوغ|گرون|گران|پولکی|نقدی|بین\s*مریض\s*حساب)")

INFO_TERMS = re.compile(r"(ام\s*ار\s*ای|mri|آزمایش|سونو|سونوگرافی|نمونه\s*برداری|عمل|جراحی|نوار|بیوپسی|جواب\s*آزمایش)")
EARLY_STAGE = re.compile(r"(منتظر\s*جواب\s*آزمایش|منتظر\s*جواب\s*ازمایش|منتظر\s*نتیجه|فعلا\s*منتظر|فعلاً\s*منتظر|در\s*حال\s*پیگیری|تازه\s*شروع\s*کردم|هنوز\s*زوده\s*نظر\s*بدم)")

SELF_HARM = re.compile(r"(خودکشی|خود\s*کشی)")
STATE_IMPROVED = re.compile(r"(الان|حالا).*(بهترم|بهتر\s*شدم|خیلی\s*بهتر)")
DOCTOR_COMPLAINT_TERMS = re.compile(r"(منشی|نوبت|مطب|ویزیت|هزینه|پولکی|معطلی|بد\s*برخورد|تشخیص.*اشتباه|دارو.*اشتباه|داروخانه|مکمل)")


def score_patterns(t: str, patterns):
    t = norm_text(t)
    score = 0.0
    hits = []
    for name, pat, w in patterns:
        if pat.search(t):
            score += w
            hits.append(name)
    return score, hits


def weak_label(text: str, params: dict) -> tuple[int, float, str]:
    t = norm_text(text)
    w = word_count(t)

    neg_kw, neg_hits = score_patterns(t, NEG_PATTERNS)
    pos_kw, pos_hits = score_patterns(t, POS_PATTERNS)

    has_service = bool(SERVICE_TERMS.search(t))
    has_service_strong_neg = bool(has_service and SERVICE_NEG_STRONG.search(t))

    # informational -> neutral
    if (w <= params["info_max_words"] and neg_kw == 0 and pos_kw == 0) or (INFO_TERMS.search(t) and neg_kw == 0 and pos_kw == 0):
        return 0, 0.94, "informational"

    # early stage waiting (no complaint) -> neutral (important)
    if EARLY_STAGE.search(t) and not DOCTOR_COMPLAINT_TERMS.search(t) and (neg_kw < params["neg_th"]) and (pos_kw < params["pos_th"]):
        return 0, 0.90, "early_stage"

    # self-harm narrative with improvement -> positive/neutral, not negative
    if SELF_HARM.search(t) and STATE_IMPROVED.search(t) and not DOCTOR_COMPLAINT_TERMS.search(t):
        return 1, 0.72, "state_improved_despite_selfharm"

    # strong positive override
    if pos_kw >= params["pos_strong_th"] and neg_kw <= params["pos_strong_neg_max"]:
        conf = min(0.95, 0.70 + 0.08 * (pos_kw - neg_kw))
        return 1, float(conf), "pos_strong"

    # mixed_service: service strong neg + praise => neutral
    if has_service_strong_neg and pos_kw > 0:
        return 0, 0.85, "mixed_service"

    # mixed_money: money + praise => neutral
    if ("money" in neg_hits or "pharmacy_profit" in neg_hits) and pos_kw > 0:
        return 0, 0.84, "mixed_money"

    # mixed cues close -> neutral
    margin = abs(pos_kw - neg_kw)
    if neg_kw > 0 and pos_kw > 0 and margin < params["mixed_margin"]:
        return 0, 0.72, "mixed"

    # negative by keywords (require margin)
    neg_margin = neg_kw - pos_kw
    if (neg_kw >= params["neg_th"]) and (neg_margin >= params["neg_margin_th"]):
        conf = min(0.95, 0.62 + 0.10 * neg_margin)
        return -1, float(max(0.55, conf)), "neg_kw"

    # service negative
    if has_service_strong_neg and pos_kw <= params["service_pos_max"]:
        return -1, 0.80, "neg_service"

    # positive by keywords (require margin)
    pos_margin = pos_kw - neg_kw
    if (pos_kw >= params["pos_th"]) and (pos_margin >= params["pos_margin_th"]):
        conf = min(0.95, 0.62 + 0.10 * pos_margin)
        return 1, float(max(0.55, conf)), "pos_kw"

    return 0, 0.45, "default_neu"


def tune_params_on_gold(df_gold: pd.DataFrame, text_col: str, y_col: str) -> dict:
    X = df_gold[text_col].fillna("").astype(str).tolist()
    y = df_gold[y_col].astype(int).to_numpy()

    grid = []
    for neg_th in [1.6, 2.0, 2.4, 2.8]:
        for pos_th in [1.6, 2.0, 2.4, 2.8]:
            for info_max_words in [3, 4, 5]:
                for mixed_margin in [0.8, 1.2, 1.6]:
                    for neg_margin_th in [0.6, 1.0, 1.4]:
                        for pos_margin_th in [0.6, 1.0, 1.4]:
                            grid.append({
                                "neg_th": float(neg_th),
                                "pos_th": float(pos_th),
                                "info_max_words": int(info_max_words),
                                "mixed_margin": float(mixed_margin),
                                "neg_margin_th": float(neg_margin_th),
                                "pos_margin_th": float(pos_margin_th),
                                "service_pos_max": 1.2,
                                "pos_strong_th": 2.8,
                                "pos_strong_neg_max": 1.2,
                            })

    n_splits = min(5, max(2, int(len(y) / 60)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    best = None
    best_score = -1.0

    for params in grid:
        scores = []
        for _, te_idx in skf.split(np.zeros(len(y)), y):
            y_true = y[te_idx]
            y_pred = []
            for i in te_idx:
                lab, _, _ = weak_label(X[i], params)
                y_pred.append(lab)
            scores.append(f1_score(y_true, y_pred, average="macro", labels=[-1, 0, 1], zero_division=0))
        m = float(np.mean(scores)) if scores else 0.0
        if m > best_score:
            best_score = m
            best = dict(params)

    best["cv_macro_f1"] = float(best_score)
    best["cv_folds"] = int(n_splits)
    return best


def stratified_split_gold(df: pd.DataFrame, label_col: str, seed: int = 42):
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    parts_tr, parts_dev, parts_te = [], [], []
    for lab, g in df.groupby(label_col):
        g = g.sample(frac=1.0, random_state=rng)
        n = len(g)
        if n < 18:
            parts_tr.append(g)
            continue

        n_te = max(3, int(round(n * 0.15)))
        n_dev = max(3, int(round(n * 0.15)))
        n_tr = n - n_dev - n_te
        if n_tr < 1:
            parts_tr.append(g)
            continue

        parts_tr.append(g.iloc[:n_tr])
        parts_dev.append(g.iloc[n_tr:n_tr + n_dev])
        parts_te.append(g.iloc[n_tr + n_dev:])

    tr = pd.concat(parts_tr).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    dev = pd.concat(parts_dev).sample(frac=1.0, random_state=rng).reset_index(drop=True) if parts_dev else df.iloc[0:0].copy()
    te = pd.concat(parts_te).sample(frac=1.0, random_state=rng).reset_index(drop=True) if parts_te else df.iloc[0:0].copy()
    return tr, dev, te


def cap_weak_by_gold(weak_df: pd.DataFrame, gold_train: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    out_parts = []
    gold_counts = gold_train["label_human_int"].value_counts().to_dict()

    for lab in [-1, 0, 1]:
        max_n = int(MAX_WEAK_PER_CLASS_MULT * int(gold_counts.get(lab, 0) or 0))
        g = weak_df[weak_df["label_final"] == lab].copy()
        if max_n <= 0:
            continue
        if len(g) > max_n:
            g = g.sample(n=max_n, random_state=rng)
        out_parts.append(g)

    if not out_parts:
        return weak_df.head(0)
    return pd.concat(out_parts, ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = pd.read_csv(POOL_CSV, encoding="utf-8")
    batch = pd.read_csv(BATCH1_CSV, encoding="utf-8")

    pool["item_id"] = pool["item_id"].astype(str)
    pool["text_label"] = pool["text_label"].fillna("").astype(str).apply(norm_text)

    batch["item_id"] = batch["item_id"].astype(str)
    batch["label_human_int"] = batch["label_human"].apply(parse_label)

    gold = batch[batch["label_human_int"].notna()].copy()
    gold = gold.sort_values(["item_id"]).drop_duplicates("item_id", keep="first")
    gold = gold[["item_id", "label_human_int"]].copy()

    df = pool.merge(gold, on="item_id", how="left")
    df["is_gold"] = df["label_human_int"].notna().astype(int)

    df_gold = df[df["label_human_int"].notna()].copy()
    df_unlab = df[df["label_human_int"].isna()].copy()

    if len(df_gold) < 80:
        raise RuntimeError(f"Not enough gold labels found. gold_rows={len(df_gold)}")

    best_params = tune_params_on_gold(df_gold, "text_label", "label_human_int")

    # weak on unlabeled
    wl, wc, wr = [], [], []
    for t in df_unlab["text_label"].tolist():
        lab, conf, reason = weak_label(t, best_params)
        wl.append(lab); wc.append(conf); wr.append(reason)

    df_unlab["label_weak"] = wl
    df_unlab["weak_confidence"] = wc
    df_unlab["weak_reason"] = wr

    # weak on gold (diagnostic)
    g_wl, g_wc, g_wr = [], [], []
    for t in df_gold["text_label"].tolist():
        lab, conf, reason = weak_label(t, best_params)
        g_wl.append(lab); g_wc.append(conf); g_wr.append(reason)

    df_gold["label_weak"] = g_wl
    df_gold["weak_confidence"] = g_wc
    df_gold["weak_reason"] = g_wr

    gold_train, gold_dev, gold_test = stratified_split_gold(df_gold, "label_human_int", seed=SEED)

    # select weak for train
    df_unlab["label_final"] = np.nan

    # neg
    neg_mask = (
        (df_unlab["label_weak"] == -1) &
        (df_unlab["weak_confidence"] >= CONF_TH_NEG) &
        (df_unlab["weak_reason"].isin(["neg_kw", "neg_service"]))
    )
    df_unlab.loc[neg_mask, "label_final"] = -1

    # pos
    pos_mask = (
        (df_unlab["label_weak"] == 1) &
        (df_unlab["weak_confidence"] >= CONF_TH_POS) &
        (df_unlab["weak_reason"].isin(["pos_kw", "pos_strong"]))
    )
    df_unlab.loc[pos_mask, "label_final"] = 1

    # neutral informational + mixed neutrals (controlled)
    neu_info = (
        (df_unlab["label_weak"] == 0) &
        (df_unlab["weak_confidence"] >= CONF_TH_NEU_INFO) &
        (df_unlab["weak_reason"].isin(["informational", "early_stage"]))
    )
    df_unlab.loc[neu_info, "label_final"] = 0

    neu_mixed = (
        (df_unlab["label_weak"] == 0) &
        (df_unlab["weak_confidence"] >= CONF_TH_NEU_MIXED) &
        (df_unlab["weak_reason"].isin(["mixed_money", "mixed_service", "mixed"]))
    )
    df_unlab.loc[neu_mixed, "label_final"] = 0

    weak_train = df_unlab[df_unlab["label_final"].notna()].copy()
    weak_train["label_final"] = weak_train["label_final"].astype(int)

    weak_train_capped = cap_weak_by_gold(weak_train, gold_train)

    gold_train_rep = pd.concat([gold_train.copy() for _ in range(max(1, GOLD_REPEAT))], ignore_index=True)

    train = pd.concat([gold_train_rep, weak_train_capped], ignore_index=True)
    train["label_human_int"] = train["label_human_int"].fillna(train["label_final"]).astype(int)

    gold_dev["label_human_int"] = gold_dev["label_human_int"].astype(int)
    gold_test["label_human_int"] = gold_test["label_human_int"].astype(int)

    df_all = pd.concat([df_gold, df_unlab], ignore_index=True)
    df_all.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    train.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    gold_dev.to_csv(OUT_DEV, index=False, encoding="utf-8-sig")
    gold_test.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

    def dist(s):
        vc = s.value_counts(dropna=False).sort_index()
        return {str(k): int(v) for k, v in vc.items()}

    rep = []
    rep.append("SENTIMENT HYBRID BUILD REPORT (v1 - rewritten v2)")
    rep.append(f"pool_rows={len(pool):,}")
    rep.append(f"gold_rows={len(df_gold):,}")
    rep.append(f"unlabeled_rows={len(df_unlab):,}")
    rep.append("")
    rep.append("Tuned weak params (CV on gold):")
    rep.append(json.dumps(best_params, ensure_ascii=False, indent=2))
    rep.append("")
    rep.append("Thresholds:")
    rep.append(f"  CONF_TH_NEG={CONF_TH_NEG}")
    rep.append(f"  CONF_TH_POS={CONF_TH_POS}")
    rep.append(f"  CONF_TH_NEU_INFO={CONF_TH_NEU_INFO}")
    rep.append(f"  CONF_TH_NEU_MIXED={CONF_TH_NEU_MIXED}")
    rep.append(f"  GOLD_REPEAT={GOLD_REPEAT}")
    rep.append(f"  MAX_WEAK_PER_CLASS_MULT={MAX_WEAK_PER_CLASS_MULT}")
    rep.append("")
    rep.append("Gold label distribution:")
    rep.append(str(dist(df_gold["label_human_int"])))
    rep.append("Weak label distribution on unlabeled (all):")
    rep.append(str(dist(df_unlab["label_weak"])))
    rep.append("Weak kept for train (before cap):")
    rep.append(str(dist(weak_train["label_final"]) if len(weak_train) else {}))
    rep.append("Weak kept for train (after cap):")
    rep.append(str(dist(weak_train_capped["label_final"]) if len(weak_train_capped) else {}))
    rep.append("")
    rep.append(f"train_rows={len(train):,}")
    rep.append(f"dev_rows={len(gold_dev):,} (gold only)")
    rep.append(f"test_rows={len(gold_test):,} (gold only)")
    rep.append("")

    y_true = df_gold["label_human_int"].astype(int).to_numpy()
    y_weak = df_gold["label_weak"].astype(int).to_numpy()
    rep.append("Weak vs Gold (on gold subset):")
    rep.append(f"  macro_f1_weak_on_gold={f1_score(y_true, y_weak, average='macro', labels=[-1,0,1], zero_division=0):.4f}")
    rep.append("  confusion_matrix labels=[-1,0,1]:")
    rep.append(str(confusion_matrix(y_true, y_weak, labels=[-1,0,1])))
    rep.append("  classification_report:")
    rep.append(classification_report(y_true, y_weak, digits=4, zero_division=0))

    OUT_REPORT.write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()