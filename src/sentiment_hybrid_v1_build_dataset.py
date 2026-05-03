from __future__ import annotations

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# -----------------------
# Paths (DO NOT overwrite old outputs)
# -----------------------
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


# -----------------------
# Robust label parsing
# -----------------------
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


# -----------------------
# Text utils
# -----------------------
SPACE_RE = re.compile(r"\s+")
ZWNJ_RE = re.compile(r"\u200c")

def norm_text(t: str) -> str:
    t = str(t or "")
    t = ZWNJ_RE.sub(" ", t)
    t = t.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    t = SPACE_RE.sub(" ", t).strip()
    return t

def n_words(t: str) -> int:
    t = norm_text(t)
    return 0 if not t else len(t.split())


# -----------------------
# Rule features (keyword patterns)
# -----------------------
NEG_PATTERNS = [
    ("strong_negative", re.compile(r"(افتضاح|مزخرف|فاجعه|بدترین|خیلی\s*بد|اصلا\s*خوب\s*نیست|اصلاً\s*خوب\s*نیست)"), 2.6),
    ("not_recommend", re.compile(r"(پیشنهاد\s*نمی\s*کنم|توصیه\s*نمی\s*کنم|دیگه\s*نمیرم|دیگر\s*نمیرم|پشیمون|پشیمان)"), 2.6),
    ("rude_behavior", re.compile(r"(بد\s*برخورد|بی\s*ادب|بی\s*احترام|توهین|تحقیر|داد\s*زد|بی\s*حوصله)"), 2.2),
    ("long_wait", re.compile(r"(معطلی|منتظر|تاخیر|تأخیر|ساعت\s*ها|صف|شلوغ)"), 1.8),
    ("no_response", re.compile(r"(پاسخگو\s*نبود|جواب\s*نمی\s*ده|تلفن.*جواب|منشی.*جواب)"), 1.9),
    ("money", re.compile(r"(هزینه|گران|گرون|پولکی|پول\s*اضافی|ویزیت.*بالا|هزینه\s*تراشی)"), 1.4),
    ("misdiagnosis", re.compile(r"(تشخیص.*اشتباه|اشتباه\s*تشخیص|غلط\s*تشخیص|دارو.*اشتباه)"), 2.2),
    ("no_improve", re.compile(r"(بهتر\s*نشد|بدتر\s*شدم|اثر\s*نداشت|فایده\s*نداشت|نتیجه\s*نگرفتم|درمان\s*نشد|هیچ\s*تغییری)"), 2.0),
    ("careless", re.compile(r"(سرسری|بی\s*دقت|بدون\s*معاینه|توجه\s*نکرد|گوش\s*نکرد|بی\s*تفاوت)"), 1.6),
    ("disappointed_soft", re.compile(r"(متاسفانه|متأسفانه|ناراضی|راضی\s*نبودم|کاش)"), 1.6),
]

POS_PATTERNS = [
    ("strong_positive", re.compile(r"(عالی|فوق\s*العاده|بی\s*نظیر|محشر|بهترین)"), 2.5),
    ("thanks", re.compile(r"(ممنون|سپاس|تشکر|مرسی|خدا\s*خیر|دستتون\s*درد\s*نکنه|دستش\s*درد\s*نکنه)"), 1.3),
    ("skilled", re.compile(r"(حاذق|ماهر|کاربلد|متخصص|تشخیص\s*دقیق|تجربه\s*بالا)"), 1.6),
    ("kind", re.compile(r"(مهربان|خوش\s*برخورد|با\s*حوصله|صبور|آرام)"), 1.0),
    ("improved", re.compile(r"(بهتر\s*شدم|خوب\s*شدم|درمان\s*شدم|نتیجه\s*گرفتم)"), 1.8),
]

SERVICE_TERMS = re.compile(r"(نوبت|نوبت\s*دهی|منشی|مطب|تلفن|تماس|ویزیت|هزینه|معطلی|انتظار|صف|شلوغ)")
MILD_NEG_CUE = re.compile(r"(بد|ضعیف|اشتباه|ناراضی|راضی\s*نبود|متاسفانه|متأسفانه|پشیمون|پشیمان|بی\s*نظم|بی\s*مسئولیت|بی\s*توجه|سرسری)")

INFO_TERMS = re.compile(r"(ام\s*ار\s*ای|mri|آزمایش|سونو|سونوگرافی|نمونه\s*برداری|عمل|جراحی|نوار|بیوپسی|جواب\s*آزمایش)")


def score_patterns(t: str, patterns):
    t = norm_text(t)
    score = 0.0
    hits = []
    for name, pat, w in patterns:
        if pat.search(t):
            score += w
            hits.append(name)
    return score, hits


# -----------------------
# Parametrized weak labeler (we will tune params on gold)
# -----------------------
def weak_label_with_params(text: str, params: dict) -> tuple[int, float, str]:
    """
    returns (label, confidence, reason)
    """
    t = norm_text(text)
    w = n_words(t)

    neg_kw, neg_hits = score_patterns(t, NEG_PATTERNS)
    pos_kw, pos_hits = score_patterns(t, POS_PATTERNS)

    has_service = bool(SERVICE_TERMS.search(t))
    has_service_neg = bool(has_service and MILD_NEG_CUE.search(t))

    # informational short -> neutral
    if (w <= params["info_max_words"] and neg_kw == 0 and pos_kw == 0) or (INFO_TERMS.search(t) and neg_kw == 0 and pos_kw == 0):
        return 0, 0.75, "informational"

    # mixed cues -> neutral (unless strong one side)
    if neg_kw > 0 and pos_kw > 0 and (abs(neg_kw - pos_kw) < params["mixed_margin"]):
        return 0, 0.55, "mixed"

    # negative
    if (neg_kw >= params["neg_th"] and pos_kw <= params["neg_pos_max"]):
        conf = min(0.95, 0.60 + 0.12 * (neg_kw - pos_kw))
        return -1, float(max(0.55, conf)), "neg_kw"

    if has_service_neg and pos_kw <= params["service_pos_max"]:
        return -1, 0.70, "neg_service"

    # positive
    if (pos_kw >= params["pos_th"] and neg_kw <= params["pos_neg_max"]):
        conf = min(0.95, 0.60 + 0.12 * (pos_kw - neg_kw))
        return 1, float(max(0.55, conf)), "pos_kw"

    # default neutral
    return 0, 0.45, "default_neu"


def tune_params_on_gold(df_gold: pd.DataFrame, text_col: str, y_col: str) -> dict:
    """
    Grid search parameters maximizing mean macro-F1 with 5-fold CV on gold subset.
    """
    X = df_gold[text_col].fillna("").astype(str).tolist()
    y = df_gold[y_col].astype(int).to_numpy()

    grid = []
    for neg_th in [1.2, 1.6, 2.0, 2.2, 2.6]:
        for pos_th in [1.3, 1.6, 2.0, 2.5, 3.0]:
            for info_max_words in [3, 4, 5]:
                for mixed_margin in [0.6, 1.0, 1.4]:
                    for neg_pos_max in [0.8, 1.0, 1.2]:
                        for pos_neg_max in [0.8, 1.0, 1.2]:
                            grid.append({
                                "neg_th": neg_th,
                                "pos_th": pos_th,
                                "info_max_words": info_max_words,
                                "mixed_margin": mixed_margin,
                                "neg_pos_max": neg_pos_max,
                                "pos_neg_max": pos_neg_max,
                                "service_pos_max": 1.0,
                            })

    skf = StratifiedKFold(n_splits=min(5, max(2, int(len(y) / 50))), shuffle=True, random_state=SEED)

    best = None
    best_score = -1.0

    for params in grid:
        scores = []
        for tr_idx, te_idx in skf.split(np.zeros(len(y)), y):
            y_true = y[te_idx]
            y_pred = []
            for i in te_idx:
                lab, _, _ = weak_label_with_params(X[i], params)
                y_pred.append(lab)
            scores.append(f1_score(y_true, y_pred, average="macro", labels=[-1, 0, 1], zero_division=0))
        m = float(np.mean(scores)) if scores else 0.0
        if m > best_score:
            best_score = m
            best = params

    best["cv_macro_f1"] = best_score
    return best


def stratified_split_gold(df: pd.DataFrame, label_col: str, seed: int = 42):
    """
    dev/test from GOLD only (so evaluation is meaningful).
    """
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    parts_tr, parts_dev, parts_te = [], [], []
    for lab, g in df.groupby(label_col):
        g = g.sample(frac=1.0, random_state=rng)
        n = len(g)
        if n < 12:
            parts_tr.append(g)  # too small -> all train
            continue
        n_te = max(2, int(round(n * 0.15)))
        n_dev = max(2, int(round(n * 0.15)))
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not POOL_CSV.exists():
        raise FileNotFoundError(POOL_CSV)
    if not BATCH1_CSV.exists():
        raise FileNotFoundError(BATCH1_CSV)

    pool = pd.read_csv(POOL_CSV, encoding="utf-8")
    batch = pd.read_csv(BATCH1_CSV, encoding="utf-8")

    if "item_id" not in pool.columns:
        raise RuntimeError("labeling_pool.csv must contain item_id")
    if "text_label" not in pool.columns:
        raise RuntimeError("labeling_pool.csv must contain text_label")

    if "item_id" not in batch.columns:
        raise RuntimeError("labeling_batch_01.csv must contain item_id")
    if "label_human" not in batch.columns:
        raise RuntimeError("labeling_batch_01.csv must contain label_human")

    pool["item_id"] = pool["item_id"].astype(str)
    pool["text_label"] = pool["text_label"].fillna("").astype(str).apply(norm_text)

    batch["item_id"] = batch["item_id"].astype(str)
    batch["label_human_int"] = batch["label_human"].apply(parse_label)

    # only keep labeled rows; collapse duplicates by item_id
    gold = batch[batch["label_human_int"].notna()].copy()
    gold = gold.sort_values(["item_id"]).drop_duplicates("item_id", keep="first")
    gold = gold[["item_id", "label_human_int"]].copy()

    # merge gold labels into pool
    df = pool.merge(gold, on="item_id", how="left")

    gold_mask = df["label_human_int"].notna()
    df_gold = df[gold_mask].copy()
    df_unlab = df[~gold_mask].copy()

    if len(df_gold) < 50:
        raise RuntimeError(f"Not enough gold labels found in batch_01. gold_rows={len(df_gold)}")

    # tune weak labeling params on gold
    best_params = tune_params_on_gold(df_gold, text_col="text_label", y_col="label_human_int")

    # apply weak labels to unlabeled
    weak_labels = []
    weak_conf = []
    weak_reason = []
    for t in df_unlab["text_label"].tolist():
        lab, conf, reason = weak_label_with_params(t, best_params)
        weak_labels.append(lab)
        weak_conf.append(conf)
        weak_reason.append(reason)

    df_unlab["label_weak"] = weak_labels
    df_unlab["weak_confidence"] = weak_conf
    df_unlab["weak_reason"] = weak_reason

    # for gold rows, weak label too (for analysis)
    gold_weak = []
    gold_conf = []
    gold_reason = []
    for t in df_gold["text_label"].tolist():
        lab, conf, reason = weak_label_with_params(t, best_params)
        gold_weak.append(lab)
        gold_conf.append(conf)
        gold_reason.append(reason)
    df_gold["label_weak"] = gold_weak
    df_gold["weak_confidence"] = gold_conf
    df_gold["weak_reason"] = gold_reason

    # choose final label for training:
    # - GOLD rows: use label_human_int
    # - unlabeled rows: use label_weak but only if confidence >= threshold
    conf_th = 0.65
    df_unlab["label_final"] = np.where(df_unlab["weak_confidence"] >= conf_th, df_unlab["label_weak"], np.nan)
    df_gold["label_final"] = df_gold["label_human_int"]

    df_all = pd.concat([df_gold, df_unlab], ignore_index=True)

    # Build splits:
    # dev/test ONLY from gold (honest evaluation)
    gold_train, gold_dev, gold_test = stratified_split_gold(df_gold, "label_human_int", seed=SEED)

    # Train set = gold_train + high-confidence weak labels from unlabeled
    df_unlab_train = df_unlab[df_unlab["label_final"].notna()].copy()

    train = pd.concat([gold_train, df_unlab_train], ignore_index=True)

    # For trainer compatibility: name label column as label_human_int
    train["label_human_int"] = train["label_final"].astype(int)
    gold_dev["label_human_int"] = gold_dev["label_human_int"].astype(int)
    gold_test["label_human_int"] = gold_test["label_human_int"].astype(int)

    # Save
    df_all.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")
    train.to_csv(OUT_TRAIN, index=False, encoding="utf-8-sig")
    gold_dev.to_csv(OUT_DEV, index=False, encoding="utf-8-sig")
    gold_test.to_csv(OUT_TEST, index=False, encoding="utf-8-sig")

    # Report
    def dist(series):
        vc = series.value_counts(dropna=False).sort_index()
        return {str(k): int(v) for k, v in vc.items()}

    rep = []
    rep.append("SENTIMENT HYBRID BUILD REPORT (v1)")
    rep.append(f"pool_rows={len(pool):,}")
    rep.append(f"gold_rows_found_in_batch1={len(df_gold):,}")
    rep.append(f"unlabeled_rows={len(df_unlab):,}")
    rep.append("")
    rep.append("Best weak-label params (tuned on gold):")
    rep.append(json.dumps(best_params, ensure_ascii=False, indent=2))
    rep.append("")
    rep.append(f"weak_conf_threshold={conf_th}")
    rep.append(f"unlabeled_kept_as_train_weak={df_unlab_train.shape[0]:,}")
    rep.append("")
    rep.append("Gold label distribution:")
    rep.append(str(dist(df_gold['label_human_int'])))
    rep.append("Weak label distribution on unlabeled (all):")
    rep.append(str(dist(df_unlab['label_weak'])))
    rep.append("Weak label distribution on unlabeled (kept>=th):")
    rep.append(str(dist(df_unlab_train['label_final'])))
    rep.append("")
    rep.append(f"train_rows={len(train):,} (gold_train + weak_highconf)")
    rep.append(f"dev_rows={len(gold_dev):,} (gold only)")
    rep.append(f"test_rows={len(gold_test):,} (gold only)")
    OUT_REPORT.write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()