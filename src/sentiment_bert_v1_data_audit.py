# src/sentiment_bert_v1_data_audit.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Transformers is required for true WordPiece token length stats
from transformers import AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent

IN_DIR = BASE_DIR / "processed_data" / "sentiment_hybrid_v1"
TRAIN_CSV = IN_DIR / "train.csv"
DEV_CSV = IN_DIR / "dev.csv"
TEST_CSV = IN_DIR / "test.csv"

OUT_DIR = BASE_DIR / "processed_data" / "sentiment_bert_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_AUDIT_TXT = OUT_DIR / "data_audit.txt"
OUT_AUDIT_JSON = OUT_DIR / "data_audit.json"
OUT_SAMPLES_CSV = OUT_DIR / "length_samples.csv"
OUT_LABEL_DIST_CSV = OUT_DIR / "label_distribution.csv"
OUT_LENGTH_STATS_CSV = OUT_DIR / "length_stats.csv"

# You can change this later, but for audit it's good to start with ParsBERT
DEFAULT_MODEL = "HooshvareLab/bert-base-parsbert-uncased"


def section(title: str, width: int = 80) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


@dataclass
class TextColumnChoice:
    chosen: str
    candidates_checked: List[str]


def pick_text_column(df: pd.DataFrame) -> TextColumnChoice:
    """
    Prefer columns that keep natural language (best for BERT).
    Priority order:
      1) text_step04
      2) text_label
      3) text
      4) comment_text
      5) text_raw
      6) document
      7) fallback: any column that starts with "text"
    """
    candidates = [
        "text_step04",
        "text_label",
        "text",
        "comment_text",
        "text_raw",
        "document",
    ]
    checked = []
    for c in candidates:
        checked.append(c)
        if c in df.columns:
            return TextColumnChoice(chosen=c, candidates_checked=checked)

    # fallback
    text_like = [c for c in df.columns if c.lower().startswith("text")]
    checked.extend(text_like)
    if text_like:
        return TextColumnChoice(chosen=text_like[0], candidates_checked=checked)

    raise ValueError(
        "No suitable text column found. "
        "Expected one of: text_step04, text_label, text, comment_text, text_raw, document, or any column starting with 'text'."
    )


def safe_str_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    # keep_default_na=False already, but do it anyway
    return s


def compute_basic_lengths(texts: pd.Series) -> pd.DataFrame:
    """
    Compute char length and whitespace word length.
    """
    t = safe_str_series(texts)
    char_len = t.str.len()
    word_len = t.str.split().str.len()
    return pd.DataFrame({"char_len": char_len, "word_len": word_len})


def compute_wordpiece_lengths(tokenizer, texts: List[str], batch_size: int = 512) -> np.ndarray:
    """
    Compute number of wordpiece tokens including special tokens ([CLS],[SEP]) for each text.
    Uses tokenizer(..., add_special_tokens=True, truncation=False) to measure true length.
    """
    out = np.zeros(len(texts), dtype=np.int32)
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lens = [len(ids) for ids in enc["input_ids"]]
        out[i : i + len(lens)] = np.array(lens, dtype=np.int32)
    return out


def quantiles(x: np.ndarray, qs=(0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}


def label_distribution(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    tmp = df.copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    vc = tmp[label_col].value_counts(dropna=False).sort_index()
    out = vc.rename("count").reset_index().rename(columns={"index": "label"})
    out["ratio"] = out["count"] / out["count"].sum()
    return out


def audit_split(
    name: str,
    df: pd.DataFrame,
    text_col: str,
    tokenizer,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      stats_df: per-row lengths and some flags
      summary: dict with aggregate metrics
    """
    texts = safe_str_series(df[text_col])
    basic = compute_basic_lengths(texts)

    # empty flags
    empty = texts.str.strip().eq("")
    basic["is_empty"] = empty

    # wordpiece lengths
    wp = compute_wordpiece_lengths(tokenizer, texts.tolist(), batch_size=512)
    basic["wp_len"] = wp

    # summary
    total = len(df)
    empty_n = int(empty.sum())

    wp_nonempty = wp[~empty.values] if total > 0 else np.array([], dtype=np.int32)

    summ = {
        "split": name,
        "rows": int(total),
        "empty_text_rows": int(empty_n),
        "empty_text_pct": float(empty_n / total) if total else 0.0,
        "char_len_min": float(basic["char_len"].min()) if total else float("nan"),
        "char_len_mean": float(basic["char_len"].mean()) if total else float("nan"),
        "char_len_median": float(basic["char_len"].median()) if total else float("nan"),
        "char_len_max": float(basic["char_len"].max()) if total else float("nan"),
        "word_len_min": float(basic["word_len"].min()) if total else float("nan"),
        "word_len_mean": float(basic["word_len"].mean()) if total else float("nan"),
        "word_len_median": float(basic["word_len"].median()) if total else float("nan"),
        "word_len_max": float(basic["word_len"].max()) if total else float("nan"),
        "wp_len_min": float(wp_nonempty.min()) if wp_nonempty.size else float("nan"),
        "wp_len_mean": float(wp_nonempty.mean()) if wp_nonempty.size else float("nan"),
        "wp_len_max": float(wp_nonempty.max()) if wp_nonempty.size else float("nan"),
        "wp_len_quantiles": quantiles(wp_nonempty),
        # how many exceed common max_length values (including special tokens)
        "wp_gt_64": int(np.sum(wp_nonempty > 64)),
        "wp_gt_128": int(np.sum(wp_nonempty > 128)),
        "wp_gt_256": int(np.sum(wp_nonempty > 256)),
        "wp_gt_384": int(np.sum(wp_nonempty > 384)),
        "wp_gt_512": int(np.sum(wp_nonempty > 512)),
    }

    return basic, summ


def main():
    t0 = time.time()

    # Existence checks
    for p in [TRAIN_CSV, DEV_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(str(p.resolve()))

    train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    dev = pd.read_csv(DEV_CSV, keep_default_na=False)
    test = pd.read_csv(TEST_CSV, keep_default_na=False)

    # Pick text column once, but verify it exists in all splits
    choice = pick_text_column(train)
    text_col = choice.chosen
    for name, df in [("dev", dev), ("test", test)]:
        if text_col not in df.columns:
            raise ValueError(f"Chosen text column '{text_col}' not found in split: {name}")

    # Load tokenizer (WordPiece length matters for max_length decision)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    # Label distributions (per split)
    dist_train = label_distribution(train, label_col="label")
    dist_dev = label_distribution(dev, label_col="label")
    dist_test = label_distribution(test, label_col="label")

    dist_train["split"] = "train"
    dist_dev["split"] = "dev"
    dist_test["split"] = "test"
    dist_all = pd.concat([dist_train, dist_dev, dist_test], ignore_index=True)
    dist_all.to_csv(OUT_LABEL_DIST_CSV, index=False, encoding="utf-8-sig")

    # Split audits
    stats_train, summ_train = audit_split("train", train, text_col, tokenizer)
    stats_dev, summ_dev = audit_split("dev", dev, text_col, tokenizer)
    stats_test, summ_test = audit_split("test", test, text_col, tokenizer)

    # Collect length stats table
    length_rows = []
    for s in [summ_train, summ_dev, summ_test]:
        length_rows.append(
            {
                "split": s["split"],
                "rows": s["rows"],
                "empty_text_rows": s["empty_text_rows"],
                "wp_len_mean": s["wp_len_mean"],
                "wp_len_p50": s["wp_len_quantiles"]["p50"],
                "wp_len_p90": s["wp_len_quantiles"]["p90"],
                "wp_len_p95": s["wp_len_quantiles"]["p95"],
                "wp_len_p99": s["wp_len_quantiles"]["p99"],
                "wp_gt_64": s["wp_gt_64"],
                "wp_gt_128": s["wp_gt_128"],
                "wp_gt_256": s["wp_gt_256"],
                "wp_gt_384": s["wp_gt_384"],
                "wp_gt_512": s["wp_gt_512"],
            }
        )
    length_df = pd.DataFrame(length_rows)
    length_df.to_csv(OUT_LENGTH_STATS_CSV, index=False, encoding="utf-8-sig")

    # Samples for manual inspection (short + long + random)
    def make_samples(df: pd.DataFrame, stats_df: pd.DataFrame, split: str) -> pd.DataFrame:
        t = safe_str_series(df[text_col])
        tmp = df.copy()
        tmp["_text"] = t
        tmp["_char_len"] = stats_df["char_len"].values
        tmp["_word_len"] = stats_df["word_len"].values
        tmp["_wp_len"] = stats_df["wp_len"].values
        tmp["_split"] = split

        # Short but non-empty
        short = tmp[tmp["_text"].str.strip().ne("")].sort_values("_wp_len", ascending=True).head(40)
        # Long
        long = tmp.sort_values("_wp_len", ascending=False).head(40)
        # Random
        rng = np.random.default_rng(42)
        if len(tmp) > 80:
            ridx = rng.choice(np.arange(len(tmp)), size=80, replace=False)
            rnd = tmp.iloc[ridx].copy()
        else:
            rnd = tmp.copy()

        out = pd.concat([short, long, rnd], ignore_index=True)
        # keep key cols if exist
        keep_cols = []
        for c in ["doctor_id", "comment_id", "source", "label"]:
            if c in out.columns:
                keep_cols.append(c)
        keep_cols += ["_split", "_char_len", "_word_len", "_wp_len", "_text"]
        out = out[keep_cols].drop_duplicates()
        return out

    samp_train = make_samples(train, stats_train, "train")
    samp_dev = make_samples(dev, stats_dev, "dev")
    samp_test = make_samples(test, stats_test, "test")
    samples = pd.concat([samp_train, samp_dev, samp_test], ignore_index=True)
    samples.to_csv(OUT_SAMPLES_CSV, index=False, encoding="utf-8-sig")

    # Build text report (self-contained)
    lines: List[str] = []
    lines.append("BERT DATA AUDIT REPORT (v1)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"model_for_token_length={DEFAULT_MODEL}")
    lines.append(f"input_train={TRAIN_CSV.resolve()}")
    lines.append(f"input_dev={DEV_CSV.resolve()}")
    lines.append(f"input_test={TEST_CSV.resolve()}")
    lines.append("")
    lines.append(f"text_column_chosen={text_col}")
    lines.append(f"text_column_candidates_checked={choice.candidates_checked}")
    lines.append("")
    lines.append(f"output_audit_txt={OUT_AUDIT_TXT.resolve()}")
    lines.append(f"output_audit_json={OUT_AUDIT_JSON.resolve()}")
    lines.append(f"output_label_distribution_csv={OUT_LABEL_DIST_CSV.resolve()}")
    lines.append(f"output_length_stats_csv={OUT_LENGTH_STATS_CSV.resolve()}")
    lines.append(f"output_samples_csv={OUT_SAMPLES_CSV.resolve()}")

    def add_split(s: Dict):
        lines.append(section(f"SPLIT={s['split']}"))
        lines.append(f"rows={s['rows']:,}")
        lines.append(f"empty_text_rows={fmt_pct(s['empty_text_rows'], s['rows'])}")
        lines.append("")
        lines.append("BASIC LENGTHS (chars/words)")
        lines.append(f"char_len: min={s['char_len_min']:.0f} mean={s['char_len_mean']:.1f} median={s['char_len_median']:.0f} max={s['char_len_max']:.0f}")
        lines.append(f"word_len: min={s['word_len_min']:.0f} mean={s['word_len_mean']:.1f} median={s['word_len_median']:.0f} max={s['word_len_max']:.0f}")
        lines.append("")
        lines.append("WORDPIECE LENGTHS (incl. special tokens)")
        q = s["wp_len_quantiles"]
        lines.append(f"wp_len: min={s['wp_len_min']:.0f} mean={s['wp_len_mean']:.1f} max={s['wp_len_max']:.0f}")
        lines.append(f"wp_len quantiles: p50={q['p50']:.0f}  p90={q['p90']:.0f}  p95={q['p95']:.0f}  p99={q['p99']:.0f}")
        lines.append("")
        lines.append("TRUNCATION RISK COUNTS")
        lines.append(f"wp>64 : {fmt_pct(s['wp_gt_64'], s['rows'])}")
        lines.append(f"wp>128: {fmt_pct(s['wp_gt_128'], s['rows'])}")
        lines.append(f"wp>256: {fmt_pct(s['wp_gt_256'], s['rows'])}")
        lines.append(f"wp>384: {fmt_pct(s['wp_gt_384'], s['rows'])}")
        lines.append(f"wp>512: {fmt_pct(s['wp_gt_512'], s['rows'])}")

    add_split(summ_train)
    add_split(summ_dev)
    add_split(summ_test)

    lines.append(section("LABEL DISTRIBUTION (train/dev/test)"))
    lines.append(dist_all.to_string(index=False))

    dt = time.time() - t0
    lines.append(section("RUNTIME"))
    lines.append(f"seconds={dt:.2f}")

    OUT_AUDIT_TXT.write_text("\n".join(lines), encoding="utf-8")

    # JSON for programmatic use
    payload = {
        "model_for_token_length": DEFAULT_MODEL,
        "text_column_chosen": text_col,
        "text_column_candidates_checked": choice.candidates_checked,
        "splits": {"train": summ_train, "dev": summ_dev, "test": summ_test},
        "label_distribution": json.loads(dist_all.to_json(orient="records")),
        "outputs": {
            "audit_txt": str(OUT_AUDIT_TXT.resolve()),
            "audit_json": str(OUT_AUDIT_JSON.resolve()),
            "label_distribution_csv": str(OUT_LABEL_DIST_CSV.resolve()),
            "length_stats_csv": str(OUT_LENGTH_STATS_CSV.resolve()),
            "samples_csv": str(OUT_SAMPLES_CSV.resolve()),
        },
    }
    OUT_AUDIT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nsaved_audit_txt={OUT_AUDIT_TXT.resolve()}")
    print(f"saved_audit_json={OUT_AUDIT_JSON.resolve()}")


if __name__ == "__main__":
    main()