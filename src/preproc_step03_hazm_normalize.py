import time
from pathlib import Path

import numpy as np
import pandas as pd

from hazm import Normalizer


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "preprocess_steps" / "step02_pre_hazm.csv"
OUT_DIR = BASE_DIR / "processed_data" / "preprocess_steps"

OUT_CSV = OUT_DIR / "step03_hazm_normalized.csv"
OUT_REPORT = OUT_DIR / "step03_report.txt"
OUT_SAMPLES = OUT_DIR / "step03_samples.csv"

PRINT_EVERY = 4000

GLUED_TERMS = [
    "درحال",
    "درمانم",
    "ازبچگی",
    "مداومبه",
    "باسلام",
    "هردکتری",
    "اززحماتشون",
    "اماایشان",
    "وسپس",
    "شدودوبارم",
    "برطرفشد",
    "خداروشکر",
    "فوقالعاده",
]


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


def desc(s: pd.Series) -> str:
    s = s.astype(float)
    p = np.percentile(s, [10, 50, 90])
    return f"min={s.min():.1f}  mean={s.mean():.1f}  p10={p[0]:.1f}  median={p[1]:.1f}  p90={p[2]:.1f}  max={s.max():.1f}"


def count_terms(series: pd.Series, terms: list[str]) -> dict:
    out = {}
    for t in terms:
        out[t] = int(series.str.contains(t, regex=False, na=False).sum())
    return out


def build_normalizer():
    return Normalizer(
        correct_spacing=True,
        remove_diacritics=True,
        remove_specials_chars=False,
        decrease_repeated_chars=False,
        persian_style=True,
        persian_numbers=False,
    )


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()
    df = pd.read_csv(IN_CSV)

    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_step02"] = df["text_step02"].astype(str)

    total = len(df)
    print(f"rows={total:,}", flush=True)

    norm = build_normalizer()

    before_glued = count_terms(df["text_step02"], GLUED_TERMS)

    texts = df["text_step02"].tolist()
    out = [""] * total
    errors = []

    for i, txt in enumerate(texts):
        try:
            out[i] = norm.normalize(txt)
        except Exception as e:
            out[i] = ""
            errors.append((i, f"{type(e).__name__}: {e}"))
        if (i + 1) % PRINT_EVERY == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt if dt > 0 else 0
            eta = (total - (i + 1)) / rate if rate > 0 else 0
            print(f"progress={i+1:,}/{total:,}  rows_per_sec={rate:.1f}  eta_secs={eta:.1f}  errors={len(errors):,}", flush=True)

    df["text_step03"] = out

    after_glued = count_terms(df["text_step03"], GLUED_TERMS)

    df["s02_char_len"] = df["text_step02"].str.len()
    df["s03_char_len"] = df["text_step03"].str.len()
    df["s02_word_len"] = df["text_step02"].str.split().str.len()
    df["s03_word_len"] = df["text_step03"].str.split().str.len()

    df["changed_02_to_03"] = df["text_step02"] != df["text_step03"]
    df["empty_after_03"] = df["text_step03"].str.strip().eq("")

    df["char_delta_02_to_03"] = (df["s03_char_len"] - df["s02_char_len"]).fillna(0).astype(int)
    df["word_delta_02_to_03"] = (df["s03_word_len"] - df["s02_word_len"]).fillna(0).astype(int)

    placeholder = (df["label"] == -1) & (df["text_step03"] == "عدم رضایت")

    out_cols = [
        "doctor_id", "label", "rate", "date",
        "text_raw", "text_step01", "text_step02", "text_step03",
        "s02_char_len", "s03_char_len",
        "s02_word_len", "s03_word_len",
        "char_delta_02_to_03", "word_delta_02_to_03",
        "changed_02_to_03", "empty_after_03",
        "is_garbage",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("PREPROCESS STEP 03 — HAZM NORMALIZER REPORT")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input={IN_CSV.resolve()}")
    lines.append(f"output={OUT_CSV.resolve()}")

    lines.append(section("0) Snapshot"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_changed_02_to_03={fmt_pct(int(df['changed_02_to_03'].sum()), total)}")
    lines.append(f"rows_empty_after_03={fmt_pct(int(df['empty_after_03'].sum()), total)}")
    lines.append(f"rows_placeholder_negative={fmt_pct(int(placeholder.sum()), total)}")

    lines.append(section("1) Length stats  step02 -> step03"))
    lines.append(f"char_len step02: {desc(df['s02_char_len'])}")
    lines.append(f"char_len step03: {desc(df['s03_char_len'])}")
    lines.append(f"word_len step02: {desc(df['s02_word_len'])}")
    lines.append(f"word_len step03: {desc(df['s03_word_len'])}")

    lines.append(section("2) Glued-term counts  (before -> after)"))
    for t in GLUED_TERMS:
        lines.append(f"{t}\t{before_glued.get(t,0):,}\t->\t{after_glued.get(t,0):,}")

    lines.append(section("3) Biggest word count increases (top 20)"))
    top = df.sort_values("word_delta_02_to_03", ascending=False).head(20)
    for r in top.itertuples(index=False):
        s2 = str(r.text_step02).replace("\n", " ")[:220]
        s3 = str(r.text_step03).replace("\n", " ")[:220]
        lines.append(f"doctor_id={r.doctor_id}\tlabel={r.label}\trate={r.rate}\twd={int(r.word_delta_02_to_03)}\tcd={int(r.char_delta_02_to_03)}")
        lines.append(f"step02='{s2}'")
        lines.append(f"step03='{s3}'")
        lines.append("")

    lines.append(section("4) Errors"))
    lines.append(f"errors_total={len(errors):,}")
    for i, e in errors[:30]:
        lines.append(f"row={i}\t{e}")
    if len(errors) > 30:
        lines.append(f"... {len(errors)-30} more")

    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(total), size=min(80, total), replace=False)
    sample_df = df.loc[idx, ["doctor_id", "label", "rate", "text_step02", "text_step03", "word_delta_02_to_03"]].copy()
    sample_df.to_csv(OUT_SAMPLES, index=False, encoding="utf-8-sig")

    lines.append(section("5) Runtime"))
    dt = time.time() - t0
    lines.append(f"seconds={dt:.2f}")

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)

    print(f"\nsaved_report={OUT_REPORT.resolve()}")
    print(f"saved_csv={OUT_CSV.resolve()}")
    print(f"saved_samples={OUT_SAMPLES.resolve()}")


if __name__ == "__main__":
    main()