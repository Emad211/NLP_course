import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "preprocess_steps" / "step03_hazm_normalized.csv"
OUT_DIR = BASE_DIR / "processed_data" / "preprocess_steps"

OUT_CSV = OUT_DIR / "step04_deglued.csv"
OUT_REPORT = OUT_DIR / "step04_report.txt"
OUT_SAMPLES = OUT_DIR / "step04_samples.csv"

MULTI_SPACE = re.compile(r"\s+")


def section(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


def fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return f"{n:,} (N/A)"
    return f"{n:,} ({(n / total) * 100:.1f}%)"


def token_count(series: pd.Series, token: str) -> int:
    pat = re.compile(rf"(?<!\S){re.escape(token)}(?!\S)")
    return int(series.str.contains(pat, na=False).sum())


def apply_rules(text: str, rules):
    t = text
    for pat, repl in rules:
        t = pat.sub(repl, t)
    t = MULTI_SPACE.sub(" ", t).strip()
    return t


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()

    df = pd.read_csv(IN_CSV)
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_step03"] = df["text_step03"].astype(str)

    s3 = df["text_step03"]

    rules = [
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

    glued_tokens = [
        "درحال", "باسلام", "خداروشکر", "فوقالعاده",
        "اززحماتشون", "ازبچگی", "مداومبه", "هردکتری",
        "اماایشان", "وسپس", "برطرفشد",
    ]

    target_phrases = [
        "در حال", "با سلام", "خدا رو شکر", "فوق العاده",
        "از زحماتشون", "از بچگی", "مداوم به", "هر دکتری",
        "اما ایشان", "و سپس", "برطرف شد",
    ]

    before_counts = {t: token_count(s3, t) for t in glued_tokens}
    before_targets = {t: int(s3.str.contains(t, regex=False, na=False).sum()) for t in target_phrases}

    out = [None] * len(df)
    for i, txt in enumerate(s3.tolist()):
        out[i] = apply_rules(txt, rules)

    df["text_step04"] = out
    s4 = df["text_step04"]

    after_counts = {t: token_count(s4, t) for t in glued_tokens}
    after_targets = {t: int(s4.str.contains(t, regex=False, na=False).sum()) for t in target_phrases}

    df["changed_03_to_04"] = df["text_step03"] != df["text_step04"]
    df["empty_after_04"] = df["text_step04"].str.strip().eq("")

    df["s03_char_len"] = df["text_step03"].str.len()
    df["s04_char_len"] = df["text_step04"].str.len()
    df["s03_word_len"] = df["text_step03"].str.split().str.len()
    df["s04_word_len"] = df["text_step04"].str.split().str.len()
    df["word_delta_03_to_04"] = (df["s04_word_len"] - df["s03_word_len"]).fillna(0).astype(int)

    out_cols = [
        "doctor_id", "label", "rate", "date",
        "text_step02", "text_step03", "text_step04",
        "changed_03_to_04", "empty_after_04",
        "s03_word_len", "s04_word_len", "word_delta_03_to_04",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    total = len(df)

    lines = []
    lines.append("PREPROCESS STEP 04 — DOMAIN DEGLUE REPORT")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input={IN_CSV.resolve()}")
    lines.append(f"output={OUT_CSV.resolve()}")

    lines.append(section("0) Snapshot"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"rows_changed_03_to_04={fmt_pct(int(df['changed_03_to_04'].sum()), total)}")
    lines.append(f"rows_empty_after_04={fmt_pct(int(df['empty_after_04'].sum()), total)}")

    lines.append(section("1) Glued token counts (before -> after)"))
    for t in glued_tokens:
        lines.append(f"{t}\t{before_counts[t]:,}\t->\t{after_counts[t]:,}")

    lines.append(section("2) Target phrase coverage (before -> after)"))
    for t in target_phrases:
        lines.append(f"{t}\t{before_targets[t]:,}\t->\t{after_targets[t]:,}")

    lines.append(section("3) Most changed samples (largest word increase)"))
    top = df[df["changed_03_to_04"]].sort_values("word_delta_03_to_04", ascending=False).head(25)
    if len(top) == 0:
        lines.append("none")
    else:
        for r in top.itertuples(index=False):
            s3p = str(r.text_step03).replace("\n", " ")[:220]
            s4p = str(r.text_step04).replace("\n", " ")[:220]
            lines.append(f"doctor_id={r.doctor_id}\tlabel={r.label}\trate={r.rate}\twd={int(r.word_delta_03_to_04)}")
            lines.append(f"step03='{s3p}'")
            lines.append(f"step04='{s4p}'")
            lines.append("")

    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(total), size=min(80, total), replace=False)
    sample_df = df.loc[idx, ["doctor_id", "label", "rate", "text_step03", "text_step04", "word_delta_03_to_04"]].copy()
    sample_df.to_csv(OUT_SAMPLES, index=False, encoding="utf-8-sig")

    dt = time.time() - t0
    lines.append(section("4) Runtime"))
    lines.append(f"seconds={dt:.2f}")

    report = "\n".join(lines)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(report)

    print(f"\nsaved_report={OUT_REPORT.resolve()}")
    print(f"saved_csv={OUT_CSV.resolve()}")
    print(f"saved_samples={OUT_SAMPLES.resolve()}")


if __name__ == "__main__":
    main()