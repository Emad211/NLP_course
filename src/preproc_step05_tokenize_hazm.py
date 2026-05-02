import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from hazm import WordTokenizer, stopwords_list


BASE_DIR = Path(__file__).resolve().parent.parent
IN_CSV = BASE_DIR / "processed_data" / "preprocess_steps" / "step04_deglued.csv"
OUT_DIR = BASE_DIR / "processed_data" / "preprocess_steps"

OUT_CSV = OUT_DIR / "step05_tokenized.csv"
OUT_REPORT = OUT_DIR / "step05_report.txt"
OUT_SAMPLES = OUT_DIR / "step05_samples.csv"

MIN_TOKEN_LEN = 2
PRINT_EVERY_UNIQ = 500


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


def build_stopwords():
    sw = set(stopwords_list())
    keep = {"نه", "نخیر", "نیست", "نمی", "نمي", "نبود", "نشد", "بدون"}
    return sw, keep


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(str(IN_CSV.resolve()))

    t0 = time.time()
    print(f"input={IN_CSV.resolve()}", flush=True)

    df = pd.read_csv(IN_CSV)
    df["doctor_id"] = df["doctor_id"].astype(str)
    df["text_step04"] = df["text_step04"].astype(str)

    total = len(df)
    uniq_counts = df["text_step04"].value_counts()
    uniq_total = int(uniq_counts.shape[0])

    print(f"rows_total={total:,}", flush=True)
    print(f"unique_texts={uniq_total:,}", flush=True)

    sw, keep = build_stopwords()
    print(f"stopwords_size={len(sw):,} keep_size={len(keep):,}", flush=True)

    tokenizer = WordTokenizer(separate_emoji=False, replace_links=False, replace_ids=False, replace_emails=False, replace_numbers=False)

    tok_all_map = {}
    tok_ns_map = {}
    tok_all_cnt = {}
    tok_ns_cnt = {}

    c_all = Counter()
    c_ns = Counter()

    print("tokenization_started...", flush=True)

    done = 0
    for txt, freq in uniq_counts.items():
        toks = tokenizer.tokenize(txt) if txt else []
        toks = [t for t in toks if len(t) >= MIN_TOKEN_LEN]

        toks_ns = [t for t in toks if (t not in sw) or (t in keep)]

        tok_all_map[txt] = " ".join(toks)
        tok_ns_map[txt] = " ".join(toks_ns)
        tok_all_cnt[txt] = len(toks)
        tok_ns_cnt[txt] = len(toks_ns)

        if toks:
            for t in toks:
                c_all[t] += int(freq)
        if toks_ns:
            for t in toks_ns:
                c_ns[t] += int(freq)

        done += 1
        if done % PRINT_EVERY_UNIQ == 0:
            dt = time.time() - t0
            rate = done / dt if dt > 0 else 0.0
            eta = (uniq_total - done) / rate if rate > 0 else 0.0
            print(f"progress_uniq={done:,}/{uniq_total:,} uniq_per_sec={rate:.1f} eta_secs={eta:.1f}", flush=True)

    print("tokenization_done.", flush=True)
    print("mapping_to_rows...", flush=True)

    df["tok_all"] = df["text_step04"].map(tok_all_map)
    df["tok_nostop"] = df["text_step04"].map(tok_ns_map)
    df["tok_all_count"] = df["text_step04"].map(tok_all_cnt).astype(int)
    df["tok_nostop_count"] = df["text_step04"].map(tok_ns_cnt).astype(int)

    df["empty_tok_all"] = df["tok_all"].astype(str).str.strip().eq("")
    df["empty_tok_nostop"] = df["tok_nostop"].astype(str).str.strip().eq("")

    df["stopword_removed_tokens"] = (df["tok_all_count"] - df["tok_nostop_count"]).clip(lower=0)
    df["stopword_removed_ratio"] = np.where(
        df["tok_all_count"] > 0,
        df["stopword_removed_tokens"] / df["tok_all_count"],
        np.nan,
    )

    out_cols = [
        "doctor_id", "label", "rate", "date",
        "text_step04",
        "tok_all", "tok_nostop",
        "tok_all_count", "tok_nostop_count",
        "empty_tok_all", "empty_tok_nostop",
        "stopword_removed_tokens", "stopword_removed_ratio",
    ]

    print("writing_csv...", flush=True)
    df[out_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("PREPROCESS STEP 05 — HAZM TOKENIZATION REPORT (v3 WordTokenizer)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"input={IN_CSV.resolve()}")
    lines.append(f"output={OUT_CSV.resolve()}")

    lines.append(section("0) Snapshot"))
    lines.append(f"rows_total={total:,}")
    lines.append(f"unique_text_step04={uniq_total:,}")
    lines.append(f"min_token_len={MIN_TOKEN_LEN}")
    lines.append(f"stopwords_size={len(sw):,}")
    lines.append(f"stopwords_keep_size={len(keep):,}")

    lines.append(section("1) Empty after tokenization"))
    lines.append(f"empty_tok_all={fmt_pct(int(df['empty_tok_all'].sum()), total)}")
    lines.append(f"empty_tok_nostop={fmt_pct(int(df['empty_tok_nostop'].sum()), total)}")

    lines.append(section("2) Token count stats"))
    lines.append(f"tok_all_count:    {desc(df['tok_all_count'])}")
    lines.append(f"tok_nostop_count: {desc(df['tok_nostop_count'])}")

    lines.append(section("3) Stopword removal impact"))
    ratio = df["stopword_removed_ratio"].dropna()
    lines.append(f"mean_removed_tokens={df['stopword_removed_tokens'].mean():.3f}")
    lines.append(f"mean_removed_ratio={ratio.mean():.3f}")
    lines.append(f"p90_removed_ratio={np.percentile(ratio, 90):.3f}")

    lines.append(section("4) Top tokens (all, weighted)"))
    for tok, cnt in c_all.most_common(60):
        lines.append(f"{tok}\t{cnt:,}")

    lines.append(section("5) Top tokens (no-stop, weighted)"))
    for tok, cnt in c_ns.most_common(60):
        lines.append(f"{tok}\t{cnt:,}")

    rng = np.random.default_rng(42)
    idx = rng.choice(np.arange(total), size=min(160, total), replace=False)
    sample_df = df.loc[idx, ["doctor_id", "label", "rate", "text_step04", "tok_all", "tok_nostop", "tok_all_count", "tok_nostop_count"]].copy()
    sample_df.to_csv(OUT_SAMPLES, index=False, encoding="utf-8-sig")

    dt = time.time() - t0
    lines.append(section("6) Runtime"))
    lines.append(f"seconds={dt:.2f}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nsaved_report={OUT_REPORT.resolve()}", flush=True)
    print(f"saved_csv={OUT_CSV.resolve()}", flush=True)
    print(f"saved_samples={OUT_SAMPLES.resolve()}", flush=True)


if __name__ == "__main__":
    main()