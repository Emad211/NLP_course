# src/retriever_stopwords_finalize.py
import argparse
import time
from pathlib import Path

from retriever_medical_stopwords import (
    MEDICAL_ALLOWLIST_LATIN,
    MEDICAL_STOPWORDS_SEED_V02,
    norm_tok,
)


BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = BASE_DIR / "processed_data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Finalize retriever medical stopwords: seed + (safe) auto.")
    p.add_argument(
        "--auto-suggested",
        type=str,
        default=str(FINAL_DIR / "retriever_stopwords_auto_suggested_medical.txt"),
        help="Path to auto-suggested stopwords (one token per line).",
    )
    p.add_argument(
        "--out-final",
        type=str,
        default=str(FINAL_DIR / "retriever_medical_stopwords_final_v03.txt"),
        help="Final merged stopword list output.",
    )
    p.add_argument(
        "--out-report",
        type=str,
        default=str(FINAL_DIR / "retriever_medical_stopwords_final_v03_report.txt"),
        help="Human-readable report output.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    auto_path = Path(args.auto_suggested)
    out_final = Path(args.out_final)
    out_report = Path(args.out_report)

    if not auto_path.exists():
        raise FileNotFoundError(f"auto_suggested file not found: {auto_path.resolve()}")

    # High-risk tokens that are COMMON but still useful for medical matching / bigrams.
    # We DO NOT auto-add these as stopwords at this stage.
    HIGH_RISK_KEEP = {
        "درد",
        "دارو",
        "درمان",
        "تشخیص",
        "عمل",
        "جراحی",
        # also keep "کار" for now because it supports important bigrams like "کار لیزر"
        "کار",
    }

    # Read auto suggested tokens
    raw_lines = auto_path.read_text(encoding="utf-8").splitlines()
    auto_raw = [norm_tok(x) for x in raw_lines if norm_tok(x)]

    # Filter auto tokens (safe subset)
    auto_kept = []
    auto_excluded = []  # (token, reason)
    for t in auto_raw:
        if len(t) < 2:
            auto_excluded.append((t, "too_short"))
            continue
        if t in MEDICAL_ALLOWLIST_LATIN:
            auto_excluded.append((t, "latin_allowlist"))
            continue
        if t in HIGH_RISK_KEEP:
            auto_excluded.append((t, "high_risk_keep"))
            continue
        auto_kept.append(t)

    auto_kept_set = set(auto_kept)

    # Merge: seed + safe auto
    final_set = set(MEDICAL_STOPWORDS_SEED_V02) | auto_kept_set

    # Write final list
    out_final.write_text("\n".join(sorted(final_set)), encoding="utf-8")

    # Build report
    lines = []
    lines.append("RETRIEVER MEDICAL STOPWORDS — FINALIZE REPORT (v03)")
    lines.append(f"project_root={BASE_DIR.resolve()}")
    lines.append(f"auto_suggested_in={auto_path.resolve()}")
    lines.append(f"out_final={out_final.resolve()}")
    lines.append(f"out_report={out_report.resolve()}")
    lines.append("")
    lines.append(f"seed_count={len(MEDICAL_STOPWORDS_SEED_V02):,}")
    lines.append(f"auto_raw_count={len(auto_raw):,}")
    lines.append(f"auto_kept_count={len(auto_kept_set):,}")
    lines.append(f"final_count={len(final_set):,}")
    lines.append("")

    lines.append("AUTO KEPT (safe additions):")
    for t in sorted(auto_kept_set):
        lines.append(f"  + {t}")
    lines.append("")

    lines.append("AUTO EXCLUDED (with reasons):")
    for t, reason in auto_excluded:
        if not t:
            continue
        lines.append(f"  - {t}\t{reason}")

    dt = time.time() - t0
    lines.append("")
    lines.append(f"seconds={dt:.2f}")

    out_report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()