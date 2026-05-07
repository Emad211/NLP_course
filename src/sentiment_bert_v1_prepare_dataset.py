"""
sentiment_bert_v1_prepare_dataset.py
ورودی : processed_data/sentiment_hybrid_v1/{train,dev,test}.csv
خروجی : processed_data/sentiment_bert_v1/{train,dev,test}.csv
         (ستون‌های: text_bert, label_id)
"""

import os
import pandas as pd
from pathlib import Path

# ─── مسیرها ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
INPUT_DIR   = ROOT / "processed_data" / "sentiment_hybrid_v1"
OUTPUT_DIR  = ROOT / "processed_data" / "sentiment_bert_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COL    = "text_label"   # همان ستونی که audit انتخاب کرد
LABEL_COL   = "label"

# ─── پردازش ──────────────────────────────────────────────────────────────────
for split in ("train", "dev", "test"):
    src = INPUT_DIR / f"{split}.csv"
    df  = pd.read_csv(src)

    # تمیزکاری ساده
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df = df[df[TEXT_COL] != ""].reset_index(drop=True)

    out = pd.DataFrame({
        "text_bert" : df[TEXT_COL],
        "label_id"  : df[LABEL_COL].astype(int),
    })

    dst = OUTPUT_DIR / f"{split}.csv"
    out.to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"[{split}] rows={len(out)}  →  {dst}")

print("Done.")