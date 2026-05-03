from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

IN_POOL = BASE_DIR / "processed_data" / "sentiment" / "labeling_pool.csv"
OUT_BATCH = BASE_DIR / "processed_data" / "sentiment" / "labeling_batch_01.csv"
SEED = 42

# Batch 01: منفی‌محور برای اینکه MLP سریع یاد بگیرد
QUOTAS = {
    "high_neg": 220,
    "service_neg": 260,
    "rate_neg": 520,
    "mid_neg": 160,
    "mixed": 180,
    "topup_pref": 260,   # مهم
    "high_pos": 120,
    "random": 80,
}

DUPLICATES = 80  # برای کنترل کیفیت/سازگاری لیبلر

def main():
    df = pd.read_csv(IN_POOL, encoding="utf-8")
    rng = np.random.RandomState(SEED)

    parts = []
    for b, n in QUOTAS.items():
        d = df[df["bucket"] == b]
        if len(d) == 0:
            continue
        take = min(n, len(d))
        parts.append(d.sample(n=take, random_state=rng))

    batch = pd.concat(parts, ignore_index=True)

    target = int(sum(QUOTAS.values()))
    if len(batch) < target:
        need = target - len(batch)
        left = df[~df["item_id"].isin(batch["item_id"])]
        batch = pd.concat([batch, left.sample(n=min(need, len(left)), random_state=rng)], ignore_index=True)

    # duplicate QA rows
    dup_src = batch.sample(n=min(DUPLICATES, len(batch)), random_state=rng).copy()
    dup_src["is_duplicate"] = 1
    batch["is_duplicate"] = 0

    out = pd.concat([batch, dup_src], ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    out["task_id"] = [f"task01_{i:05d}" for i in range(len(out))]

    keep = [
        "task_id", "item_id", "text_label", "bucket", "specialty", "rate",
        "is_duplicate", "label_human", "label_human_note", "rater_id"
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    OUT_BATCH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_BATCH, index=False, encoding="utf-8-sig")
    print(f"saved={OUT_BATCH.resolve()} rows={len(out):,}")

if __name__ == "__main__":
    main()