from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "processed_data" / "sentiment_hybrid_v1"

DEV_CSV = DATA_DIR / "dev.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_PATH = BASE_DIR / "models" / "sentiment_mlp_hybrid_v1.joblib"

OUT_DEV_ERRORS = DATA_DIR / "errors_dev.csv"
OUT_TEST_ERRORS = DATA_DIR / "errors_test.csv"
OUT_SUMMARY = DATA_DIR / "error_analysis_summary.txt"

LABELS = [-1, 0, 1]


def run_one(path: Path, vec, model):
    df = pd.read_csv(path, encoding="utf-8")
    x = df["text_label"].fillna("").astype(str).tolist()
    y = df["label_human_int"].astype(int).to_numpy()

    X = vec.transform(x)
    pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y, pred, labels=LABELS)

    out = df.copy()
    out["y_true"] = y
    out["y_pred"] = pred

    if proba is not None:
        cls = list(model.classes_)
        for lab in LABELS:
            out[f"p_{lab}"] = proba[:, cls.index(lab)] if lab in cls else np.nan

    errors = out[out["y_true"] != out["y_pred"]].copy()
    return out, errors, cm


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    pack = joblib.load(MODEL_PATH)
    vec = pack["vectorizer"]
    model = pack["model"]

    all_dev, dev_err, cm_dev = run_one(DEV_CSV, vec, model)
    all_test, test_err, cm_test = run_one(TEST_CSV, vec, model)

    dev_err.to_csv(OUT_DEV_ERRORS, index=False, encoding="utf-8-sig")
    test_err.to_csv(OUT_TEST_ERRORS, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("HYBRID v1 ERROR ANALYSIS SUMMARY")
    lines.append(f"dev_rows={len(all_dev):,} dev_errors={len(dev_err):,}")
    lines.append("dev_confusion_matrix labels=[-1,0,1]:")
    lines.append(str(cm_dev))
    lines.append("")
    lines.append(f"test_rows={len(all_test):,} test_errors={len(test_err):,}")
    lines.append("test_confusion_matrix labels=[-1,0,1]:")
    lines.append(str(cm_test))
    lines.append("")
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"saved_dev_errors={OUT_DEV_ERRORS.resolve()}")
    print(f"saved_test_errors={OUT_TEST_ERRORS.resolve()}")
    print(f"saved_summary={OUT_SUMMARY.resolve()}")


if __name__ == "__main__":
    main()