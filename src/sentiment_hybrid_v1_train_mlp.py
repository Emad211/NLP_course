from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "processed_data" / "sentiment_hybrid_v1"

TRAIN_CSV = DATA_DIR / "train.csv"
DEV_CSV = DATA_DIR / "dev.csv"
TEST_CSV = DATA_DIR / "test.csv"

MODEL_OUT = BASE_DIR / "models" / "sentiment_mlp_hybrid_v1.joblib"
REPORT_OUT = DATA_DIR / "mlp_report.txt"

SEED = 42
LABELS = [-1, 0, 1]


def load_split(path: Path):
    df = pd.read_csv(path, encoding="utf-8")
    if "text_label" not in df.columns:
        raise RuntimeError(f"Missing text_label in {path.name}")
    if "label_human_int" not in df.columns:
        raise RuntimeError(f"Missing label_human_int in {path.name}")
    x = df["text_label"].fillna("").astype(str).tolist()
    y = df["label_human_int"].astype(int).to_numpy()
    return df, x, y


def eval_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    rep = classification_report(y_true, y_pred, digits=4, zero_division=0)

    lines = []
    lines.append(f"=== {name} ===")
    lines.append(f"accuracy={acc:.4f}")
    lines.append(f"macro_f1={f1_macro:.4f}")
    lines.append("confusion_matrix labels=[-1,0,1]:")
    lines.append(str(cm))
    lines.append("classification_report:")
    lines.append(rep)
    lines.append("")
    return "\n".join(lines)


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Inverse-frequency weights.
    """
    vc = pd.Series(y).value_counts().to_dict()
    total = len(y)
    weights = {}
    for lab in vc:
        weights[int(lab)] = total / (len(vc) * vc[lab])
    return np.array([weights[int(v)] for v in y], dtype=float)


def main():
    for p in [TRAIN_CSV, DEV_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(p)

    df_tr, x_tr, y_tr = load_split(TRAIN_CSV)
    df_dev, x_dev, y_dev = load_split(DEV_CSV)
    df_te, x_te, y_te = load_split(TEST_CSV)

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=120000,
    )

    X_tr = vec.fit_transform(x_tr)
    X_dev = vec.transform(x_dev)
    X_te = vec.transform(x_te)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 64),
        activation="relu",
        solver="adam",
        alpha=3e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=5,
        random_state=SEED,
        verbose=True,
    )

    sw = make_sample_weights(y_tr)

    # Fit with sample_weight if supported, else fallback
    try:
        clf.fit(X_tr, y_tr, sample_weight=sw)
        used_sw = True
    except TypeError:
        clf.fit(X_tr, y_tr)
        used_sw = False

    dev_pred = clf.predict(X_dev)
    test_pred = clf.predict(X_te)

    lines = []
    lines.append("SENTIMENT MLP REPORT (HYBRID v1)")
    lines.append(f"train_rows={len(df_tr):,} dev_rows={len(df_dev):,} test_rows={len(df_te):,}")
    lines.append(f"sample_weight_used={used_sw}")
    lines.append("label_dist_train:")
    vc = pd.Series(y_tr).value_counts().sort_index()
    for k, v in vc.items():
        lines.append(f"  {int(k)}: {int(v):,}")
    lines.append("")
    lines.append(eval_split("DEV (gold)", y_dev, dev_pred))
    lines.append(eval_split("TEST (gold)", y_te, test_pred))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vec,
            "model": clf,
            "labels": LABELS,
            "seed": SEED,
            "meta": {"sample_weight_used": used_sw},
        },
        MODEL_OUT,
    )

    print("\n".join(lines))
    print(f"saved_model={MODEL_OUT.resolve()}")
    print(f"saved_report={REPORT_OUT.resolve()}")


if __name__ == "__main__":
    main()