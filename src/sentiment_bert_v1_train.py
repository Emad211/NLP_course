"""
sentiment_bert_v1_train.py
Fine-tune ParsBERT روی دیتاست sentiment فارسی
خروجی: models/sentiment_bert_v1/{best_model, metrics.json, confusion_matrix.png, errors.csv}
"""

import os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# ─── تنظیمات ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR     = ROOT / "processed_data" / "sentiment_bert_v1"
MODEL_PATH   = ROOT / "models" / "parsbert_base"
OUTPUT_DIR   = ROOT / "models" / "sentiment_bert_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# هایپرپارامترها
MAX_LEN      = 128
BATCH_SIZE   = 16      # برای RTX 4060 8GB کافیه؛ اگر OOM شد → 8
EPOCHS       = 5
LR           = 2e-5
WARMUP_RATIO = 0.1
NUM_LABELS   = 2
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        df = pd.read_csv(csv_path)
        self.texts  = df["text_bert"].tolist()
        self.labels = df["label_id"].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["label"].to(device)

            out  = model(input_ids, attention_mask=attn_mask, labels=labels)
            total_loss += out.loss.item()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc, all_preds, all_labels

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)

    print("Loading datasets...")
    train_ds = SentimentDataset(DATA_DIR / "train.csv", tok, MAX_LEN)
    dev_ds   = SentimentDataset(DATA_DIR / "dev.csv",   tok, MAX_LEN)
    test_ds  = SentimentDataset(DATA_DIR / "test.csv",  tok, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        num_labels=NUM_LABELS,
        local_files_only=True
    ).to(DEVICE)

    # Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ─── Train Loop ──────────────────────────────────────────────────────────
    best_dev_f1   = 0.0
    best_epoch    = 0
    history       = []

    print(f"\nTraining for {EPOCHS} epochs on {len(train_ds)} samples...")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out  = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            if step % 20 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{EPOCHS}  Step {step}/{len(train_loader)}"
                      f"  loss={train_loss/step:.4f}", end="\r")

        avg_train_loss = train_loss / len(train_loader)
        dev_loss, dev_f1, dev_acc, _, _ = evaluate(model, dev_loader, DEVICE)

        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f}  "
              f"dev_loss={dev_loss:.4f}  dev_f1={dev_f1:.4f}  dev_acc={dev_acc:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_loss,
            "dev_f1": dev_f1,
            "dev_acc": dev_acc
        })

        # ذخیره بهترین مدل
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch  = epoch
            model.save_pretrained(str(OUTPUT_DIR / "best_model"))
            tok.save_pretrained(str(OUTPUT_DIR / "best_model"))
            print(f"  ✓ Best model saved (dev_f1={dev_f1:.4f})")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min. Best epoch={best_epoch}, best_dev_f1={best_dev_f1:.4f}")

    # ─── Test با بهترین مدل ──────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        str(OUTPUT_DIR / "best_model"), local_files_only=True
    ).to(DEVICE)

    test_loss, test_f1, test_acc, test_preds, test_labels = evaluate(
        best_model, test_loader, DEVICE
    )

    print(f"\nTest Results: loss={test_loss:.4f}  f1={test_f1:.4f}  acc={test_acc:.4f}")
    print("\nClassification Report:")
    report = classification_report(test_labels, test_preds,
                                   target_names=["negative", "positive"])
    print(report)

    # ─── ذخیره نتایج ─────────────────────────────────────────────────────────
    metrics = {
        "best_epoch"  : best_epoch,
        "best_dev_f1" : best_dev_f1,
        "test_loss"   : test_loss,
        "test_f1_macro": test_f1,
        "test_acc"    : test_acc,
        "history"     : history,
        "hyperparams" : {
            "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR,
            "warmup_ratio": WARMUP_RATIO, "seed": SEED
        }
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"metrics.json saved.")

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["negative", "positive"])
    ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Test Confusion Matrix\nF1={test_f1:.4f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "confusion_matrix.png"), dpi=150)
    plt.close()
    print("confusion_matrix.png saved.")

    # Error Analysis
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["pred_label"]    = test_preds
    test_df["true_label"]    = test_labels
    errors = test_df[test_df["pred_label"] != test_df["true_label"]].copy()
    errors.to_csv(OUTPUT_DIR / "errors.csv", index=False, encoding="utf-8-sig")
    print(f"errors.csv saved ({len(errors)} errors out of {len(test_df)} test samples).")

    print("\n=== DONE ===")
    print(f"Output dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()