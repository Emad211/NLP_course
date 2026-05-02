from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_CSV = BASE_DIR / "processed_data" / "retriever" / "doctor_documents.csv"
OUT_TXT = BASE_DIR / "processed_data" / "retriever" / "df_audit_report.txt"

TOKENS = [
    "حوصله",
    "برخورد",
    "تشخیص",
    "عمل",
    "درد",
    "سنگ",
    "کلیه",
    "دیسک",
    "زانو",
    "ریزش",
    "مو",
    "نوبت",
    "معطلی",
    "هزینه",
]

def main():
    df = pd.read_csv(DOCS_CSV, keep_default_na=False)
    df["document"] = df["document"].astype(str)

    n_docs = len(df)
    lines = []
    lines.append("DF AUDIT (doctor documents)")
    lines.append(f"doctor_docs={n_docs}")
    lines.append("token\tdoc_freq\tdoc_ratio")

    for tok in TOKENS:
        dfreq = int(df["document"].str.contains(rf"(?<!\S){tok}(?!\S)", regex=True).sum())
        ratio = dfreq / n_docs if n_docs else 0.0
        lines.append(f"{tok}\t{dfreq}\t{ratio:.3f}")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved_report={OUT_TXT.resolve()}")

if __name__ == "__main__":
    main()