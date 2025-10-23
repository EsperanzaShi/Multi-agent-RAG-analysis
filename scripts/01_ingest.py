from __future__ import annotations

from pathlib import Path
import sys
# Ensure project root is on sys.path so `app/...` imports work when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, glob
from app.ingest.parse_pdf import parse_pdf_to_outputs

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def main():
    all_chunks = []
    report = []
    pdfs = sorted(glob.glob(str(RAW_DIR / "*.pdf")))
    if not pdfs:
        print("No PDFs found in data/raw/. Put your 10-K PDFs there.")
        return
    for pdf in pdfs:
        print(f"Parsing {pdf} ...")
        out = parse_pdf_to_outputs(pdf, PROC_DIR)
        chunks = out["text_chunks"] + out["table_chunks"]
        all_chunks.extend(chunks)
        report.append({
            "pdf": pdf,
            "company": out["company"],
            "year": out["year"],
            "n_text_chunks": len(out["text_chunks"]),
            "n_table_chunks": len(out["table_chunks"]),
            "low_text_pages": out["low_text_pages"],
        })
    with open(PROC_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(PROC_DIR / "ingest_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Done. Wrote {len(all_chunks)} chunks to data/processed/chunks.jsonl")
    print("Ingest report: data/processed/ingest_report.json")

if __name__ == "__main__":
    main()