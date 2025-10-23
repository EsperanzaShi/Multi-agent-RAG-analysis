from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import unicodedata


ITEM_RE = re.compile(r"\bItem\s+(\d+[A-Z]?)\b", re.I)

# --- Text normalization helpers ---

def normalize_text(s: str) -> str:
    """Normalize PDF-extracted text to improve retrieval and model grounding.
    - Unicode NFKC normalize (fixes curly quotes, widths)
    - Remove soft hyphens (\u00AD) and NBSPs (\xa0)
    - De-hyphenate words broken across line breaks: e.g., "decarbon-\nization" -> "decarbonization"
    - Collapse odd whitespace before newlines
    """
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    # Remove soft hyphens and NBSPs
    s = s.replace("\u00AD", "")
    s = s.replace("\xa0", " ")
    # De-hyphenate word breaks across newlines
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # Collapse tabs/multiple spaces that precede newlines
    s = re.sub(r"[ \t]+\n", "\n", s)
    return s

def infer_company_year_from_name(path: str) -> Tuple[str, int]:
    name = Path(path).stem
    m = re.search(r"(\d{4})", name)
    year = int(m.group(1)) if m else -1
    company = re.sub(r"[_\- ]?\d{4}.*$", "", name).replace("_", " ").strip() or "Unknown"
    return company, year

def split_into_chunks(text: str, chunk_size: int = 1400, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

def detect_item_label(page_text: str, last_item: str | None) -> str | None:
    m = ITEM_RE.search(page_text)
    return m.group(0) if m else last_item

def extract_text_chunks(pdf_path: str, company: str, year: int) -> tuple[List[Dict[str, Any]], List[int]]:
    doc = fitz.open(pdf_path) # PyMuPDF for text extraction
    chunks, low_text_pages, last_item = [], [], None
    for pno in range(len(doc)):
        page = doc[pno]
        text_raw = page.get_text("text")
        text = normalize_text(text_raw) #involve steps like converting to lowercase, removing punctuation, and fixing extra whitespace
        if len(text.strip()) < 50:
            low_text_pages.append(pno + 1)
        curr_item = detect_item_label(text, last_item)
        last_item = curr_item
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()] # Split into paragraphs
        joined = "\n\n".join(paras)
        for idx, chunk_text in enumerate(split_into_chunks(joined)):
            chunks.append({
                "id": f"{Path(pdf_path).stem}_p{pno+1}_c{idx}",
                "doc": Path(pdf_path).name,
                "company": company,
                "year": year,
                "type": "text",
                "item": curr_item,
                "page_start": pno + 1,
                "page_end": pno + 1,
                "text": chunk_text
            })
    return chunks, low_text_pages

def _clean(values):
    return [str(v).strip() if v is not None else "" for v in values]

def extract_tables(pdf_path: str, out_dir: Path, company: str, year: int) -> List[Dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_chunks: List[Dict[str, Any]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables, start=1):
                if not table or len(table) < 2:  # need header + at least 1 row
                    continue
                header = _clean(table[0])
                rows = [_clean(r) for r in table[1:]]
                df = pd.DataFrame(rows, columns=header if any(header) else None)
                csv_path = out_dir / f"{Path(pdf_path).stem}_p{pno}_t{t_idx}.csv"
                try:
                    df.to_csv(csv_path, index=False)
                except Exception:
                    df.to_csv(csv_path, index=False, header=False)
                preview = "; ".join([", ".join(map(str, r)) for r in df.head(5).values])
                textified = f"TABLE p.{pno} #{t_idx}: columns=[{', '.join(map(str, df.columns))}]. rows: {preview}"
                table_chunks.append({
                    "id": f"{Path(pdf_path).stem}_p{pno}_t{t_idx}",
                    "doc": Path(pdf_path).name,
                    "company": company,
                    "year": year,
                    "type": "table",
                    "item": None,
                    "page_start": pno,
                    "page_end": pno,
                    "text": textified,
                    "artifact": str(csv_path)
                })
    return table_chunks

def parse_pdf_to_outputs(pdf_path: str, processed_dir: Path) -> Dict[str, Any]:
    company, year = infer_company_year_from_name(pdf_path)
    text_chunks, low_text = extract_text_chunks(pdf_path, company, year)
    table_chunks = extract_tables(pdf_path, processed_dir / "tables", company, year)
    return {
        "company": company,
        "year": year,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "low_text_pages": low_text
    }