from __future__ import annotations
import re
from typing import Dict, Any, List

# Accept a trailing punctuation (period/semicolon/comma/paren/quote) after the citation.
# Examples accepted: "]", "].", "],", "])", "]”"
CITE_RE = re.compile(
    r"\[CITATION:\s*([^,]+)\s+(\d{4}),\s*p\.\s*(\d+)\]\s*(?:[.;,)\]\"'”’]{0,2})\s*$",
    re.IGNORECASE,
)

def split_sentences(text: str) -> List[str]:
    # Split on newlines OR on whitespace after sentence enders
    parts = [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", text) if s.strip()]
    return parts

def validate_answer(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    sentences = split_sentences(answer)
    feedback = []
    valid = set((c.get("company"), int(c.get("year")), int(c.get("page_start"))) for c in retrieved)

    for i, s in enumerate(sentences, start=1):
        if "Insufficient support from retrieved context." in s:
            continue
        m = CITE_RE.search(s)
        if not m:
            feedback.append({"idx": i, "issue": "missing_citation",
                             "message": "Sentence missing [CITATION: Company Year, p. N]."})
            continue
        comp, yr, page = m.group(1).strip(), int(m.group(2)), int(m.group(3))
        # Lenient company match (handles 'ExxonMobil' vs 'ExxonMobil Corporation')
        ok = any(((comp == vc) or vc.lower() in comp.lower() or comp.lower() in vc.lower())
                 and (yr == vy) and (page == vp)
                 for (vc, vy, vp) in valid)
        if not ok:
            feedback.append({"idx": i, "issue": "bad_reference",
                             "message": f"Cited page not in retrieved set: {comp} {yr} p.{page}."})

    return {"approved": len(feedback) == 0, "feedback": feedback}