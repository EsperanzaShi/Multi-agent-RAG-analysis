from __future__ import annotations
import os, re
from typing import Dict, Any, List
from dotenv import load_dotenv
from .retriever import Retriever
from .prompts import ANSWERER_SYSTEM
from ..llm_client import get_client, LLMClient

load_dotenv(dotenv_path=".env", override=True)

CITE_RE = re.compile(r"\[CITATION:\s*([^,]+)\s+(\d{4}),\s*p\.\s*(\d+)\]")

def _fmt_context(chunks: List[Dict[str, Any]]) -> str:
    blocks = []
    for c in chunks:
        meta = f"{c['company']} {c['year']} — {c['doc']} — p.{c['page_start']}"
        blocks.append(f"### {meta}\n{c['text']}")
    return "\n\n".join(blocks)

def answer_question(
    question: str,
    company: str,
    years: List[int],
    top_k: int = 6,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    r = Retriever()
    item_hint = "Item 1A" if "risk" in question.lower() else None
    chunks = r.search(question, top_k=top_k, company=company, years=years, item_hint=item_hint)
    context = _fmt_context(chunks)

    client = (
        LLMClient(provider, model) if (provider and model) else get_client("ANSWERER_PROVIDER", "ANSWERER_MODEL")
    )

    sys_prompt = ANSWERER_SYSTEM
    user_prompt = (
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\n"
        "Write 4–8 short sentences.\n"
        "- Each sentence MUST end with exactly one citation in this form: [CITATION: Company Year, p. N].\n"
        "- Place the period AFTER the citation (e.g., ... [CITATION: ExxonMobil 2024, p. 47].)\n"
        "- Use a SINGLE page number (no ranges like 46–47).\n"
        "- Only respond with 'Insufficient support from retrieved context.' if the context contains NO relevant information to answer the question.\n"
        "- If the context contains any relevant information (even if limited), provide a summary based on what is available.\n"
        "- Focus on the most relevant information from the context.\n"
        "- Be specific and factual in your statements.\n"
        "- For climate-related questions, include any information about emissions, climate change, energy transition, or environmental risks, even if not perfectly matching the specific question.\n"
        "- Extract and summarize the most relevant climate-related information available in the context.\n"
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = client.generate(messages, temperature=temperature, max_tokens=500)

    found = CITE_RE.findall(text)
    used_pages = set()
    for comp, yr, page in found:
        try:
            used_pages.add((comp.strip(), int(yr), int(page)))
        except:
            pass

    # Attach trace info for dashboard
    trace: Dict[str, Any] = {
        "providers": {
            "answerer": {
                "provider": provider or os.getenv("ANSWERER_PROVIDER", "openai"),
                "model": model or os.getenv("ANSWERER_MODEL", "gpt-4o-mini"),
            }
        },
        "retriever": getattr(r, "stats", {}),
    }
    return {"answer": text, "retrieved": chunks, "citations_found": list(used_pages), "trace": trace}