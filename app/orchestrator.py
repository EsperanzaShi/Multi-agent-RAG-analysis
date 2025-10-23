from __future__ import annotations
from typing import Dict, Any, List
from app.rag.qa_agent import answer_question
from app.rag.enhanced_critic import validate_answer

def run_qa(
    question: str,
    company: str,
    years: List[int],
    top_k: int = 6,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    out = answer_question(question, company, years, top_k=top_k, provider=provider, model=model, temperature=temperature)
    t0_ok = True
    verdict = validate_answer(out["answer"], out["retrieved"])
    if verdict["approved"]:
        out["critic"] = verdict
        out["iterations"] = 1
        # Combine traces from all agents
        combined_trace = out.get("trace", {})
        if "_trace" in verdict:
            combined_trace["critic"] = verdict["_trace"]
        out["trace"] = combined_trace
        return out
    # single retry with stricter hint
    retry_q = question + "\n\nIMPORTANT: Ensure every sentence ends with a valid [CITATION: Company Year, p. N] taken from the retrieved pages."
    out2 = answer_question(retry_q, company, years, top_k=top_k, provider=provider, model=model, temperature=temperature)
    verdict2 = validate_answer(out2["answer"], out2["retrieved"])
    out2["critic"] = verdict2
    out2["iterations"] = 2
    # Bubble up trace combining attempts
    trace = out.get("trace", {})
    trace2 = out2.get("trace", {})
    if "_trace" in verdict:
        trace["critic_attempt1"] = verdict["_trace"]
    if "_trace" in verdict2:
        trace2["critic_attempt2"] = verdict2["_trace"]
    out2["trace"] = {"attempt1": trace, "attempt2": trace2}
    return out2