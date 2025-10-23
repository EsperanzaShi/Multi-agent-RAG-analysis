"""
Enhanced critic agent that can use Ollama for sophisticated answer validation.
"""
from __future__ import annotations
import re
from typing import Dict, Any, List
from ..llm_client import get_client

# Accept a trailing punctuation (period/semicolon/comma/paren/quote) after the citation.
# Examples accepted: "]", "].", "],", "])", "]"""
# Permit small variations: optional comma after page number and optional trailing punctuation
CITE_RE = re.compile(
    r"\[CITATION:\s*([^,]+)\s+(\d{4}),\s*p\.\s*(\d+)\s*,?\]\s*(?:[.;,)\]’”\"']{0,2})\s*$",
    re.IGNORECASE,
)

def split_sentences(text: str) -> List[str]:
    # Split on newlines OR on whitespace after sentence enders
    parts = [s.strip() for s in re.split(r"\n+|(?<=[.!?])\s+", text) if s.strip()]
    return parts

def validate_answer_rule_based(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rule-based validation (original method) - made less strict."""
    sentences = split_sentences(answer)
    feedback = []
    valid = set((c.get("company"), int(c.get("year")), int(c.get("page_start"))) for c in retrieved)

    # If answer is "Insufficient support from retrieved context", approve it
    if "Insufficient support from retrieved context" in answer:
        return {"approved": True, "feedback": []}

    # Count sentences with proper citations
    sentences_with_citations = 0
    total_sentences = len(sentences)
    
    for i, s in enumerate(sentences, start=1):
        if "Insufficient support from retrieved context" in s:
            continue
        m = CITE_RE.search(s)
        if m:
            sentences_with_citations += 1
            comp, yr, page = m.group(1).strip(), int(m.group(2)), int(m.group(3))
            # Lenient company match (handles 'ExxonMobil' vs 'ExxonMobil Corporation')
            # Accept exact page or neighbor pages (±1) to account for chunk boundaries and TOC offsets.
            ok = any(((comp == vc) or vc.lower() in comp.lower() or comp.lower() in vc.lower())
                     and (yr == vy) and (vp in (page - 1, page, page + 1))
                     for (vc, vy, vp) in valid)
            if not ok:
                feedback.append({"idx": i, "issue": "bad_reference",
                                 "message": f"Cited page not in retrieved set: {comp} {yr} p.{page}."})
        else:
            # Only flag missing citations if the sentence makes factual claims
            if any(keyword in s.lower() for keyword in ["climate", "risk", "emission", "carbon", "greenhouse", "net zero", "2050"]):
                feedback.append({"idx": i, "issue": "missing_citation",
                                 "message": "Sentence missing [CITATION: Company Year, p. N]."})

    # Approve if at least 50% of sentences have citations or if there are few issues
    citation_ratio = sentences_with_citations / max(total_sentences, 1)
    if citation_ratio >= 0.5 or len(feedback) <= 2:
        return {"approved": True, "feedback": feedback}
    
    return {"approved": False, "feedback": feedback}

def validate_answer_llm(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LLM-based validation using Ollama - made less strict."""
    try:
        import os
        client = get_client("CRITIC_PROVIDER", "CRITIC_MODEL")
        provider = os.getenv("CRITIC_PROVIDER", "openai")
        model = os.getenv("CRITIC_MODEL", "gpt-4o-mini")
        
        # Format retrieved sources for context
        sources_text = "\n".join([
            f"- {c.get('company', 'Unknown')} {c.get('year', 'Unknown')} — {c.get('doc', 'Unknown')} — p.{c.get('page_start', 'Unknown')}\n  {c.get('text', '')[:200]}..."
            for c in retrieved
        ])
        
        system_prompt = """You are a quality control expert for financial document analysis. 
Your job is to validate answers for accuracy, completeness, and proper citation.

Be REASONABLE in your assessment:
- Approve answers that provide useful information, even if not perfect
- Only flag serious issues (completely wrong citations, major factual errors)
- Minor citation format issues should not cause rejection
- If the answer addresses the question with relevant information, approve it
 - Treat off-by-one page differences (±1) as acceptable if the cited pages are contiguous in the retrieved set and content plausibly matches.

Check for:
1. Are citations properly formatted as [CITATION: Company Year, p. N]?
2. Do the cited pages exist in the retrieved sources?
3. Is the answer factually accurate based on the sources?
4. Does the answer address the question asked?

Respond with JSON in this format:
{
    "approved": true/false,
    "feedback": [
        {"idx": 1, "issue": "missing_citation", "message": "Description of the issue"}
    ]
}"""

        user_prompt = f"""Please validate this answer:

ANSWER TO VALIDATE:
{answer}

RETRIEVED SOURCES:
{sources_text}

Respond with JSON only. Be reasonable - approve if the answer is useful."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.generate(messages, temperature=0.1, max_tokens=1000)
        
        # Try to parse JSON response
        import json
        try:
            result = json.loads(response)
            # Add provider info to result
            result["_trace"] = {"provider": provider, "model": model}
            return result
        except json.JSONDecodeError:
            # Fallback to rule-based if JSON parsing fails
            fallback_result = validate_answer_rule_based(answer, retrieved)
            fallback_result["_trace"] = {"provider": provider, "model": model, "fallback": True}
            return fallback_result
            
    except Exception as e:
        print(f"LLM critic failed, falling back to rule-based: {e}")
        fallback_result = validate_answer_rule_based(answer, retrieved)
        fallback_result["_trace"] = {"provider": provider, "model": model, "error": str(e)}
        return fallback_result

def validate_answer(answer: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Main validation function that chooses between rule-based and LLM-based validation."""
    # Use LLM-based validation if Ollama is configured, otherwise use rule-based
    import os
    critic_provider = os.getenv("CRITIC_PROVIDER", "openai")
    
    if critic_provider.lower() == "ollama":
        return validate_answer_llm(answer, retrieved)
    else:
        result = validate_answer_rule_based(answer, retrieved)
        result["_trace"] = {"provider": "rule-based", "model": "rule-based"}
        return result