#!/usr/bin/env python3
"""
Extractor Agent - Converts unstructured risk text into structured tables
"""
from __future__ import annotations
import re
from typing import Dict, Any, List
from .retriever import Retriever
from ..llm_client import get_client, LLMClient
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

def extract_risk_table(
    company: str,
    years: List[int],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Extract structured risk table from company documents
    
    Args:
        company: Company name
        years: List of years to analyze
        provider: LLM provider (openai/ollama)
        model: LLM model name
        temperature: Temperature for generation (low for structured output)
    
    Returns:
        Dict with extracted table data and metadata
    """
    
    # Initialize retriever
    r = Retriever()
    
    # Search for risk-related content
    risk_query = f"Item 1A Risk Factors {company}"
    chunks = r.search(
        risk_query, 
        top_k=15,  # Get more chunks for comprehensive extraction
        company=company, 
        years=years, 
        item_hint="Item 1A"
    )
    
    if not chunks:
        return {
            "success": False,
            "error": "No risk-related content found",
            "table": [],
            "metadata": {"company": company, "years": years, "chunks_found": 0}
        }
    
    # Format context for extraction
    context = _format_context_for_extraction(chunks)
    
    # Initialize LLM client
    client = (
        LLMClient(provider, model) if (provider and model) else 
        get_client("EXTRACTOR_PROVIDER", "EXTRACTOR_MODEL")
    )
    
    # Extraction prompt
    system_prompt = """You are a financial document analyst specializing in risk factor extraction.
Your task is to extract risk factors from 10-K documents and structure them into a clean table.

For each risk factor, identify:
1. Risk: The specific risk name/category
2. Description: Brief description of the risk
3. Mitigation: Any mitigation strategies mentioned
4. Citations: Specific page references [CITATION: Company Year, p. N]

Be precise and factual. Only include risks that are clearly stated in the text.
Each row must have at least one citation."""
    
    user_prompt = f"""Extract risk factors from the following {company} 10-K document sections:

CONTEXT:
{context}

Create a structured table with these columns:
- Risk: Risk factor name/category
- Description: Brief description of the risk
- Mitigation: Mitigation strategies (if mentioned)
- Citations: Page references [CITATION: Company Year, p. N]

Format as a clean table without separator rows. Use this exact format:
| Risk | Description | Mitigation | Citations |
| [Risk Name] | [Description] | [Mitigation] | [CITATION: Company Year, p. N] |
| [Risk Name] | [Description] | [Mitigation] | [CITATION: Company Year, p. N] |

Be comprehensive but concise. Do not include separator rows with dashes."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Generate extraction
    text = client.generate(messages, temperature=temperature, max_tokens=1500)
    
    # Parse the structured output
    table_data = _parse_extraction_output(text, chunks)
    
    # Add trace information
    trace = {
        "extractor": {
            "provider": provider or os.getenv("EXTRACTOR_PROVIDER", "openai"),
            "model": model or os.getenv("EXTRACTOR_MODEL", "gpt-4o-mini"),
            "temperature": temperature
        },
        "retriever": getattr(r, "stats", {}),
        "chunks_processed": len(chunks)
    }
    
    return {
        "success": True,
        "table": table_data,
        "metadata": {
            "company": company,
            "years": years,
            "chunks_found": len(chunks),
            "risks_extracted": len(table_data)
        },
        "trace": trace,
        "raw_output": text
    }

def _format_context_for_extraction(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks for extraction with clear page references"""
    blocks = []
    for c in chunks:
        meta = f"{c['company']} {c['year']} — {c['doc']} — p.{c['page_start']}"
        blocks.append(f"### {meta}\n{c['text']}")
    return "\n\n".join(blocks)

def _parse_extraction_output(text: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse the LLM output into structured table data"""
    
    # Simple parsing - look for table-like structure
    lines = text.split('\n')
    table_data = []
    
    # Find table rows (look for patterns like "| Risk | Description | ...")
    in_table = False
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and headers
        if not line or (line.startswith('|') and ('Risk' in line or 'Description' in line)):
            in_table = True
            continue
            
        if in_table and line.startswith('|'):
            # Parse table row
            parts = [p.strip() for p in line.split('|') if p.strip()]
            
            # Skip separator rows (rows with only dashes or similar)
            if len(parts) >= 3 and not _is_separator_row(parts):
                risk_data = {
                    "risk": parts[0] if len(parts) > 0 else "",
                    "description": parts[1] if len(parts) > 1 else "",
                    "mitigation": parts[2] if len(parts) > 2 else "",
                    "citations": parts[3] if len(parts) > 3 else ""
                }
                # Only add if risk is not empty and not a separator
                if (risk_data["risk"] and 
                    risk_data["risk"] != "----------" and 
                    not risk_data["risk"].startswith("-") and
                    risk_data["risk"] != "Risk" and
                    risk_data["risk"] != "risk"):
                    table_data.append(risk_data)
        elif in_table and not line.startswith('|'):
            # End of table
            break
    
    # If no table structure found, try to extract from prose
    if not table_data:
        table_data = _extract_from_prose(text)
    
    # Post-process to remove any remaining empty or invalid rows
    table_data = _clean_table_data(table_data)
    
    return table_data

def _is_separator_row(parts: List[str]) -> bool:
    """Check if a row is a separator row (dashes, etc.)"""
    if not parts:
        return True
    
    # Check if all parts are dashes or similar separators
    for part in parts:
        if not (part.startswith("-") or part == "---" or part == "----------" or 
                part == "|" or part == "" or part.isspace()):
            return False
    return True

def _clean_table_data(table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean table data by removing empty or invalid rows"""
    cleaned_data = []
    
    for row in table_data:
        # Check if row has meaningful content
        if (row.get("risk") and 
            row["risk"].strip() and 
            not row["risk"].startswith("-") and
            row["risk"] != "----------" and
            row["risk"] != "Risk" and
            row["risk"] != "risk" and
            len(row["risk"].strip()) > 1):
            cleaned_data.append(row)
    
    return cleaned_data

def _extract_from_prose(text: str) -> List[Dict[str, Any]]:
    """Fallback: extract risk information from prose format"""
    
    # Look for risk patterns
    risk_patterns = [
        r"Risk[:\s]+([^.]*?)(?:\.|Mitigation[:\s]+([^.]*?))",
        r"([A-Z][^.]*?risk[^.]*?)(?:\.|Mitigation[:\s]+([^.]*?))",
    ]
    
    risks = []
    for pattern in risk_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if len(match) >= 1:
                risk_data = {
                    "risk": match[0].strip(),
                    "description": match[0].strip(),
                    "mitigation": match[1].strip() if len(match) > 1 else "",
                    "citations": ""
                }
                risks.append(risk_data)
    
    return risks
