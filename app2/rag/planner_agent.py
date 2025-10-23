#!/usr/bin/env python3
"""
Planner Agent - Routes user queries to appropriate workflow paths
"""
from __future__ import annotations
import re
from typing import Dict, Any, List
from ..llm_client import get_client, LLMClient
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

def plan_workflow(
    question: str,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Determine if user query should go to Q&A or Extraction workflow
    
    Args:
        question: User's question/request
        provider: LLM provider (openai/ollama)
        model: LLM model name
        temperature: Temperature for planning (low for consistency)
    
    Returns:
        Dict with path decision and reasoning
    """
    
    # Get LLM client
    if provider is None:
        provider = os.getenv("PLANNER_PROVIDER", "ollama")
    if model is None:
        model = os.getenv("PLANNER_MODEL", "llama3.1:8b")
    
    client = get_client(provider, model)
    
    # System prompt for workflow planning
    system_prompt = """You are a workflow planner for a financial document analysis system.

Your job is to determine if a user query should be routed to:
1. Q&A PATH: For conversational answers, summaries, explanations, analysis
2. EXTRACTION PATH: For structured data extraction, tables, comparisons, CSV exports

Q&A PATH examples:
- "Summarize climate risks for ExxonMobil"
- "What are the transition risks?"
- "How does Chevron address emissions?"
- "Explain the financial performance"

EXTRACTION PATH examples:
- "Compare risk factors between Chevron and ExxonMobil"
- "Extract risk table for 2024"
- "Track how risks evolved from 2022-2024"
- "Export risk factors to CSV"
- "Show me a structured comparison"

Respond with JSON format:
{
    "path": "qa" or "extraction",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of decision"
}"""

    user_prompt = f"""Analyze this user query and determine the appropriate workflow path:

Query: "{question}"

Consider:
- Keywords like "compare", "extract", "table", "csv", "export" → EXTRACTION
- Keywords like "summarize", "explain", "what", "how", "why" → Q&A
- Context: structured data needs vs conversational answers
- User intent: analysis vs data extraction"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.generate(messages, temperature=temperature, max_tokens=200)
        
        # Parse JSON response
        import json
        try:
            plan_result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to keyword-based planning if JSON parsing fails
            plan_result = _fallback_keyword_planning(question)
        
        # Add tracing information
        plan_result["_trace"] = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "raw_response": response
        }
        
        return plan_result
        
    except Exception as e:
        # Fallback to keyword-based planning
        plan_result = _fallback_keyword_planning(question)
        plan_result["_trace"] = {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "error": str(e),
            "fallback_used": True
        }
        return plan_result

def _fallback_keyword_planning(question: str) -> Dict[str, Any]:
    """Fallback keyword-based planning when LLM fails"""
    
    question_lower = question.lower()
    
    # Keywords that strongly indicate extraction
    extraction_keywords = [
        "compare", "comparison", "extract", "table", "csv", "export", "download",
        "track", "evolution", "timeline", "side by side", "structured",
        "risk factors", "factors table", "matrix", "grid", "evolved", "2022", "2023", "2024"
    ]
    
    # Keywords that indicate Q&A
    qa_keywords = [
        "summarize", "summary", "explain", "what", "how", "why", "describe",
        "analysis", "risks", "mitigation", "strategy", "approach",
        "tell me", "can you", "please", "question"
    ]
    
    # Count keyword matches
    extraction_score = sum(1 for kw in extraction_keywords if kw in question_lower)
    qa_score = sum(1 for kw in qa_keywords if kw in question_lower)
    
    # Decision logic
    if extraction_score > qa_score:
        return {
            "path": "extraction",
            "confidence": min(0.8, 0.5 + (extraction_score * 0.1)),
            "reasoning": f"Extraction keywords detected: {extraction_score} vs Q&A: {qa_score}"
        }
    elif qa_score > extraction_score:
        return {
            "path": "qa", 
            "confidence": min(0.8, 0.5 + (qa_score * 0.1)),
            "reasoning": f"Q&A keywords detected: {qa_score} vs Extraction: {extraction_score}"
        }
    else:
        # Default to Q&A for ambiguous cases
        return {
            "path": "qa",
            "confidence": 0.6,
            "reasoning": "Ambiguous query, defaulting to Q&A path"
        }

def plan_workflow_simple(question: str) -> Dict[str, Any]:
    """Simple keyword-based planning without LLM"""
    return _fallback_keyword_planning(question)
