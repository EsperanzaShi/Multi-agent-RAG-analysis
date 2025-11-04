#!/usr/bin/env python3
"""
Unified Orchestrator - Routes queries through Planner Agent to appropriate workflows
"""
from __future__ import annotations
from typing import Dict, Any, List
from .rag.qa_agent import answer_question
from .rag.enhanced_critic import validate_answer
from .rag.extractor_agent import extract_risk_table
from .rag.planner_agent import plan_workflow

def run_unified(
    question: str,
    company: str,
    years: List[int],
    top_k: int = 6,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    planner_provider: str | None = None,       
    planner_model: str | None = None,
    planner_temperature: float = 0.1
) -> Dict[str, Any]:
    """
    Unified orchestrator that routes queries through planner to appropriate agents
    
    Args:
        question: User's question/request
        company: Company name
        years: List of years to analyze
        top_k: Number of chunks to retrieve
        provider: LLM provider for main agents
        model: LLM model for main agents
        temperature: Temperature for main agents
        planner_provider: LLM provider for planner
        planner_model: LLM model for planner
        planner_temperature: Temperature for planner
    
    Returns:
        Unified response with trace information
    """
    
    # Step 1: Plan the workflow
    plan = plan_workflow(
        question, 
        provider=planner_provider,
        model=planner_model,
        temperature=planner_temperature
    )
    
    # Step 2: Route to appropriate workflow
    if plan["path"] == "extraction":
        # Route to extraction workflow
        result = _run_extraction_workflow(
            question, company, years, 
            provider=provider, model=model, temperature=temperature
        )
    else:
        # Route to Q&A workflow (default)
        result = _run_qa_workflow(
            question, company, years, top_k,
            provider=provider, model=model, temperature=temperature
        )
    
    # Step 3: Combine traces
    combined_trace = {
        "planner": plan,
        "workflow": result.get("trace", {}),
        "workflow_type": plan["path"]
    }
    
    result["trace"] = combined_trace
    # Preserve the specific workflow type from extraction results (e.g., "timeseries", "cross_company")
    # Only override if it's not already set by the extraction workflow
    if "workflow_type" not in result:
        result["workflow_type"] = plan["path"]
    result["planning_confidence"] = plan.get("confidence", 0.0)
    result["planning_reasoning"] = plan.get("reasoning", "")
    
    return result

def _run_qa_workflow(
    question: str,
    company: str,
    years: List[int],
    top_k: int,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2
) -> Dict[str, Any]:
    """Run Q&A workflow with critic validation"""
    
    # Get answer from QA agent
    out = answer_question(
        question, company, years, 
        top_k=top_k, provider=provider, model=model, temperature=temperature
    )
    
    # Validate with critic
    verdict = validate_answer(out["answer"], out["retrieved"])
    
    if verdict["approved"]:
        out["critic"] = verdict
        out["iterations"] = 1
        return out
    
    # Retry with stricter citation requirements
    retry_q = question + "\n\nIMPORTANT: Ensure every sentence ends with a valid [CITATION: Company Year, p. N] taken from the retrieved pages."
    out2 = answer_question(
        retry_q, company, years, 
        top_k=top_k, provider=provider, model=model, temperature=temperature
    )
    verdict2 = validate_answer(out2["answer"], out2["retrieved"])
    
    out2["critic"] = verdict2
    out2["iterations"] = 2
    
    # Combine traces from both attempts
    trace = out.get("trace", {})
    trace2 = out2.get("trace", {})
    if "_trace" in verdict:
        trace["critic_attempt1"] = verdict["_trace"]
    if "_trace" in verdict2:
        trace2["critic_attempt2"] = verdict2["_trace"]
    
    out2["trace"] = {"attempt1": trace, "attempt2": trace2}
    return out2

def _run_extraction_workflow(
    question: str,
    company: str,
    years: List[int],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Run extraction workflow"""
    
    # For extraction, we use the question to determine comparison mode
    comparison_mode = _determine_comparison_mode(question, company, years)
    
    # Use OpenAI for extraction tasks for better performance and reliability
    extraction_provider = provider if provider == "openai" else "openai"
    extraction_model = model if model and provider == "openai" else "gpt-4o-mini"
    
    if comparison_mode == "single":
        # Single company extraction
        return extract_risk_table(
            company, years,
            provider=extraction_provider, model=extraction_model, temperature=temperature
        )
    elif comparison_mode == "cross":
        # Cross-company comparison
        return _run_cross_company_extraction(
            company, years, extraction_provider, extraction_model, temperature
        )
    elif comparison_mode == "timeseries":
        # Time-series analysis
        return _run_timeseries_extraction(
            company, years, extraction_provider, extraction_model, temperature
        )
    else:
        # Default to single company
        return extract_risk_table(
            company, years,
            provider=extraction_provider, model=extraction_model, temperature=temperature
        )

def _determine_comparison_mode(question: str, company: str, years: List[int]) -> str:
    """Determine extraction mode based on question"""
    
    question_lower = question.lower()
    
    # Cross-company keywords (check first - higher priority)
    cross_keywords = ["compare", "between", "vs", "versus", "side by side", "different from", "contrast", "versus"]
    if any(kw in question_lower for kw in cross_keywords):
        return "cross"
    
    # Time-series keywords (only if no cross-company indicators)
    timeseries_keywords = ["track", "evolution", "timeline", "over time", "evolved", "changed", "progression"]
    # Only trigger time-series if it contains time-series specific words AND year ranges
    if any(kw in question_lower for kw in timeseries_keywords):
        # Check for year ranges or multiple years
        year_indicators = ["2022", "2023", "2024", "from 2022", "to 2024", "2022-2024"]
        if any(year in question_lower for year in year_indicators):
            return "timeseries"
    
    # Default to single company
    return "single"

def _run_cross_company_extraction(
    company: str, years: List[int], provider: str, model: str, temperature: float
) -> Dict[str, Any]:
    """Run cross-company risk comparison"""
    
    # Hardcode to Chevron vs ExxonMobil 2024 for demo
    result_chevron = extract_risk_table(
        "Chevron", [2024],
        provider=provider, model=model, temperature=temperature
    )
    
    result_exxon = extract_risk_table(
        "ExxonMobil", [2024], 
        provider=provider, model=model, temperature=temperature
    )
    
    # Combine results
    combined_result = {
        "success": result_chevron["success"] and result_exxon["success"],
        "workflow_type": "cross_company",
        "chevron_table": result_chevron.get("table", []),
        "exxon_table": result_exxon.get("table", []),
        "metadata": {
            "chevron_risks": len(result_chevron.get("table", [])),
            "exxon_risks": len(result_exxon.get("table", [])),
            "comparison_year": 2024
        },
        "trace": {
            "chevron": result_chevron.get("trace", {}),
            "exxon": result_exxon.get("trace", {})
        }
    }
    
    return combined_result

def _run_timeseries_extraction(
    company: str, years: List[int], provider: str, model: str, temperature: float
) -> Dict[str, Any]:
    """Run time-series risk analysis"""
    
    # Use the provided company and years, but fallback to ExxonMobil 2022-2024 if needed
    if company == "Chevron" and len(years) == 1 and years[0] == 2024:
        # Chevron only has 2024 data, so we'll use ExxonMobil for time-series demo
        target_company = "ExxonMobil"
        target_years = [2022, 2023, 2024]
    else:
        target_company = company
        target_years = years if years else [2022, 2023, 2024]
    
    results = {}
    for year in target_years:
        results[year] = extract_risk_table(
            target_company, [year],
            provider=provider, model=model, temperature=temperature
        )
    
    # Combine results
    combined_result = {
        "success": all(r["success"] for r in results.values()),
        "workflow_type": "timeseries",
        "tables": {year: r.get("table", []) for year, r in results.items()},
        "metadata": {
            "company": target_company,
            "years": target_years,
            "risk_counts": {year: len(r.get("table", [])) for year, r in results.items()}
        },
        "trace": {year: r.get("trace", {}) for year, r in results.items()}
    }
    
    return combined_result
