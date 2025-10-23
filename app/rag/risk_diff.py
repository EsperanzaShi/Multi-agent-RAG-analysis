#!/usr/bin/env python3
"""
Risk Diff Helper - Compare risk factors across companies and time periods
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd

def compare_risks(
    risk_data_1: List[Dict[str, Any]], 
    risk_data_2: List[Dict[str, Any]],
    label_1: str = "Company 1",
    label_2: str = "Company 2"
) -> Dict[str, Any]:
    """
    Compare two sets of risk data and identify differences
    
    Args:
        risk_data_1: First set of risk data
        risk_data_2: Second set of risk data  
        label_1: Label for first dataset
        label_2: Label for second dataset
    
    Returns:
        Dict with comparison results
    """
    
    # Convert to sets for comparison
    risks_1 = {_normalize_risk(risk["risk"]) for risk in risk_data_1}
    risks_2 = {_normalize_risk(risk["risk"]) for risk in risk_data_2}
    
    # Find differences
    added_risks = risks_2 - risks_1
    removed_risks = risks_1 - risks_2
    common_risks = risks_1 & risks_2
    
    # Get detailed info for added/removed risks
    added_details = [risk for risk in risk_data_2 if _normalize_risk(risk["risk"]) in added_risks]
    removed_details = [risk for risk in risk_data_1 if _normalize_risk(risk["risk"]) in removed_risks]
    
    return {
        "summary": {
            "total_1": len(risks_1),
            "total_2": len(risks_2),
            "common": len(common_risks),
            "added": len(added_risks),
            "removed": len(removed_risks)
        },
        "added_risks": added_details,
        "removed_risks": removed_details,
        "common_risks": list(common_risks),
        "labels": {"dataset_1": label_1, "dataset_2": label_2}
    }

def create_risk_timeline(
    risk_data_by_year: Dict[int, List[Dict[str, Any]]],
    company: str
) -> Dict[str, Any]:
    """
    Create a timeline showing how risks evolved over time
    
    Args:
        risk_data_by_year: Dict mapping years to risk data
        company: Company name
    
    Returns:
        Timeline analysis
    """
    
    years = sorted(risk_data_by_year.keys())
    if len(years) < 2:
        return {"error": "Need at least 2 years for timeline analysis"}
    
    timeline = []
    
    for i, year in enumerate(years):
        if i == 0:
            # First year - all risks are "new"
            timeline.append({
                "year": year,
                "status": "baseline",
                "risks": risk_data_by_year[year]
            })
        else:
            # Compare with previous year
            prev_year = years[i-1]
            comparison = compare_risks(
                risk_data_by_year[prev_year],
                risk_data_by_year[year],
                f"{company} {prev_year}",
                f"{company} {year}"
            )
            
            timeline.append({
                "year": year,
                "status": "comparison",
                "added_risks": comparison["added_risks"],
                "removed_risks": comparison["removed_risks"],
                "summary": comparison["summary"]
            })
    
    return {
        "company": company,
        "timeline": timeline,
        "years_analyzed": years
    }

def _normalize_risk(risk_name: str) -> str:
    """Normalize risk names for comparison"""
    return risk_name.lower().strip().replace(" ", "_")

def export_risk_comparison(
    comparison: Dict[str, Any],
    format: str = "csv"
) -> str:
    """
    Export risk comparison to CSV format
    
    Args:
        comparison: Comparison results
        format: Export format (csv, json)
    
    Returns:
        Formatted string
    """
    
    if format == "csv":
        # Create CSV content
        lines = []
        lines.append("Risk,Status,Description,Mitigation,Citations")
        
        # Added risks
        for risk in comparison["added_risks"]:
            lines.append(f'"{risk["risk"]}",Added,"{risk["description"]}","{risk["mitigation"]}","{risk["citations"]}"')
        
        # Removed risks  
        for risk in comparison["removed_risks"]:
            lines.append(f'"{risk["risk"]}",Removed,"{risk["description"]}","{risk["mitigation"]}","{risk["citations"]}"')
        
        return "\n".join(lines)
    
    elif format == "json":
        import json
        return json.dumps(comparison, indent=2)
    
    return str(comparison)
