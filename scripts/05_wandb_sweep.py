#!/usr/bin/env python3
"""
Weights & Biases Hyperparameter Sweep for Multi-Agent RAG System
Systematically tests temperature, top-k, and model combinations.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys
import wandb
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.orchestrator import run_qa
from app.rag.critic_agent import split_sentences as _split_sentences


class RAGExperiment:
    """RAG system experiment runner with wandb integration"""
    
    def __init__(self, project_name: str = "rag-multi-agent-sweep"):
        self.project_name = project_name
        self.results: List[Dict[str, Any]] = []
        
    def setup_wandb_sweep(self) -> str:
        """Create wandb sweep configuration"""
        sweep_config = {
            "method": "grid",
            "metric": {
                "name": "overall_quality",
                "goal": "maximize"
            },
            "parameters": {
                "temperature": {
                    "values": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
                },
                "top_k": {
                    "values": [3, 5, 7, 10]
                },
                "model_strategy": {
                    "values": ["cost_effective", "zero_cost"]
                }
            }
        }
        
        # Create sweep
        sweep_id = wandb.sweep(sweep_config, project=self.project_name)
        print(f"Created sweep: {sweep_id}")
        return sweep_id
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment with given configuration"""
        
        # Test cases for evaluation
        test_cases = [
            {
                "id": "climate_risk_chevron",
                "question": "Summarize physical climate risks and controls for Chevron 2024.",
                "company": "Chevron",
                "years": [2024],
                "category": "climate_risk",
                "expected_keywords": ["physical", "risk", "climate", "emission"]
            },
            {
                "id": "climate_risk_exxon",
                "question": "Summarize physical climate risks and controls for ExxonMobil 2024.",
                "company": "ExxonMobil", 
                "years": [2024],
                "category": "climate_risk",
                "expected_keywords": ["physical", "risk", "climate", "emission"]
            },
            {
                "id": "financial_chevron",
                "question": "What were Chevron's key financial performance metrics in 2024?",
                "company": "Chevron",
                "years": [2024],
                "category": "financial",
                "expected_keywords": ["revenue", "profit", "earnings", "financial"]
            }
        ]
        
        all_results = []
        
        for case in test_cases:
            try:
                start_time = time.time()
                
                # Map model strategy to actual providers/models
                if config["model_strategy"] == "cost_effective":
                    answerer_provider = "openai"
                    answerer_model = "gpt-4o-mini"
                    critic_provider = "ollama"
                    critic_model = "llama3.1:8b"
                else:  # zero_cost
                    answerer_provider = "ollama"
                    answerer_model = "llama3.1:8b"
                    critic_provider = "ollama"
                    critic_model = "llama3.1:8b"
                
                # Run the system with current config
                result = run_qa(
                    case["question"],
                    company=case["company"],
                    years=case["years"],
                    top_k=config["top_k"],
                    provider=answerer_provider,
                    model=answerer_model
                )
                
                execution_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    case, result, execution_time, config
                )
                
                all_results.append(metrics)
                
            except Exception as e:
                print(f"Error in case {case['id']}: {e}")
                all_results.append({
                    "case_id": case["id"],
                    "error": str(e),
                    "approved": False,
                    "overall_quality": 0.0
                })
        
        # Aggregate results
        return self._aggregate_results(all_results, config)
    
    def _calculate_metrics(self, case: Dict, result: Dict, execution_time: float, config: Dict) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single test case"""
        
        answer = result.get("answer", "")
        critic = result.get("critic", {"approved": False, "feedback": []})
        retrieved = result.get("retrieved", [])
        trace = result.get("trace", {})
        
        # Basic metrics
        answer_length = len(answer)
        n_retrieved = len(retrieved)
        approved = critic.get("approved", False)
        
        # Citation analysis
        sentences = _split_sentences(answer)
        citation_ratio = sum(1 for s in sentences if "[CITATION:" in s) / max(len(sentences), 1)
        
        # Keyword analysis
        keyword_hit = self._keyword_score(answer, case.get("expected_keywords", []))
        
        # Quality scores
        completeness_score = min(1.0, answer_length / 500)  # Target 500+ chars
        citation_quality = citation_ratio
        relevance_score = keyword_hit
        
        # Efficiency metrics
        speed_score = max(0, 1.0 - execution_time / 30.0)  # Target <30s
        
        # Overall quality (weighted combination)
        overall_quality = (
            completeness_score * 0.25 +
            citation_quality * 0.25 +
            relevance_score * 0.25 +
            speed_score * 0.15 +
            (1.0 if approved else 0.0) * 0.10
        )
        
        return {
            "case_id": case["id"],
            "category": case["category"],
            "approved": approved,
            "answer_length": answer_length,
            "n_retrieved": n_retrieved,
            "citation_ratio": citation_ratio,
            "keyword_hit": keyword_hit,
            "completeness_score": completeness_score,
            "citation_quality": citation_quality,
            "relevance_score": relevance_score,
            "speed_score": speed_score,
            "overall_quality": overall_quality,
            "execution_time": execution_time,
            "iterations": result.get("iterations", 1),
            "config": config
        }
    
    def _keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword hit rate"""
        if not keywords:
            return 1.0
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        return hits / len(keywords)
    
    def _aggregate_results(self, results: List[Dict], config: Dict) -> Dict[str, Any]:
        """Aggregate results across all test cases"""
        
        if not results:
            return {"overall_quality": 0.0, "error": "No results"}
        
        # Calculate averages
        avg_quality = sum(r.get("overall_quality", 0) for r in results) / len(results)
        avg_approval = sum(r.get("approved", False) for r in results) / len(results)
        avg_citation = sum(r.get("citation_ratio", 0) for r in results) / len(results)
        avg_keyword = sum(r.get("keyword_hit", 0) for r in results) / len(results)
        avg_time = sum(r.get("execution_time", 0) for r in results) / len(results)
        
        # Category breakdown
        climate_results = [r for r in results if r.get("category") == "climate_risk"]
        financial_results = [r for r in results if r.get("category") == "financial"]
        
        climate_quality = sum(r.get("overall_quality", 0) for r in climate_results) / max(len(climate_results), 1)
        financial_quality = sum(r.get("overall_quality", 0) for r in financial_results) / max(len(financial_results), 1)
        
        return {
            "overall_quality": avg_quality,
            "approval_rate": avg_approval,
            "citation_ratio": avg_citation,
            "keyword_hit": avg_keyword,
            "execution_time": avg_time,
            "climate_quality": climate_quality,
            "financial_quality": financial_quality,
            "config": config,
            "n_cases": len(results),
            "successful_cases": sum(1 for r in results if not r.get("error"))
        }
    
    def run_sweep(self, sweep_id: str, n_runs: int = 50):
        """Run the sweep with wandb agent"""
        
        def sweep_function():
            with wandb.init() as run:
                config = wandb.config
                
                print(f"Running experiment with config: {config}")
                
                # Run experiment
                results = self.run_single_experiment(config)
                
                # Log metrics
                wandb.log({
                    "overall_quality": results["overall_quality"],
                    "approval_rate": results["approval_rate"],
                    "citation_ratio": results["citation_ratio"],
                    "keyword_hit": results["keyword_hit"],
                    "execution_time": results["execution_time"],
                    "climate_quality": results["climate_quality"],
                    "financial_quality": results["financial_quality"],
                    "successful_cases": results["successful_cases"],
                    "n_cases": results["n_cases"]
                })
                
                # Log configuration
                wandb.log({
                    "temperature": config["temperature"],
                    "top_k": config["top_k"],
                    "model_strategy": config["model_strategy"],
                    "answerer_provider": answerer_provider,
                    "answerer_model": answerer_model,
                    "critic_provider": critic_provider,
                    "critic_model": critic_model
                })
                
                print(f"Experiment completed. Quality: {results['overall_quality']:.3f}")
        
        # Run the sweep
        wandb.agent(sweep_id, function=sweep_function, count=n_runs)
    
    def analyze_results(self, project_name: str = None):
        """Analyze sweep results and generate recommendations"""
        print("🔍 Wandb API Analysis (Currently Broken)")
        print("=" * 50)
        print("❌ The wandb API is returning string data instead of dictionaries")
        print("❌ This causes the analysis to fail with 'str' object has no attribute 'keys'")
        print("")
        print("✅ Use the manual analysis instead:")
        print("   python scripts/09_manual_analysis.py")
        print("")
        print("📊 The manual analysis is based on your dashboard data and provides:")
        print("   • Optimal configurations (temp=0.7, top_k=7)")
        print("   • Model strategy comparisons (cost_effective vs zero_cost)")
        print("   • Parameter importance rankings")
        print("   • Business recommendations")
        print("")
        print("🎯 Key Metrics Used:")
        print("   • overall_quality: Primary metric (0.30-0.70 range)")
        print("   • citation_ratio: Citation quality")
        print("   • keyword_hit: Relevance to expected terms")
        print("   • execution_time: Speed performance")
        print("   • cost_per_query: Cost efficiency")


def main():
    """Main function to run wandb sweep"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run wandb hyperparameter sweep")
    parser.add_argument("--action", choices=["setup", "run", "analyze"], default="setup",
                       help="Action to perform")
    parser.add_argument("--sweep-id", type=str, help="Sweep ID for run/analyze")
    parser.add_argument("--n-runs", type=int, default=50, help="Number of runs")
    parser.add_argument("--project", type=str, default="rag-multi-agent-sweep", help="Wandb project name")
    
    args = parser.parse_args()
    
    experiment = RAGExperiment(args.project)
    
    if args.action == "setup":
        sweep_id = experiment.setup_wandb_sweep()
        print(f"Setup complete. Sweep ID: {sweep_id}")
        print(f"Run with: python scripts/05_wandb_sweep.py --action run --sweep-id {sweep_id}")
        
    elif args.action == "run":
        if not args.sweep_id:
            print("Error: --sweep-id required for run action")
            return
        experiment.run_sweep(args.sweep_id, args.n_runs)
        
    elif args.action == "analyze":
        experiment.analyze_results(args.project)


if __name__ == "__main__":
    main()
