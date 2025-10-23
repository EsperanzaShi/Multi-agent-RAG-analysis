#!/usr/bin/env python3
"""
Test script for the unified orchestration system
"""
from app2.orchestrator2 import run_unified
from app2.rag.planner_agent import plan_workflow_simple

def test_planner_routing():
    """Test the planner agent routing decisions"""
    print("🧠 Testing Planner Agent Routing")
    print("=" * 50)
    
    test_queries = [
        ("Summarize climate risks for ExxonMobil", "Q&A"),
        ("What are the transition risks?", "Q&A"), 
        ("Compare risk factors between Chevron and ExxonMobil", "Extraction"),
        ("Extract risk table for 2024", "Extraction"),
        ("Track how risks evolved from 2022-2024", "Extraction"),
        ("How does ExxonMobil address emissions?", "Q&A")
    ]
    
    for query, expected in test_queries:
        result = plan_workflow_simple(query)
        decision = result["path"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]
        
        status = "✅" if decision == expected.lower() else "❌"
        print(f"{status} Query: {query}")
        print(f"   Decision: {decision} (confidence: {confidence:.1%})")
        print(f"   Expected: {expected.lower()}")
        print(f"   Reasoning: {reasoning}")
        print()

def test_unified_orchestrator():
    """Test the unified orchestrator (without actually running agents)"""
    print("🎯 Testing Unified Orchestrator")
    print("=" * 50)
    
    # Test planning integration
    from app2.rag.planner_agent import plan_workflow
    
    try:
        # Test with simple planning first
        plan = plan_workflow_simple("Summarize climate risks")
        print(f"✅ Planner integration working")
        print(f"   Decision: {plan['path']}")
        print(f"   Confidence: {plan['confidence']:.1%}")
    except Exception as e:
        print(f"❌ Planner integration failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing Unified Multi-Agent System")
    print("=" * 60)
    print()
    
    test_planner_routing()
    test_unified_orchestrator()
    
    print("✅ All tests completed!")
    print()
    print("🎯 To run the unified frontend:")
    print("   streamlit run app2/frontend_streamlit2.py")
    print()
    print("📊 Original system still available:")
    print("   streamlit run app/frontend_streamlit.py")
