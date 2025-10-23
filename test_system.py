#!/usr/bin/env python3
"""
Test script to verify the PDF ingestion and summarization system works.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.orchestrator import run_qa

def test_system():
    """Test the system with a simple query."""
    print("Testing PDF ingestion and summarization system...")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY not set. Please set your OpenAI API key in .env file.")
        print("   Example: OPENAI_API_KEY=sk-your-actual-key-here")
        return False
    
    # Check if index exists
    index_path = Path("data/index/index.faiss")
    if not index_path.exists():
        print("❌ FAISS index not found. Please run the ingestion pipeline first:")
        print("   python scripts/01_ingest.py")
        print("   python scripts/02_build_index.py")
        return False
    
    try:
        # Test with a simple query
        print("Running test query...")
        result = run_qa(
            question="What are the main climate risks mentioned?",
            company="Chevron",
            years=[2024],
            top_k=5
        )
        
        print(f"✅ System working! Answer: {result['answer']}")
        print(f"   Retrieved {len(result['retrieved'])} sources")
        print(f"   Iterations: {result.get('iterations', 1)}")
        print(f"   Approved: {result['critic']['approved']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
