#!/usr/bin/env python3
"""
Setup script for the PDF ingestion and summarization system.
"""
import os
import sys
from pathlib import Path

def main():
    print("🔧 Setting up PDF Summarization System")
    print("=" * 50)
    
    # Check if .env exists
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found. Creating template...")
        with open(env_path, "w") as f:
            f.write("# OpenAI API Key - Replace with your actual API key\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n\n")
            f.write("# Optional: Specify the model to use for answering (defaults to gpt-4o-mini)\n")
            f.write("ANSWERER_MODEL=gpt-4o-mini\n")
        print("✅ Created .env template. Please edit it with your OpenAI API key.")
    else:
        print("✅ .env file exists")
    
    # Check if PDFs exist
    raw_dir = Path("data/raw")
    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No PDFs found in data/raw/")
        print("   Please add your 10-K PDF files to data/raw/ directory")
        return False
    else:
        print(f"✅ Found {len(pdfs)} PDF files")
        for pdf in pdfs:
            print(f"   - {pdf.name}")
    
    # Check if processed data exists
    chunks_path = Path("data/processed/chunks.jsonl")
    if not chunks_path.exists():
        print("❌ Processed chunks not found. Running ingestion...")
        print("   Run: python scripts/01_ingest.py")
        return False
    else:
        print("✅ Processed chunks found")
    
    # Check if index exists
    index_path = Path("data/index/index.faiss")
    if not index_path.exists():
        print("❌ FAISS index not found. Building index...")
        print("   Run: python scripts/02_build_index.py")
        return False
    else:
        print("✅ FAISS index found")
    
    print("\n🎉 Setup complete! You can now run:")
    print("   streamlit run app/frontend_streamlit.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
