#!/usr/bin/env python3
"""
Main entry point for the PDF Summarization System.
"""
import sys
from pathlib import Path

def main():
    print("📄 PDF Summarization System")
    print("=" * 40)
    print()
    print("Available commands:")
    print("  python setup.py           - Check system setup")
    print("  python test_system.py      - Test the system")
    print("  python scripts/01_ingest.py - Ingest PDFs")
    print("  python scripts/02_build_index.py - Build search index")
    print("  streamlit run app/frontend_streamlit.py - Launch web interface")
    print()
    print("For help, see README.md")

if __name__ == "__main__":
    main()
