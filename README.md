# PDF Summarization System

A sophisticated RAG (Retrieval-Augmented Generation) system for analyzing and summarizing 10-K financial documents, with a focus on climate risk disclosure.

## 🚀 Quick Start

1. **Set up your OpenAI API key:**
   ```bash
   # Edit .env file and add your OpenAI API key
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

2. **Add your PDF files:**
   ```bash
   # Place your 10-K PDF files in data/raw/
   cp your-10k-files.pdf data/raw/
   ```

3. **Run the setup:**
   ```bash
   python setup.py
   ```

4. **If needed, run ingestion:**
   ```bash
   python scripts/01_ingest.py    # Parse PDFs into chunks
   python scripts/02_build_index.py  # Build search index
   ```

5. **Launch the web interface:**
   ```bash
   streamlit run app/frontend_streamlit.py
   ```

## 🔧 Troubleshooting

### "Insufficient support from retrieved context" Error

This error occurs when:
1. **Missing OpenAI API Key**: Set `OPENAI_API_KEY` in `.env` file
2. **No relevant content found**: The system couldn't find climate-related information
3. **Index not built**: Run the ingestion pipeline first

### Common Issues

1. **No PDFs found**: Add PDF files to `data/raw/` directory
2. **Index not found**: Run `python scripts/02_build_index.py`
3. **API errors**: Check your OpenAI API key and billing

## 📁 Project Structure

```
├── app/
│   ├── frontend_streamlit.py    # Web interface
│   ├── orchestrator.py          # Main coordination logic
│   ├── rag/                     # RAG system components
│   │   ├── qa_agent.py         # Question answering
│   │   ├── critic_agent.py     # Answer validation
│   │   ├── retriever.py        # Document retrieval
│   │   └── prompts.py          # AI prompts
│   └── ingest/                  # PDF processing
│       ├── parse_pdf.py        # PDF parsing
│       └── build_index.py      # Index building
├── data/
│   ├── raw/                     # Input PDFs
│   ├── processed/               # Parsed chunks
│   └── index/                   # Search index
├── scripts/                     # Utility scripts
└── main.py                     # Entry point
```

## 🎯 Features

- **Intelligent Retrieval**: Climate-focused document search
- **Multi-document Analysis**: Compare across years and companies
- **Citation Tracking**: Every claim backed by source citations
- **Quality Control**: Automated answer validation
- **Web Interface**: Easy-to-use Streamlit frontend

## 🔍 Usage Examples

- "Summarize physical climate risks for Chevron 2024"
- "Compare climate disclosures between ExxonMobil 2022 and 2024"
- "What are the main transition risks mentioned in the documents?"

## 🛠️ Development

The system uses:
- **FAISS** for vector search
- **OpenAI GPT-4** for answer generation
- **Streamlit** for web interface
- **PyMuPDF** for PDF parsing

## 📝 Notes

- The system is optimized for 10-K financial documents
- Climate-related queries get enhanced retrieval
- All answers include proper citations
- The system validates answer quality automatically
