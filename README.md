# Multi-Agent RAG System for 10-K Document Analysis

A sophisticated multi-agent RAG (Retrieval-Augmented Generation) system for analyzing and summarizing 10-K financial documents, featuring intelligent workflow routing, hyperparameter optimization, and comprehensive evaluation capabilities.

## 🚀 Quick Start

1. **Set up your environment:**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file and add your API keys
   OPENAI_API_KEY=sk-your-actual-key-here
   OLLAMA_HOST=http://localhost:11434  # For local Ollama
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

4. **Process your documents:**
   ```bash
   python scripts/01_ingest.py    # Parse PDFs into chunks
   python scripts/02_build_index.py  # Build search index
   ```

5. **Launch the web interface:**
   ```bash
   # Original system (dual-tab interface)
   streamlit run app/frontend_streamlit.py
   
   # Unified system (intelligent routing)
   streamlit run app2/frontend_streamlit2.py
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
├── app/                         # Original system
│   ├── frontend_streamlit.py    # Dual-tab web interface
│   ├── orchestrator.py          # Q&A coordination logic
│   ├── rag/                     # RAG system components
│   │   ├── qa_agent.py         # Question answering
│   │   ├── enhanced_critic.py  # Answer validation
│   │   ├── extractor_agent.py  # Risk table extraction
│   │   ├── retriever.py        # Document retrieval
│   │   └── risk_diff.py        # Risk comparison utilities
│   └── ingest/                  # PDF processing
│       ├── parse_pdf.py        # PDF parsing
│       └── build_index.py      # Index building
├── app2/                        # Unified multi-agent system
│   ├── frontend_streamlit2.py   # Intelligent routing interface
│   ├── orchestrator2.py         # Unified orchestration
│   └── rag/                     # Enhanced agents
│       ├── planner_agent.py     # Intelligent workflow routing
│       ├── qa_agent.py         # Question answering
│       ├── enhanced_critic.py  # Answer validation
│       ├── extractor_agent.py  # Risk table extraction
│       └── retriever.py        # Document retrieval
├── data/
│   ├── raw/                     # Input PDFs
│   ├── processed/               # Parsed chunks
│   ├── index/                   # FAISS search index
│   └── eval/                    # Evaluation results
├── scripts/                     # Processing & evaluation
│   ├── 01_ingest.py            # PDF ingestion
│   ├── 02_build_index.py       # Index building
│   ├── 05_wandb_sweep.py       # Hyperparameter optimization
│   └── 08_wandb_export_analysis.py  # Results analysis
├── .notes/                      # Documentation & analysis
└── main.py                     # Entry point
```

## 🎯 Features

### **Core Capabilities**
- **Intelligent Retrieval**: Climate-focused document search with query expansion
- **Multi-Agent Architecture**: Planner, QA Agent, Critic, and Extractor agents
- **Dual System Design**: Original system + Unified intelligent routing
- **Citation Tracking**: Every claim backed by source citations with validation
- **Quality Control**: Automated answer validation with retry logic

### **Advanced Features**
- **Intelligent Workflow Routing**: AI-powered decision making between Q&A and extraction
- **Hyperparameter Optimization**: Systematic tuning with Weights & Biases
- **Risk Factor Extraction**: Structured table generation from unstructured text
- **Cross-Company Analysis**: Compare risk factors between companies
- **Time-Series Analysis**: Track risk evolution over multiple years
- **Comprehensive Evaluation**: Multi-metric assessment with detailed reporting

### **Technical Features**
- **Multi-LLM Support**: OpenAI GPT-4o-mini + Ollama llama3.1:8b
- **Vector Search**: FAISS index with 384D embeddings
- **Batch Processing**: Efficient document processing pipeline
- **Real-time Monitoring**: Performance dashboards and trace information

## 🔍 Usage Examples

### **Q&A Queries**
- "Summarize physical climate risks for Chevron 2024"
- "What are the main transition risks mentioned in the documents?"
- "How does ExxonMobil address carbon emissions in their 2024 report?"

### **Extraction Queries**
- "Extract risk factors for ExxonMobil 2024"
- "Compare risk factors between Chevron and ExxonMobil for 2024"
- "Track how ExxonMobil's risks evolved from 2022 to 2024"

### **System Comparison**
```bash
# Original system (manual routing)
streamlit run app/frontend_streamlit.py
# - Separate tabs for Q&A and Risk Extractor
# - Manual selection of workflow

# Unified system (intelligent routing)
streamlit run app2/frontend_streamlit2.py
# - Single interface with AI-powered routing
# - Automatic workflow selection based on query intent
```

## 🛠️ Development

### **Technology Stack**
- **Vector Search**: FAISS with sentence-transformers/all-MiniLM-L6-v2
- **LLMs**: OpenAI GPT-4o-mini + Ollama llama3.1:8b
- **Web Interface**: Streamlit with real-time dashboards
- **PDF Processing**: PyMuPDF + pdfplumber for text and table extraction
- **Evaluation**: Weights & Biases for hyperparameter optimization
- **Data Processing**: Pandas, NumPy for structured data manipulation

### **Hyperparameter Optimization**
```bash
# Run hyperparameter sweep
python scripts/05_wandb_sweep.py --action setup
python scripts/05_wandb_sweep.py --action run --sweep-id YOUR_SWEEP_ID

# Analyze results
python scripts/08_wandb_export_analysis.py
```

### **Evaluation Metrics**
- **Overall Quality**: Weighted combination of completeness, citation quality, relevance, and speed
- **Citation Ratio**: Percentage of sentences with proper citations
- **Keyword Hit Rate**: Relevance to expected climate/risk terms
- **Approval Rate**: Percentage of answers passing critic validation
- **Execution Time**: End-to-end processing speed

## 📊 Performance Analysis

The system includes comprehensive evaluation capabilities:
- **Hyperparameter Sweeps**: Temperature, top-k, and model strategy optimization
- **Multi-Metric Assessment**: Quality, speed, and cost analysis
- **Comparative Studies**: Original vs unified system performance
- **Detailed Reporting**: Markdown reports with visualizations and recommendations

## 📝 Notes

- **Dual System Architecture**: Original system preserved for comparison, unified system for production
- **Climate-Focused**: Enhanced retrieval for climate-related queries
- **Citation Validation**: Automatic validation with ±1 page tolerance
- **Quality Assurance**: Multi-level validation with retry logic
- **Scalable Design**: Modular architecture supporting easy extension
