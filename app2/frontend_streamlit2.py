#!/usr/bin/env python3
"""
Unified Frontend - Single interface for both Q&A and Extraction workflows
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import streamlit as st


# Ensure project root is on sys.path so absolute `app2.*` imports work when Streamlit runs from app2/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app2.orchestrator2 import run_unified
from app2.rag.risk_diff import compare_risks, create_risk_timeline, export_risk_comparison
import pandas as pd

# Helper functions for displaying results
def _display_qa_results(result: dict):
    """Display Q&A workflow results"""
    st.subheader("🤖 Answer")
    st.write(result["answer"])
    st.caption(f"Iterations: {result.get('iterations', 1)} | Approved: {result['critic']['approved']}")
    
    if result["critic"]["feedback"]:
        st.warning("Critic feedback:")
        for fb in result["critic"]["feedback"]:
            st.write(f"- Sentence {fb['idx']}: {fb['issue']} — {fb['message']}")
    
    st.subheader("📚 Retrieved Sources")
    for c in result["retrieved"]:
        st.write(f"- **{c['company']} {c['year']}** — {c['doc']} — p.{c['page_start']}")
        with st.expander("Preview chunk"):
            st.code(c["text"][:1200])

def _display_extraction_results(result: dict):
    """Display extraction workflow results"""
    workflow_type = result.get("workflow_type", "single")
    
    if workflow_type == "cross_company":
        _display_cross_company_results(result)
    elif workflow_type == "timeseries":
        _display_timeseries_results(result)
    else:
        _display_single_company_results(result)

def _display_single_company_results(result: dict):
    """Display single company extraction results"""
    if result["success"]:
        actual_count = len(result.get("table", [])) if result.get("table") else 0
        st.success(f"✅ Extracted {actual_count} risk factors")
        
        # Display the table
        if result.get("table"):
            st.subheader("📋 Risk Factors Table")
            df = pd.DataFrame(result["table"])
            st.dataframe(df, use_container_width=True)
            
            # CSV download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"risk_factors_{company}_{default_year}.csv",
                mime="text/csv"
            )
    else:
        st.error(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")

def _display_cross_company_results(result: dict):
    """Display cross-company comparison results"""
    st.subheader("🔄 Cross-Company Comparison: Chevron vs ExxonMobil 2024")
    
    # Show Chevron risks
    st.write("**Chevron 2024 Risks**")
    if result.get("chevron_table"):
        df_chevron = pd.DataFrame(result["chevron_table"])
        st.dataframe(df_chevron, use_container_width=True)
    
    # Show ExxonMobil risks
    st.write("**ExxonMobil 2024 Risks**")
    if result.get("exxon_table"):
        df_exxon = pd.DataFrame(result["exxon_table"])
        st.dataframe(df_exxon, use_container_width=True)
    
    # Comparison summary
    st.subheader("📊 Comparison Summary")
    chevron_count = len(result.get("chevron_table", []))
    exxon_count = len(result.get("exxon_table", []))
    st.write(f"**Chevron**: {chevron_count} risk factors")
    st.write(f"**ExxonMobil**: {exxon_count} risk factors")
    
    # Combined CSV download
    if result.get("chevron_table") and result.get("exxon_table"):
        combined_data = []
        for row in result["chevron_table"]:
            combined_data.append({**row, "Company": "Chevron"})
        for row in result["exxon_table"]:
            combined_data.append({**row, "Company": "ExxonMobil"})
        
        df_combined = pd.DataFrame(combined_data)
        csv = df_combined.to_csv(index=False)
        st.download_button(
            label="📥 Download Combined CSV",
            data=csv,
            file_name="chevron_vs_exxon_risks_2024.csv",
            mime="text/csv"
        )

def _display_timeseries_results(result: dict):
    """Display time-series analysis results"""
    st.subheader("📈 Time-Series Analysis: ExxonMobil 2022-2024")
    
    # Show each year's risks
    tables = result.get("tables", {})
    for year, table in tables.items():
        st.write(f"**{year} Risks**")
        if table:
            df = pd.DataFrame(table)
            st.dataframe(df, use_container_width=True)
    
    # Risk evolution summary
    st.subheader("📊 Risk Evolution Summary")
    metadata = result.get("metadata", {})
    risk_counts = metadata.get("risk_counts", {})
    for year, count in risk_counts.items():
        st.write(f"**{year}**: {count} risk factors")
    
    # Time-series CSV download
    all_data = []
    for year, table in tables.items():
        for row in table:
            all_data.append({**row, "Year": year})
    
    if all_data:
        df_timeseries = pd.DataFrame(all_data)
        csv = df_timeseries.to_csv(index=False)
        st.download_button(
            label="📥 Download Time-Series CSV",
            data=csv,
            file_name="exxon_risks_2022_2024.csv",
            mime="text/csv"
        )

def _display_unified_trace(result: dict):
    """Display unified trace information"""
    st.subheader("🔍 Unified Workflow Trace")
    
    trace = result.get("trace", {})
    workflow_type = result.get("workflow_type", "unknown")
    
    # Show current hyperparameters
    st.info(f"🎛️ **Current Settings**: Temperature={temperature}, Top-K={top_k}")
    
    # Planner information
    if "planner" in trace:
        planner_info = trace["planner"]
        st.write("**🧠 Planner Agent**")
        st.write(f"• **Decision**: {planner_info.get('path', 'unknown').title()} workflow")
        st.write(f"• **Confidence**: {planner_info.get('confidence', 0):.1%}")
        st.write(f"• **Reasoning**: {planner_info.get('reasoning', 'N/A')}")
        
        if "_trace" in planner_info:
            planner_trace = planner_info["_trace"]
            st.write(f"• **Provider**: {planner_trace.get('provider', 'unknown')} ({planner_trace.get('model', 'unknown')})")
    
    # Workflow information
    if "workflow" in trace:
        workflow_trace = trace["workflow"]
        st.write("**⚙️ Workflow Execution**")
        
        if workflow_type == "qa":
            # Q&A workflow trace
            if "providers" in workflow_trace:
                for agent, info in workflow_trace["providers"].items():
                    st.write(f"• **{agent.title()}**: {info.get('provider', 'unknown')} ({info.get('model', 'unknown')})")
            
            if "retriever" in workflow_trace:
                ret_stats = workflow_trace["retriever"]
                st.write(f"• **Retriever**: {ret_stats.get('results', 0)} chunks, {ret_stats.get('time_s', 0):.2f}s")
        
        elif workflow_type == "extraction":
            # Extraction workflow trace
            if "extractor" in workflow_trace:
                ext_trace = workflow_trace["extractor"]
                st.write(f"• **Extractor**: {ext_trace.get('provider', 'unknown')} ({ext_trace.get('model', 'unknown')})")
                st.write(f"• **Chunks Processed**: {ext_trace.get('chunks_processed', 0)}")
    
    # Workflow type indicator
    st.write(f"**🎯 Workflow Type**: {workflow_type.title()}")
    
    # Show detailed trace if available
    if st.expander("🔍 Detailed Trace (JSON)"):
        st.json(trace)

st.set_page_config(page_title="10-K Unified Multi-Agent System", layout="wide")
st.title("🤖 10-K Unified Multi-Agent System")
st.caption("Intelligent routing: Ask questions or request structured analysis")

# derive available companies/years from ingest report (fallbacks if missing)
report_path = Path("data/processed/ingest_report.json")
company_years = {}
if report_path.exists():
    r = json.loads(report_path.read_text())
    for row in r:
        company_years.setdefault(row["company"], set()).add(row["year"])
companys = sorted(company_years.keys()) or ["ExxonMobil", "Chevron"]

# Sidebar configuration
with st.sidebar:
    st.header("🎛️ Configuration")
    
    # Company and year selection
    company = st.selectbox("Company", companys, index=0)
    years_list = sorted(company_years.get(company, {2024}))
    years = st.multiselect("Years", years_list, default=years_list)
    
    st.markdown("---")
    st.subheader("🤖 Agent Settings")
    
    # Temperature control with session state
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    temperature = st.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=st.session_state.temperature, 
        step=0.1,
        help="Controls creativity/randomness. 0.0 = deterministic, 1.0 = most creative"
    )
    st.session_state.temperature = temperature
    
    # Top-K control with session state
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 7
    top_k = st.slider(
        "Top-K passages", 
        min_value=3, 
        max_value=10, 
        value=st.session_state.top_k, 
        step=1,
        help="Number of document chunks to retrieve for context"
    )
    st.session_state.top_k = top_k
    
    # Optimal configurations from analysis
    st.markdown("---")
    st.subheader("🏆 Optimal Configurations")
    st.caption("Based on 56-experiment sweep analysis")
    
    if st.button("🎯 Zero-Cost Optimal", help="Best overall: Ollama all, temp=1.0, top_k=3"):
        st.session_state.temperature = 1.0
        st.session_state.top_k = 3
        st.session_state.provider = "ollama"
        st.session_state.model = "llama3.1:8b"
        st.rerun()
    
    if st.button("⚡ Cost-Effective Optimal", help="Best cost-effective: OpenAI+Ollama, temp=0.3, top_k=7"):
        st.session_state.temperature = 0.3
        st.session_state.top_k = 7
        st.session_state.provider = "openai"
        st.session_state.model = "gpt-4o-mini"
        st.rerun()
    
    if st.button("🌡️ High Creativity", help="High creativity: temp=1.0, top_k=7"):
        st.session_state.temperature = 1.0
        st.session_state.top_k = 7
        st.rerun()
    
    if st.button("🎯 Focused Precision", help="Focused: temp=0.2, top_k=3"):
        st.session_state.temperature = 0.2
        st.session_state.top_k = 3
        st.rerun()
    
    st.markdown("---")
    st.caption("Tip: queries containing 'risk' bias retrieval to Item 1A.")
    st.markdown("---")
    st.subheader("Models")
    
    # Provider selection with session state
    if 'provider' not in st.session_state:
        st.session_state.provider = "openai"
    
    provider = st.selectbox(
        "Main Agent Provider", 
        ["openai", "ollama"], 
        index=0 if st.session_state.provider == "openai" else 1
    )
    st.session_state.provider = provider
    
    # Model selection with session state
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-4o-mini" if provider == "openai" else "llama3.1:8b"
    
    model = st.text_input(
        "Main Agent Model", 
        value=st.session_state.model
    )
    st.session_state.model = model

default_year = max(years) if years else (max(years_list) if years_list else 2024)

# Main interface
st.header("💬 Unified Query Interface")
st.caption("Ask questions or request structured analysis - the system will automatically route to the appropriate workflow")

# Unified textarea
question = st.text_area(
    "Ask a question or request analysis",
    value=f"Summarize physical climate risks and controls for {company} {default_year}.",
    height=120,
    help="Examples: 'Summarize climate risks' (Q&A) or 'Compare risk factors between companies' (Extraction)"
)

# Example queries
st.subheader("💡 Example Queries")
st.caption("Click to copy any of these example queries:")

# Q&A Examples
st.write("**🤖 Q&A Examples (Conversational)**")
qa_examples = [
    f"Summarize physical climate risks and controls for {company} {default_year}.",
    f"What are the transition risks and mitigation strategies for {company}?",
    f"How does {company} address greenhouse gas emissions and climate change?",
    f"What climate-related regulations and policies affect {company}?"
]

for i, example in enumerate(qa_examples):
    st.code(example, language=None)

# Extraction Examples  
st.write("**📊 Extraction Examples (Structured)**")
extraction_examples = [
    f"Compare risk factors between {company} and Chevron for 2024",
    f"Extract risk table for {company} {default_year}",
    f"Track how {company}'s risks evolved from 2022 to 2024",
    f"Export risk factors to CSV for {company}"
]

for i, example in enumerate(extraction_examples):
    st.code(example, language=None)

# Run button
if st.button("🚀 Run Analysis", type="primary"):
    if not years:
        st.error("Select at least one year.")
    else:
        with st.spinner("🤖 Planning and executing workflow..."):
            result = run_unified(
                question, 
                company=company, 
                years=years, 
                top_k=top_k, 
                provider=provider, 
                model=model, 
                temperature=temperature
            )
        
        # Display workflow decision
        st.subheader("🎯 Workflow Decision")
        workflow_type = result.get("workflow_type", "unknown")
        confidence = result.get("planning_confidence", 0.0)
        reasoning = result.get("planning_reasoning", "")
        
        if workflow_type == "extraction":
            st.success(f"📊 **Extraction Workflow** (Confidence: {confidence:.1%})")
        else:
            st.success(f"🤖 **Q&A Workflow** (Confidence: {confidence:.1%})")
        
        st.caption(f"Reasoning: {reasoning}")
        
        # Display results based on workflow type
        if workflow_type in ["extraction", "timeseries", "cross_company", "single"]:
            _display_extraction_results(result)
        else:
            _display_qa_results(result)
        
        # Display unified trace
        _display_unified_trace(result)

