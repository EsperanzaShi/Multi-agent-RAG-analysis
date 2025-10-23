from __future__ import annotations
import json
from pathlib import Path
import sys
import streamlit as st

# Ensure project root is on sys.path so absolute `app.*` imports work when Streamlit runs from app/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.orchestrator import run_qa
from app.rag.extractor_agent import extract_risk_table
from app.rag.risk_diff import compare_risks, create_risk_timeline, export_risk_comparison
import pandas as pd

st.set_page_config(page_title="10-K Multi-Agent System", layout="wide")
st.title("10-K Multi-Agent System")

# Create tabs
tab1, tab2 = st.tabs(["🤖 Q&A Agent", "📊 Risk Extractor"])

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
    st.header("Filters")
    company = st.selectbox("Company", companys, index=0)
    years_list = sorted(company_years.get(company, {2024}))
    years = st.multiselect("Years (time-series)", years_list, default=years_list)
    
    st.markdown("---")
    st.subheader("🎛️ Hyperparameters")
    
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
        "Answerer provider", 
        ["openai", "ollama"], 
        index=0 if st.session_state.provider == "openai" else 1
    )
    st.session_state.provider = provider
    
    # Model selection with session state
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-4o-mini" if provider == "openai" else "llama3.1:8b"
    
    model = st.text_input(
        "Answerer model", 
        value=st.session_state.model
    )
    st.session_state.model = model

default_year = max(years) if years else (max(years_list) if years_list else 2024)

# Q&A Agent Tab
with tab1:
    st.header("🤖 Q&A Agent")
    
    # Simple textarea without complex state management
    question = st.text_area(
        "Ask a question",
        value=f"Summarize physical climate risks and controls for {company} {default_year}.",
        height=110,
        help="Enter your question about the company's 10-K documents"
    )

    # Example questions section
    st.subheader("💡 Example Questions")
    st.caption("Click to copy any of these example questions:")

    # Climate Risk Examples
    st.write("**🌍 Climate Risk Analysis**")
    climate_examples = [
        f"Summarize physical climate risks and controls for {company} {default_year}.",
        f"What are the transition risks and mitigation strategies for {company}?",
        f"How does {company} address greenhouse gas emissions and climate change?",
        f"What climate-related regulations and policies affect {company}?"
    ]

    for i, example in enumerate(climate_examples):
        st.code(example, language=None)

    # Financial Examples
    st.write("**💰 Financial Performance**")
    financial_examples = [
        f"What were {company}'s key financial performance metrics in {default_year}?",
        f"What are {company}'s main revenue sources and business segments?",
        f"How did {company} perform financially compared to previous years?",
        f"What are {company}'s key financial risks and challenges?"
    ]

    for i, example in enumerate(financial_examples):
        st.code(example, language=None)

    # Operational Examples
    st.write("**⚙️ Operations & Strategy**")
    operational_examples = [
        f"What are {company}'s main operational activities and capabilities?",
        f"What is {company}'s business strategy and competitive position?",
        f"What are {company}'s key operational risks and challenges?",
        f"How does {company} manage supply chain and operational efficiency?"
    ]

    for i, example in enumerate(operational_examples):
        st.code(example, language=None)

    # Simple run button
    if st.button("Run", type="primary"):
        if not years:
            st.error("Select at least one year.")
        else:
            with st.spinner("Thinking..."):
                out = run_qa(question, company=company, years=years, top_k=top_k, provider=provider, model=model, temperature=temperature)
            
            st.subheader("Answer")
            st.write(out["answer"])
            st.caption(f"Iterations: {out.get('iterations', 1)} | Approved: {out['critic']['approved']}")
            if out["critic"]["feedback"]:
                st.warning("Critic feedback:")
                for fb in out["critic"]["feedback"]:
                    st.write(f"- Sentence {fb['idx']}: {fb['issue']} — {fb['message']}")
            st.subheader("Retrieved Sources")
            for c in out["retrieved"]:
                st.write(f"- **{c['company']} {c['year']}** — {c['doc']} — p.{c['page_start']}")
                with st.expander("Preview chunk"):
                    st.code(c["text"][:1200])

            # Real-time performance dashboard panel
            st.subheader("Run Trace & Performance")
            
            # Show current hyperparameters
            st.info(f"🎛️ **Current Settings**: Temperature={temperature}, Top-K={top_k}")
            
            trace = out.get("trace", {})
            if trace:
                # Show all agents and their providers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Agent Providers**")
                    if "providers" in trace:
                        for agent, info in trace["providers"].items():
                            st.write(f"• **{agent.title()}**: {info.get('provider', 'unknown')} ({info.get('model', 'unknown')})")
                    
                    if "critic" in trace:
                        critic_info = trace["critic"]
                        st.write(f"• **Critic**: {critic_info.get('provider', 'unknown')} ({critic_info.get('model', 'unknown')})")
                        if critic_info.get('fallback'):
                            st.caption("⚠️ Used fallback rule-based validation")
                        if critic_info.get('error'):
                            st.caption(f"⚠️ Error: {critic_info['error']}")
                
                with col2:
                    st.write("**Retriever Stats**")
                    if "retriever" in trace:
                        ret_stats = trace["retriever"]
                        st.write(f"• **Query**: {ret_stats.get('query', 'N/A')[:50]}...")
                        st.write(f"• **Expanded**: {ret_stats.get('expanded', False)}")
                        st.write(f"• **Pool Size**: {ret_stats.get('pool', 0)}")
                        st.write(f"• **Candidates**: {ret_stats.get('candidates_considered', 0)}")
                        st.write(f"• **Results**: {ret_stats.get('results', 0)}")
                        st.write(f"• **Time**: {ret_stats.get('time_s', 0)}s")
                
                # Multi-iteration details
                if isinstance(trace.get("attempt1"), dict):
                    st.write("**Multi-Iteration Details**")
                    st.json(trace)
            else:
                st.caption("No trace available for this run.")

# Risk Extractor Tab
with tab2:
    st.header("📊 Risk Extractor")
    st.caption("Extract structured risk tables from 10-K documents")
    
    # Extractor controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Extraction Settings")
        extractor_provider = st.selectbox(
            "Extractor Provider", 
            ["openai", "ollama"], 
            index=0,
            key="extractor_provider"
        )
        extractor_model = st.text_input(
            "Extractor Model", 
            value="gpt-4o-mini" if extractor_provider == "openai" else "llama3.1:8b",
            key="extractor_model"
        )
        extractor_temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.1,
            help="Low temperature for structured output",
            key="extractor_temperature"
        )
    
    with col2:
        st.subheader("📈 Analysis Options")
        comparison_mode = st.selectbox(
            "Analysis Mode",
            ["Single Company", "Cross-Company", "Time-Series"],
            help="Choose analysis type"
        )
        
        if comparison_mode == "Single Company":
            st.info("Extract risks for the selected company and year(s)")
        elif comparison_mode == "Cross-Company":
            st.info("Compare Chevron 2024 vs ExxonMobil 2024 (automatically selected)")
        elif comparison_mode == "Time-Series":
            st.info("Track ExxonMobil risks from 2022-2024 (automatically selected)")
    
    # Extract button
    if st.button("🔍 Extract Risk Table", type="primary"):
        
        if comparison_mode == "Single Company":
            # Single company analysis
            with st.spinner("Extracting risk factors..."):
                result = extract_risk_table(
                    company=company,
                    years=years,
                    provider=extractor_provider,
                    model=extractor_model,
                    temperature=extractor_temperature
                )
            
            if result["success"]:
                actual_count = len(result["table"]) if result["table"] else 0
                st.success(f"✅ Extracted {actual_count} risk factors")
                
                # Display the table
                if result["table"]:
                    st.subheader("📋 Risk Factors Table")
                    df = pd.DataFrame(result["table"])
                    st.dataframe(df, use_container_width=True)
                    
                    # Download CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{company}_risks_{'-'.join(map(str, years))}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No risk factors found in the documents")
            else:
                st.error(f"❌ Extraction failed: {result['error']}")
        
        elif comparison_mode == "Cross-Company":
            # Cross-company comparison: Chevron 2024 vs ExxonMobil 2024
            with st.spinner("Extracting Chevron 2024 risks..."):
                result_chevron = extract_risk_table(
                    company="Chevron",
                    years=[2024],
                    provider=extractor_provider,
                    model=extractor_model,
                    temperature=extractor_temperature
                )
            
            with st.spinner("Extracting ExxonMobil 2024 risks..."):
                result_exxon = extract_risk_table(
                    company="ExxonMobil",
                    years=[2024],
                    provider=extractor_provider,
                    model=extractor_model,
                    temperature=extractor_temperature
                )
            
            if result_chevron["success"] and result_exxon["success"]:
                chevron_count = len(result_chevron["table"]) if result_chevron["table"] else 0
                exxon_count = len(result_exxon["table"]) if result_exxon["table"] else 0
                st.success(f"✅ Cross-company comparison completed (Chevron: {chevron_count}, ExxonMobil: {exxon_count})")
                
                # Create comparison table
                st.subheader("🔄 Cross-Company Comparison: Chevron vs ExxonMobil 2024")
                
                # Show Chevron risks
                st.write("**Chevron 2024 Risks:**")
                df_chevron = pd.DataFrame(result_chevron["table"])
                st.dataframe(df_chevron, use_container_width=True)
                
                # Show ExxonMobil risks
                st.write("**ExxonMobil 2024 Risks:**")
                df_exxon = pd.DataFrame(result_exxon["table"])
                st.dataframe(df_exxon, use_container_width=True)
                
                # Comparison analysis
                comparison = compare_risks(
                    result_chevron["table"], 
                    result_exxon["table"],
                    "Chevron 2024",
                    "ExxonMobil 2024"
                )
                
                st.write("**Comparison Summary:**")
                st.write(f"• Chevron: {comparison['summary']['total_1']} risks")
                st.write(f"• ExxonMobil: {comparison['summary']['total_2']} risks")
                st.write(f"• Common: {comparison['summary']['common']} risks")
                st.write(f"• Chevron-specific: {comparison['summary']['removed']} risks")
                st.write(f"• ExxonMobil-specific: {comparison['summary']['added']} risks")
                
                # Download combined CSV
                combined_df = pd.concat([
                    df_chevron.assign(company="Chevron"),
                    df_exxon.assign(company="ExxonMobil")
                ], ignore_index=True)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Combined CSV",
                    data=csv,
                    file_name="chevron_vs_exxonmobil_2024_risks.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Cross-company extraction failed")
        
        elif comparison_mode == "Time-Series":
            # Time-series analysis: ExxonMobil 2022-2024
            with st.spinner("Extracting ExxonMobil risks across 2022-2024..."):
                result_2022 = extract_risk_table(
                    company="ExxonMobil",
                    years=[2022],
                    provider=extractor_provider,
                    model=extractor_model,
                    temperature=extractor_temperature
                )
                
                result_2023 = extract_risk_table(
                    company="ExxonMobil",
                    years=[2023],
                    provider=extractor_provider,
                    model=extractor_model,
                    temperature=extractor_temperature
                )
                
                result_2024 = extract_risk_table(
                    company="ExxonMobil",
                    years=[2024],
                    provider=extractor_provider,
                    model=extractor_model,
                    temperature=extractor_temperature
                )
            
            if all(r["success"] for r in [result_2022, result_2023, result_2024]):
                counts = [len(r["table"]) if r["table"] else 0 for r in [result_2022, result_2023, result_2024]]
                st.success(f"✅ Time-series analysis completed (2022: {counts[0]}, 2023: {counts[1]}, 2024: {counts[2]})")
                
                st.subheader("📈 Time-Series Analysis: ExxonMobil 2022-2024")
                
                # Show each year's risks
                for year, result in [(2022, result_2022), (2023, result_2023), (2024, result_2024)]:
                    st.write(f"**ExxonMobil {year} Risks:**")
                    df = pd.DataFrame(result["table"])
                    st.dataframe(df, use_container_width=True)
                
                # Create timeline analysis
                timeline = create_risk_timeline({
                    2022: result_2022["table"],
                    2023: result_2023["table"],
                    2024: result_2024["table"]
                }, "ExxonMobil")
                
                st.write("**Risk Evolution Summary:**")
                for entry in timeline["timeline"]:
                    if entry["status"] == "comparison":
                        st.write(f"**{entry['year']}**: {entry['summary']['added']} new risks, {entry['summary']['removed']} removed risks")
                
                # Download combined CSV
                combined_df = pd.concat([
                    pd.DataFrame(result_2022["table"]).assign(year=2022),
                    pd.DataFrame(result_2023["table"]).assign(year=2023),
                    pd.DataFrame(result_2024["table"]).assign(year=2024)
                ], ignore_index=True)
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Time-Series CSV",
                    data=csv,
                    file_name="exxonmobil_risks_2022-2024.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ Time-series extraction failed")
        
        # Show trace information for any mode
        if "result" in locals() and "trace" in result:
            with st.expander("🔍 Extraction Details"):
                trace = result["trace"]
                st.write(f"**Provider**: {trace['extractor']['provider']} ({trace['extractor']['model']})")
                st.write(f"**Temperature**: {trace['extractor']['temperature']}")
                st.write(f"**Chunks Processed**: {trace['chunks_processed']}")
                st.write(f"**Retriever Stats**: {trace['retriever']}")
    
    # Example usage
    st.subheader("💡 Usage Examples")
    st.write("**Single Company Analysis:**")
    st.code(f"Extract all risk factors for {company} {default_year}")
    
    st.write("**Cross-Company Comparison:**")
    st.code(f"Compare risk factors between {company} and Chevron for 2024")
    
    st.write("**Time-Series Analysis:**")
    st.code(f"Track how {company}'s risks evolved from 2022 to 2024")