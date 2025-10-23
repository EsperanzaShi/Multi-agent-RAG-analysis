#!/usr/bin/env python3
"""
Wandb Python Export API - Download CSV of runs data
Uses the official wandb export method to get all run data
"""
import pandas as pd
import wandb
import json

def download_wandb_data():
    """Download all runs data using wandb Python Export API"""
    
    print("🔍 Downloading Wandb Runs Data...")
    print("=" * 50)
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs("Week6/rag-multi-agent-sweep")
    
    print(f"📊 Found {len(runs)} runs in project")
    
    # Extract data using the official method
    summary_list, config_list, name_list = [], [], []
    
    for i, run in enumerate(runs):
        print(f"Processing run {i+1}/{len(runs)}: {run.id} ({run.name})")
        
        try:
            # Handle summary data - it might be a string or dict
            summary_data = run.summary
            if hasattr(summary_data, '_json_dict'):
                summary_list.append(summary_data._json_dict)
            elif isinstance(summary_data, dict):
                summary_list.append(summary_data)
            else:
                # If it's a string, try to parse it
                try:
                    import json
                    summary_list.append(json.loads(str(summary_data)))
                except:
                    summary_list.append({})
            
            # Handle config data - it might be a string or dict
            config_data = run.config
            if isinstance(config_data, dict):
                # Remove special values that start with _
                filtered_config = {k: v for k, v in config_data.items() if not k.startswith('_')}
                config_list.append(filtered_config)
            else:
                # If it's a string, try to parse it
                try:
                    import json
                    parsed_config = json.loads(str(config_data))
                    filtered_config = {k: v for k, v in parsed_config.items() if not k.startswith('_')}
                    config_list.append(filtered_config)
                except:
                    config_list.append({})
            
            # .name is the human-readable name of the run
            name_list.append(run.name)
            
        except Exception as e:
            print(f"  ❌ Error processing run {run.id}: {e}")
            # Add empty entries to maintain alignment
            summary_list.append({})
            config_list.append({})
            name_list.append(f"error_{run.id}")
    
    # Create DataFrame
    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
    })
    
    # Save to CSV
    runs_df.to_csv("data/eval/wandb_runs_export.csv", index=False)
    print(f"\n💾 Data saved to data/eval/wandb_runs_export.csv")
    
    # Show sample of the data
    print(f"\n📊 Sample Data Preview:")
    print(f"   Total runs: {len(runs_df)}")
    print(f"   Columns: {list(runs_df.columns)}")
    
    if len(runs_df) > 0:
        print(f"\n🔍 Sample Summary Data:")
        sample_summary = runs_df.iloc[0]['summary']
        if isinstance(sample_summary, dict):
            for key, value in list(sample_summary.items())[:5]:
                print(f"   {key}: {value}")
        else:
            print(f"   Summary type: {type(sample_summary)}")
            print(f"   Summary preview: {str(sample_summary)[:100]}...")
        
        print(f"\n🔧 Sample Config Data:")
        sample_config = runs_df.iloc[0]['config']
        if isinstance(sample_config, dict):
            for key, value in list(sample_config.items())[:5]:
                print(f"   {key}: {value}")
        else:
            print(f"   Config type: {type(sample_config)}")
            print(f"   Config preview: {str(sample_config)[:100]}...")
    
    return runs_df

def analyze_exported_data():
    """Generate comprehensive analysis report of the exported wandb data"""
    
    print("\n🔍 Comprehensive Wandb Analysis Report")
    print("=" * 60)
    
    # Load the exported data
    try:
        runs_df = pd.read_csv("data/eval/wandb_runs_export.csv")
    except FileNotFoundError:
        print("❌ No exported data found. Run download first.")
        return
    
    print(f"📊 Loaded {len(runs_df)} runs from wandb sweep")
    
    # Extract metrics from summary data
    analysis_data = []
    
    for idx, row in runs_df.iterrows():
        try:
            # Parse summary data (it's stored as string in CSV)
            import ast
            summary = ast.literal_eval(row['summary']) if isinstance(row['summary'], str) else row['summary']
            config = ast.literal_eval(row['config']) if isinstance(row['config'], str) else row['config']
            
            # Extract key metrics with proper value extraction
            def extract_value(value):
                """Extract actual value from wandb config format"""
                if isinstance(value, dict) and 'value' in value:
                    return value['value']
                return value
            
            run_data = {
                "run_id": idx,
                "name": row['name'],
                "overall_quality": summary.get("overall_quality", 0),
                "approval_rate": summary.get("approval_rate", 0),
                "citation_ratio": summary.get("citation_ratio", 0),
                "execution_time": summary.get("execution_time", 0),
                "temperature": extract_value(config.get("temperature", 0)),
                "top_k": extract_value(config.get("top_k", 0)),
                "model_strategy": extract_value(config.get("model_strategy", "")),
                "cost_per_query": summary.get("cost_per_query", 0),
                "climate_quality": summary.get("climate_quality", 0),
                "financial_quality": summary.get("financial_quality", 0),
                "keyword_hit": summary.get("keyword_hit", 0),
                "successful_cases": summary.get("successful_cases", 0),
                "n_cases": summary.get("n_cases", 0)
            }
            
            analysis_data.append(run_data)
            
        except Exception as e:
            print(f"  ❌ Error parsing run {idx}: {e}")
            continue
    
    if not analysis_data:
        print("❌ No valid data found for analysis")
        return
    
    # Create analysis DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Generate comprehensive report
    generate_comprehensive_report(df)
    
    return df

def generate_comprehensive_report(df):
    """Generate a comprehensive analysis report"""
    
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE WANDB SWEEP ANALYSIS REPORT")
    print("="*80)
    
    # Create markdown report
    report_content = []
    report_content.append("# 📋 Comprehensive Wandb Sweep Analysis Report")
    report_content.append("")
    report_content.append("## 📊 Metrics Design & Measurement")
    report_content.append("")
    report_content.append("### 🎯 Primary Metric: overall_quality")
    report_content.append("- **Weighted combination of:**")
    report_content.append("  - Completeness (25%): Answer length vs target (500+ chars)")
    report_content.append("  - Citation Quality (25%): Ratio of sentences with proper citations")
    report_content.append("  - Relevance (25%): Keyword hit rate vs expected terms")
    report_content.append("  - Speed (15%): Execution time vs target (<30s)")
    report_content.append("  - Approval (10%): Critic validation (approved=True)")
    report_content.append("- **Range:** 0.0-1.0 (higher is better)")
    report_content.append("")
    report_content.append("### 📈 Secondary Metrics:")
    report_content.append("- **approval_rate:** Critic validation success rate")
    report_content.append("- **citation_ratio:** Percentage of sentences with [CITATION: ...] format")
    report_content.append("- **keyword_hit:** Relevance to expected domain terms")
    report_content.append("- **execution_time:** Response time in seconds")
    report_content.append("- **cost_per_query:** Economic efficiency ($/query)")
    report_content.append("")
    report_content.append("### 🎨 Domain-Specific Quality:")
    report_content.append("- **climate_quality:** Performance on climate risk queries")
    report_content.append("- **financial_quality:** Performance on financial metrics queries")
    
    # Overall Statistics
    report_content.append("")
    report_content.append("## 📈 Overall Statistics")
    report_content.append("")
    report_content.append(f"- **Total Runs:** {len(df)}")
    report_content.append(f"- **Average Quality:** {df['overall_quality'].mean():.3f}")
    report_content.append(f"- **Best Quality:** {df['overall_quality'].max():.3f}")
    report_content.append(f"- **Quality Range:** {df['overall_quality'].min():.3f} - {df['overall_quality'].max():.3f}")
    report_content.append(f"- **Average Approval Rate:** {df['approval_rate'].mean():.3f}")
    report_content.append(f"- **Average Citation Ratio:** {df['citation_ratio'].mean():.3f}")
    report_content.append(f"- **Average Execution Time:** {df['execution_time'].mean():.1f}s")
    
    # Model Strategy Comparison
    report_content.append("")
    report_content.append("## 🤖 Model Strategy Comparison")
    report_content.append("")
    
    strategies = df['model_strategy'].unique()
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            report_content.append(f"### 📊 {strategy.upper()} Strategy:")
            report_content.append(f"- **Runs:** {len(strategy_df)}")
            report_content.append(f"- **Average Quality:** {strategy_df['overall_quality'].mean():.3f} ± {strategy_df['overall_quality'].std():.3f}")
            report_content.append(f"- **Approval Rate:** {strategy_df['approval_rate'].mean():.3f}")
            report_content.append(f"- **Citation Ratio:** {strategy_df['citation_ratio'].mean():.3f}")
            report_content.append(f"- **Execution Time:** {strategy_df['execution_time'].mean():.1f}s")
            report_content.append(f"- **Cost per Query:** ${strategy_df['cost_per_query'].mean():.3f}")
            report_content.append("")
    
    # Strategy-Specific Parameter Analysis
    report_content.append("## 🔍 Strategy-Specific Parameter Analysis")
    report_content.append("")
    
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            
            report_content.append(f"### 🎯 {strategy.upper()} Strategy Parameter Analysis:")
            report_content.append("")
            
            # Temperature analysis for this strategy
            report_content.append(f"#### 🌡️ Temperature Impact ({strategy}):")
            temp_stats = strategy_df.groupby('temperature')['overall_quality'].agg(['mean', 'std', 'count'])
            for temp, stats in temp_stats.iterrows():
                if temp > 0:
                    report_content.append(f"- **{temp}:** {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
            report_content.append("")
            
            # Top-K analysis for this strategy
            report_content.append(f"#### 📚 Top-K Impact ({strategy}):")
            topk_stats = strategy_df.groupby('top_k')['overall_quality'].agg(['mean', 'std', 'count'])
            for topk, stats in topk_stats.iterrows():
                if topk > 0:
                    report_content.append(f"- **{topk}:** {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
            report_content.append("")
            
            # Find best combination for this strategy
            best_combo = strategy_df.loc[strategy_df['overall_quality'].idxmax()]
            report_content.append(f"#### 🏆 Best {strategy} Configuration:")
            report_content.append(f"- **Quality:** {best_combo['overall_quality']:.3f}")
            report_content.append(f"- **Temperature:** {best_combo['temperature']}")
            report_content.append(f"- **Top-K:** {best_combo['top_k']}")
            report_content.append(f"- **Cost:** ${best_combo['cost_per_query']:.3f}")
            report_content.append("")
    
    # Best Configurations
    report_content.append("## 🏆 Best Configurations")
    report_content.append("")
    
    # Overall best
    best_overall = df.loc[df['overall_quality'].idxmax()]
    report_content.append("### 🥇 Overall Best Configuration:")
    report_content.append(f"- **Quality:** {best_overall['overall_quality']:.3f}")
    report_content.append(f"- **Strategy:** {best_overall['model_strategy']}")
    report_content.append(f"- **Temperature:** {best_overall['temperature']}")
    report_content.append(f"- **Top-K:** {best_overall['top_k']}")
    report_content.append(f"- **Cost:** ${best_overall['cost_per_query']:.3f}")
    report_content.append("")
    
    # Best by strategy
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            best_strategy = strategy_df.loc[strategy_df['overall_quality'].idxmax()]
            report_content.append(f"### 🥈 Best {strategy} Configuration:")
            report_content.append(f"- **Quality:** {best_strategy['overall_quality']:.3f}")
            report_content.append(f"- **Temperature:** {best_strategy['temperature']}")
            report_content.append(f"- **Top-K:** {best_strategy['top_k']}")
            report_content.append(f"- **Cost:** ${best_strategy['cost_per_query']:.3f}")
            report_content.append("")
    
    # Business Recommendations
    report_content.append("## 💡 Business Recommendations")
    report_content.append("")
    
    # Compare strategies
    cost_effective = df[df['model_strategy'] == 'cost_effective']
    zero_cost = df[df['model_strategy'] == 'zero_cost']
    
    if len(cost_effective) > 0 and len(zero_cost) > 0:
        ce_quality = cost_effective['overall_quality'].mean()
        zc_quality = zero_cost['overall_quality'].mean()
        quality_diff = ce_quality - zc_quality
        
        report_content.append("### 📊 Strategy Performance Comparison:")
        report_content.append(f"- **Cost-Effective:** {ce_quality:.3f} average quality")
        report_content.append(f"- **Zero-Cost:** {zc_quality:.3f} average quality")
        report_content.append(f"- **Quality Difference:** {quality_diff:+.3f}")
        report_content.append("")
        
        if quality_diff > 0.05:
            report_content.append(f"### ✅ Recommendation: Use cost-effective strategy for {quality_diff:.3f} better quality")
        elif quality_diff < -0.05:
            report_content.append(f"### ✅ Recommendation: Use zero-cost strategy for {-quality_diff:.3f} better quality")
        else:
            report_content.append("### ✅ Recommendation: Both strategies perform similarly - choose based on cost")
        report_content.append("")
    
    # Parameter recommendations
    report_content.append("### 🎯 Parameter Recommendations:")
    
    # Temperature recommendation
    temp_stats = df.groupby('temperature')['overall_quality'].mean()
    best_temp = temp_stats.idxmax()
    report_content.append(f"- **Optimal Temperature:** {best_temp} (quality: {temp_stats[best_temp]:.3f})")
    
    # Top-K recommendation
    topk_stats = df.groupby('top_k')['overall_quality'].mean()
    best_topk = topk_stats.idxmax()
    report_content.append(f"- **Optimal Top-K:** {best_topk} (quality: {topk_stats[best_topk]:.3f})")
    report_content.append("")
    
    report_content.append("### 🚀 Implementation Strategy:")
    report_content.append("- **Development:** Use zero-cost strategy for rapid iteration")
    report_content.append("- **Production:** Choose based on quality vs cost trade-offs")
    report_content.append("- **Monitoring:** Track overall_quality, approval_rate, and cost_per_query")
    
    # Additional Analysis: Speed vs Quality Trade-offs
    report_content.append("")
    report_content.append("## ⚡ Speed vs Quality Analysis")
    report_content.append("")
    
    # Calculate speed-quality correlation
    speed_quality_corr = df['execution_time'].corr(df['overall_quality'])
    report_content.append(f"### 📊 Speed-Quality Correlation: {speed_quality_corr:.3f}")
    
    if speed_quality_corr > 0.1:
        report_content.append("- **Insight:** Longer execution time correlates with higher quality")
    elif speed_quality_corr < -0.1:
        report_content.append("- **Insight:** Faster execution correlates with higher quality")
    else:
        report_content.append("- **Insight:** Speed and quality are largely independent")
    
    # Fast vs Slow configurations
    fast_configs = df[df['execution_time'] < df['execution_time'].quantile(0.25)]
    slow_configs = df[df['execution_time'] > df['execution_time'].quantile(0.75)]
    
    report_content.append("")
    report_content.append("### 🏃 Fast Configurations (Bottom 25% execution time):")
    report_content.append(f"- **Average Quality:** {fast_configs['overall_quality'].mean():.3f}")
    report_content.append(f"- **Average Time:** {fast_configs['execution_time'].mean():.1f}s")
    report_content.append(f"- **Best Strategy:** {fast_configs.groupby('model_strategy')['overall_quality'].mean().idxmax()}")
    
    report_content.append("")
    report_content.append("### 🐌 Slow Configurations (Top 25% execution time):")
    report_content.append(f"- **Average Quality:** {slow_configs['overall_quality'].mean():.3f}")
    report_content.append(f"- **Average Time:** {slow_configs['execution_time'].mean():.1f}s")
    report_content.append(f"- **Best Strategy:** {slow_configs.groupby('model_strategy')['overall_quality'].mean().idxmax()}")
    
    # Citation Quality Analysis
    report_content.append("")
    report_content.append("## 📝 Citation Quality Deep Dive")
    report_content.append("")
    
    high_citation = df[df['citation_ratio'] > df['citation_ratio'].quantile(0.75)]
    low_citation = df[df['citation_ratio'] < df['citation_ratio'].quantile(0.25)]
    
    report_content.append("### 📊 High Citation Quality (Top 25%):")
    report_content.append(f"- **Average Quality:** {high_citation['overall_quality'].mean():.3f}")
    report_content.append(f"- **Average Citation Ratio:** {high_citation['citation_ratio'].mean():.3f}")
    report_content.append(f"- **Best Strategy:** {high_citation.groupby('model_strategy')['overall_quality'].mean().idxmax()}")
    report_content.append(f"- **Best Temperature:** {high_citation.groupby('temperature')['overall_quality'].mean().idxmax()}")
    report_content.append(f"- **Best Top-K:** {high_citation.groupby('top_k')['overall_quality'].mean().idxmax()}")
    
    report_content.append("")
    report_content.append("### 📊 Low Citation Quality (Bottom 25%):")
    report_content.append(f"- **Average Quality:** {low_citation['overall_quality'].mean():.3f}")
    report_content.append(f"- **Average Citation Ratio:** {low_citation['citation_ratio'].mean():.3f}")
    report_content.append(f"- **Best Strategy:** {low_citation.groupby('model_strategy')['overall_quality'].mean().idxmax()}")
    
    # Domain-Specific Performance
    report_content.append("")
    report_content.append("## 🎯 Domain-Specific Performance Analysis")
    report_content.append("")
    
    if 'climate_quality' in df.columns and 'financial_quality' in df.columns:
        climate_quality = df['climate_quality'].mean()
        financial_quality = df['financial_quality'].mean()
        
        report_content.append("### 🌍 Climate Risk Queries:")
        report_content.append(f"- **Average Quality:** {climate_quality:.3f}")
        report_content.append(f"- **Best Strategy:** {df.groupby('model_strategy')['climate_quality'].mean().idxmax()}")
        
        report_content.append("")
        report_content.append("### 💰 Financial Metrics Queries:")
        report_content.append(f"- **Average Quality:** {financial_quality:.3f}")
        report_content.append(f"- **Best Strategy:** {df.groupby('model_strategy')['financial_quality'].mean().idxmax()}")
        
        if abs(climate_quality - financial_quality) > 0.05:
            report_content.append("")
            report_content.append("### 🎯 Domain Specialization Insight:")
            if climate_quality > financial_quality:
                report_content.append("- **Climate queries perform better** - system optimized for environmental analysis")
            else:
                report_content.append("- **Financial queries perform better** - system optimized for numerical analysis")
    
    # Parameter Interaction Analysis
    report_content.append("")
    report_content.append("## 🔄 Parameter Interaction Analysis")
    report_content.append("")
    
    # Temperature-TopK interaction
    temp_topk_interaction = df.groupby(['temperature', 'top_k'])['overall_quality'].mean().unstack()
    best_temp_topk = temp_topk_interaction.stack().idxmax()
    
    report_content.append("### 🌡️📚 Temperature-TopK Interaction:")
    report_content.append(f"- **Best Combination:** temp={best_temp_topk[0]}, top_k={best_temp_topk[1]}")
    report_content.append(f"- **Quality:** {temp_topk_interaction.loc[best_temp_topk[0], best_temp_topk[1]]:.3f}")
    
    # Strategy-Parameter interaction
    report_content.append("")
    report_content.append("### 🤖 Strategy-Parameter Synergy:")
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            best_temp = strategy_df.groupby('temperature')['overall_quality'].mean().idxmax()
            best_topk = strategy_df.groupby('top_k')['overall_quality'].mean().idxmax()
            report_content.append(f"- **{strategy}:** temp={best_temp}, top_k={best_topk}")
    
    # Cost-Benefit Analysis
    report_content.append("")
    report_content.append("## 💰 Cost-Benefit Analysis")
    report_content.append("")
    
    # Calculate cost per quality point
    df['cost_per_quality'] = df['cost_per_query'] / df['overall_quality']
    df['cost_per_quality'] = df['cost_per_quality'].replace([float('inf'), -float('inf')], 0)
    
    # Recalculate strategy groups with updated dataframe
    cost_effective = df[df['model_strategy'] == 'cost_effective']
    zero_cost = df[df['model_strategy'] == 'zero_cost']
    
    cost_effective_roi = cost_effective['cost_per_quality'].mean()
    zero_cost_roi = zero_cost['cost_per_quality'].mean()
    
    report_content.append("### 📊 Return on Investment (Cost per Quality Point):")
    report_content.append(f"- **Cost-Effective:** ${cost_effective_roi:.4f} per quality point")
    report_content.append(f"- **Zero-Cost:** ${zero_cost_roi:.4f} per quality point")
    
    if cost_effective_roi < zero_cost_roi:
        report_content.append("- **Insight:** Cost-effective provides better ROI despite higher absolute cost")
    else:
        report_content.append("- **Insight:** Zero-cost provides infinite ROI (free with better quality)")
    
    # Quality Distribution Analysis
    report_content.append("")
    report_content.append("## 📊 Quality Distribution Analysis")
    report_content.append("")
    
    quality_quartiles = df['overall_quality'].quantile([0.25, 0.5, 0.75])
    report_content.append("### 📈 Quality Quartiles:")
    report_content.append(f"- **Q1 (25th percentile):** {quality_quartiles[0.25]:.3f}")
    report_content.append(f"- **Q2 (Median):** {quality_quartiles[0.5]:.3f}")
    report_content.append(f"- **Q3 (75th percentile):** {quality_quartiles[0.75]:.3f}")
    
    # Consistency analysis
    ce_consistency = cost_effective['overall_quality'].std()
    zc_consistency = zero_cost['overall_quality'].std()
    
    report_content.append("")
    report_content.append("### 🎯 Consistency Analysis (Lower std = More Consistent):")
    report_content.append(f"- **Cost-Effective Consistency:** {ce_consistency:.3f}")
    report_content.append(f"- **Zero-Cost Consistency:** {zc_consistency:.3f}")
    
    if ce_consistency < zc_consistency:
        report_content.append("- **Insight:** Cost-effective strategy is more predictable")
    else:
        report_content.append("- **Insight:** Zero-cost strategy is more predictable")
    
    # Save markdown report
    report_path = "data/eval/wandb_sweep_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    print(f"\n💾 Comprehensive report saved to {report_path}")
    
    # Also print to console
    print_metrics_design()
    print_overall_statistics(df)
    print_strategy_comparison(df)
    print_strategy_specific_analysis(df)
    print_best_configurations(df)
    print_business_recommendations(df)
    
    # Save detailed results
    df.to_csv("data/eval/wandb_analysis_results.csv", index=False)
    print(f"\n💾 Detailed results saved to data/eval/wandb_analysis_results.csv")

def print_metrics_design():
    """Explain how metrics are designed and measured"""
    
    print("\n📊 METRICS DESIGN & MEASUREMENT")
    print("-" * 50)
    print("🎯 Primary Metric: overall_quality")
    print("   • Weighted combination of:")
    print("     - Completeness (25%): Answer length vs target (500+ chars)")
    print("     - Citation Quality (25%): Ratio of sentences with proper citations")
    print("     - Relevance (25%): Keyword hit rate vs expected terms")
    print("     - Speed (15%): Execution time vs target (<30s)")
    print("     - Approval (10%): Critic validation (approved=True)")
    print("   • Range: 0.0-1.0 (higher is better)")
    
    print("\n📈 Secondary Metrics:")
    print("   • approval_rate: Critic validation success rate")
    print("   • citation_ratio: Percentage of sentences with [CITATION: ...] format")
    print("   • keyword_hit: Relevance to expected domain terms")
    print("   • execution_time: Response time in seconds")
    print("   • cost_per_query: Economic efficiency ($/query)")
    
    print("\n🎨 Domain-Specific Quality:")
    print("   • climate_quality: Performance on climate risk queries")
    print("   • financial_quality: Performance on financial metrics queries")

def print_overall_statistics(df):
    """Print overall statistics"""
    
    print("\n📈 OVERALL STATISTICS")
    print("-" * 30)
    print(f"   Total Runs: {len(df)}")
    print(f"   Average Quality: {df['overall_quality'].mean():.3f}")
    print(f"   Best Quality: {df['overall_quality'].max():.3f}")
    print(f"   Quality Range: {df['overall_quality'].min():.3f} - {df['overall_quality'].max():.3f}")
    print(f"   Average Approval Rate: {df['approval_rate'].mean():.3f}")
    print(f"   Average Citation Ratio: {df['citation_ratio'].mean():.3f}")
    print(f"   Average Execution Time: {df['execution_time'].mean():.1f}s")

def print_strategy_comparison(df):
    """Compare model strategies"""
    
    print("\n🤖 MODEL STRATEGY COMPARISON")
    print("-" * 40)
    
    strategies = df['model_strategy'].unique()
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            print(f"\n📊 {strategy.upper()} Strategy:")
            print(f"   Runs: {len(strategy_df)}")
            print(f"   Average Quality: {strategy_df['overall_quality'].mean():.3f} ± {strategy_df['overall_quality'].std():.3f}")
            print(f"   Approval Rate: {strategy_df['approval_rate'].mean():.3f}")
            print(f"   Citation Ratio: {strategy_df['citation_ratio'].mean():.3f}")
            print(f"   Execution Time: {strategy_df['execution_time'].mean():.1f}s")
            print(f"   Cost per Query: ${strategy_df['cost_per_query'].mean():.3f}")

def print_strategy_specific_analysis(df):
    """Analyze parameters by model strategy"""
    
    print("\n🔍 STRATEGY-SPECIFIC PARAMETER ANALYSIS")
    print("-" * 50)
    
    strategies = df['model_strategy'].unique()
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            
            print(f"\n🎯 {strategy.upper()} Strategy Parameter Analysis:")
            
            # Temperature analysis for this strategy
            print(f"\n🌡️ Temperature Impact ({strategy}):")
            temp_stats = strategy_df.groupby('temperature')['overall_quality'].agg(['mean', 'std', 'count'])
            for temp, stats in temp_stats.iterrows():
                if temp > 0:
                    print(f"   {temp}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
            
            # Top-K analysis for this strategy
            print(f"\n📚 Top-K Impact ({strategy}):")
            topk_stats = strategy_df.groupby('top_k')['overall_quality'].agg(['mean', 'std', 'count'])
            for topk, stats in topk_stats.iterrows():
                if topk > 0:
                    print(f"   {topk}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")
            
            # Find best combination for this strategy
            best_combo = strategy_df.loc[strategy_df['overall_quality'].idxmax()]
            print(f"\n🏆 Best {strategy} Configuration:")
            print(f"   Quality: {best_combo['overall_quality']:.3f}")
            print(f"   Temperature: {best_combo['temperature']}")
            print(f"   Top-K: {best_combo['top_k']}")
            print(f"   Cost: ${best_combo['cost_per_query']:.3f}")

def print_best_configurations(df):
    """Print best configurations overall and by strategy"""
    
    print("\n🏆 BEST CONFIGURATIONS")
    print("-" * 30)
    
    # Overall best
    best_overall = df.loc[df['overall_quality'].idxmax()]
    print(f"\n🥇 Overall Best Configuration:")
    print(f"   Quality: {best_overall['overall_quality']:.3f}")
    print(f"   Strategy: {best_overall['model_strategy']}")
    print(f"   Temperature: {best_overall['temperature']}")
    print(f"   Top-K: {best_overall['top_k']}")
    print(f"   Cost: ${best_overall['cost_per_query']:.3f}")
    
    # Best by strategy
    strategies = df['model_strategy'].unique()
    for strategy in strategies:
        if strategy:
            strategy_df = df[df['model_strategy'] == strategy]
            best_strategy = strategy_df.loc[strategy_df['overall_quality'].idxmax()]
            print(f"\n🥈 Best {strategy} Configuration:")
            print(f"   Quality: {best_strategy['overall_quality']:.3f}")
            print(f"   Temperature: {best_strategy['temperature']}")
            print(f"   Top-K: {best_strategy['top_k']}")
            print(f"   Cost: ${best_strategy['cost_per_query']:.3f}")

def print_business_recommendations(df):
    """Print business recommendations based on analysis"""
    
    print("\n💡 BUSINESS RECOMMENDATIONS")
    print("-" * 35)
    
    # Compare strategies
    cost_effective = df[df['model_strategy'] == 'cost_effective']
    zero_cost = df[df['model_strategy'] == 'zero_cost']
    
    if len(cost_effective) > 0 and len(zero_cost) > 0:
        ce_quality = cost_effective['overall_quality'].mean()
        zc_quality = zero_cost['overall_quality'].mean()
        quality_diff = ce_quality - zc_quality
        
        print(f"\n📊 Strategy Performance Comparison:")
        print(f"   Cost-Effective: {ce_quality:.3f} average quality")
        print(f"   Zero-Cost: {zc_quality:.3f} average quality")
        print(f"   Quality Difference: {quality_diff:+.3f}")
        
        if quality_diff > 0.05:
            print(f"\n✅ Recommendation: Use cost-effective strategy for {quality_diff:.3f} better quality")
        elif quality_diff < -0.05:
            print(f"\n✅ Recommendation: Use zero-cost strategy for {-quality_diff:.3f} better quality")
        else:
            print(f"\n✅ Recommendation: Both strategies perform similarly - choose based on cost")
    
    # Parameter recommendations
    print(f"\n🎯 Parameter Recommendations:")
    
    # Temperature recommendation
    temp_stats = df.groupby('temperature')['overall_quality'].mean()
    best_temp = temp_stats.idxmax()
    print(f"   • Optimal Temperature: {best_temp} (quality: {temp_stats[best_temp]:.3f})")
    
    # Top-K recommendation
    topk_stats = df.groupby('top_k')['overall_quality'].mean()
    best_topk = topk_stats.idxmax()
    print(f"   • Optimal Top-K: {best_topk} (quality: {topk_stats[best_topk]:.3f})")
    
    print(f"\n🚀 Implementation Strategy:")
    print(f"   • Development: Use zero-cost strategy for rapid iteration")
    print(f"   • Production: Choose based on quality vs cost trade-offs")
    print(f"   • Monitoring: Track overall_quality, approval_rate, and cost_per_query")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wandb Export and Analysis")
    parser.add_argument("--action", choices=["download", "analyze", "both"], default="both",
                       help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action in ["download", "both"]:
        download_wandb_data()
    
    if args.action in ["analyze", "both"]:
        analyze_exported_data()

if __name__ == "__main__":
    main()
