import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm


def render():
    """Main render function for merged AI Insights & Report page."""
    
    st.markdown("## 📋 AI Insights & Report")
    st.markdown("Comprehensive AI-powered analysis and recommendations based on all agent outputs.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("⚠️ No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    file_name = st.session_state.get("file_name", "data.csv")
    
    # Show pipeline status
    show_pipeline_status()
    st.markdown("---")
    
    # Report Configuration
    st.markdown("### ⚙️ Report Configuration")
    
    col_opts1, col_opts2 = st.columns(2)
    with col_opts1:
        report_tone = st.selectbox(
            "🎨 Report Tone",
            ["Executive (non-technical)", "Technical (data science)", "Mixed (balanced)"],
            help="Choose the writing style that best suits your audience"
        )
    with col_opts2:
        include_recs = st.checkbox("💡 Include Recommendations Section", value=True)
    
    # Generate Report Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        generate_btn = st.button("🚀 Generate AI Report", use_container_width=True, type="primary")
    
    if generate_btn:
        try:
            # Build context for LLM
            context = build_full_context(df)
            
            system_prompt = f"""You are a senior data analyst writing a professional analysis report.
Tone: {report_tone}.
Write clearly, use structure (headings, bullet points), and be specific with numbers.
Use markdown formatting for better readability."""
            
            user_prompt = f"""Write a comprehensive data analysis report based on the following dataset analysis.

{context}

Structure your report with these sections:

## Executive Summary
- 3-4 sentences overview of key findings

## Data Quality & Cleaning Summary
- Brief assessment of data completeness, quality score, and cleaning operations

## Statistical Analysis Highlights
- Key statistics, correlations, and patterns found in the data

## Model Performance Analysis
- Best performing model and its key metrics
- What the model reveals about the data

{f'## Strategic Recommendations' if include_recs else ''}
{f'- 3-5 actionable, specific next steps' if include_recs else ''}

## Conclusion
- Final thoughts and overall assessment

Keep the report professional, specific, and under 800 words. Use markdown formatting."""
            
            with st.spinner("📝 Generating comprehensive report..."):
                llm_report = call_llm(user_prompt, system_prompt, max_tokens=1800)
                st.session_state.ai_report = llm_report
            
            st.success("✅ Report generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
    
    # Display Report
    if st.session_state.get("ai_report"):
        st.markdown("---")
        st.markdown("### 📄 Generated Report")
        
        formatted_report = format_report_text(st.session_state.ai_report)
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(124,58,237,0.3);
                    border-radius: 20px;
                    padding: 2rem;
                    line-height: 1.7;
                    font-size: 0.95rem;">
            {formatted_report}
        </div>
        """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("---")
        render_export_options(df)
    
    # Navigation
    st.markdown("---")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Back to Modeling", use_container_width=True):
            st.session_state.current_page = "modeling"
            st.rerun()
    with col_nav2:
        if st.button("🔄 New Analysis", use_container_width=True):
            # Reset relevant session states
            for key in ["quality_report", "cleaning_report", "stats_done", "viz_done", 
                       "modeling_done", "ai_report"]:
                if key in st.session_state:
                    st.session_state[key] = None
            st.session_state.current_page = "home"
            st.rerun()


def show_pipeline_status():
    """Show which agents have completed."""
    
    steps = [
        ("🔍 Quality", st.session_state.get("quality_report") is not None),
        ("🧹 Cleaning", st.session_state.get("cleaning_report") is not None),
        ("📊 Stats", st.session_state.get("stats_done", False)),
        ("📈 Viz", st.session_state.get("viz_done", False)),
        ("🤖 Modeling", st.session_state.get("modeling_done", False)),
        ("📋 Report", st.session_state.get("ai_report") is not None),
    ]
    
    st.markdown("### 📋 Pipeline Status")
    
    cols = st.columns(len(steps))
    for col, (label, done) in zip(cols, steps):
        icon = "✅" if done else "⏳"
        color = "#34d399" if done else "#64748b"
        col.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:1.5rem;">{icon}</div>
            <div style="font-size:0.7rem; color:{color};">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    completed = sum(1 for _, done in steps if done)
    st.progress(completed / len(steps))
    st.caption(f"Pipeline Progress: {completed}/{len(steps)} agents complete")


def build_full_context(df: pd.DataFrame) -> str:
    """Build comprehensive context for the LLM report."""
    
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    parts = []
    parts.append(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    parts.append(f"Columns: {', '.join(df.columns.tolist()[:15])}")
    
    if len(df.columns) > 15:
        parts.append(f"... and {len(df.columns) - 15} more columns")
    
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        parts.append("\nNumeric Statistics:\n" + desc.to_string())
    
    if cat_cols:
        parts.append("\nCategorical Columns:")
        for col in cat_cols[:4]:
            top = df[col].value_counts().head(3)
            parts.append(f"  {col}: {dict(top)}")
    
    quality_report = st.session_state.get("quality_report")
    if quality_report and isinstance(quality_report, dict):
        score = quality_report.get('score', 'N/A')
        completeness = quality_report.get('completeness', 'N/A')
        parts.append(f"\nData Quality Score: {score}/100")
        parts.append(f"Completeness: {completeness}%")
    
    cleaning_report = st.session_state.get("cleaning_report")
    if cleaning_report and isinstance(cleaning_report, dict):
        before = cleaning_report.get('before_shape', [0, 0])[0]
        after = cleaning_report.get('after_shape', [0, 0])[0]
        parts.append(f"\nCleaning Summary:")
        parts.append(f"  - Rows before: {before}")
        parts.append(f"  - Rows after: {after}")
        parts.append(f"  - Rows removed: {before - after}")
    
    if st.session_state.get("modeling_done"):
        parts.append(f"\nMachine Learning Model: {st.session_state.trained_model_name}")
        parts.append(f"Predicting Target: {st.session_state.model_target_col}")
        parts.append(f"Task type: {st.session_state.model_task_type}")
        metrics = st.session_state.model_metrics[st.session_state.trained_model_name]
        parts.append("Model Performance Metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in ["confusion_matrix"]:
                parts.append(f"  {k}: {v:.4f}")
        
        if st.session_state.get("model_recommendation"):
            parts.append("\nAI Model Recommendation:")
            parts.append(st.session_state.model_recommendation[:500])
    
    return "\n".join(parts)


def format_report_text(text: str) -> str:
    """Format report text with HTML styling."""
    
    if not text:
        return "<p>No report content available.</p>"
    
    # Headers
    text = re.sub(r'^# (.*?)$', r'<h1 style="color: #a78bfa; font-size: 2rem; margin-top: 0; margin-bottom: 0.5rem;">\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2 style="color: #8b5cf6; font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 0.75rem; border-left: 4px solid #8b5cf6; padding-left: 0.8rem;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3 style="color: #c4b5fd; font-size: 1.2rem; margin-top: 1rem; margin-bottom: 0.5rem;">\1</h3>', text, flags=re.MULTILINE)
    
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color: #06b6d4;">\1</strong>', text)
    
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<em style="color: rgba(255,255,255,0.8);">\1</em>', text)
    
    # Lists
    text = re.sub(r'^[\-\*•]\s+(.*?)$', r'<li style="margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);">• \1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\.\s+(.*?)$', r'<li style="margin-bottom: 0.5rem; color: rgba(255,255,255,0.9);">\1</li>', text, flags=re.MULTILINE)
    
    # Wrap lists
    text = re.sub(r'(<li.*?>.*?</li>\n?)+', r'<ul style="margin: 0.8rem 0; padding-left: 1.5rem; list-style-type: none;">\g<0></ul>', text, flags=re.DOTALL)
    
    # Line breaks
    text = text.replace('\n\n', '<br><br>')
    
    return text


def render_export_options(df: pd.DataFrame):
    """Render export options for report."""
    
    st.markdown("### 📤 Export Options")
    st.markdown("Download the report and data in your preferred format:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📄 CSV",
            data=csv_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with col2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name="Report_Data", index=False)
        excel_data = output.getvalue()
        st.download_button(
            "📊 Excel",
            data=excel_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    
    with col3:
        json_data = df.to_json(orient="records", indent=2)
        st.download_button(
            "📋 JSON",
            data=json_data,
            file_name=f"report_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    
    with col4:
        if st.session_state.get("ai_report"):
            st.download_button(
                "📝 Report (Markdown)",
                data=st.session_state.ai_report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )