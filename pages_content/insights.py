import streamlit as st
import pandas as pd
import numpy as np
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm, get_active_model
import re


def build_data_summary(df: pd.DataFrame) -> str:
    """Build a compact text summary of the dataframe to send to LLM."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    lines = []
    lines.append(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"Numeric columns: {', '.join(numeric_cols) if numeric_cols else 'None'}")
    lines.append(f"Categorical columns: {', '.join(cat_cols) if cat_cols else 'None'}")
    lines.append(f"Missing values: {df.isnull().sum().sum()} total cells")
    lines.append("")

    if numeric_cols:
        lines.append("=== Numeric Summary ===")
        desc = df[numeric_cols].describe().round(3)
        lines.append(desc.to_string())
        lines.append("")

    if cat_cols:
        lines.append("=== Categorical Summary ===")
        for col in cat_cols[:5]:
            top = df[col].value_counts().head(5)
            lines.append(f"{col}: {dict(top)}")
        lines.append("")

    if len(numeric_cols) >= 2:
        lines.append("=== Top Correlations ===")
        corr = df[numeric_cols].corr()
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], round(corr.iloc[i, j], 3)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for a, b, val in pairs[:5]:
            lines.append(f"  {a} <-> {b}: {val}")

    return "\n".join(lines)


def format_ai_response(text: str) -> str:
    """Format AI response with better HTML styling."""
    # Replace markdown headers
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'### (.*?)\n', r'<h3 style="color: #f093fb; margin-top: 1rem; margin-bottom: 0.5rem;">\1</h3>\n', text)
    text = re.sub(r'## (.*?)\n', r'<h2 style="color: #a78bfa; margin-top: 1.5rem; margin-bottom: 0.75rem;">\1</h2>\n', text)
    
    # Format bullet points
    text = re.sub(r'^\- (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^• (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\. (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Wrap lists
    text = re.sub(r'(<li>.*?</li>\n?)+', r'<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">\g<0></ul>', text, flags=re.DOTALL)
    
    # Format tables (simple markdown tables)
    lines = text.split('\n')
    formatted_lines = []
    in_table = False
    table_html = ""
    
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if not in_table:
                in_table = True
                table_html = '<table style="width: 100%; border-collapse: collapse; margin: 1rem 0; background: rgba(255,255,255,0.05); border-radius: 8px;">'
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if '---' in line:
                continue
            table_html += '<tr>'
            for cell in cells:
                table_html += f'<td style="border: 1px solid rgba(255,255,255,0.2); padding: 0.5rem; color: rgba(255,255,255,0.9);">📊 {cell}</td>'
            table_html += '</tr>'
        else:
            if in_table:
                table_html += '</table>'
                formatted_lines.append(table_html)
                in_table = False
                table_html = ""
            formatted_lines.append(line)
    
    if in_table:
        table_html += '</table>'
        formatted_lines.append(table_html)
    
    text = '\n'.join(formatted_lines)
    
    return text


def render():
    st.markdown("## 💡 AI Insights Agent")
    st.markdown("Uses an LLM via OpenRouter to generate business-level insights from your data.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()

    # Model info
    model_name = get_active_model()
    st.markdown(f"""
    <div style="background:linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
                border:1px solid rgba(102,126,234,0.5);
                border-radius:12px;
                padding:0.7rem 1rem;
                margin-bottom:1.5rem;">
        <span style="color:rgba(255,255,255,0.8);font-size:0.85rem;">🤖 Active Model: </span>
        <code style="color:#a78bfa;font-size:0.85rem;background:rgba(0,0,0,0.3);padding:2px 8px;border-radius:6px;">
            {model_name}
        </code>
        <span style="color:#34d399;font-size:0.75rem;margin-left:0.8rem;">✓ OpenRouter</span>
    </div>
    """, unsafe_allow_html=True)

    # Insight type selector with icons
    insight_options = {
        "General business insights": "📊",
        "Anomalies and red flags": "⚠️",
        "Growth opportunities": "🚀",
        "Customer behavior patterns": "👥",
        "Operational efficiency observations": "⚙️",
        "Executive summary (non-technical)": "📋"
    }
    
    insight_type = st.selectbox(
        "Select insight type",
        list(insight_options.keys()),
        format_func=lambda x: f"{insight_options[x]} {x}"
    )

    extra_context = st.text_area(
        "Additional context (optional)",
        placeholder="Example: This is monthly sales data from an e-commerce company targeting Southeast Asia...",
        height=80,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Generate Insights", use_container_width=True, type="primary")

    if run_btn:
        data_summary = build_data_summary(df)

        system_prompt = """You are a senior data analyst and business intelligence expert. 
You analyze datasets and provide clear, actionable, business-focused insights.
Write in a professional but accessible tone. Use bullet points and short paragraphs.
Always be specific and reference actual column names and values from the data.
Format your response with clear sections using markdown (## for sections, ### for subsections, - for bullet points).
Use **bold** for key metrics and important numbers.
Avoid generic statements. Every insight must be grounded in the data provided."""

        user_prompt = f"""Analyze the following dataset and provide {insight_type.lower()}.

{extra_context if extra_context.strip() else ''}

=== DATASET SUMMARY ===
{data_summary}

Please provide a well-structured response with:

## Key Findings
- 3-5 bullet points with specific numbers and **bold** for important metrics

## Patterns & Trends
- What stands out in the data
- Use a table if comparing multiple categories

## Actionable Recommendations
- 2-3 concrete next steps with clear justifications

## Data Quality Notes
- Any concerns worth flagging

Keep the total response under 600 words. Be specific and data-driven.
Use markdown formatting for better readability."""

        with st.spinner(f"🧠 Analyzing with {model_name}..."):
            response = call_llm(user_prompt, system_prompt, max_tokens=1500)
            st.session_state.insights_text = response
            st.session_state.insights_type = insight_type

    # Display insights with better formatting
    if st.session_state.get("insights_text"):
        st.markdown("---")
        
        # Header with icon
        insight_icon = insight_options.get(st.session_state.get('insights_type', ''), "💡")
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">{insight_icon}</span>
            <h2 style="display: inline-block; margin-left: 0.5rem;">{st.session_state.get('insights_type', 'AI Insights')}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Format and display the response
        formatted_response = format_ai_response(st.session_state.insights_text)
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(102,126,234,0.3);
                    border-radius: 16px;
                    padding: 1.8rem;
                    line-height: 1.8;
                    font-size: 0.95rem;">
            {formatted_response}
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.download_button(
                "📥 Download as Text",
                data=st.session_state.insights_text,
                file_name=f"insights_{st.session_state.get('insights_type', '').replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.insights_text = None
                st.rerun()
        
        with col3:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.write("💡 Select and copy the text above (Ctrl+C / Cmd+C)")

        # Custom question section
        st.markdown("---")
        st.markdown("### ❓ Ask a Follow-up Question")
        
        col_q1, col_q2 = st.columns([4, 1])
        with col_q1:
            custom_q = st.text_input(
                "",
                placeholder="e.g., What is the correlation between age and heart disease?",
                label_visibility="collapsed",
                key="custom_question"
            )
        with col_q2:
            if st.button("Ask", key="custom_ask", use_container_width=True):
                if custom_q.strip():
                    data_summary = build_data_summary(df)
                    prompt = f"""Dataset summary:
{data_summary}

Question: {custom_q}

Answer concisely and specifically using the data above. Use markdown formatting."""
                    with st.spinner("Thinking..."):
                        answer = call_llm(prompt, max_tokens=600)
                        st.markdown(f"""
                        <div style="background:rgba(102,126,234,0.1);
                                    border-left: 4px solid #667eea;
                                    border-radius: 12px;
                                    padding: 1rem;
                                    margin-top: 0.5rem;">
                            <div style="color: #a78bfa; font-weight: 600; margin-bottom: 0.5rem;">💬 Answer:</div>
                            {answer.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("Please enter a question.")

        # Feedback section
        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);
                    border:1px solid rgba(255,255,255,0.1);
                    border-radius:10px;
                    padding:1rem;
                    text-align: center;">
            <p style="color:rgba(255,255,255,0.6); margin:0; font-size:0.85rem;">
                📊 Insights generated using AI | Always verify critical findings with domain experts
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Next step button
        if st.button("➡️ Proceed to Report Generation", use_container_width=True):
            st.session_state.current_page = "report"
            st.rerun()

    else:
        # Empty state with better styling
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Ready to Generate Insights</div>
            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">
                Select an insight type and click the button above to begin
            </div>
        </div>
        """, unsafe_allow_html=True)
