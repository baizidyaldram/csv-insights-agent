import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm


def render():
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
    show_pipeline_status()
    st.markdown("---")

    st.markdown("### ⚙️ Report Configuration")
    col1, col2 = st.columns(2)
    with col1:
        report_tone = st.selectbox(
            "🎨 Report Tone",
            ["Executive (non-technical)", "Technical (data science)", "Mixed (balanced)"],
        )
    with col2:
        include_recs = st.checkbox("💡 Include Recommendations", value=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        generate_btn = st.button("🚀 Generate AI Report", use_container_width=True, type="primary")

    if generate_btn:
        try:
            context = build_full_context(df)

            system_prompt = f"""You are a senior data analyst writing a professional analysis report.
Tone: {report_tone}.
CRITICAL FORMATTING RULES:
- Do NOT use markdown tables (no | pipe characters).
- Present all statistics as labeled bullet points or numbered lists.
- Use plain prose and bullet points only.
- Use headings (## and ###) and bullets (-) for structure.
- Be specific with numbers."""

            user_prompt = f"""Write a comprehensive data analysis report from the dataset analysis below.

{context}

Structure the report with these exact sections:

## Executive Summary
3-4 sentence overview of key findings.

## Dataset Overview
- Bullet points covering: total rows, columns, numeric columns, categorical columns, missing data %, data quality score.

## Statistical Highlights
Describe the key numeric variables (mean, range, notable patterns) as bullet points — no tables.
Describe the key categorical variables and their top values as bullet points.

## Model Performance
- Best model name and algorithm type
- Key metrics as bullet points (e.g. "Accuracy: 0.8762", "F1-Score: 0.8544")
- What the model reveals about the data

{f'## Recommendations' if include_recs else ''}
{f'- 3 to 5 actionable next steps, each as a clear bullet point.' if include_recs else ''}

## Conclusion
2-3 sentences summarising overall quality and readiness for production use.

Keep the total under 700 words. No markdown tables whatsoever."""

            with st.spinner("📝 Generating report..."):
                report = call_llm(user_prompt, system_prompt, max_tokens=1800)
                st.session_state.ai_report = report

            st.success("✅ Report generated!")
            st.rerun()

        except Exception as e:
            st.error(f"Error generating report: {e}")

    if st.session_state.get("ai_report"):
        st.markdown("---")
        st.markdown("### 📄 Generated Report")

        # Render the report cleanly — strip any stray table characters
        clean_report = clean_report_text(st.session_state.ai_report)

        st.markdown(
            f"""<div style="background:#FFFBF5; border:1.5px solid #D85A30; border-radius:16px;
                            padding:2rem; line-height:1.75; font-size:0.94rem; color:#221E1B;">
            {format_report_html(clean_report)}
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("---")
        render_export_options(df)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Modeling", use_container_width=True):
            st.session_state.current_page = "modeling"
            st.rerun()
    with c2:
        if st.button("🔄 New Analysis", use_container_width=True):
            for key in ["quality_report","cleaning_report","stats_done","viz_done","modeling_done","ai_report"]:
                st.session_state[key] = None
            st.session_state.current_page = "home"
            st.rerun()


def clean_report_text(text: str) -> str:
    """Remove any markdown table rows (lines containing | characters)."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        # Skip lines that look like table rows or separator rows
        stripped = line.strip()
        if stripped.startswith("|") or re.match(r"^\|?[-|: ]+\|", stripped):
            # Convert table row to bullet points instead
            cells = [c.strip() for c in stripped.strip("|").split("|") if c.strip() and c.strip() not in ["---","----","-----"]]
            if cells:
                cleaned.append("- " + " | ".join(cells))
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def format_report_html(text: str) -> str:
    if not text:
        return "<p>No report content.</p>"

    text = re.sub(r'^## (.*?)$', r'<h2 style="color:#3D2800;font-size:1.3rem;margin-top:1.5rem;margin-bottom:0.5rem;border-left:4px solid #EF9F27;padding-left:0.75rem;">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3 style="color:#5C3800;font-size:1.1rem;margin-top:1rem;margin-bottom:0.4rem;">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#B83A00;">\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^[-•]\s+(.*?)$', r'<li style="margin-bottom:0.4rem;">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^[0-9]+\.\s+(.*?)$', r'<li style="margin-bottom:0.4rem;">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li.*?</li>\n?)+', r'<ul style="margin:0.6rem 0;padding-left:1.4rem;">\g<0></ul>', text, flags=re.DOTALL)
    text = text.replace('\n\n', '<br>')
    return text


def show_pipeline_status():
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
        color = "#3B6D11" if done else "#854F0B"
        col.markdown(f'<div style="text-align:center;"><div style="font-size:1.4rem;">{icon}</div><div style="font-size:0.68rem;color:{color};">{label}</div></div>', unsafe_allow_html=True)
    completed = sum(1 for _, d in steps if d)
    st.progress(completed / len(steps))
    st.caption(f"Pipeline Progress: {completed}/{len(steps)} agents complete")


def build_full_context(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    parts = [
        f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"Columns: {', '.join(df.columns.tolist()[:15])}" + (f" ... and {len(df.columns)-15} more" if len(df.columns) > 15 else ""),
    ]

    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        for col in numeric_cols[:6]:
            s = desc[col]
            parts.append(f"  {col}: mean={s['mean']}, std={s['std']}, min={s['min']}, max={s['max']}")

    if cat_cols:
        parts.append("Categorical columns:")
        for col in cat_cols[:4]:
            top = df[col].value_counts().head(3)
            parts.append(f"  {col}: {dict(top)}")

    qr = st.session_state.get("quality_report")
    if qr and isinstance(qr, dict):
        parts.append(f"Data quality score: {qr.get('score','N/A')}/100")
        parts.append(f"Completeness: {qr.get('completeness','N/A')}%")

    cr = st.session_state.get("cleaning_report")
    if cr and isinstance(cr, dict):
        before = cr.get('before_shape', [0,0])[0]
        after  = cr.get('after_shape',  [0,0])[0]
        parts.append(f"Cleaning: {before} → {after} rows (removed {before-after})")

    if st.session_state.get("modeling_done"):
        parts.append(f"Best model: {st.session_state.trained_model_name}")
        parts.append(f"Target: {st.session_state.model_target_col}  |  Task: {st.session_state.model_task_type}")
        m = st.session_state.model_metrics.get(st.session_state.trained_model_name, {})
        for k, v in m.items():
            if isinstance(v, (int, float)) and k != "confusion_matrix":
                parts.append(f"  {k}: {v:.4f}")

    return "\n".join(parts)


def render_export_options(df: pd.DataFrame):
    st.markdown("### 📤 Export Options")
    c1, c2, c3, c4 = st.columns(4)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with c1:
        st.download_button("📄 CSV", data=df.to_csv(index=False),
                           file_name=f"data_{ts}.csv", mime="text/csv", use_container_width=True)
    with c2:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as w:
            df.to_excel(w, sheet_name="Data", index=False)
        st.download_button("📊 Excel", data=out.getvalue(),
                           file_name=f"data_{ts}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c3:
        st.download_button("📋 JSON", data=df.to_json(orient="records", indent=2),
                           file_name=f"data_{ts}.json", mime="application/json", use_container_width=True)
    with c4:
        if st.session_state.get("ai_report"):
            st.download_button("📝 Report (.md)", data=st.session_state.ai_report,
                               file_name=f"report_{ts}.md", mime="text/markdown", use_container_width=True)
