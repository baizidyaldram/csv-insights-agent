import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.session import get_df, is_data_loaded

WARM = {
    "primary":   "#EF9F27",
    "secondary": "#D85A30",
    "tertiary":  "#BA7517",
    "palette":   ["#EF9F27","#D85A30","#854F0B","#FAC775","#F0997B","#BA7517"],
    "seq":       "YlOrBr",
    "template":  "plotly_white",
}


def _layout(height=420):
    return dict(
        template=WARM["template"],
        height=height,
        margin=dict(t=48, b=40, l=40, r=20),
        font=dict(family="Inter, sans-serif", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


def chart_numeric_distribution(df, col):
    """Histogram with KDE overlay for numeric cols."""
    data = df[col].dropna()
    fig = px.histogram(
        data_frame=pd.DataFrame({col: data}),
        x=col, nbins=30,
        title=f"Distribution — {col}",
        color_discrete_sequence=[WARM["primary"]],
        marginal="box",
    )
    fig.update_layout(**_layout())
    fig.update_traces(marker_line_width=0.5, marker_line_color="white", opacity=0.85)
    return fig


def chart_binary_pie(df, col):
    """Pie chart for binary / low-cardinality numeric columns."""
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    fig = px.pie(
        vc, names=col, values="count",
        title=f"Split — {col}",
        color_discrete_sequence=WARM["palette"],
        hole=0.4,
    )
    fig.update_layout(**_layout(360))
    fig.update_traces(textinfo="percent+label")
    return fig


def chart_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdYlBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(**_layout(500))
    return fig


def chart_scatter_top_pair(df, numeric_cols):
    """Scatter of the most correlated numeric pair."""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().abs()
    pairs = [
        (corr.columns[i], corr.columns[j], corr.iloc[i, j])
        for i in range(len(corr.columns))
        for j in range(i + 1, len(corr.columns))
    ]
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[2], reverse=True)
    col_a, col_b, r = pairs[0]
    fig = px.scatter(
        df, x=col_a, y=col_b,
        title=f"Scatter — {col_a} vs {col_b}  (r={r:.2f})",
        color_discrete_sequence=[WARM["primary"]],
        trendline="ols",
        opacity=0.65,
    )
    fig.update_layout(**_layout())
    return fig


def chart_categorical_bar(df, col):
    vc = df[col].value_counts().head(12).reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(
        vc, x=col, y="count",
        title=f"Top values — {col}",
        color="count",
        color_continuous_scale=WARM["seq"],
        text_auto=True,
    )
    fig.update_layout(**_layout())
    fig.update_xaxes(tickangle=-30)
    fig.update_coloraxes(showscale=False)
    return fig


def chart_time_series(df, date_col, numeric_col):
    tmp = df[[date_col, numeric_col]].dropna().sort_values(date_col)
    fig = px.line(
        tmp, x=date_col, y=numeric_col,
        title=f"{numeric_col} over time",
        color_discrete_sequence=[WARM["primary"]],
        markers=True,
    )
    fig.update_layout(**_layout())
    fig.update_traces(line_width=2)
    return fig


def render():
    st.markdown("## 📈 Visualization Agent")
    st.markdown("Detects your data's structure and picks the most fitting chart type for each column.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    for drop_col in ["Unnamed: 0", "index"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    if st.button("▶ Generate Visualizations", use_container_width=False):
        st.session_state.viz_done = True

    if not st.session_state.get("viz_done"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col2:
            st.info("Click **Generate Visualizations** to build charts.")
        return

    # ── Classify columns ──────────────────────────────────
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ["index", "Unnamed: 0", "level_0"]
    ]
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

    binary_num = [c for c in numeric_cols if df[c].nunique() <= 4]
    cont_num   = [c for c in numeric_cols if c not in binary_num]

    charts_rendered = 0

    # ── 1. Binary / flag columns → pie charts ────────────
    if binary_num:
        st.markdown("### 🎯 Binary / Flag Columns")
        cols_ui = st.columns(min(len(binary_num), 3))
        for i, col in enumerate(binary_num[:3]):
            with cols_ui[i]:
                fig = chart_binary_pie(df, col)
                st.plotly_chart(fig, use_container_width=True)
        charts_rendered += 1

    # ── 2. Continuous numeric → histograms (2-col grid) ──
    if cont_num:
        st.markdown("### 📊 Continuous Distributions")
        for i in range(0, min(len(cont_num), 6), 2):
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(chart_numeric_distribution(df, cont_num[i]), use_container_width=True)
            if i + 1 < len(cont_num):
                with c2:
                    st.plotly_chart(chart_numeric_distribution(df, cont_num[i+1]), use_container_width=True)
        charts_rendered += 1

    # ── 3. Correlation heatmap ────────────────────────────
    if len(numeric_cols) >= 3:
        st.markdown("### 🔗 Correlation Heatmap")
        st.plotly_chart(chart_correlation_heatmap(df, numeric_cols), use_container_width=True)
        charts_rendered += 1

    # ── 4. Top-correlated pair scatter ───────────────────
    if len(cont_num) >= 2:
        st.markdown("### 🔵 Strongest Relationship")
        fig_scatter = chart_scatter_top_pair(df, cont_num)
        if fig_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True)
        charts_rendered += 1

    # ── 5. Time series if date column detected ───────────
    if date_cols and cont_num:
        date_col = date_cols[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            target_col = cont_num[0]
            st.markdown("### 📅 Time Series")
            st.plotly_chart(chart_time_series(df, date_col, target_col), use_container_width=True)
            charts_rendered += 1
        except Exception:
            pass

    # ── 6. Categorical bar charts ─────────────────────────
    if cat_cols:
        meaningful_cat = [
            c for c in cat_cols
            if 1 < df[c].nunique() <= 30
            and c.lower() not in ["id", "customerid", "customer_id", "name", "email"]
        ]
        if meaningful_cat:
            st.markdown("### 🏷️ Categorical Distributions")
            for col in meaningful_cat[:3]:
                st.plotly_chart(chart_categorical_bar(df, col), use_container_width=True)
            charts_rendered += 1

    if charts_rendered == 0:
        st.warning("Not enough data variety to generate charts.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
    with col2:
        if st.button("➡️ Next: Modeling & Evaluation", use_container_width=True):
            st.session_state.current_page = "modeling"
            st.rerun()
