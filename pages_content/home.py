import streamlit as st
import pandas as pd
import numpy as np
from utils.session import set_df, is_data_loaded, get_df


def render():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>📊 CSV Insight Agents</h1>
        <p style="color: #6B6055; font-size: 1.05rem;">
            Multi-agent AI system for automated data analysis and visualization
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔄 Agent Pipeline")

    col1, col2 = st.columns(2, gap="medium")

    agents_left = [
        ("🔍", "Data Quality Agent",    "Automatic quality assessment with scoring"),
        ("🧹", "Data Cleaning Agent",   "Smart cleaning for missing values & duplicates"),
        ("📊", "Statistical Analysis",  "Descriptive stats, correlations & distributions"),
    ]

    # Merged: AI Insights + Report → one card
    agents_right = [
        ("📈", "Visualization Agent",         "Interactive charts auto-selected by data type"),
        ("🤖", "Modeling & Evaluation Agent", "ML algorithms, GridSearchCV tuning & metrics"),
        ("📋", "AI Report Agent",             "LLM-powered insights + downloadable report"),
    ]

    def agent_card(icon, title, desc):
        return f"""
        <div class="agent-card">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div style="font-size:1.4rem;">{icon}</div>
                <div>
                    <h4 style="margin:0;">{title}</h4>
                    <p style="margin:0; font-size:0.78rem;">{desc}</p>
                </div>
            </div>
        </div>
        """

    with col1:
        for icon, title, desc in agents_left:
            st.markdown(agent_card(icon, title, desc), unsafe_allow_html=True)

    with col2:
        for icon, title, desc in agents_right:
            st.markdown(agent_card(icon, title, desc), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"],
        help="Upload any CSV file — sales, surveys, financial records, etc.",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            set_df(df, uploaded_file.name)
            st.balloons()
            st.success(f"✅ Successfully loaded **{uploaded_file.name}**!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("<div style='text-align:center; margin:1rem 0; color:#9E9385;'>— or —</div>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
                np.random.seed(42)
                sample_df = pd.DataFrame({
                    'order_id':        range(1001, 1051),
                    'order_date':      pd.date_range('2024-01-01', periods=50, freq='D'),
                    'customer_name':   [f'Customer_{i}' for i in range(1, 51)],
                    'product_category': np.random.choice(['Electronics','Clothing','Books','Home','Sports'], 50),
                    'quantity':        np.random.randint(1, 10, 50),
                    'price':           np.random.uniform(10, 500, 50).round(2),
                    'region':          np.random.choice(['North','South','East','West'], 50),
                })
                sample_df['total_amount'] = (sample_df['quantity'] * sample_df['price']).round(2)
                set_df(sample_df, "sample_orders.csv")
                st.balloons()
                st.success("✅ Sample dataset loaded (50 rows)!")
            except Exception as e:
                st.error(f"Error: {e}")

    if is_data_loaded():
        df = get_df()
        st.markdown("---")
        st.markdown("### 📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[0]:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
        with col3:
            nc = df.select_dtypes(include="number").shape[1]
            st.markdown(f'<div class="metric-card"><div class="metric-value">{nc}</div><div class="metric-label">Numeric Cols</div></div>', unsafe_allow_html=True)
        with col4:
            mp = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mp:.1f}%</div><div class="metric-label">Missing Data</div></div>', unsafe_allow_html=True)

        st.markdown("#### 👀 Data Preview (First 10 rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🚀 Quick Actions")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("🔍 Quality Check", key="btn_quality_home", use_container_width=True):
                st.session_state.current_page = "quality"; st.rerun()
        with c2:
            if st.button("🧹 Clean Data", key="btn_cleaning_home", use_container_width=True):
                st.session_state.current_page = "cleaning"; st.rerun()
        with c3:
            if st.button("📊 Statistics", key="btn_stats_home", use_container_width=True):
                st.session_state.current_page = "stats"; st.rerun()
        with c4:
            if st.button("🤖 Modeling", key="btn_modeling_home", use_container_width=True):
                st.session_state.current_page = "modeling"; st.rerun()

        st.info("💡 **Pro Tip:** Navigate through the agents in order for the best analysis experience!")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:0.75rem 0 0; color:#9E9385; font-size:0.73rem;">
        Powered by Streamlit · Plotly · XGBoost · OpenRouter AI
    </div>
    """, unsafe_allow_html=True)
