import streamlit as st
import pandas as pd
from utils.session import set_df, is_data_loaded, get_df


def render():
    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">
            🤖 CSV Insight Agents
        </h1>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.8); margin-bottom: 1rem;">
            Multi-agent AI system for automated data analysis and visualization
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two column layout ──────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🔄 Agent Pipeline")
        
        agents = [
            ("🔍", "Data Quality Agent", "Assesses completeness, validity, and consistency", "Local"),
            ("🧹", "Data Cleaning Agent", "Fixes missing values, duplicates, and type issues", "Local"),
            ("📊", "Statistical Analysis Agent", "Descriptive stats, correlations, distributions", "Local"),
            ("📈", "Visualization Agent", "Auto-generates interactive charts with Plotly", "Local"),
            ("💡", "AI Insights Agent", "LLM-powered business insights and analysis", "AI"),
            ("📄", "Report Writing Agent", "Compiles full analysis into downloadable report", "AI"),
        ]
        
        for icon, name, desc, agent_type in agents:
            badge_color = "#00b09b" if agent_type == "Local" else "#f093fb"
            st.markdown(f"""
            <div class="agent-card">
                <h4>{icon} {name} 
                    <span style="background:{badge_color};padding:2px 8px;border-radius:12px;font-size:0.7rem;margin-left:8px;">
                        {agent_type}
                    </span>
                </h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📁 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload any CSV file - sales data, surveys, financial records, etc."
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                set_df(df, uploaded_file.name)
                st.balloons()
                st.success(f"✅ Successfully loaded **{uploaded_file.name}**!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

        st.markdown("<div style='text-align: center; margin: 1rem 0'>— or —</div>", unsafe_allow_html=True)

        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
                # Create sample data if file doesn't exist
                sample_df = pd.DataFrame({
                    'order_id': range(1001, 1011),
                    'order_date': pd.date_range('2024-01-15', periods=10, freq='D'),
                    'customer_name': ['John Smith', 'Emma Wilson', 'Michael Brown', 'Sarah Davis', 'James Johnson',
                                     'Lisa Anderson', 'Robert Taylor', 'Maria Garcia', 'David Martinez', 'Jennifer Lee'],
                    'product_category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Clothing',
                                        'Home', 'Electronics', 'Books', 'Clothing', 'Home'],
                    'quantity': [2, 3, 1, 1, 4, 2, 1, 5, 2, 1],
                    'price': [299.99, 49.99, 19.99, 899.99, 29.99, 159.99, 499.99, 12.99, 39.99, 299.99],
                    'total_amount': [599.98, 149.97, 19.99, 899.99, 119.96, 319.98, 499.99, 64.95, 79.98, 299.99],
                    'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South']
                })
                set_df(sample_df, "sample_orders.csv")
                st.success("✅ Sample dataset loaded!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

        # ── Data Preview Section ──────────────────────────────────────────────
        if is_data_loaded():
            df = get_df()
            st.markdown("---")
            st.markdown("#### 📊 Dataset Overview")
            
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with m2:
                st.metric("Total Columns", df.shape[1])
            with m3:
                numeric_count = df.select_dtypes(include="number").shape[1]
                st.metric("Numeric Columns", numeric_count)
            with m4:
                missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Data preview
            st.markdown("#### 👀 Data Preview (first 10 rows)")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Quick action button
            if st.button("🚀 Start Analysis", use_container_width=True):
                st.session_state.current_page = "quality"
                st.rerun()
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; margin-top: 1rem;">
                <p style="color: rgba(255,255,255,0.8); margin: 0;">
                    📂 Upload a CSV or load the sample dataset to begin
                </p>
            </div>
            """, unsafe_allow_html=True)
