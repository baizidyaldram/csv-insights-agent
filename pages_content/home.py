import streamlit as st
import pandas as pd
from utils.session import set_df, is_data_loaded, get_df


def render():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3.5rem;">🤖 CSV Insight Agents</h1>
        <p style="font-size: 1.2rem; color: rgba(255,255,255,0.8);">
            Multi-agent AI system for automated data analysis and visualization
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 🔄 Agent Pipeline")
        agents = [
            ("🔍", "Data Quality Agent", "Assesses completeness, validity, and consistency"),
            ("🧹", "Data Cleaning Agent", "Fixes missing values, duplicates, and type issues"),
            ("📊", "Statistical Analysis Agent", "Descriptive stats, correlations, distributions"),
            ("📈", "Visualization Agent", "Auto-generates interactive charts with Plotly"),
            ("💡", "AI Insights Agent", "LLM-powered business insights and analysis"),
            ("📄", "Report Writing Agent", "Compiles full analysis into downloadable report"),
        ]
        
        for icon, name, desc in agents:
            st.markdown(f"""
            <div class="agent-card">
                <h4>{icon} {name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📁 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload any CSV file"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                set_df(df, uploaded_file.name)
                st.balloons()
                st.success(f"✅ Successfully loaded **{uploaded_file.name}**!")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("<div style='text-align: center; margin: 1rem 0'>— or —</div>", unsafe_allow_html=True)

        if st.button("📦 Load Sample Dataset", use_container_width=True):
            # Create sample data
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

        if is_data_loaded():
            df = get_df()
            st.markdown("---")
            st.markdown("#### 📊 Dataset Overview")
            
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Rows", f"{df.shape[0]:,}")
            with m2:
                st.metric("Total Columns", df.shape[1])
            with m3:
                st.metric("Numeric Cols", df.select_dtypes(include="number").shape[1])
            with m4:
                missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
                st.metric("Missing %", f"{missing_pct:.1f}%")
            
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Start Analysis", use_container_width=True):
                st.session_state.current_page = "quality"
                st.rerun()
