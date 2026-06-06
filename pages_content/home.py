import streamlit as st
import pandas as pd
from utils.session import set_df, is_data_loaded, get_df
import plotly.express as px
import plotly.graph_objects as go


def render():
    # Hero Section with Animation
    st.markdown("""
    <style>
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        border-radius: 32px;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #f093fb, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.8);
        margin-bottom: 1rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.08));
        border-color: rgba(255,255,255,0.4);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateX(10px);
        border-color: rgba(255,255,255,0.3);
        background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #fff, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        line-height: 1.4;
    }
    
    .upload-area {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border: 2px dashed rgba(255,255,255,0.3);
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(255,255,255,0.1);
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    </style>
    
    <div class="hero-section">
        <div class="hero-title">🤖 CSV Insight Agents</div>
        <div class="hero-subtitle">Multi-agent AI system for automated data analysis and visualization</div>
        <div class="hero-badge">
            ⚡ 6 Intelligent Agents | 🎨 Interactive Visualizations | 🤖 AI-Powered Insights
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🚀 Agent Pipeline")
        
        agents = [
            ("🔍", "Data Quality Agent", "Automatic quality assessment with scoring"),
            ("🧹", "Data Cleaning Agent", "Smart cleaning for missing values & duplicates"),
            ("📊", "Statistical Analysis", "Descriptive stats, correlations & distributions"),
            ("📈", "Visualization Agent", "Interactive charts powered by Plotly"),
            ("💡", "AI Insights Agent", "LLM-powered business insights (OpenRouter)"),
            ("📄", "Report Agent", "Generate comprehensive analysis reports"),
        ]
        
        for icon, title, desc in agents:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📁 Upload Your Data")
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "",
            type=["csv"],
            help="Upload any CSV file - sales data, surveys, financial records, etc.",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                set_df(df, uploaded_file.name)
                st.balloons()
                
                # Success animation
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #00b09b, #96c93d); 
                            border-radius: 12px; padding: 1rem; margin: 1rem 0; text-align: center;">
                    <div style="font-size: 1.2rem; font-weight: 600;">✅ Successfully Loaded!</div>
                    <div style="font-size: 0.9rem;">{uploaded_file.name} ({df.shape[0]:,} rows × {df.shape[1]} columns)</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Divider
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0; position: relative;">
            <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);">
            <span style="position: relative; top: -0.8rem; background: rgba(255,255,255,0.1); padding: 0 1rem; border-radius: 20px;">or</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample dataset button
        if st.button("📦 Load Sample Dataset", use_container_width=True):
            try:
                # Create sample dataset
                sample_df = pd.DataFrame({
                    'order_id': range(1001, 1051),
                    'order_date': pd.date_range('2024-01-01', periods=50, freq='D'),
                    'customer_name': [f'Customer_{i}' for i in range(1, 51)],
                    'product_category': ['Electronics', 'Clothing', 'Books', 'Home', 'Sports'] * 10,
                    'quantity': np.random.randint(1, 10, 50),
                    'price': np.random.uniform(10, 500, 50).round(2),
                    'region': ['North', 'South', 'East', 'West'] * 12 + ['North', 'South'],
                })
                sample_df['total_amount'] = (sample_df['quantity'] * sample_df['price']).round(2)
                set_df(sample_df, "sample_orders.csv")
                st.balloons()
                st.success("✅ Sample e-commerce dataset loaded (50 orders)!")
            except Exception as e:
                st.error(f"Error: {e}")
        
        # Data Preview Section (when data is loaded)
        if is_data_loaded():
            df = get_df()
            
            st.markdown("---")
            st.markdown("### 📊 Dataset Overview")
            
            # Stats Grid
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{df.shape[0]:,}</div>
                    <div class="stat-label">Rows</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{df.shape[1]}</div>
                    <div class="stat-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                numeric_count = df.select_dtypes(include="number").shape[1]
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{numeric_count}</div>
                    <div class="stat-label">Numeric Cols</div>
                </div>
                """, unsafe_allow_html=True)
            with col_d:
                missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{missing_pct:.1f}%</div>
                    <div class="stat-label">Missing Data</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Data Preview
            st.markdown("#### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True, height=300)
            
            # Column info
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2).values,
                    'Unique': df.nunique().values
                })
                st.dataframe(col_info, use_container_width=True)
            
            # Action Buttons
            st.markdown("---")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("🔍 Start Quality Check", use_container_width=True):
                    st.session_state.current_page = "quality"
                    st.rerun()
            with btn_col2:
                if st.button("🧹 Clean Data", use_container_width=True):
                    st.session_state.current_page = "cleaning"
                    st.rerun()
            with btn_col3:
                if st.button("📊 View Statistics", use_container_width=True):
                    st.session_state.current_page = "stats"
                    st.rerun()
            
            # Quick tip
            st.info("💡 **Pro Tip:** Navigate through the 6 agents in order for the best analysis experience!")
        
        else:
            # Empty state
            st.markdown("""
            <div style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📂</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">No Data Loaded</div>
                <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">
                    Upload a CSV file or load the sample dataset to begin
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: rgba(255,255,255,0.5); font-size: 0.8rem;">
        ⚡ Powered by Streamlit | 🤖 AI by OpenRouter | 📊 Visualizations with Plotly
    </div>
    """, unsafe_allow_html=True)
