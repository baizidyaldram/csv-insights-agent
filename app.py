import streamlit as st
from utils.session import init_session, get_df, is_data_loaded
import pandas as pd
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CSV Insight Agents | AI-Powered Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Warm, Professional Theme CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global Reset ── */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* ── Main Background ── */
.stApp {
    background: #f5f0eb;
}

/* ── Main Container ── */
.main .block-container {
    padding: 2rem 2rem !important;
    max-width: 1300px !important;
    margin: 0 auto !important;
    background: rgba(255, 252, 248, 0.92);
    backdrop-filter: blur(12px);
    border-radius: 24px;
    margin-top: 1rem !important;
    margin-bottom: 1rem !important;
    border: 1px solid rgba(200, 180, 165, 0.25);
    box-shadow: 0 8px 32px rgba(160, 140, 120, 0.12);
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f3ee 0%, #efe8e2 100%);
    border-right: 1px solid rgba(200, 180, 165, 0.3);
}

[data-testid="stSidebar"] * {
    color: #3d3530 !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #2d2520 !important;
}

/* ── Sidebar Buttons ── */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 248, 240, 0.6);
    border: 1px solid rgba(200, 180, 165, 0.25);
    border-radius: 12px;
    text-align: left;
    padding: 0.7rem 1rem;
    transition: all 0.25s ease;
    font-weight: 500;
    color: #3d3530 !important;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(200, 180, 165, 0.2);
    border-color: rgba(180, 150, 130, 0.5);
    transform: translateX(4px);
}

[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #c9b09b, #b89a84);
    border: none;
    color: white !important;
    box-shadow: 0 4px 12px rgba(180, 150, 130, 0.25);
}

/* ── Typography ── */
h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #2d2520 !important;
    text-align: center;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #3d3530 !important;
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
}

h3 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #4d4540 !important;
    margin-bottom: 0.5rem !important;
}

p, .stMarkdown {
    color: #5d5550;
    line-height: 1.7;
}

/* ── Agent Cards ── */
.agent-card {
    background: rgba(255, 252, 248, 0.8);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(200, 180, 165, 0.2);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.agent-card:hover {
    transform: translateY(-3px);
    border-color: rgba(180, 150, 130, 0.4);
    background: rgba(255, 248, 240, 0.9);
    box-shadow: 0 8px 24px rgba(160, 140, 120, 0.12);
}

.agent-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
    color: #2d2520;
}

.agent-card p {
    font-size: 0.8rem;
    color: #7d756f;
    margin: 0;
}

/* ── Metric Cards ── */
.metric-card {
    background: rgba(255, 252, 248, 0.8);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(200, 180, 165, 0.2);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(180, 150, 130, 0.3);
    box-shadow: 0 8px 24px rgba(160, 140, 120, 0.1);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #2d2520;
}

.metric-label {
    font-size: 0.7rem;
    color: #8d857f;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    font-weight: 500;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c9b09b, #b89a84);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(180, 150, 130, 0.2);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(180, 150, 130, 0.3);
    background: linear-gradient(135deg, #d4bca8, #c4a690);
}

.stButton > button[data-testid="baseButton-secondary"] {
    background: rgba(240, 232, 224, 0.6);
    box-shadow: none;
    border: 1px solid rgba(200, 180, 165, 0.25);
    color: #3d3530 !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: rgba(200, 180, 165, 0.2);
    border-color: rgba(180, 150, 130, 0.4);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(255, 248, 240, 0.6);
    border-radius: 12px;
    border: 1px solid rgba(200, 180, 165, 0.15);
}

[data-testid="stDataFrame"] table {
    color: #3d3530;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255, 248, 240, 0.4);
    border: 2px dashed rgba(200, 180, 165, 0.3);
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(180, 150, 130, 0.5);
    background: rgba(255, 248, 240, 0.6);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255, 248, 240, 0.4);
    border: 1px solid rgba(200, 180, 165, 0.15);
    border-radius: 12px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(240, 232, 224, 0.4);
    padding: 0.4rem;
    border-radius: 40px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 30px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    color: #6d655f;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #c9b09b, #b89a84);
    color: white !important;
}

/* ── Alerts ── */
.stAlert {
    background: rgba(255, 248, 240, 0.6) !important;
    border: 1px solid rgba(200, 180, 165, 0.2) !important;
    border-radius: 12px !important;
    color: #3d3530 !important;
}

.stAlert [data-testid="stMarkdown"] {
    color: #3d3530 !important;
}

/* ── Badges ── */
.badge-ready {
    background: linear-gradient(135deg, #c9b09b, #b89a84);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    color: white !important;
}

.badge-waiting {
    background: linear-gradient(135deg, #e8d5c8, #dcc8b8);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    color: #5d5550 !important;
}

/* ── Divider ── */
hr {
    margin: 1.5rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200, 180, 165, 0.3), transparent);
}

/* ── Inputs ── */
[data-testid="stTextInput"] input {
    background: rgba(255, 248, 240, 0.6);
    border: 1px solid rgba(200, 180, 165, 0.2);
    border-radius: 12px;
    color: #3d3530;
}

[data-testid="stTextInput"] input:focus {
    border-color: #c9b09b;
    box-shadow: 0 0 0 2px rgba(200, 180, 165, 0.2);
}

[data-testid="stTextArea"] textarea {
    background: rgba(255, 248, 240, 0.6);
    border: 1px solid rgba(200, 180, 165, 0.2);
    border-radius: 12px;
    color: #3d3530;
}

[data-baseweb="select"] {
    background: rgba(255, 248, 240, 0.6);
    border-radius: 12px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(200, 180, 165, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #c9b09b, #b89a84);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #d4bca8, #c4a690);
}

/* ── Slider ── */
[data-testid="stSlider"] {
    color: #b89a84;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
        margin: 0.5rem !important;
    }
    
    h1 {
        font-size: 1.8rem !important;
    }
    
    .metric-value {
        font-size: 1.5rem !important;
    }
}

/* ── Dark Mode Support ── */
@media (prefers-color-scheme: dark) {
    .stApp {
        background: #1a1614;
    }
    
    .main .block-container {
        background: rgba(30, 26, 23, 0.95);
        border-color: rgba(80, 70, 60, 0.3);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1a17 0%, #161210 100%);
        border-right: 1px solid rgba(80, 70, 60, 0.3);
    }
    
    [data-testid="stSidebar"] * {
        color: #d5cdc5 !important;
    }
    
    h1, h2, h3, .metric-value {
        color: #e8e0d8 !important;
    }
    
    p, .stMarkdown {
        color: #b5ada5;
    }
    
    .agent-card {
        background: rgba(40, 35, 30, 0.6);
        border-color: rgba(80, 70, 60, 0.2);
    }
    
    .agent-card h4 {
        color: #e8e0d8;
    }
    
    .metric-card {
        background: rgba(40, 35, 30, 0.6);
        border-color: rgba(80, 70, 60, 0.2);
    }
    
    [data-testid="stDataFrame"] {
        background: rgba(40, 35, 30, 0.4);
    }
}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 CSV Insight Agents")
    st.markdown("### AI-Powered Analysis")
    st.markdown("---")
    
    pages = {
        "🏠 Home": "home",
        "🔍 Data Quality": "quality",
        "🧹 Data Cleaning": "cleaning",
        "📊 Statistical Analysis": "stats",
        "📈 Visualization": "visualization",
        "🤖 Modeling & Evaluation": "modeling",
        "📋 AI Insights & Report": "ai_report",
    }
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    
    for label, key in pages.items():
        is_active = st.session_state.current_page == key
        disabled = key != "home" and not is_data_loaded()
        
        button_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{key}", use_container_width=True, disabled=disabled, type=button_type):
            st.session_state.current_page = key
            st.rerun()
    
    st.markdown("---")
    
    # ── Agent Directory Panel ─────────────────────────────────────────────────
    st.markdown("### 🤖 Agent Status")
    
    quality_ready = st.session_state.get("quality_report") is not None
    cleaning_ready = st.session_state.get("cleaning_report") is not None
    stats_ready = st.session_state.get("stats_done", False)
    viz_ready = st.session_state.get("viz_done", False)
    modeling_ready = st.session_state.get("modeling_done", False)
    report_ready = st.session_state.get("ai_report") is not None
    
    agent_statuses = [
        ("System Orchestrator", "Ready", "⚡"),
        ("Quality Agent", "Ready" if quality_ready else "Waiting", "🔍"),
        ("Cleaning Agent", "Ready" if cleaning_ready else "Waiting", "🧹"),
        ("Statistical Agent", "Ready" if stats_ready else "Waiting", "📊"),
        ("Visuals Agent", "Ready" if viz_ready else "Waiting", "📈"),
        ("Modeling Agent", "Ready" if modeling_ready else "Waiting", "🤖"),
        ("AI Insights & Report", "Ready" if report_ready else "Waiting", "📋"),
    ]
    
    for name, status, icon in agent_statuses:
        is_ready = status == "Ready"
        bg_color = "rgba(200, 180, 165, 0.15); border:1px solid rgba(200, 180, 165, 0.2); color:#4d4540;" if is_ready else "rgba(180, 170, 160, 0.08); border:1px solid rgba(180, 170, 160, 0.1); color:#8d857f;"
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.4rem 0.6rem; background: rgba(255,248,240,0.3); border-radius: 8px; margin-bottom: 0.3rem;">
            <span style="font-size: 0.9rem; margin-right: 0.5rem;">{icon}</span>
            <span style="font-size: 0.75rem; font-weight: 500; color: #5d5550; flex-grow: 1;">{name}</span>
            <span style="font-size: 0.65rem; font-weight: 600; padding: 2px 10px; border-radius: 12px; background:{bg_color}">{status}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # ── API Key configurator ──────────────────────────────────────────────────
    from utils.llm import get_api_key
    if not get_api_key():
        st.markdown("### 🔑 OpenRouter API Key")
        api_key_input = st.text_input("Enter Key", type="password", help="Provide your OpenRouter API key to activate the AI Insights and Report Writing agents.")
        if api_key_input:
            st.session_state.openrouter_api_key = api_key_input
            st.rerun()
        st.markdown("---")

    if is_data_loaded():
        df = get_df()
        st.markdown(f'<span class="badge-ready">✓ Data Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"### {df.shape[0]:,}")
        st.caption("rows")
        st.markdown(f"### {df.shape[1]}")
        st.caption("columns")
        
        with st.expander("📊 Quick Preview"):
            st.dataframe(df.head(3), use_container_width=True)
    else:
        st.markdown(f'<span class="badge-waiting">⏳ No Data</span>', unsafe_allow_html=True)
        st.caption("Upload a CSV to begin")
    
    st.markdown("---")
    st.caption("💡 Click the ◀ arrow to collapse")

# ── Page Router ───────────────────────────────────────────────────────────────
page = st.session_state.current_page

try:
    if page == "home":
        from pages_content.home import render
        render()
        
    elif page == "quality":
        from pages_content.quality import render
        render()
        
    elif page == "cleaning":
        from pages_content.cleaning import render
        render()
        
    elif page == "stats":
        from pages_content.stats import render
        render()
        
    elif page == "visualization":
        from pages_content.visualization import render
        render()
        
    elif page == "modeling":
        from pages_content.modeling import render
        render()
        
    elif page == "ai_report":
        from pages_content.ai_report import render
        render()
        
except ImportError as e:
    st.error(f"Page not found: {e}")
    st.info("Please ensure all page files exist in the 'pages_content' folder")
