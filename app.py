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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Theme CSS with #1371A0 Color Palette ──────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* Global Reset */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Premium Dark Gradient Background with #1371A0 Theme */
.stApp {
    background: radial-gradient(ellipse at 30% 20%, #1a1a2e, #0b0f1a);
    background-attachment: fixed;
}

/* Animated particles effect */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
        radial-gradient(circle at 20% 40%, rgba(19, 113, 160, 0.15) 0%, transparent 50%),
        radial-gradient(circle at 80% 60%, rgba(49, 136, 173, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 50% 80%, rgba(19, 113, 160, 0.08) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* Main content container */
.main .block-container {
    padding: 2rem 2rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: rgba(18, 18, 35, 0.7);
    backdrop-filter: blur(20px);
    border-radius: 32px;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid rgba(19, 113, 160, 0.2);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
}

/* Hide Streamlit Branding */
#MainMenu, footer, header { visibility: hidden; }

/* ============ SIDEBAR STYLING ============ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121f 0%, #0a0a14 100%);
    border-right: 1px solid rgba(19, 113, 160, 0.2);
}

[data-testid="stSidebar"] * {
    color: #e2e2f0 !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(19, 113, 160, 0.2);
    border-radius: 12px;
    text-align: left;
    padding: 0.7rem 1rem;
    transition: all 0.25s ease;
    font-weight: 500;
}

[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(19, 113, 160, 0.15);
    border-color: rgba(49, 136, 173, 0.4);
    transform: translateX(4px);
}

[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #1371A0, #3188AD);
    border: none;
    box-shadow: 0 4px 12px rgba(19, 113, 160, 0.3);
}

/* ============ CARD STYLES ============ */
.agent-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(19, 113, 160, 0.2);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.agent-card:hover {
    transform: translateY(-3px);
    border-color: rgba(49, 136, 173, 0.5);
    background: linear-gradient(135deg, rgba(19, 113, 160, 0.08), rgba(49, 136, 173, 0.04));
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
}

.agent-card h4 {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    background: linear-gradient(135deg, #ffffff, #77B4C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.agent-card p {
    font-size: 0.8rem;
    color: #a0a0c0;
    margin: 0;
}

/* ============ METRIC CARDS ============ */
.metric-card {
    background: linear-gradient(135deg, rgba(19, 113, 160, 0.1), rgba(49, 136, 173, 0.05));
    backdrop-filter: blur(10px);
    border: 1px solid rgba(19, 113, 160, 0.2);
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(49, 136, 173, 0.5);
    background: linear-gradient(135deg, rgba(19, 113, 160, 0.15), rgba(49, 136, 173, 0.08));
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #77B4C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.7rem;
    color: #a0a0c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
    font-weight: 500;
}

/* ============ BUTTON STYLES ============ */
.stButton > button {
    background: linear-gradient(135deg, #1371A0, #3188AD);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(19, 113, 160, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(19, 113, 160, 0.4);
    background: linear-gradient(135deg, #3188AD, #529EBA);
}

/* Secondary button */
.stButton > button[data-testid="baseButton-secondary"] {
    background: rgba(255, 255, 255, 0.05);
    box-shadow: none;
    border: 1px solid rgba(19, 113, 160, 0.2);
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: rgba(19, 113, 160, 0.15);
    border-color: rgba(49, 136, 173, 0.4);
}

/* ============ TYPOGRAPHY ============ */
h1 {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #ffffff, #77B4C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #ffffff, #77B4C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #e2e2f0 !important;
    margin-bottom: 0.5rem !important;
}

p, .stMarkdown {
    color: #c0c0e0;
    line-height: 1.6;
}

/* ============ DATAFRAME STYLING ============ */
[data-testid="stDataFrame"] {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 16px;
    border: 1px solid rgba(19, 113, 160, 0.15);
}

[data-testid="stDataFrame"] table {
    color: #e2e2f0;
}

/* ============ FILE UPLOADER ============ */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02);
    border: 2px dashed rgba(19, 113, 160, 0.3);
    border-radius: 20px;
    padding: 2rem;
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(49, 136, 173, 0.6);
    background: rgba(19, 113, 160, 0.05);
}

/* ============ EXPANDER ============ */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(19, 113, 160, 0.15);
    border-radius: 12px;
}

[data-testid="stExpander"] details {
    background: transparent;
}

/* ============ TABS ============ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.03);
    padding: 0.5rem;
    border-radius: 60px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 50px;
    padding: 0.5rem 1.5rem;
    font-weight: 500;
    color: #a0a0c0;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1371A0, #3188AD);
    color: white;
}

/* ============ PROGRESS BAR ============ */
.stProgress > div > div {
    background: linear-gradient(90deg, #1371A0, #3188AD);
}

/* ============ ALERTS ============ */
.stAlert {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(19, 113, 160, 0.2) !important;
    border-radius: 12px !important;
    color: #e2e2f0 !important;
}

.stAlert [data-testid="stMarkdown"] {
    color: #e2e2f0 !important;
}

/* ============ BADGES ============ */
.badge-ready {
    background: linear-gradient(135deg, #1371A0, #2E7C9E);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    color: white !important;
}

.badge-waiting {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
    color: white !important;
}

/* ============ DIVIDER ============ */
hr {
    margin: 1.5rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(19, 113, 160, 0.3), transparent);
}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #1371A0, #3188AD);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #3188AD, #529EBA);
}

/* ============ CODE BLOCKS ============ */
code, pre {
    background: rgba(0, 0, 0, 0.4);
    border-radius: 8px;
    padding: 0.2rem 0.4rem;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.85rem;
    color: #77B4C7;
}

/* ============ SELECT BOX ============ */
[data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
}

[data-baseweb="select"] * {
    color: #e2e2f0;
}

/* ============ TEXT INPUT ============ */
[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(19, 113, 160, 0.2);
    border-radius: 12px;
    color: #e2e2f0;
}

[data-testid="stTextInput"] input:focus {
    border-color: #1371A0;
    box-shadow: 0 0 0 2px rgba(19, 113, 160, 0.2);
}

/* ============ TEXT AREA ============ */
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(19, 113, 160, 0.2);
    border-radius: 12px;
    color: #e2e2f0;
}

/* ============ MULTISELECT ============ */
[data-baseweb="tag"] {
    background: rgba(19, 113, 160, 0.2);
    border-radius: 8px;
}

/* ============ SLIDER ============ */
[data-testid="stSlider"] {
    color: #1371A0;
}

/* ============ REPORT TABLE STYLES ============ */
.report-table-container {
    background: linear-gradient(135deg, rgba(19, 113, 160, 0.05), rgba(49, 136, 173, 0.02));
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(19, 113, 160, 0.15);
    margin: 1rem 0;
    overflow-x: auto;
}

.report-table {
    width: 100%;
    border-collapse: collapse;
    color: #e2e2f0;
    font-size: 0.9rem;
    border-spacing: 0;
}

.report-table thead tr {
    background: rgba(19, 113, 160, 0.1);
    border-bottom: 2px solid rgba(19, 113, 160, 0.3);
}

.report-table th {
    text-align: left;
    padding: 0.75rem 1rem;
    color: #77B4C7;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
    border-bottom: 2px solid rgba(19, 113, 160, 0.2);
}

.report-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    vertical-align: middle;
}

.report-table tbody tr:hover {
    background: rgba(19, 113, 160, 0.05);
}

.report-table .status-success {
    color: #2E7C9E;
    font-weight: 600;
}

.report-table .status-warning {
    color: #f59e0b;
    font-weight: 600;
}

.report-table .status-danger {
    color: #ef4444;
    font-weight: 600;
}

/* Column width classes */
.report-table .col-metric { width: 35%; }
.report-table .col-value { width: 35%; }
.report-table .col-status { width: 30%; }
.report-table .col-column { width: 40%; }
.report-table .col-count { width: 25%; }
.report-table .col-pct { width: 25%; }
.report-table .col-action { width: 10%; }
.report-table .col-stat { width: 15%; }
.report-table .col-numeric { width: auto; }

/* Insights box */
.insights-box {
    background: linear-gradient(135deg, rgba(19,113,160,0.08), rgba(49,136,173,0.04));
    border-left: 4px solid #1371A0;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
}

.insights-box h4 {
    color: #77B4C7;
    margin-bottom: 0.5rem;
}

.insights-box p {
    color: #c0c0e0;
    margin: 0.25rem 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem !important;
        margin: 0.5rem !important;
    }
    
    h1 {
        font-size: 2rem !important;
    }
    
    .metric-value {
        font-size: 1.5rem !important;
    }
    
    .report-table-container {
        padding: 0.5rem !important;
    }
    
    .report-table th,
    .report-table td {
        padding: 0.3rem 0.4rem !important;
        font-size: 0.7rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Session init ──────────────────────────────────────────────────────────────
init_session()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 CSV Agent Analyzer")
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
    st.markdown("### 🤖 Agent Directory")
    
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
        status_color = "color:#2E7C9E; background:rgba(46,124,158,0.12); border:1px solid rgba(46,124,158,0.2);" if status == "Ready" else "color:#64748b; background:rgba(148,163,184,0.06); border:1px solid rgba(148,163,184,0.15);"
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.45rem 0.6rem; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 6px; margin-bottom: 0.3rem;">
            <span style="font-size: 0.9rem; margin-right: 0.5rem;">{icon}</span>
            <span style="font-size: 0.78rem; font-weight: 500; color: #94a3b8; flex-grow: 1;">{name}</span>
            <span style="font-size: 0.68rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; {status_color}">{status}</span>
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

# ── Report Table Helper Function - Only used on AI Report Page ──────────────
def render_report_table(df: pd.DataFrame = None):
    """Render a styled report table with metrics - ONLY on AI Report page."""
    
    # Get data if not provided
    if df is None and is_data_loaded():
        df = get_df()
    
    # If no data, show message and return
    if df is None:
        st.info("📊 No data loaded. Upload a CSV to see the report summary.")
        return
    
    # Safely get quality_report
    quality_report = st.session_state.get("quality_report")
    if quality_report is None or not isinstance(quality_report, dict):
        quality_report = {}
    
    cleaning_report = st.session_state.get("cleaning_report")
    if cleaning_report is None or not isinstance(cleaning_report, dict):
        cleaning_report = {}
    
    # Get values
    completeness = quality_report.get('completeness', 0)
    score = quality_report.get('score', 0)
    rows_before = len(df)
    rows_after = cleaning_report.get('after_shape', [rows_before])[0] if cleaning_report else rows_before
    rows_removed = rows_before - rows_after
    
    # ── Data Quality & Cleaning Summary ──
    st.markdown(f"""
    <div class="report-table-container">
        <h3 style="color: #77B4C7; margin-bottom: 1rem;">📊 Data Quality & Cleaning Summary</h3>
        <table class="report-table">
            <thead>
                <tr>
                    <th class="col-metric">Metric</th>
                    <th class="col-value">Value</th>
                    <th class="col-status">Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Total Rows (Original)</strong></td>
                    <td>{rows_before:,}</td>
                    <td class="status-success">✅</td>
                </tr>
                <tr>
                    <td><strong>Rows After Cleaning</strong></td>
                    <td>{rows_after:,}</td>
                    <td class="status-success">✅</td>
                </tr>
                <tr>
                    <td><strong>Rows Removed</strong></td>
                    <td>{rows_removed:,}</td>
                    <td class="{'status-success' if rows_removed == 0 else 'status-warning'}">{'✅' if rows_removed == 0 else '⚠️'}</td>
                </tr>
    """, unsafe_allow_html=True)
    
    if completeness > 0 or score > 0:
        status_class = "status-success" if completeness >= 95 else "status-warning" if completeness >= 80 else "status-danger"
        status_icon = "✅" if completeness >= 95 else "⚠️" if completeness >= 80 else "❌"
        score_class = "status-success" if score >= 85 else "status-warning" if score >= 70 else "status-danger"
        score_icon = "⭐" if score >= 85 else "📊" if score >= 70 else "⚠️"
        
        st.markdown(f"""
                <tr>
                    <td><strong>Completeness</strong></td>
                    <td>{completeness:.1f}%</td>
                    <td class="{status_class}">{status_icon}</td>
                </tr>
                <tr>
                    <td><strong>Data Quality Score</strong></td>
                    <td>{score:.1f}/100</td>
                    <td class="{score_class}">{score_icon}</td>
                </tr>
        """, unsafe_allow_html=True)
    
    st.markdown("""
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Missingness Analysis ──
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) > 0:
        st.markdown("""
        <div class="report-table-container">
            <h3 style="color: #77B4C7; margin-bottom: 1rem;">🔍 Missingness Analysis</h3>
            <table class="report-table">
                <thead>
                    <tr>
                        <th class="col-column">Column</th>
                        <th class="col-count">Missing Count</th>
                        <th class="col-pct">Missing %</th>
                        <th class="col-action">Action</th>
                    </tr>
                </thead>
                <tbody>
        """, unsafe_allow_html=True)
        
        for col, count in missing_cols.head(10).items():
            pct = (count / len(df)) * 100
            if pct < 5:
                action = "Imputed"
                status_class = "status-success"
            elif pct < 30:
                action = "Retained"
                status_class = "status-warning"
            else:
                action = "Investigate"
                status_class = "status-danger"
            
            st.markdown(f"""
            <tr>
                <td><code>{col}</code></td>
                <td>{count:,}</td>
                <td>{pct:.1f}%</td>
                <td class="{status_class}">{action}</td>
            </tr>
            """, unsafe_allow_html=True)
        
        st.markdown("""
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Numeric Summary ──
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        display_cols = numeric_cols[:6]
        
        # Build table header
        header_html = """
        <div class="report-table-container">
            <h3 style="color: #77B4C7; margin-bottom: 1rem;">📈 Numeric Summary</h3>
            <table class="report-table">
                <thead>
                    <tr>
                        <th class="col-stat">Statistic</th>
        """
        
        for col in display_cols:
            header_html += f"<th class='col-numeric'>{col}</th>"
        
        header_html += "</tr></thead><tbody>"
        st.markdown(header_html, unsafe_allow_html=True)
        
        # Add statistics rows
        try:
            desc_df = df[display_cols].describe()
            
            stats_to_display = [
                ('count', 'Count'),
                ('mean', 'Mean'),
                ('std', 'Std. Dev.'),
                ('min', 'Min'),
                ('25%', '25th %'),
                ('50%', 'Median'),
                ('75%', '75th %'),
                ('max', 'Max')
            ]
            
            for stat, label in stats_to_display:
                row_html = f"<tr><td><strong>{label}</strong></td>"
                for col in display_cols:
                    try:
                        val = desc_df.loc[stat, col]
                        if stat == 'count':
                            val_str = f"{int(val):,}"
                        else:
                            val_str = f"{val:,.2f}"
                    except:
                        val_str = "—"
                    row_html += f"<td>{val_str}</td>"
                row_html += "</tr>"
                st.markdown(row_html, unsafe_allow_html=True)
                
        except Exception as e:
            st.markdown(f"<tr><td colspan='{len(display_cols)+1}'>Error: {str(e)}</td></tr>", unsafe_allow_html=True)
        
        st.markdown("""
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Key Insights ──
        st.markdown(f"""
        <div class="insights-box">
            <h4>💡 Key Insights</h4>
            <p><strong>Dataset Overview</strong>: {df.shape[0]:,} rows × {df.shape[1]} columns with 
            {len(numeric_cols)} numeric features and {len(df.select_dtypes(include='object').columns)} categorical features.</p>
            <p><strong>Data Quality</strong>: {completeness:.1f}% completeness with a quality score of {score:.1f}/100.</p>
            <p><strong>Memory Usage</strong>: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB</p>
        </div>
        """, unsafe_allow_html=True)


# ── Page Router - NO report tables on any page except AI Report ─────────────
try:
    if page == "home":
        from pages_content.home import render
        render()
        # Clean home page - no report tables
        
    elif page == "quality":
        from pages_content.quality import render
        render()
        # No report table here - keep it clean
        
    elif page == "cleaning":
        from pages_content.cleaning import render
        render()
        # No report table here - keep it clean
        
    elif page == "stats":
        from pages_content.stats import render
        render()
        # No report table here - keep it clean
        
    elif page == "visualization":
        from pages_content.visualization import render
        render()
        
    elif page == "modeling":
        from pages_content.modeling import render
        render()
        
    elif page == "ai_report":
        from pages_content.ai_report import render
        render()
        # ONLY on the AI Report page - show the full report summary
        if is_data_loaded():
            st.markdown("---")
            st.markdown("### 📊 Detailed Report Summary")
            render_report_table()
            
except ImportError as e:
    st.error(f"Page not found: {e}")
    st.info("Please ensure all page files exist in the 'pages_content' folder")
