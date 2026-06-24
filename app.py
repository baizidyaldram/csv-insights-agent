import streamlit as st
from utils.session import init_session, get_df, is_data_loaded
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="CSV Insight Agents | AI-Powered Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE CSS WITH DARK MODE SUPPORT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    box-sizing: border-box;
}

/* ── WARM ADAPTIVE PALETTE ─────────────────────────────── */
:root {
    --amber-50:  #FAEEDA;
    --amber-100: #FAC775;
    --amber-200: #EF9F27;
    --amber-400: #BA7517;
    --amber-600: #854F0B;
    --amber-800: #633806;
    --coral-50:  #FAECE7;
    --coral-100: #F5C4B3;
    --coral-200: #F0997B;
    --coral-400: #D85A30;
    --coral-600: #993C1D;
    --slate-50:  #F8F5F0;
    --slate-100: #EDE8DF;
    --slate-200: #D5CCBF;
    --slate-400: #9E9385;
    --slate-600: #6B6055;
    --slate-800: #3D3530;
    --slate-900: #221E1B;
}

/* Light mode */
.stApp {
    background-color: var(--slate-50);
}

.stApp .main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: #FFFFFF;
    border-radius: 20px;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
    border: 1px solid var(--slate-100);
    box-shadow: 0 2px 12px rgba(61,53,48,0.06);
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    .stApp { background-color: #1A1714 !important; }
    .stApp .main .block-container {
        background: #241F1B !important;
        border-color: #3A3028 !important;
        box-shadow: 0 2px 20px rgba(0,0,0,0.4) !important;
    }
}

#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFF8F0;
    border-right: 1px solid var(--slate-100);
}

@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] {
        background: #1E1915 !important;
        border-right-color: #3A3028 !important;
    }
}

[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    text-align: left;
    padding: 0.6rem 1rem;
    transition: all 0.2s ease;
    font-weight: 500;
    color: var(--slate-800);
}

@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] .stButton button {
        color: #EDE8DF !important;
    }
}

[data-testid="stSidebar"] .stButton button:hover {
    background: var(--amber-50);
    border-color: var(--amber-100);
}

[data-testid="stSidebar"] .stButton button[data-testid="baseButton-primary"] {
    background: var(--amber-200);
    border-color: var(--amber-400);
    color: #fff !important;
    font-weight: 600;
}

/* ── AGENT CARDS ────────────────────────────────────────── */
.agent-card {
    background: var(--slate-50);
    border: 1px solid var(--slate-100);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    transition: all 0.25s ease;
}

.agent-card:hover {
    border-color: var(--amber-100);
    background: var(--amber-50);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(186,117,23,0.1);
}

.agent-card h4 {
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: var(--slate-800);
}

.agent-card p {
    font-size: 0.78rem;
    color: var(--slate-400);
    margin: 0;
}

/* Dark mode cards */
@media (prefers-color-scheme: dark) {
    .agent-card { background: #2A2420; border-color: #3A3028; }
    .agent-card h4 { color: #EDE8DF; }
    .agent-card p { color: #9E9385; }
    .agent-card:hover { background: #352B22; border-color: var(--amber-600); }
}

/* ── METRIC CARDS ───────────────────────────────────────── */
.metric-card {
    background: var(--amber-50);
    border: 1px solid var(--amber-100);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(186,117,23,0.12);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--amber-600);
}

.metric-label {
    font-size: 0.68rem;
    color: var(--slate-400);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.4rem;
    font-weight: 500;
}

@media (prefers-color-scheme: dark) {
    .metric-card {
        background: #2A2420;
        border-color: #3A3028;
    }
    .metric-value {
        color: #F0997B;
    }
    .metric-label {
        color: #9E9385;
    }
}

/* ── BUTTONS ────────────────────────────────────────────── */
.stButton > button {
    background: var(--amber-200);
    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.55rem 1.4rem;
    transition: all 0.25s ease;
    box-shadow: 0 2px 8px rgba(186,117,23,0.2);
}

.stButton > button:hover {
    background: var(--amber-400) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(186,117,23,0.3);
}

.stButton > button[data-testid="baseButton-secondary"] {
    background: transparent !important;
    color: var(--slate-600) !important;
    border: 1px solid var(--slate-200) !important;
    box-shadow: none;
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: var(--amber-50) !important;
    border-color: var(--amber-100) !important;
}

@media (prefers-color-scheme: dark) {
    .stButton > button[data-testid="baseButton-secondary"] {
        color: #EDE8DF !important;
        border-color: #4A3F37 !important;
    }
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #2A2420 !important;
        border-color: #EF9F27 !important;
    }
    .stDownloadButton button {
        background: #EF9F27 !important;
        color: #1A1714 !important;
    }
    .stDownloadButton button:hover {
        background: #D85A30 !important;
        color: #FFFFFF !important;
    }
}

/* ── TYPOGRAPHY ─────────────────────────────────────────── */
h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: var(--slate-800) !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem !important;
}

h2 {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--slate-800) !important;
    margin-top: 1rem !important;
    margin-bottom: 0.6rem !important;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--slate-600) !important;
    margin-bottom: 0.4rem !important;
}

@media (prefers-color-scheme: dark) {
    h1, h2 { color: #EDE8DF !important; }
    h3 { color: #C8BFB4 !important; }
}

/* ── TABS ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: var(--slate-50);
    padding: 0.4rem;
    border-radius: 50px;
    border: 1px solid var(--slate-100);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 50px;
    padding: 0.45rem 1.2rem;
    font-weight: 500;
    color: var(--slate-400);
}

.stTabs [aria-selected="true"] {
    background: var(--amber-200) !important;
    color: white !important;
}

@media (prefers-color-scheme: dark) {
    .stTabs [data-baseweb="tab-list"] {
        background: #2A2420 !important;
        border-color: #3A3028 !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9E9385 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #EDE8DF !important;
    }
    .stTabs [aria-selected="true"] {
        background: #EF9F27 !important;
        color: #1A1714 !important;
    }
    .stTabs [aria-selected="true"]:hover {
        color: #1A1714 !important;
    }
}

/* ── PROGRESS ───────────────────────────────────────────── */
.stProgress > div > div {
    background: var(--amber-200);
}

@media (prefers-color-scheme: dark) {
    .stProgress > div > div {
        background: #EF9F27 !important;
    }
    .stProgress > div {
        background: #2A2420 !important;
    }
}

/* ── ALERTS ─────────────────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    border-left: 4px solid var(--amber-200) !important;
}

@media (prefers-color-scheme: dark) {
    .stAlert {
        background: #2A2420 !important;
        border-color: #4A3F37 !important;
    }
    .stAlert p {
        color: #EDE8DF !important;
    }
    .stAlert .st-ae {
        color: #EDE8DF !important;
    }
}

/* ── FILE UPLOADER ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--amber-100);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--amber-200);
    background: var(--amber-50);
}

@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploader"] {
        border-color: #4A3F37 !important;
        background: #2A2420 !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #EF9F27 !important;
        background: #3A3028 !important;
    }
    [data-testid="stFileUploader"] p {
        color: #EDE8DF !important;
    }
}

/* ── DIVIDERS ────────────────────────────────────────────── */
hr {
    margin: 1.5rem 0;
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--slate-200), transparent);
}

@media (prefers-color-scheme: dark) {
    hr {
        background: linear-gradient(90deg, transparent, #3A3028, transparent);
    }
}

/* ── SCROLLBAR ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--slate-50); border-radius: 10px; }
::-webkit-scrollbar-thumb { background: var(--amber-100); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-200); }

@media (prefers-color-scheme: dark) {
    ::-webkit-scrollbar-track { background: #2A2420; }
    ::-webkit-scrollbar-thumb { background: #4A3F37; }
    ::-webkit-scrollbar-thumb:hover { background: #5A4F45; }
}

/* ── BADGES ──────────────────────────────────────────────── */
.badge-ready {
    background: #EAF3DE;
    color: #3B6D11;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
}

.badge-waiting {
    background: var(--amber-50);
    color: var(--amber-600);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    display: inline-block;
}

@media (prefers-color-scheme: dark) {
    .badge-ready {
        background: #2A4A1A;
        color: #A0D080;
    }
    .badge-waiting {
        background: #3A3028;
        color: #FAC775;
    }
}

/* ── CODE ────────────────────────────────────────────────── */
code, pre {
    background: var(--slate-50);
    border-radius: 8px;
    border: 1px solid var(--slate-100);
    color: var(--coral-600);
}

@media (prefers-color-scheme: dark) {
    code, pre {
        background: #2A2420;
        border-color: #3A3028;
        color: #F0997B;
    }
}

/* ── SELECT / INPUT ─────────────────────────────────────── */
[data-baseweb="select"] { border-radius: 10px; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    border: 1px solid var(--slate-200);
    border-radius: 10px;
}
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
    border-color: var(--amber-200);
    box-shadow: 0 0 0 2px rgba(239,159,39,0.15);
}

[data-baseweb="tag"] {
    background: var(--amber-50);
    border-radius: 8px;
}

@media (prefers-color-scheme: dark) {
    /* Fix all select boxes */
    [data-baseweb="select"] {
        background: #2A2420 !important;
    }
    [data-baseweb="select"] [data-testid="stSelectbox"] {
        background: #2A2420 !important;
        color: #EDE8DF !important;
    }
    [data-baseweb="select"] input {
        color: #EDE8DF !important;
    }
    [data-baseweb="select"] .st-bm {
        color: #EDE8DF !important;
    }
    
    /* Fix all multi-select */
    [data-baseweb="select"] [data-testid="stMultiSelect"] {
        background: #2A2420 !important;
    }
    [data-baseweb="select"] [data-testid="stMultiSelect"] input {
        color: #EDE8DF !important;
    }
    
    /* Fix dropdown menus */
    [data-baseweb="select"] ul {
        background: #2A2420 !important;
        border-color: #4A3F37 !important;
    }
    [data-baseweb="select"] li {
        color: #EDE8DF !important;
    }
    [data-baseweb="select"] li:hover {
        background: #3A3028 !important;
    }
    [data-baseweb="select"] li[aria-selected="true"] {
        background: #4A3F37 !important;
        color: #F0E6D8 !important;
    }
    
    /* Fix tags in multi-select */
    [data-baseweb="tag"] {
        background: #4A3F37 !important;
        color: #EDE8DF !important;
        border-color: #5A4F45 !important;
    }
    [data-baseweb="tag"] span {
        color: #EDE8DF !important;
    }
    [data-baseweb="tag"] button {
        color: #EDE8DF !important;
    }
    [data-baseweb="tag"] button:hover {
        color: #F0997B !important;
        background: rgba(240, 153, 123, 0.2) !important;
    }
    
    /* Fix all text inputs */
    [data-testid="stTextInput"] input {
        color: #EDE8DF !important;
        background: #2A2420 !important;
        border-color: #4A3F37 !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #EF9F27 !important;
        box-shadow: 0 0 0 2px rgba(239,159,39,0.2) !important;
    }
    
    /* Fix all number inputs */
    [data-testid="stNumberInput"] input {
        color: #EDE8DF !important;
        background: #2A2420 !important;
        border-color: #4A3F37 !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: #EF9F27 !important;
        box-shadow: 0 0 0 2px rgba(239,159,39,0.2) !important;
    }
    
    /* Fix text areas */
    [data-testid="stTextArea"] textarea {
        color: #EDE8DF !important;
        background: #2A2420 !important;
        border-color: #4A3F37 !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #EF9F27 !important;
        box-shadow: 0 0 0 2px rgba(239,159,39,0.2) !important;
    }
    
    /* Fix radio buttons */
    .stRadio label {
        color: #EDE8DF !important;
    }
    .stRadio .st-ae {
        color: #EDE8DF !important;
    }
    .stRadio [data-testid="stRadio"] {
        color: #EDE8DF !important;
    }
    
    /* Fix checkboxes */
    .stCheckbox label {
        color: #EDE8DF !important;
    }
    .stCheckbox .st-ae {
        color: #EDE8DF !important;
    }
    
    /* Fix sliders */
    .stSlider label {
        color: #EDE8DF !important;
    }
    .stSlider .st-bm {
        color: #EDE8DF !important;
    }
    
    /* Fix dataframes */
    .stDataFrame {
        background: #1A1714 !important;
    }
    .stDataFrame table {
        color: #EDE8DF !important;
    }
    .stDataFrame th {
        background: #2A2420 !important;
        color: #EDE8DF !important;
        border-color: #3A3028 !important;
    }
    .stDataFrame td {
        color: #D5CCBF !important;
        border-color: #2A2420 !important;
    }
    .stDataFrame tr:hover td {
        background: #2A2420 !important;
    }
    
    /* Fix expanders */
    .streamlit-expanderHeader {
        color: #EDE8DF !important;
        background: #2A2420 !important;
        border-color: #3A3028 !important;
    }
    .streamlit-expanderHeader:hover {
        background: #3A3028 !important;
    }
    .streamlit-expanderContent {
        background: #1A1714 !important;
        border-color: #3A3028 !important;
    }
    
    /* Fix report container */
    .report-wrapper {
        background: #2A2420 !important;
        border-color: #4A3F37 !important;
    }
    .report-container h2 {
        color: #F5E6D0 !important;
        border-left-color: #F0997B !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
    }
    .report-container h3 {
        color: #E8D5C0 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
    }
    .report-container strong {
        color: #F5A07A !important;
    }
    .report-container li {
        color: #D5CCBF !important;
    }
    .report-container p {
        color: #D5CCBF !important;
    }
    .report-container ul {
        color: #D5CCBF !important;
        list-style-type: disc !important;
    }
    .report-container li::marker {
        color: #F0997B !important;
    }
    .report-container {
        color: #D5CCBF !important;
    }
}

@media (max-width: 768px) {
    .stApp .main .block-container { padding: 1rem !important; }
    h1 { font-size: 1.8rem !important; }
    .metric-value { font-size: 1.4rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# INIT SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
init_session()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
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
        "📋 AI Report": "ai_report",
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
        ("AI Report Agent", "Ready" if report_ready else "Waiting", "📋"),
    ]
    
    for name, status, icon in agent_statuses:
        if status == "Ready":
            status_style = "color:#3B6D11; background:#EAF3DE; border:1px solid #C0DD97;"
        else:
            status_style = "color:#854F0B; background:#FAEEDA; border:1px solid #FAC775;"
        
        # Dark mode overrides for status badges
        if status == "Ready":
            status_style_dark = "color:#A0D080; background:#2A4A1A; border:1px solid #3A6A2A;"
        else:
            status_style_dark = "color:#FAC775; background:#3A3028; border:1px solid #5A4F45;"
        
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; padding:0.4rem 0.6rem; background:rgba(0,0,0,0.03); border:1px solid rgba(0,0,0,0.06); border-radius:8px; margin-bottom:0.3rem;">
            <span style="font-size:0.9rem; margin-right:0.5rem;">{icon}</span>
            <span style="font-size:0.78rem; font-weight:500; flex-grow:1; color:#3D3530;">{name}</span>
            <span style="font-size:0.68rem; font-weight:600; padding:2px 7px; border-radius:5px; {status_style}">{status}</span>
        </div>
        <style>
        @media (prefers-color-scheme: dark) {{
            div[style*="display:flex; align-items:center;"] {{
                background: #2A2420 !important;
                border-color: #3A3028 !important;
            }}
            div[style*="display:flex; align-items:center;"] span {{
                color: #EDE8DF !important;
            }}
            div[style*="display:flex; align-items:center;"] span:last-child {{
                {status_style_dark}
            }}
        }}
        </style>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    from utils.llm import get_api_key
    if not get_api_key():
        st.markdown("### 🔑 OpenRouter API Key")
        api_key_input = st.text_input("Enter Key", type="password", help="Required for AI Insights & Report agents.")
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

# ──────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ──────────────────────────────────────────────────────────────────────────────
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
