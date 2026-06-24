import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm


def render():
    # ── CUSTOM CSS FOR REPORT DARK MODE ──────────────────────────────────
    st.markdown("""
    <style>
    /* Report container dark mode fixes */
    @media (prefers-color-scheme: dark) {
        .report-container h2 { 
            color: #F0E6D8 !important; 
            border-left-color: #F0997B !important; 
        }
        .report-container h3 { 
            color: #E0D4C8 !important; 
        }
        .report-container strong { 
            color: #F0997B !important; 
        }
        .report-container li { 
            color: #D5CCBF !important; 
        }
        .report-container p { 
            color: #D5CCBF !important; 
        }
        .report-container ul { 
            color: #D5CCBF !important; 
        }
        .report-container {
            color: #
