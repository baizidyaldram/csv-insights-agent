import streamlit as st
import pandas as pd
import numpy as np
from utils.session import get_df, is_data_loaded


def render():
    st.markdown("## 🔍 Data Quality Agent")
    st.markdown("Automated quality assessment — no AI required.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()
    
    if st.button("▶ Run Data Quality Analysis"):
        with st.spinner("Analyzing data quality..."):
            # Simple quality metrics
            missing_cells = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()
            total_cells = df.size
            
            completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
            duplicate_rate = (duplicate_rows / len(df)) * 100 if len(df) > 0 else 0
            
            st.session_state.quality_report = {
                "score": round((completeness - duplicate_rate), 1),
                "completeness": round(completeness, 1),
                "duplicate_rate": round(duplicate_rate, 1),
                "duplicate_rows": int(duplicate_rows),
                "missing_cells": int(missing_cells),
                "total_rows": len(df),
                "total_cols": len(df.columns),
            }
    
    if st.session_state.get("quality_report"):
        r = st.session_state.quality_report
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Completeness", f"{r['completeness']}%")
        col2.metric("Duplicate Rate", f"{r['duplicate_rate']}%")
        col3.metric("Missing Cells", f"{r['missing_cells']:,}")
        col4.metric("Overall Score", f"{r['score']}/100")
        
        st.markdown("---")
        st.dataframe(df.head(10), use_container_width=True)
        
        if st.button("➡️ Next: Data Cleaning"):
            st.session_state.current_page = "cleaning"
            st.rerun()
    else:
        st.info("Click the button above to analyze your data quality.")
