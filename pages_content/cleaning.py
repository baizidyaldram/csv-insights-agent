import streamlit as st
import pandas as pd
import numpy as np
import io
from utils.session import get_df, get_raw_df, update_clean_df, is_data_loaded


def render():
    st.markdown("## 🧹 Data Cleaning Agent")
    st.markdown("Automatically detects and fixes common data issues.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded.")
        if st.button("← Back to Home", use_container_width=False):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df_raw = get_raw_df()
    
    # Show current data statistics
    st.markdown("### 📊 Current Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df_raw.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df_raw.shape[1])
    with col3:
        duplicates = df_raw.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicates:,}")
    with col4:
        missing = df_raw.isnull().sum().sum()
        st.metric("Missing Values", f"{missing:,}")
    
    st.markdown("---")
    
    # Simple cleaning options
    col1, col2 = st.columns(2)
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    with col2:
        fill_missing = st.checkbox("Fill missing numeric values with median", value=True)
    
    if st.button("▶ Run Data Cleaning"):
        df_clean = df_raw.copy()
        log = []
        
        # Remove duplicates
        if remove_duplicates:
            before = len(df_clean)
            df_clean.drop_duplicates(inplace=True)
            removed = before - len(df_clean)
            if removed > 0:
                log.append(f"✅ Removed {removed:,} duplicate rows")
        
        # Fill missing numeric with median
        if fill_missing:
            numeric_cols = df_clean.select_dtypes(include="number").columns
            filled = 0
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    filled += 1
            if filled > 0:
                log.append(f"✅ Filled missing values in {filled} numeric columns")
        
        update_clean_df(df_clean)
        st.session_state.cleaning_report = {
            "log": log,
            "before_shape": df_raw.shape,
            "after_shape": df_clean.shape,
        }
        st.success("✅ Cleaning complete!")
    
    if st.session_state.get("cleaning_report"):
        r = st.session_state.cleaning_report
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows Before", f"{r['before_shape'][0]:,}")
        col2.metric("Rows After", f"{r['after_shape'][0]:,}")
        col3.metric("Rows Removed", f"{r['before_shape'][0] - r['after_shape'][0]:,}")
        
        if r.get("log"):
            for item in r["log"]:
                st.info(item)
        
        st.dataframe(get_df().head(10), use_container_width=True)
        
        # Download button
        df_clean = get_df()
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col2:
            if st.button("➡️ Next: Statistical Analysis", use_container_width=True):
                st.session_state.current_page = "stats"
                st.rerun()
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.current_page = "home"
                st.rerun()
        with col2:
            st.info("Configure options above and click **Run Data Cleaning** to begin.")
