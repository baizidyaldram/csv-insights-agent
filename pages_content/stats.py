import streamlit as st
import pandas as pd
from utils.session import get_df, is_data_loaded


def render():
    st.markdown("## 📊 Statistical Analysis Agent")
    st.markdown("Descriptive statistics and correlations.")
    st.markdown("---")

    if not is_data_loaded():
        st.warning("No data loaded.")
        return

    df = get_df()
    
    if st.button("▶ Run Statistical Analysis"):
        st.session_state.stats_done = True
    
    if st.session_state.get("stats_done"):
        numeric_cols = df.select_dtypes(include="number").columns
        
        if len(numeric_cols) > 0:
            st.markdown("### Descriptive Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            if len(numeric_cols) >= 2:
                st.markdown("### Correlation Matrix")
                st.dataframe(df[numeric_cols].corr(), use_container_width=True)
        
        if st.button("➡️ Next: Visualization"):
            st.session_state.current_page = "visualization"
            st.rerun()
    else:
        st.info("Click the button above to run statistical analysis.")
