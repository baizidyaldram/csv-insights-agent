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
    elif page == "insights":
        from pages_content.insights import render
        render()
    elif page == "report":
        from pages_content.report import render
        render()
except ImportError as e:
    st.error(f"Page not found: {e}")
    st.info("Please ensure all page files exist in the 'pages_content' folder")
