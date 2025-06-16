import streamlit as st
import sys
from pathlib import Path

# Add the frontend directory to Python path
frontend_path = Path(__file__).parent
if str(frontend_path) not in sys.path:
    sys.path.append(str(frontend_path))

from components.sidebar import render_sidebar
from components.input_section import render_input_section
from components.analysis_section import render_analysis
from components.reddit_section import render_reddit_section
from utils.api import analyze_content

# Page config
st.set_page_config(
    page_title="Content Moderation System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("styles/main.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render components
render_sidebar()
st.title("Content Moderation System")

# Create tabs for different analysis types
tab1, tab2 = st.tabs(["📝 Text Analysis", "🤖 Reddit Analysis"])

with tab1:
    user_input = render_input_section()
    analyze_button = st.button("🔍 Analyze Content", type="primary", disabled=not user_input)

    if analyze_button and user_input.strip():
        with st.spinner("Analyzing content..."):
            try:
                result = analyze_content(user_input)
                render_analysis(result)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Please make sure the FastAPI server is running and try again.")

with tab2:
    render_reddit_section()

# Footer
st.divider()
st.caption("Content Moderation System v1.0 - Made with Streamlit & FastAPI")