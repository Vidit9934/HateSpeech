import streamlit as st
from utils.constants import EXAMPLE_TEXTS

def render_sidebar():
    """Render the sidebar with about info and examples"""
    st.sidebar.title("🛡️ About")
    st.sidebar.markdown("""
        This tool helps moderators analyze content for policy violations
        and recommends appropriate actions.
    """)
    
    st.sidebar.header("🎯 Classification Labels")
    st.sidebar.markdown("""
        - 🔴 **Hate**: Severe policy violations
        - 🟠 **Toxic**: Harmful/abusive content
        - 🟡 **Offensive**: Inappropriate content  
        - 🟢 **Neutral**: Safe content
        - 🔵 **Ambiguous**: Needs review
    """)
    
    st.sidebar.header("🔍 Try Examples")
    for text_type in EXAMPLE_TEXTS:
        st.sidebar.button(
            f"Test {text_type}", 
            on_click=lambda t=text_type: st.session_state.update({"user_input": EXAMPLE_TEXTS[t]})
        )