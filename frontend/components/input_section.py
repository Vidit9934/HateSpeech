import streamlit as st
from utils.audio import record_voice_input

def render_input_section():
    """Render the content input section"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            input_type = st.radio("Input Method:", ["Text", "Voice"], horizontal=True)
            
            if input_type == "Text":
                user_input = st.text_area(
                    "Enter content to analyze:",
                    height=100,
                    key="user_input",
                    help="Paste the content you want to analyze for policy violations"
                )
            else:
                st.write("📢 Voice Input")
                if st.button("🎤 Start Recording"):
                    recorded_text = record_voice_input()
                    if recorded_text:
                        st.session_state.user_input = recorded_text
                        st.write("**Transcribed Text:**")
                        st.write(recorded_text)
        
        with col2:
            st.markdown("### Options")
            if st.button("Clear"):
                st.session_state.user_input = ""
                
    return st.session_state.get("user_input", "")