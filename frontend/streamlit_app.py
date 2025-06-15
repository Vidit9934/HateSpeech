import streamlit as st
import speech_recognition as sr
import requests
import json
from datetime import datetime
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Content Moderation System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .policy-card {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
API_BASE_URL = "http://localhost:8000"
EXAMPLE_TEXTS = {
    "Neutral": "Hello everyone, hope you're having a great day!",
    "Offensive": "You're such an idiot, learn to write properly!",
    "Toxic": "I'll make sure you never work in this industry again.",
    "Hate": "All [group] should be banned from our country.",
}

def load_example(text_type):
    st.session_state.user_input = EXAMPLE_TEXTS[text_type]

def analyze_content(text):
    """Call FastAPI backend for content analysis"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/moderate",
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to API server")
    except requests.exceptions.Timeout:
        raise TimeoutError("API request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error: {str(e)}")

def record_audio():
    """Record audio and convert to text using speech recognition"""
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Create a temporary file to store audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            # Use streamlit's audio recorder
            st.write("🎤 Click to start recording...")
            audio_bytes = st.audio_recorder()
            
            if audio_bytes:
                # Save audio bytes to temporary file
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                
                # Use speech recognition
                with sr.AudioFile(temp_audio.name) as source:
                    audio_data = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio_data)
                        return text
                    except sr.UnknownValueError:
                        st.warning("Could not understand audio. Please try again.")
                    except sr.RequestError as e:
                        st.error(f"Error with speech recognition service: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
    finally:
        # Cleanup temporary file
        if 'temp_audio' in locals():
            os.unlink(temp_audio.name)
    
    return None

def voice_search():
    """Record and transcribe voice input using speech recognition"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("🎤 Listening... (speak into your microphone)")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen for input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            try:
                st.info("Processing your speech...")
                text = recognizer.recognize_google(audio)
                st.success(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                st.warning("🔊 Could not understand audio. Please try again.")
            except sr.RequestError as e:
                st.error(f"🚫 Error with speech recognition service: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error accessing microphone: {str(e)}")
    
    return None

# Sidebar
with st.sidebar:
    st.title("🛡️ About")
    st.markdown("""
        This tool helps moderators analyze content for policy violations
        and recommends appropriate actions.
    """)
    
    st.header("🎯 Classification Labels")
    st.markdown("""
        - 🔴 **Hate**: Severe policy violations
        - 🟠 **Toxic**: Harmful/abusive content
        - 🟡 **Offensive**: Inappropriate content  
        - 🟢 **Neutral**: Safe content
        - 🔵 **Ambiguous**: Needs review
    """)
    
    st.header("🔍 Try Examples")
    for text_type in EXAMPLE_TEXTS:
        st.button(f"Test {text_type}", on_click=load_example, args=(text_type,))

# Main Content
st.title("Content Moderation System")

# Input Section
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
                recorded_text = voice_search()
                if recorded_text:
                    st.session_state.user_input = recorded_text
                    user_input = recorded_text
                    st.write("**Transcribed Text:**")
                    st.write(recorded_text)
                else:
                    user_input = ""
            else:
                user_input = st.session_state.get("user_input", "")
    
    with col2:
        st.markdown("### Options")
        clear = st.button("Clear")
        if clear:
            st.session_state.user_input = ""
            user_input = ""

analyze_button = st.button("🔍 Analyze Content", type="primary", disabled=not user_input)

# Analysis Section
if analyze_button and user_input.strip():
    with st.spinner("Analyzing content..."):
        try:
            result = analyze_content(user_input)
            
            # Display Classification
            st.header("Analysis Results")
            cols = st.columns(4)
            
            with cols[0]:
                confidence = result["classification"]["confidence"]
                st.metric(
                    "Classification",
                    result["classification"]["label"].upper(),
                    delta=f"{confidence:.0%} confidence"
                )
                
            with cols[1]:
                severity = result["classification"].get("severity", "UNKNOWN")
                st.metric("Severity", severity)
                
            with cols[2]:
                policies_count = len(result["policy_analysis"]["applicable_policies"])
                st.metric("Policies Analyzed", policies_count)
                
            with cols[3]:
                action = result["recommended_action"]["primary_action"]
                st.metric("Recommended Action", action.upper())

            # Reasoning Section
            st.subheader("📝 Reasoning")
            with st.expander("View Detailed Analysis", expanded=True):
                st.markdown(result["classification"]["reasoning"])
                
                if result["policy_analysis"]["policy_violations"]:
                    st.markdown("#### Policy Violations:")
                    for violation in result["policy_analysis"]["policy_violations"]:
                        st.warning(violation)

            # Policy Details
            st.subheader("📋 Relevant Policies")
            for policy in result["policy_analysis"]["applicable_policies"]:
                with st.expander(
                    f"📄 {policy['source']} - {policy['title']} (Score: {policy['relevance_score']:.2f})"
                ):
                    st.info(policy["content"])
                    if policy.get("section"):
                        st.caption(f"Section: {policy['section']}")

            # Action Section
            st.subheader("⚡ Recommended Action")
            action_colors = {
                "ban": "error",
                "warn": "warning",
                "flag": "warning",
                "allow": "success",
                "review": "info"
            }
            
            action_data = result["recommended_action"]
            action_type = action_data["primary_action"].lower()

            # Use appropriate Streamlit alert components
            if action_type in ["ban", "immediate_removal", "permanent_ban"]:
                st.error(action_data["justification"])
            elif action_type in ["warn", "flag", "temporary_ban"]:
                st.warning(action_data["justification"])
            elif action_type in ["allow", "no_action"]:
                st.success(action_data["justification"])
            else:
                st.info(action_data["justification"])

            if action_data.get("follow_up_actions"):
                st.markdown("#### Follow-up Actions:")
                for action in action_data["follow_up_actions"]:
                    st.info(action)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please make sure the FastAPI server is running and try again.")

# Footer
st.divider()
st.caption(
    "Content Moderation System v1.0 - Made with Streamlit & FastAPI"
)