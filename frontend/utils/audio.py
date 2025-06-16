import speech_recognition as sr
import streamlit as st

def record_voice_input() -> str:
    """Record and transcribe voice input"""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.write("🎤 Listening... (speak into your microphone)")
            recognizer.adjust_for_ambient_noise(source, duration=1)
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