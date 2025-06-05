# frontend/streamlit_app.py
import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Hate Speech Detection Tool",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Hate Speech Detection Tool")
st.markdown("Enter text content to analyze for hate speech, toxicity, and policy violations.")

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Input Section
with st.container():
    st.subheader("Content Input")
    user_input = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="Type or paste content here..."
    )
    
    analyze_button = st.button("Analyze Content", type="primary")

# Results Section
if analyze_button and user_input.strip():
    with st.spinner("Analyzing content..."):
        try:
            # Call FastAPI endpoint
            response = requests.post(
                f"{API_BASE_URL}/moderate",
                json={"text": user_input}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result["success"]:
                    # Display Classification
                    st.subheader("🎯 Classification Result")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Classification", result["classification"]["label"])
                    with col2:
                        st.metric("Confidence", f"{result['classification']['confidence']:.2f}")
                    
                    st.info(result["classification"]["explanation"])
                    
                    # Display Retrieved Policies
                    st.subheader("📋 Relevant Policies")
                    for policy in result["retrieved_policies"]:
                        with st.expander(f"📄 {policy['filename']} (Score: {policy['relevance_score']:.3f})"):
                            st.text(policy["content"])
                    
                    # Display Reasoning
                    st.subheader("🤔 Policy Reasoning")
                    st.write(result["reasoning"])
                    
                    # Display Recommended Action
                    st.subheader("⚡ Recommended Action")
                    action_colors = {
                        "ban": "🔴",
                        "warn": "🟡", 
                        "flag": "🟠",
                        "allow": "🟢",
                        "review": "🔵"
                    }
                    action = result["recommended_action"]
                    st.markdown(f"### {action_colors.get(action, '⚪')} {action.upper()}")
                    
                else:
                    st.error(f"Analysis failed: {result.get('error_message', 'Unknown error')}")
            else:
                st.error(f"API Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure FastAPI server is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

elif analyze_button and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")

# Sidebar Information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses AI to detect:
    - **Hate Speech**: Content promoting hatred
    - **Toxic Content**: Harmful or abusive material  
    - **Offensive Content**: Inappropriate material
    - **Neutral Content**: Acceptable content
    - **Ambiguous Content**: Unclear cases
    """)
    
    st.header("🔧 Classification Labels")
    st.markdown("""
    - 🔴 **Hate**: Severe violations
    - 🟠 **Toxic**: Harmful content
    - 🟡 **Offensive**: Inappropriate content
    - 🟢 **Neutral**: Safe content
    - 🔵 **Ambiguous**: Needs review
    """)