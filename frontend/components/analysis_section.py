import streamlit as st
from typing import Dict, Any
from datetime import datetime
from utils.export import export_to_csv  # Add this import

def display_confidence_meter(confidence: float, label: str):
    """Display a confidence meter with appropriate coloring"""
    color = "red" if confidence > 0.7 else ("yellow" if confidence > 0.4 else "green")
    st.progress(confidence, f"Confidence: {confidence:.1%}")
    st.markdown(f"<p style='color: {color};'><b>{label.upper()}</b></p>", unsafe_allow_html=True)

def display_policy_violations(policies: list):
    """Display policy violations in expandable sections"""
    if not policies:
        st.info("No policy violations detected")
        return
        
    for policy in policies:
        with st.expander(f"📋 {policy['title']} (Score: {policy['relevance_score']:.2f})"):
            st.markdown(policy["content"])
            if policy.get("section"):
                st.caption(f"Section: {policy['section']}")

def display_action_recommendation(action: Dict[str, Any]):
    """Display recommended actions with color coding"""
    action_colors = {
        "ban": "red",
        "warn": "orange",
        "flag": "yellow",
        "allow": "green",
        "review": "blue"
    }
    
    color = action_colors.get(action["primary_action"].lower(), "blue")
    st.markdown(f"""
    <div style='padding: 1rem; border-left: 5px solid {color}; background: rgba(0,0,0,0.05);'>
        <h4 style='color: {color};'>Recommended Action: {action["primary_action"].upper()}</h4>
        <p>{action["justification"]}</p>
    </div>
    """, unsafe_allow_html=True)

def render_analysis(result: Dict[str, Any]):
    """Render the complete analysis results"""
    try:
        st.header("Analysis Results")
        
        # Add export button at the top
        col1, col2, col3 = st.columns([2, 2, 1])
        with col3:
            export_to_csv(result, "text")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Classification",
                result["classification"]["label"].upper(),
                f"{result['classification']['confidence']:.1%} confidence"
            )
            
        with col2:
            st.metric(
                "Severity",
                result["classification"]["severity"].upper(),
                delta=None
            )
            
        with col3:
            st.metric(
                "Policies Analyzed",
                result["policy_analysis"]["policies_analyzed"],
                delta=None
            )

        # Display detailed classification
        st.subheader("📊 Classification Details")
        st.markdown(f"**Reasoning:** {result['classification']['reasoning']}")
        
        if result["classification"]["detected_categories"]:
            st.markdown("**Detected Categories:**")
            for category in result["classification"]["detected_categories"]:
                st.markdown(f"- {category}")

        # Display policy analysis
        st.subheader("📑 Policy Analysis")
        st.markdown(f"**Summary:** {result['policy_analysis']['reasoning_summary']}")
        
        if result["policy_analysis"]["applicable_policies"]:
            st.markdown("### Relevant Policies")
            display_policy_violations(result["policy_analysis"]["applicable_policies"])

        # Display recommended action
        st.subheader("⚡ Recommended Action")
        display_action_recommendation(result["recommended_action"])

        # Show processing metadata
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        with col2:
            st.caption(f"Analysis Timestamp: {datetime.fromisoformat(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyError as e:
        st.error(f"❌ Error displaying results: Missing field {str(e)}")
    except Exception as e:
        st.error(f"❌ Error displaying results: {str(e)}")