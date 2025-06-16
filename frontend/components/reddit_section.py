import streamlit as st
from utils.api import analyze_reddit_comments
from utils.export import export_to_csv  # Add this import
from datetime import datetime

def render_reddit_section():
    """Render the Reddit comment analysis section"""
    st.header("🤖 Reddit Comment Analysis")
    
    # Input for Reddit post URL
    post_url = st.text_input(
        "Reddit Post URL:",
        placeholder="https://www.reddit.com/r/subreddit/comments/...",
        help="Paste the full URL of a Reddit post to analyze its comments"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        max_comments = st.slider("Maximum comments to analyze:", 5, 100, 20)
    with col2:
        analyze_button = st.button("🔍 Analyze Comments", disabled=not post_url)
    
    if analyze_button and post_url:
        with st.spinner("Analyzing Reddit comments..."):
            try:
                results = analyze_reddit_comments(post_url, max_comments)
                
                # Display results
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.subheader(f"📊 Analysis Results ({len(results)} comments)")
                with col3:
                    export_to_csv(results, "reddit")
                
                for comment in results:
                    with st.expander(f"💬 Comment by u/{comment['author']}"):
                        st.markdown(f"**Comment text:**\n> {comment['text']}")
                        
                        result = comment['moderation_result']
                        
                        # Show classification metrics
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Classification", result['classification']['label'])
                        with cols[1]:
                            st.metric("Confidence", f"{result['confidence_overall']:.1%}")
                        with cols[2]:
                            st.metric("Severity", result['classification']['severity'])
                        
                        # Show policy analysis
                        if result['policy_analysis']['applicable_policies']:
                            st.markdown("#### 📋 Relevant Policies")
                            for policy in result['policy_analysis']['applicable_policies']:
                                st.info(
                                    f"**{policy['title']}** (Score: {policy['relevance_score']:.2f})\n\n"
                                    f"{policy['content']}"
                                )
                        
                        # Show recommended action
                        st.markdown(f"#### ⚡ Recommended Action: {result['recommended_action']['primary_action']}")
                        st.caption(result['recommended_action']['justification'])
                        
                        # Show metadata
                        st.caption(f"Comment ID: {comment['comment_id']} | "
                                 f"Created: {datetime.fromisoformat(str(comment['created_utc'])).strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"❌ Error analyzing Reddit comments: {str(e)}")