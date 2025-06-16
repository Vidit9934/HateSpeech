import pandas as pd
from typing import List, Dict, Any
from io import StringIO
import streamlit as st
from datetime import datetime

def export_to_csv(results: List[Dict[str, Any]], analysis_type: str = "text") -> None:
    """
    Convert analysis results to CSV and create a download button.
    
    Args:
        results: List of analysis results
        analysis_type: Either "text" or "reddit" to determine data structure
    """
    try:
        if analysis_type == "reddit":
            # Format Reddit results
            data = []
            for comment in results:
                mod_result = comment['moderation_result']
                data.append({
                    'Comment ID': comment['comment_id'],
                    'Author': comment['author'],
                    'Text': comment['text'],
                    'Classification': mod_result['classification']['label'],
                    'Confidence': f"{mod_result['confidence_overall']:.1%}",
                    'Severity': mod_result['classification']['severity'],
                    'Action': mod_result['recommended_action']['primary_action'],
                    'Justification': mod_result['recommended_action']['justification'],
                    'Policy Violations': len(mod_result['policy_analysis']['policy_violations']),
                    'Timestamp': mod_result['timestamp']
                })
        else:
            # Format single text analysis result
            mod_result = results
            data = [{
                'Text': mod_result['original_text'],
                'Classification': mod_result['classification']['label'],
                'Confidence': f"{mod_result['classification']['confidence']:.1%}",
                'Severity': mod_result['classification']['severity'],
                'Action': mod_result['recommended_action']['primary_action'],
                'Justification': mod_result['recommended_action']['justification'],
                'Policy Violations': len(mod_result['policy_analysis']['policy_violations']),
                'Timestamp': mod_result['timestamp']
            }]

        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert to CSV
        csv = df.to_csv(index=False)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moderation_results_{analysis_type}_{timestamp}.csv"
        
        # Create download button
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            help="Download the analysis results in CSV format"
        )

    except Exception as e:
        st.error(f"❌ Error exporting results: {str(e)}")