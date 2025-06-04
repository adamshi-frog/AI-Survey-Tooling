"""
Logging utilities for the Comprehensive Survey Analyzer
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict

class ProcessingLogger:
    """Handles logging of processing steps in the application"""
    
    def __init__(self):
        """Initialize the logger"""
        if 'processing_logs' not in st.session_state:
            st.session_state.processing_logs = []
    
    def log(self, message: str, log_type: str = "info") -> None:
        """Add a processing step to the logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.processing_logs.append({
            "timestamp": timestamp,
            "message": message,
            "type": log_type
        })
    
    def get_logs(self, limit: int = 10) -> List[Dict]:
        """Get the most recent logs"""
        return st.session_state.processing_logs[-limit:]
    
    def clear_logs(self) -> None:
        """Clear all logs"""
        st.session_state.processing_logs = []
    
    def display_logs(self) -> None:
        """Display processing logs in an attractive format"""
        if not st.session_state.processing_logs:
            return
        
        st.subheader("📋 Processing Log")
        
        for log in self.get_logs():
            if log["type"] == "success":
                st.markdown(f'<div class="success-banner">✅ {log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)
            elif log["type"] == "warning":
                st.markdown(f'<div class="warning-banner">⚠️ {log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)
            elif log["type"] == "error":
                st.markdown(f'<div class="warning-banner">❌ {log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-banner">ℹ️ {log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)

# Create a global logger instance
logger = ProcessingLogger() 