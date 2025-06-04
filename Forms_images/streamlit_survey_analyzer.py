#!/usr/bin/env python3
"""
Streamlit Web Interface for Google Drive Survey Image Analyzer

A user-friendly web interface for analyzing survey data with Google Drive images.
Features real-time image downloading, AI analysis, and interactive reports.
"""

import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from datetime import datetime
import base64
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from google_drive_image_analyzer import GoogleDriveImageAnalyzer
import requests
from io import BytesIO

# Configure Streamlit page
st.set_page_config(
    page_title="AI Survey Image Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'survey_data' not in st.session_state:
        st.session_state.survey_data = None
    if 'downloaded_images' not in st.session_state:
        st.session_state.downloaded_images = {}
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def display_image_grid(images_dict, survey_data):
    """Display images in a responsive grid layout"""
    if not images_dict:
        st.warning("No images to display")
        return
    
    # Calculate grid layout
    num_images = len(images_dict)
    cols_per_row = min(3, num_images)
    rows_needed = (num_images + cols_per_row - 1) // cols_per_row
    
    st.subheader(f"📸 Downloaded Images ({num_images} total)")
    
    image_items = list(images_dict.items())
    
    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            img_idx = row * cols_per_row + col_idx
            
            if img_idx < len(image_items):
                response_idx, img_path = image_items[img_idx]
                
                with cols[col_idx]:
                    try:
                        # Display image
                        image = Image.open(img_path)
                        st.image(image, caption=f"Response {response_idx + 1}", use_column_width=True)
                        
                        # Display survey context
                        with st.expander(f"Survey Data - Response {response_idx + 1}"):
                            response_data = survey_data.iloc[response_idx]
                            for col, value in response_data.items():
                                if col != "What apps are you using ":  # Skip the drive URL column
                                    st.write(f"**{col}**: {value}")
                    
                    except Exception as e:
                        st.error(f"Error displaying image {img_path}: {e}")

def create_analysis_dashboard(analysis_data):
    """Create an interactive dashboard for analysis results"""
    st.subheader("📊 Analysis Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Responses", 
            analysis_data.get('total_responses', 0),
            help="Number of survey responses"
        )
    
    with col2:
        st.metric(
            "Images Downloaded", 
            analysis_data.get('responses_with_images', 0),
            help="Successfully downloaded images"
        )
    
    with col3:
        success_rate = analysis_data.get('download_success_rate', 0)
        st.metric(
            "Success Rate", 
            f"{success_rate:.1f}%",
            help="Percentage of successful downloads"
        )
    
    with col4:
        timestamp = analysis_data.get('analysis_timestamp', 'Unknown')
        if timestamp != 'Unknown':
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
        st.metric(
            "Last Analysis", 
            timestamp,
            help="When the analysis was performed"
        )
    
    # Individual analysis results
    individual_analyses = analysis_data.get('individual_analyses', {})
    
    if individual_analyses:
        st.subheader("🔍 Individual Image Analysis")
        
        # Create tabs for each response
        response_tabs = st.tabs([f"Response {int(idx) + 1}" for idx in individual_analyses.keys()])
        
        for tab_idx, (response_idx, analysis) in enumerate(individual_analyses.items()):
            with response_tabs[tab_idx]:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Display image if exists
                    img_path = analysis.get('file_path')
                    if img_path and os.path.exists(img_path):
                        image = Image.open(img_path)
                        st.image(image, caption=f"Response {int(response_idx) + 1}", use_column_width=True)
                    
                    # Technical details
                    st.write("**Technical Details:**")
                    st.write(f"- Dimensions: {analysis.get('dimensions', 'N/A')}")
                    st.write(f"- Format: {analysis.get('format', 'N/A')}")
                    st.write(f"- File Size: {analysis.get('file_size', 0):,} bytes")
                
                with col2:
                    # Survey context
                    st.write("**Survey Response:**")
                    survey_context = analysis.get('survey_context', {})
                    for key, value in survey_context.items():
                        if key != "What apps are you using ":  # Skip drive URL
                            st.write(f"**{key}**: {value}")
                    
                    # AI Analysis placeholder
                    st.write("**AI Analysis:**")
                    ai_analysis = analysis.get('ai_analysis', {})
                    st.info(ai_analysis.get('description', 'No AI analysis available'))

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Survey Image Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("Analyze Google Forms survey data with integrated Google Drive image processing")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Survey CSV File",
            type=['csv'],
            help="Upload your Google Forms survey data CSV file"
        )
        
        # Output directory
        output_dir = st.text_input(
            "Output Directory",
            value="survey_analysis",
            help="Directory to save downloaded images and analysis"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        download_images = st.checkbox("Download Images from Google Drive", value=True)
        run_ai_analysis = st.checkbox("Run AI Image Analysis", value=False, help="Requires OpenAI API key")
        
        if run_ai_analysis:
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key for image analysis"
            )
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analyzer
        if st.session_state.analyzer is None:
            st.session_state.analyzer = GoogleDriveImageAnalyzer(temp_file_path, output_dir)
        
        # Load and display survey data
        if st.session_state.survey_data is None:
            with st.spinner("Loading survey data..."):
                st.session_state.survey_data = st.session_state.analyzer.load_survey_data()
        
        if st.session_state.survey_data is not None:
            st.success(f"✅ Loaded {len(st.session_state.survey_data)} survey responses")
            
            # Display data preview
            with st.expander("📋 Survey Data Preview"):
                st.dataframe(st.session_state.survey_data)
            
            # Download images button
            if download_images and not st.session_state.downloaded_images:
                if st.button("🔄 Download Images from Google Drive", type="primary"):
                    with st.spinner("Downloading images from Google Drive..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Download images with progress tracking
                        st.session_state.downloaded_images = st.session_state.analyzer.download_all_images()
                        
                        progress_bar.progress(100)
                        status_text.success(f"Downloaded {len(st.session_state.downloaded_images)} images")
            
            # Display downloaded images
            if st.session_state.downloaded_images:
                display_image_grid(st.session_state.downloaded_images, st.session_state.survey_data)
                
                # Run analysis button
                if st.button("🔍 Generate Analysis Report", type="primary"):
                    with st.spinner("Generating comprehensive analysis..."):
                        analysis_data = st.session_state.analyzer.generate_comprehensive_analysis()
                        report_path = st.session_state.analyzer.create_analysis_report()
                        
                        st.session_state.analysis_complete = True
                        
                        # Display analysis dashboard
                        create_analysis_dashboard(analysis_data)
                        
                        # Download options
                        st.subheader("📥 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download analysis JSON
                            analysis_files = list(Path(output_dir + "/analysis").glob("survey_analysis_*.json"))
                            if analysis_files:
                                latest_analysis = max(analysis_files, key=os.path.getctime)
                                with open(latest_analysis, 'r') as f:
                                    analysis_json = f.read()
                                
                                st.download_button(
                                    label="📊 Download Analysis (JSON)",
                                    data=analysis_json,
                                    file_name=f"survey_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                        
                        with col2:
                            # Download report
                            report_files = list(Path(output_dir + "/analysis").glob("report_*.md"))
                            if report_files:
                                latest_report = max(report_files, key=os.path.getctime)
                                with open(latest_report, 'r') as f:
                                    report_content = f.read()
                                
                                st.download_button(
                                    label="📄 Download Report (Markdown)",
                                    data=report_content,
                                    file_name=f"survey_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file from Google Forms to get started")
        
        # Example data format
        with st.expander("📝 Expected CSV Format"):
            st.write("Your CSV should contain survey responses with Google Drive image links. Example:")
            
            example_data = {
                "Timestamp": ["2025/06/02 3:04:22 PM AST", "2025/06/02 3:04:47 PM AST"],
                "Question 1": ["Answer 1", "Answer 2"],
                "Image Upload": [
                    "https://drive.google.com/open?id=1nlw9nECrgInNFqzohddZd0j_ytSChCFa",
                    "https://drive.google.com/open?id=13qeRxdMwCQeUiqBkQSiYPUinioC1Hwha"
                ]
            }
            
            st.dataframe(pd.DataFrame(example_data))

if __name__ == "__main__":
    main() 