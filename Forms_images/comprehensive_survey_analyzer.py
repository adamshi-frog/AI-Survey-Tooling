#!/usr/bin/env python3
"""
Comprehensive Survey Analyzer - Ultimate Unified Application

This application combines all survey analysis tools into one comprehensive platform:
- Google Forms CSV upload and analysis
- Automatic Google Drive image download and processing
- AI-powered image analysis using OpenAI Vision
- Interactive survey insights and visualizations
- Comprehensive reporting and export capabilities
"""

# =============================================================================
# STREAMLIT CONFIGURATION - MUST BE FIRST!
# =============================================================================

import streamlit as st
import io

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Tactile Survey Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Import configuration settings
try:
    from config import (
        OPENAI_API_KEY as BUILTIN_OPENAI_API_KEY,
        DEFAULT_OUTPUT_DIR,
        ENABLE_AI_BY_DEFAULT,
        SHOW_API_KEY_INPUT,
        APP_TITLE,
        APP_DESCRIPTION
    )
except ImportError:
    # Fallback configuration if config.py is not available
    BUILTIN_OPENAI_API_KEY = ""  # No default key
    DEFAULT_OUTPUT_DIR = "output"
    ENABLE_AI_BY_DEFAULT = False  # Disable AI by default since no key
    SHOW_API_KEY_INPUT = True  # Always show API key input
    APP_TITLE = "Tactile Survey Analyzer"
    APP_DESCRIPTION = ""

# Try to load API key from Streamlit secrets
try:
    if 'OPENAI_API_KEY' in st.secrets:
        BUILTIN_OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
except:
    pass

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import json
import os
import re
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO, StringIO
import zipfile
import folium
from streamlit_folium import folium_static

# =============================================================================
# CORE FUNCTIONALITY IMPORTS
# =============================================================================

from enhanced_drive_downloader import EnhancedDriveDownloader, analyze_survey_drive_links
from ai_vision_analyzer import AIVisionAnalyzer
from google_drive_image_analyzer import GoogleDriveImageAnalyzer

# =============================================================================
# OPTIONAL IMPORTS WITH ERROR HANDLING
# =============================================================================

# Function to automatically install packages
def install_package(package_name, import_name=None):
    """Automatically install a package if it's not available"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        try:
            import subprocess
            import sys
            st.info(f"Installing {package_name}... Please wait.")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            st.success(f"✅ {package_name} installed successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to install {package_name}: {e}")
            return False

# Streamlit and visualization imports with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.warning(f"Plotly not found. Attempting automatic installation...")
    if install_package("plotly"):
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            PLOTLY_AVAILABLE = True
        except ImportError:
            st.error(f"""
            ❌ **Plotly installation failed!** 
            
            Please install plotly manually:
            ```bash
            pip install plotly
            ```
            
            Error: {e}
            """)
            PLOTLY_AVAILABLE = False
            # Create dummy objects to prevent further errors
            class DummyPlotly:
                def __init__(self):
                    pass
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
            
            px = DummyPlotly()
            go = DummyPlotly()
    else:
        PLOTLY_AVAILABLE = False
        class DummyPlotly:
            def __init__(self):
                pass
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        
        px = DummyPlotly()
        go = DummyPlotly()

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    st.warning("PIL (Pillow) not found. Attempting automatic installation...")
    if install_package("Pillow", "PIL"):
        try:
            from PIL import Image
            PIL_AVAILABLE = True
        except ImportError:
            st.error("❌ PIL (Pillow) installation failed! Please install: pip install Pillow")
            PIL_AVAILABLE = False
            Image = None
    else:
        PIL_AVAILABLE = False
        Image = None

try:
    from collections import Counter
    import requests
    STANDARD_LIBS_AVAILABLE = True
except ImportError:
    st.error("❌ Standard libraries not available!")
    STANDARD_LIBS_AVAILABLE = False

# Optional import for OpenAI with auto-install fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    st.warning("OpenAI package not found. Attempting automatic installation...")
    if install_package("openai"):
        try:
            import openai
            OPENAI_AVAILABLE = True
        except ImportError:
            st.error("❌ OpenAI installation failed! Please install: pip install openai")
            OPENAI_AVAILABLE = False
            openai = None
    else:
        OPENAI_AVAILABLE = False
        openai = None

# =============================================================================
# STREAMLIT STYLING AND CUSTOM CSS
# =============================================================================

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    
    /* Streamlit tab styling */
    .stTabs [data-baseweb="tab-list"] {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab-panel"] h1, 
    .stTabs [data-baseweb="tab-panel"] h2, 
    .stTabs [data-baseweb="tab-panel"] h3 {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    .main-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #f6f6f3;
        color: #2c2c2c;
        padding: 0.75rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .metric-card h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        color: #2c2c2c;
    }
    .metric-card h2 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        margin: 0;
        color: #2c2c2c;
    }
    .success-banner {
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    .warning-banner {
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(90deg, #ff9800, #f57c00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    .info-banner {
        font-family: 'IBM Plex Sans', sans-serif;
        background: linear-gradient(90deg, #2196F3, #1976D2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    /* Primary button styling */
    .stButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #a3ba65 !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #8fa051 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    }
    /* Download button styling */
    .stDownloadButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #a3ba65 !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: #8fa051 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    }
    /* Form submit button styling */
    .stFormSubmitButton > button {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #a3ba65 !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stFormSubmitButton > button:hover {
        background-color: #8fa051 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2) !important;
    }
    /* Custom styling for dataframes */
    .styled-table th {
        font-family: 'IBM Plex Mono', monospace;
        background-color: #f6f6f3 !important;
        color: #2c2c2c !important;
        font-weight: bold !important;
        border: 1px solid #ddd !important;
    }
    .styled-table td {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #f6f6f3 !important;
        color: #2c2c2c !important;
        border: 1px solid #ddd !important;
    }
    .styled-table table {
        border-collapse: collapse !important;
        border-radius: 5px !important;
        overflow: hidden !important;
    }
    /* Streamlit default text styling */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    .stMarkdown p, .stMarkdown li {
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Utility function to style dataframes
def style_dataframe(df):
    """Apply custom styling to dataframes"""
    return df.style.set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#f6f6f3'),
                ('color', '#2c2c2c'),
                ('font-weight', 'bold'),
                ('border', '1px solid #ddd')
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('background-color', '#f6f6f3'),
                ('color', '#2c2c2c'),
                ('border', '1px solid #ddd')
            ]
        },
        {
            'selector': 'table',
            'props': [
                ('border-collapse', 'collapse'),
                ('border-radius', '5px'),
                ('overflow', 'hidden')
            ]
        }
    ])

def display_styled_dataframe(df, title=None, use_container_width=True):
    """Display a dataframe with custom styling"""
    if title:
        st.subheader(title)
    
    # Check if we have a small enough dataframe to apply styling
    if len(df) <= 100 and len(df.columns) <= 20:
        try:
            styled_df = style_dataframe(df)
            st.write(styled_df.to_html(escape=False, table_id="styled-table"), unsafe_allow_html=True)
        except Exception as e:
            # Fallback to regular dataframe display
            st.dataframe(df, use_container_width=use_container_width)
    else:
        # For large dataframes, use regular display
        st.dataframe(df, use_container_width=use_container_width)

# Load sample data function
@st.cache_data
def load_sample_data():
    """Load sample CSV data for testing purposes"""
    try:
        # Use the correct path for Sample_Survey_Data.csv
        sample_path = '../ai_survey_tool/Sample_Survey_Data.csv'
        if os.path.exists(sample_path):
            sample_data = pd.read_csv(sample_path)
            log_processing_step(f"Loaded sample data from {sample_path} with {len(sample_data)} rows", "success")
            return sample_data
        else:
            st.error(f"Sample data file {sample_path} not found.")
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        log_processing_step(f"Failed to load sample data: {str(e)}", "error")
        return None

def create_mock_survey_data():
    """Create mock survey data for demonstration purposes"""
    import random
    from datetime import datetime, timedelta
    
    # Create mock data
    num_responses = 50
    base_date = datetime.now() - timedelta(days=30)
    
    mock_data = {
        'Timestamp': [base_date + timedelta(days=random.randint(0, 30), 
                                          hours=random.randint(0, 23), 
                                          minutes=random.randint(0, 59)) for _ in range(num_responses)],
        'Name': [f"Respondent {i+1}" for i in range(num_responses)],
        'Email': [f"user{i+1}@example.com" for i in range(num_responses)],
        'Age': [random.randint(18, 65) for _ in range(num_responses)],
        'Location': [random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']) for _ in range(num_responses)],
        'Satisfaction': [random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied']) for _ in range(num_responses)],
        'Comments': [f"This is a sample comment from respondent {i+1}" for i in range(num_responses)],
        'ZIP Code': [f"{random.randint(10000, 99999)}" for _ in range(num_responses)],
        'Photo Upload': [f"https://drive.google.com/file/d/1example{i}/view" if random.random() > 0.3 else "" for i in range(num_responses)]
    }
    
    return pd.DataFrame(mock_data)

def detect_csv_structure(df: pd.DataFrame) -> Dict:
    """Enhanced CSV structure detection with better analysis"""
    analysis = {
        "total_responses": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "response_timestamps": [],
        "google_drive_columns": [],
        "text_columns": [],
        "numeric_columns": [],
        "choice_columns": [],
        "email_columns": [],
        "zip_columns": [],
        "data_quality": {},
        "sample_data": {}
    }
    
    # Analyze each column
    for col in df.columns:
        sample_values = df[col].dropna().astype(str)
        unique_values = sample_values.nunique()
        total_values = len(sample_values)
        
        # Store sample data for preview
        analysis["sample_data"][col] = sample_values.head().tolist()
        
        # Check for Google Drive links
        if any('drive.google.com' in str(val) for val in sample_values.head(10)):
            analysis["google_drive_columns"].append(col)
        
        # Check for timestamps
        elif any(keyword in col.lower() for keyword in ['timestamp', 'time', 'date', 'submitted']):
            analysis["response_timestamps"].append(col)
        
        # Check for email addresses
        elif any('@' in str(val) for val in sample_values.head(10)):
            analysis["email_columns"].append(col)
        
        # Check for ZIP codes
        elif any(keyword in col.lower() for keyword in ['zip', 'postal', 'zipcode', 'zip_code']):
            analysis["zip_columns"].append(col)
        
        # Check for numeric data
        elif df[col].dtype in ['int64', 'float64'] or col.lower() in ['age', 'score', 'rating', 'count']:
            try:
                pd.to_numeric(df[col], errors='raise')
                analysis["numeric_columns"].append(col)
            except:
                # Check if it's a choice column (limited unique values)
                if unique_values <= min(10, total_values * 0.5):
                    analysis["choice_columns"].append(col)
                else:
                    analysis["text_columns"].append(col)
        
        # Check for choice/categorical data (small number of unique values)
        elif unique_values <= min(10, total_values * 0.3):
            analysis["choice_columns"].append(col)
        
        # Default to text column
        else:
            analysis["text_columns"].append(col)
    
    # Enhanced data quality assessment
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    
    analysis["data_quality"] = {
        "missing_data_percentage": (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
        "complete_responses": len(df.dropna()),
        "completion_rate": (len(df.dropna()) / len(df)) * 100 if len(df) > 0 else 0,
        "average_response_length": df.astype(str).apply(lambda x: x.str.len()).mean().mean(),
        "columns_with_missing_data": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "duplicate_responses": df.duplicated().sum()
    }
    
    return analysis

def display_enhanced_data_preview(df: pd.DataFrame, csv_analysis: Dict):
    """Display enhanced data preview with better insights"""
    st.subheader("Data Preview")
    
    # Create tabs for different views
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["Data Sample", "Column Analysis", "Data Quality"])
    
    with preview_tab1:
        st.write("**First 10 rows of your data:**")
        display_styled_dataframe(df.head(10), use_container_width=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Data Types", df.dtypes.nunique())
        with col4:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    with preview_tab2:
        st.write("**Column Type Analysis:**")
        
        col_analysis_data = []
        for col in df.columns:
            col_type = "Unknown"
            if col in csv_analysis["google_drive_columns"]:
                col_type = "Google Drive Link"
            elif col in csv_analysis["response_timestamps"]:
                col_type = "Timestamp"
            elif col in csv_analysis["email_columns"]:
                col_type = "Email"
            elif col in csv_analysis["zip_columns"]:
                col_type = "ZIP Code"
            elif col in csv_analysis["numeric_columns"]:
                col_type = "Numeric"
            elif col in csv_analysis["choice_columns"]:
                col_type = "Multiple Choice"
            elif col in csv_analysis["text_columns"]:
                col_type = "Text"
            
            col_analysis_data.append({
                "Column": col,
                "Type": col_type,
                "Unique Values": df[col].nunique(),
                "Missing": df[col].isnull().sum(),
                "Sample Values": ", ".join(str(v) for v in df[col].dropna().head(3).tolist())
            })
        
        col_analysis_df = pd.DataFrame(col_analysis_data)
        display_styled_dataframe(col_analysis_df, use_container_width=True)
    
    with preview_tab3:
        st.write("**Data Quality Overview:**")
        
        quality = csv_analysis["data_quality"]
        
        # Quality metrics in columns
        qual_col1, qual_col2, qual_col3 = st.columns(3)
        
        with qual_col1:
            completion_rate = quality["completion_rate"]
            if completion_rate >= 90:
                st.success(f"Completion Rate: {completion_rate:.1f}%")
            elif completion_rate >= 70:
                st.warning(f"Completion Rate: {completion_rate:.1f}%")
            else:
                st.error(f"Completion Rate: {completion_rate:.1f}%")
        
        with qual_col2:
            missing_pct = quality["missing_data_percentage"]
            if missing_pct <= 5:
                st.success(f"Missing Data: {missing_pct:.1f}%")
            elif missing_pct <= 15:
                st.warning(f"Missing Data: {missing_pct:.1f}%")
            else:
                st.error(f"Missing Data: {missing_pct:.1f}%")
        
        with qual_col3:
            duplicates = quality["duplicate_responses"]
            if duplicates == 0:
                st.success(f"Duplicates: {duplicates}")
            elif duplicates <= 3:
                st.warning(f"Duplicates: {duplicates}")
            else:
                st.error(f"Duplicates: {duplicates}")
        
        # Missing data breakdown
        if quality["columns_with_missing_data"]:
            st.write("**Columns with Missing Data:**")
            missing_data = pd.DataFrame(list(quality["columns_with_missing_data"].items()), 
                                      columns=["Column", "Missing Count"])
            missing_data["Missing %"] = (missing_data["Missing Count"] / len(df) * 100).round(1)
            display_styled_dataframe(missing_data, use_container_width=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'survey_data': None,
        'analysis_complete': False,
        'downloaded_images': {},
        'drive_analysis': None,
        'ai_analysis': None,
        'processing_logs': [],
        'current_step': 1,
        'openai_api_key': "",  # Start with empty key
        'output_directory': DEFAULT_OUTPUT_DIR,
        'csv_analysis': None,
        'use_sample_data': False,
        'ai_text_insights': {},
        'vc_trend_insights': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def log_processing_step(message: str, step_type: str = "info"):
    """Add a processing step to the logs"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_logs.append({
        "timestamp": timestamp,
        "message": message,
        "type": step_type
    })

def display_processing_logs():
    """Display processing logs in an attractive format"""
    if st.session_state.processing_logs:
        st.subheader("Processing Log")
        
        for log in st.session_state.processing_logs[-10:]:  # Show last 10 logs
            if log["type"] == "success":
                st.markdown(f'<div class="success-banner">{log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)
            elif log["type"] == "warning":
                st.markdown(f'<div class="warning-banner">{log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)
            elif log["type"] == "error":
                st.markdown(f'<div class="warning-banner">{log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-banner">{log["timestamp"]} - {log["message"]}</div>', unsafe_allow_html=True)

def analyze_csv_structure(df: pd.DataFrame) -> Dict:
    """Analyze the structure and content of the uploaded CSV"""
    # Use the enhanced detection function
    return detect_csv_structure(df)

def create_survey_insights_dashboard(df: pd.DataFrame, csv_analysis: Dict):
    """Create comprehensive survey insights dashboard"""
    st.subheader("Survey Data Insights Dashboard")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Total Responses</h3>
            <h2>{csv_analysis["total_responses"]}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Questions</h3>
            <h2>{csv_analysis["total_columns"]}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Image Columns</h3>
            <h2>{len(csv_analysis["google_drive_columns"])}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        complete_pct = (csv_analysis["data_quality"]["complete_responses"] / csv_analysis["total_responses"]) * 100
        st.markdown(f'''
        <div class="metric-card">
            <h3>Completion Rate</h3>
            <h2>{complete_pct:.1f}%</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # --------- VC Trend Insights Block ---------
    if st.session_state.openai_api_key and csv_analysis['text_columns']:
        st.markdown("---")
        st.subheader("Venture Capital Trend Insights")
        default_prompt = (
            "Pretend that you are a top tier consumer venture capital firm investor, who wants to capture consumer "
            "insights to make investable decisions. Analyze the attached consumer survey data. Identify and summarize "
            "the most significant emerging consumer trends, shifts in preferences, and mentions of new products or "
            "services. Segment your findings by relevant demographics, time periods, and product categories. The firm "
            "also privatizes trends with scalable path to acquire new customers and reason for consumers to continue "
            "to stay engaged. Highlight any anomalies or unexpected patterns, and provide supporting evidence from the "
            "data for each identified trend."
        )
        if st.button("Generate VC Trend Insights"):
            with st.spinner("Analyzing survey for VC insights..."):
                result_md = generate_vc_trend_insights(
                    st.session_state.survey_data,
                    csv_analysis['text_columns'],
                    default_prompt,
                    st.session_state.openai_api_key,
                )
                st.session_state.vc_trend_insights = result_md
                st.success("VC trend insights generated!")
        if st.session_state.vc_trend_insights:
            st.markdown("### VC Trend Insights Report")
            st.markdown(st.session_state.vc_trend_insights, unsafe_allow_html=True)

def display_image_analysis_results(downloaded_images: Dict, ai_analysis: Dict):
    """Display comprehensive image analysis results"""
    st.subheader("Image Analysis Results")
    
    if not downloaded_images:
        st.warning("No images were downloaded for analysis")
        return
    
    # Image gallery with analysis
    num_images = len(downloaded_images)
    cols_per_row = min(3, num_images)
    
    image_items = list(downloaded_images.items())
    
    for i in range(0, len(image_items), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx < len(image_items):
                response_idx, img_path = image_items[idx]
                
                with cols[j]:
                    try:
                        # Display image
                        image = Image.open(img_path)
                        st.image(image, caption=f"Response {response_idx + 1}", use_container_width=True)
                        
                        # Display AI analysis if available
                        if ai_analysis and str(response_idx) in ai_analysis.get('individual_analyses', {}):
                            analysis_data = ai_analysis['individual_analyses'][str(response_idx)]
                            ai_info = analysis_data.get('ai_analysis', {})
                            
                            with st.expander(f"AI Analysis - Response {response_idx + 1}"):
                                if ai_info.get('description') and ai_info['description'] != "Image analysis placeholder - integrate with OpenAI Vision API":
                                    st.write("Description:")
                                    st.write(ai_info['description'])
                                    
                                    # Sentiment and categories
                                    if ai_info.get('sentiment'):
                                        sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                                        st.write(f"Sentiment: {sentiment_color.get(ai_info['sentiment'], '⚪')} {ai_info['sentiment'].title()}")
                                    
                                    if ai_info.get('categories'):
                                        st.write(f"Categories: {', '.join(ai_info['categories'])}")
                                else:
                                    st.info("Basic image info available - enable AI analysis for detailed insights")
                                
                                # Technical details
                                st.write(f"Dimensions: {analysis_data.get('dimensions', 'N/A')}")
                                st.write(f"Format: {analysis_data.get('format', 'N/A')}")
                                file_size_kb = analysis_data.get('file_size', 0) / 1024
                                st.write(f"Size: {file_size_kb:.1f} KB")
                        
                    except Exception as e:
                        st.error(f"Error displaying image: {e}")

    # Download JSON option (keep this even if table is removed)
    if st.session_state.ai_analysis and st.session_state.ai_analysis.get('analysis_results'):
        st.download_button("Download Raw AI Analysis (JSON)", data=safe_json_dumps(st.session_state.ai_analysis, indent=2), file_name="ai_image_analysis.json", mime="application/json")

    # Show last Vision API error if present
    if 'vision_error' in st.session_state and st.session_state['vision_error']:
        st.error(f"OpenAI Vision API error: {st.session_state['vision_error']}")

    # --- Automatically parse downloaded images for app names/dates using AI ---
    api_key = st.session_state.get('openai_api_key', '')
    if api_key and downloaded_images:
        client = openai.OpenAI(api_key=api_key)
        all_apps_data = []
        for idx, img_path in downloaded_images.items():
            try:
                image = Image.open(img_path)
                with st.spinner(f"Extracting apps from Downloaded Image {idx+1}..."):
                    apps_data, raw_response = parse_app_store_screenshot(
                        image, 
                        client,
                        survey_data=st.session_state.survey_data,
                        image_index=idx
                    )
                if apps_data:
                    for app in apps_data:
                        app['image_file'] = os.path.basename(img_path)
                    all_apps_data.extend(apps_data)
                    st.success(f"Extracted {len(apps_data)} apps from Downloaded Image {idx+1}")
                else:
                    st.warning(f"No apps found in Downloaded Image {idx+1}")
                    st.text_area(f"Raw AI Response (Downloaded Image {idx+1})", raw_response)
            except Exception as e:
                st.error(f"Error processing image {img_path}: {e}")
        if all_apps_data:
            df = pd.DataFrame(all_apps_data)
            st.dataframe(df, use_container_width=True)
    elif not api_key:
        st.warning("OpenAI API key required to extract app names and dates from images.")

def create_comprehensive_report(survey_data: pd.DataFrame, csv_analysis: Dict, 
                               drive_analysis: Dict, ai_analysis: Dict, 
                               downloaded_images: Dict) -> str:
    """Create a comprehensive analysis report"""
    
    report_content = f"""
# Comprehensive Survey Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Survey Overview:**
- Total Responses: {csv_analysis['total_responses']}
- Questions/Columns: {csv_analysis['total_columns']}
- Data Quality Score: {100 - csv_analysis['data_quality']['missing_data_percentage']:.1f}%

**Image Analysis:**
- Google Drive Links Found: {drive_analysis.get('total_links', 0)}
- Images Successfully Downloaded: {len(downloaded_images)}
- Download Success Rate: {drive_analysis.get('accessibility_rate', '0%')}

## Detailed Analysis

### Survey Data Structure
"""
    
    # Add column analysis
    report_content += "\n**Column Types:**\n"
    report_content += f"- Text Columns: {len(csv_analysis['text_columns'])}\n"
    report_content += f"- Google Drive Columns: {len(csv_analysis['google_drive_columns'])}\n"
    report_content += f"- Timestamp Columns: {len(csv_analysis['response_timestamps'])}\n"
    
    # Add image analysis if available
    if ai_analysis and 'individual_analyses' in ai_analysis:
        report_content += "\n### AI Image Analysis Summary\n"
        
        sentiments = []
        categories = []
        
        for analysis in ai_analysis['individual_analyses'].values():
            ai_info = analysis.get('ai_analysis', {})
            if ai_info.get('sentiment'):
                sentiments.append(ai_info['sentiment'])
            if ai_info.get('categories'):
                categories.extend(ai_info['categories'])
        
        if sentiments:
            sentiment_counts = Counter(sentiments)
            report_content += "\n**Sentiment Distribution:**\n"
            for sentiment, count in sentiment_counts.items():
                pct = (count / len(sentiments)) * 100
                report_content += f"- {sentiment.title()}: {count} ({pct:.1f}%)\n"
        
        if categories:
            category_counts = Counter(categories)
            report_content += "\n**Top Content Categories:**\n"
            for category, count in category_counts.most_common(5):
                report_content += f"- {category.title()}: {count} mentions\n"
    
    # Add recommendations
    report_content += "\n## Recommendations\n"
    
    if drive_analysis.get('recommendations'):
        report_content += "\n**Google Drive Access:**\n"
        for rec in drive_analysis['recommendations']:
            report_content += f"- {rec}\n"
    
    if csv_analysis['data_quality']['missing_data_percentage'] > 20:
        report_content += "- Consider improving data collection to reduce missing responses\n"
    
    if len(downloaded_images) < drive_analysis.get('total_links', 0):
        report_content += "- Some images could not be downloaded - check sharing permissions\n"
    
    report_content += "\n## Next Steps\n"
    report_content += "- Review AI analysis insights for actionable findings\n"
    report_content += "- Consider follow-up surveys based on image analysis results\n"
    report_content += "- Use sentiment analysis to improve future survey design\n"
    
    # --- AI Text Insights in report ---
    if 'ai_text_insights' in st.session_state and st.session_state['ai_text_insights']:
        report_content += "\n### AI Insights for Open-Ended Responses\n"
        for q, ins in st.session_state['ai_text_insights'].items():
            report_content += f"\n#### {q}\n"
            report_content += f"\n**Summary:** {ins.get('summary', '')}\n"
            if ins.get('actionable_insights'):
                report_content += "\n**Actionable Insights:**\n"
                for a in ins['actionable_insights']:
                    report_content += f"- {a}\n"
    
    # --- VC Trend Insights section ---
    if st.session_state.get('vc_trend_insights'):
        report_content += "\n## VC Trend Insights\n" + st.session_state['vc_trend_insights'] + "\n"
    
    report_content += f"\n---\n*Report generated by Comprehensive Survey Analyzer v1.0*"
    
    return report_content

def create_download_package(survey_data: pd.DataFrame, analysis_results: Dict, 
                          downloaded_images: Dict, report_content: str) -> BytesIO:
    """Create a downloadable ZIP package with all results"""
    
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add the main report
        zip_file.writestr("analysis_report.md", report_content)
        
        # Add survey data
        csv_buffer = StringIO()
        survey_data.to_csv(csv_buffer, index=False)
        zip_file.writestr("survey_data.csv", csv_buffer.getvalue())
        
        # Add analysis results
        zip_file.writestr("analysis_results.json", safe_json_dumps(analysis_results, indent=2))
        
        # Add images
        for response_idx, img_path in downloaded_images.items():
            if os.path.exists(img_path):
                with open(img_path, 'rb') as img_file:
                    zip_file.writestr(f"images/response_{response_idx + 1}.jpg", img_file.read())
    
    zip_buffer.seek(0)
    return zip_buffer

# -----------------------------------------------------------------------------
# OpenAI compatibility helper (supports both <1.0 and >=1.0 SDK versions)
# -----------------------------------------------------------------------------

def safe_chat_completion(messages, model="gpt-3.5-turbo", temperature=0.3, api_key: str = "") -> str:
    """Call OpenAI chat completion with backward-compatible logic."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI package not available")
    
    if not api_key:
        raise ValueError("No API key provided")
        
    try:
        version = getattr(openai, "__version__", "0")
        major = int(version.split(".")[0]) if version and version[0].isdigit() else 0
        if major >= 1:
            # New >=1.0 interface
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        else:
            # Legacy <1.0 interface
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp["choices"][0]["message"]["content"]
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg:
            raise ValueError("Invalid API key. Please check your OpenAI API key.")
        elif "429" in error_msg:
            raise ValueError("Rate limit exceeded. Please try again later.")
        else:
            raise RuntimeError(f"OpenAI API error: {error_msg}")

def generate_ai_text_insights(question: str, responses: pd.Series, api_key: str) -> Dict:
    """Use OpenAI to summarize long-form responses and suggest actionable insights."""
    if not OPENAI_AVAILABLE:
        return {"summary": "OpenAI package not available.", "actionable_insights": []}
    
    if not api_key:
        return {"summary": "No API key provided.", "actionable_insights": []}
        
    try:
        # Sample up to first 100 responses to reduce token usage
        sampled = responses.dropna().astype(str).tolist()[:100]
        sample_text = "\n".join(f"- {r.strip()}" for r in sampled)
        prompt = (
            f"You are an expert survey analyst. Based on the following responses to the question '{question}', "
            "write a concise summary (3-5 sentences) capturing the main themes, then provide a bullet list of 3-5 "
            "actionable recommendations an organization could take. Return the result as JSON with keys 'summary' "
            "and 'actionable_insights' (array of strings).\n\nResponses:\n" + sample_text
        )
        content = safe_chat_completion([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ], model="gpt-3.5-turbo", temperature=0.3, api_key=api_key)
        try:
            return json.loads(content)
        except Exception:
            # Fallback if JSON parsing fails
            return {"summary": content.strip(), "actionable_insights": []}
    except Exception as e:
        return {"summary": f"Error generating insights: {str(e)}", "actionable_insights": []}

# --- NEW FUNCTION: VC Trend Insights ----------------------------------------

def generate_vc_trend_insights(df: pd.DataFrame, text_columns: list, prompt: str, api_key: str) -> str:
    """Run a single OpenAI completion to analyze all long-form answers with investor-oriented prompt."""
    if not OPENAI_AVAILABLE:
        return "OpenAI package not available."
    
    if not api_key:
        return "No API key provided."
        
    try:
        # Build context: list up to 200 responses per column to control tokens
        parts = []
        for col in text_columns:
            responses = df[col].dropna().astype(str).tolist()[:200]
            if responses:
                joined = "\n".join(f"- {r.strip()}" for r in responses)
                parts.append(f"\n\n### Question: {col}\n{joined}")
        context_text = "".join(parts)
        full_prompt = prompt + "\n\nDATA:" + context_text + "\n\nReturn your analysis as Markdown."
        result_md = safe_chat_completion([
            {"role": "user", "content": full_prompt}
        ], model="gpt-3.5-turbo-16k", temperature=0.2, api_key=api_key)
        return result_md.strip()
    except Exception as e:
        return f"Error generating VC insights: {str(e)}"

def force_rerun():
    """Attempt to rerun the Streamlit script. Falls back gracefully if unsupported."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        # On very old Streamlit versions, just set a dummy session_state flag
        st.session_state["__force_rerun_flag"] = time.time()

def safe_json_dumps(obj: object, **kwargs) -> str:
    """JSON dumps that converts non-serialisable objects (e.g. pandas Timestamp) to strings."""
    return json.dumps(obj, default=lambda o: str(o), **kwargs)

def get_upload_info_box(message: str) -> str:
    """Returns HTML for the upload info box with custom styling"""
    return f'''
    <div style="
        background-color: #a3ba65;
        color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    ">{message}</div>
    '''

def get_city_state_coordinates(city_state: str) -> tuple:
    """Get coordinates for a city, state location using OpenStreetMap's Nominatim API."""
    try:
        # Clean and format the input
        city_state = city_state.strip()
        
        # Handle special cases
        if city_state.lower() == "new york, new york":
            return 40.7128, -74.0060  # Manhattan coordinates
        
        # Format the query for better accuracy
        query = f"{city_state}, USA"
        
        # Using OpenStreetMap's Nominatim API with more specific parameters
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            'q': query,
            'format': 'json',
            'limit': 1,
            'addressdetails': 1,
            'countrycodes': 'us',
            'featuretype': 'city'
        }
        headers = {
            'User-Agent': 'survey_analysis_app',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return lat, lon
        
        return None, None
    except Exception as e:
        st.write(f"❌ Error getting coordinates for {city_state}: {str(e)}")
        return None, None

def extract_representative_quote(location: str, data: pd.DataFrame, location_column: str, text_columns: list) -> str:
    """Extract a representative quote for a location using AI."""
    try:
        # Get all responses for this location
        location_data = data[data[location_column] == location]
        
        # Combine all text responses
        all_text = []
        for col in text_columns:
            responses = location_data[col].dropna().astype(str).tolist()
            all_text.extend([f"Response from {col}: {resp}" for resp in responses])
        
        if not all_text:
            return "No text responses available for this location."
        
        # Use OpenAI to select a representative quote
        prompt = f"""Given these survey responses from {location}, select the most insightful or representative quote.
        If there are multiple good quotes, select the one that best captures the sentiment or key message.
        Return ONLY the quote, nothing else.
        
        Responses:
        {chr(10).join(all_text[:10])}  # Limit to first 10 responses to avoid token limits
        """
        
        if not st.session_state.openai_api_key:
            return "AI analysis requires a valid OpenAI API key"
            
        quote = safe_chat_completion([
            {"role": "system", "content": "You are a helpful assistant that selects representative quotes."},
            {"role": "user", "content": prompt}
        ], model="gpt-3.5-turbo", temperature=0.1, api_key=st.session_state.openai_api_key)
        
        return quote.strip()
    except Exception as e:
        return f"Error extracting quote: {str(e)}"

def create_city_state_map(data: pd.DataFrame, location_column: str) -> folium.Map:
    """Creates a map with markers for each city, state location."""
    try:
        # Create base map centered on US
        m = folium.Map(
            location=[39.8283, -98.5795],  # Center of US
            zoom_start=4,
            tiles='CartoDB positron',
            width='100%',  # Make map use full width
            height='800px'  # Increased height
        )
        
        # Group by location to get counts
        location_counts = data[location_column].value_counts()
        st.write(f"Number of unique locations: {len(location_counts)}")
        
        # Get text columns for quote extraction
        text_columns = [col for col in data.columns if col not in [location_column, 'Timestamp', 'Email']]
        
        # Process each location
        success_count = 0
        error_count = 0
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (location, count) in enumerate(location_counts.items()):
            if pd.isna(location) or not location:
                continue
                
            try:
                # Update progress
                progress = (idx + 1) / len(location_counts)
                progress_bar.progress(progress)
                status_text.text(f"Processing location {idx + 1} of {len(location_counts)}")
                
                # Get coordinates
                lat, lon = get_city_state_coordinates(location)
                
                if lat and lon:
                    # Create marker
                    radius = min(max(count * 2, 8), 20)  # Size between 8 and 20 pixels
                    # Color scheme based on response count
                    color = '#c4d48a' if count == 1 else '#a3ba65' if count < 5 else '#7a8c4c'  # Light green for 1, base green for 2-4, dark green for 5+
                    
                    # Get location data
                    location_data = data[data[location_column] == location]
                    
                    # Extract quotes and names for this location
                    quotes_with_names = []
                    for _, row in location_data.iterrows():
                        # Combine First Name and Last Name if they exist
                        first_name = row.get('First Name', '')
                        last_name = row.get('Last Name', '')
                        name = f"{first_name} {last_name}".strip() if first_name or last_name else 'Anonymous'
                        
                        # Get all text responses for this person
                        responses = []
                        for col in text_columns:
                            if pd.notna(row[col]) and str(row[col]).strip():
                                responses.append(f"{col}: {row[col]}")
                        
                        if responses:
                            # Use AI to select the most insightful quote
                            prompt = f"""Given these responses from {name} in {location}, select the most insightful or unique quote.
                            Return ONLY the quote, nothing else.
                            
                            Responses:
                            {chr(10).join(responses)}
                            """
                            
                            if st.session_state.openai_api_key:
                                quote = safe_chat_completion([
                                    {"role": "system", "content": "You are a helpful assistant that selects representative quotes."},
                                    {"role": "user", "content": prompt}
                                ], model="gpt-3.5-turbo", temperature=0.1, api_key=st.session_state.openai_api_key)
                                quotes_with_names.append((name, quote.strip()))
                    
                    # Create popup content with quotes and navigation
                    popup_content = f"""
                    <div style='width: 300px;'>
                        <h4>{location}</h4>
                        <p><b>Responses:</b> {count}</p>
                        <div class='quotes-container'>
                    """
                    
                    # Add quotes with names
                    for name, quote in quotes_with_names:
                        popup_content += f"""
                        <div class='quote-box' data-location='{location}'>
                            <p><b>{name}</b> says:</p>
                            <p style='font-style: italic;'>{quote}</p>
                        </div>
                        """
                    
                    popup_content += """
                        </div>
                        <div style='text-align: center; margin-top: 10px; display: flex; justify-content: center; gap: 10px;'>
                            <button onclick='cycleQuotes(""" + f'"{location}", -1' + """)' style='background-color: #a3ba65; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;'>
                                ←
                            </button>
                            <button onclick='cycleQuotes(""" + f'"{location}", 1' + """)' style='background-color: #a3ba65; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;'>
                                →
                            </button>
                        </div>
                    </div>
                    """
                    
                    # Add marker
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=folium.Popup(popup_content, max_width=300),
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        weight=2,
                        tooltip=f"{location} ({count} responses)"
                    ).add_to(m)
                    
                    success_count += 1
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                error_count += 1
                st.write(f"Error with location {location}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show processing summary
        st.write(f"Successfully mapped: {success_count} locations")
        st.write(f"Failed to map: {error_count} locations")
        
        # Add legend with new color scheme
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999;
                    background-color: white;
                    padding: 10px;
                    font-size:12px;
                    border-radius: 5px;">
            <p><b>Response Count</b></p>
            <p><i class="fa fa-circle" style="color:#c4d48a"></i> 1 response</p>
            <p><i class="fa fa-circle" style="color:#a3ba65"></i> 2-4 responses</p>
            <p><i class="fa fa-circle" style="color:#7a8c4c"></i> 5+ responses</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add CSS for quote styling
        css = """
        <style>
            .quote-box {
                margin: 10px 0;
                padding: 10px;
                border-left: 3px solid #a3ba65;
                background-color: #f8f9fa;
                border-radius: 3px;
            }
            .quote-box p {
                margin: 5px 0;
            }
        </style>
        """
        m.get_root().html.add_child(folium.Element(css))
        
        # Add JavaScript for quote cycling
        quote_cycling_js = """
        <script>
        let quoteIndex = {};
        
        function cycleQuotes(location, direction) {
            if (!quoteIndex[location]) {
                quoteIndex[location] = 0;
            }
            
            // Get all quotes for this location
            const quotes = document.querySelectorAll(`[data-location="${location}"]`);
            if (quotes.length > 0) {
                // Update index with direction
                quoteIndex[location] = (quoteIndex[location] + direction + quotes.length) % quotes.length;
                const quoteElement = quotes[quoteIndex[location]];
                quoteElement.style.display = 'block';
                
                // Hide other quotes
                quotes.forEach((q, i) => {
                    if (i !== quoteIndex[location]) {
                        q.style.display = 'none';
                    }
                });
            }
        }
        </script>
        """
        m.get_root().html.add_child(folium.Element(quote_cycling_js))
        
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

def test_openai_key(api_key: str) -> bool:
    """Test if the OpenAI API key is valid."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Try a simple completion to test the key
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return True
    except Exception as e:
        st.error(f"API Key Test Failed: {str(e)}")
        return False

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def parse_app_store_screenshot(image, client, survey_data=None, image_index=None):
    """Use OpenAI Vision API to parse the screenshot and extract app data."""
    base64_image = encode_image_to_base64(image)
    
    # Get name information if survey data and index are provided
    name_info = ""
    if survey_data is not None and image_index is not None:
        try:
            first_name = survey_data.iloc[image_index].get('First Name', '')
            last_name = survey_data.iloc[image_index].get('Last Name', '')
            name_info = f"{first_name} {last_name}".strip()  # Removed "Name:" prefix
        except Exception as e:
            st.write(f"Error getting name info: {str(e)}")
    
    prompt = f"""
    Analyze this iOS App Store screenshot and extract the app information.
    For each app entry, extract:
    1. App name (full name as shown)
    2. Download/purchase date (in the format shown)

    Return the data as a JSON array with this exact structure:
    [
        {{
            "app_name": "App Name Here",
            "date": "Date as shown (e.g., Jan 14, 2025)"
        }}
    ]
    Only include apps that are clearly visible and readable. If you can't read the full app name or date clearly, skip that entry.
    If you cannot find any apps, return an empty array [] and nothing else.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    result = response.choices[0].message.content
    
    # Try to parse the JSON
    json_match = re.search(r'\[.*\]', result, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            apps_data = json.loads(json_str)
            # Add name info to each app entry if available
            if name_info:
                for app in apps_data:
                    app['user_name'] = name_info.strip()
            return apps_data, result
        except json.JSONDecodeError:
            return [], result
    else:
        return [], result

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown(f'<h1 class="main-header">Tactile Survey Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key section
        st.subheader("AI Analysis")
        
        # Always show API key input but hide the default prompt
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key for AI analysis",
            label_visibility="collapsed"
        )
        
        if openai_key:
            st.session_state.openai_api_key = openai_key
            # Test the key
            if test_openai_key(openai_key):
                st.success("API Key is valid!")
            else:
                st.error("Invalid API Key. Please check and try again.")
        else:
            st.warning("Please enter your OpenAI API key to enable AI features")
        
        # Output directory
        st.subheader("Output Settings")
        st.session_state.output_directory = st.text_input(
            "Output Directory", 
            value=st.session_state.output_directory,
            help="Directory to save downloaded images and analysis results"
        )
        
        # Processing options
        st.subheader("Processing Options")
        download_images = st.checkbox("Download Google Drive Images", value=True)
        
        # Auto-enable AI analysis if API key is valid
        run_ai_analysis = st.checkbox("Run AI Image Analysis", value=bool(st.session_state.openai_api_key))
        
        # Option to automatically build the final markdown report once AI analysis has finished
        auto_generate_report = st.checkbox("Auto Generate Report", value=True)
        
        if run_ai_analysis and not st.session_state.openai_api_key:
            st.error("AI analysis requires a valid OpenAI API key")
            run_ai_analysis = False
        
        generate_insights = st.checkbox("Generate Survey Insights", value=True)
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload & Process", "Survey Analysis", "Image Analysis", "Reports", "Location Map"])
    
    with tab1:
        st.header("Upload and Process Survey Data")
        
        # Sample data option
        st.subheader("Data Source")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            use_sample = st.checkbox("Use Sample Data", 
                                   value=st.session_state.use_sample_data,
                                   help="Load pre-built sample data for testing the application")
            st.session_state.use_sample_data = use_sample
        
        with col2:
            if use_sample:
                st.info("Sample data will be loaded automatically - perfect for testing all features!")
        
        # File upload or sample data loading
        uploaded_file = None
        survey_data = None
        
        if use_sample:
            # Load sample data
            if st.button("Load Sample Data", type="primary"):
                with st.spinner("Loading sample data..."):
                    survey_data = load_sample_data()
                    if survey_data is not None:
                        st.session_state.survey_data = survey_data
                        log_processing_step(f"Loaded sample data with {len(survey_data)} responses", "success")
        else:
            # File upload
            uploaded_file = st.file_uploader(
                "Upload your Google Forms CSV file",
                type=['csv'],
                help="Upload the CSV file downloaded from Google Forms"
            )
        
        # Process uploaded file
        if uploaded_file is not None and not use_sample:
            # Load and preview data
            try:
                survey_data = pd.read_csv(uploaded_file)
                st.session_state.survey_data = survey_data
                
                log_processing_step(f"Loaded CSV with {len(survey_data)} survey responses", "success")
                
                st.success(f"Successfully loaded {len(survey_data)} survey responses")
                
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
                log_processing_step(f"Failed to load CSV: {str(e)}", "error")
        
        # If we have survey data (either uploaded or sample), analyze it
        if st.session_state.survey_data is not None:
            survey_data = st.session_state.survey_data
            
            # Analyze CSV structure using enhanced function
            csv_analysis = analyze_csv_structure(survey_data)
            st.session_state.csv_analysis = csv_analysis
            
            # Enhanced data preview
            display_enhanced_data_preview(survey_data, csv_analysis)
            
            # Show structure analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Structure")
                st.write(f"**Total Responses:** {csv_analysis['total_responses']}")
                st.write(f"**Total Questions:** {csv_analysis['total_columns']}")
                st.write(f"**Google Drive Columns:** {len(csv_analysis['google_drive_columns'])}")
                st.write(f"**Text Columns:** {len(csv_analysis['text_columns'])}")
                st.write(f"**Multiple Choice Columns:** {len(csv_analysis['choice_columns'])}")
                st.write(f"**Email Columns:** {len(csv_analysis['email_columns'])}")
                st.write(f"**ZIP Code Columns:** {len(csv_analysis['zip_columns'])}")
            
            with col2:
                st.subheader("Data Quality")
                missing_pct = csv_analysis['data_quality']['missing_data_percentage']
                completion_rate = csv_analysis['data_quality']['completion_rate']
                duplicates = csv_analysis['data_quality']['duplicate_responses']
                
                st.write(f"**Missing Data:** {missing_pct:.1f}%")
                st.write(f"**Completion Rate:** {completion_rate:.1f}%")
                st.write(f"**Complete Responses:** {csv_analysis['data_quality']['complete_responses']}")
                st.write(f"**Duplicate Responses:** {duplicates}")
                
                # Overall quality score
                quality_score = completion_rate - (missing_pct * 0.5) - (duplicates * 2)
                if quality_score >= 90:
                    st.success(f"Excellent quality ({quality_score:.1f}/100)")
                elif quality_score >= 70:
                    st.warning(f"Good quality ({quality_score:.1f}/100)")
                else:
                    st.error(f"Needs improvement ({quality_score:.1f}/100)")
            
            # Google Drive image analysis is now handled entirely in the Image Analysis tab.
            if csv_analysis['google_drive_columns']:
                st.info(f"Found {len(csv_analysis['google_drive_columns'])} Google Drive image column(s). Head over to the Image Analysis tab to analyze links and download images.")
            
            # Mark as ready for analysis
            if st.session_state.survey_data is not None:
                st.session_state.analysis_complete = True
        
        elif not use_sample:
            st.info("Please upload a CSV file or use sample data to begin analysis")
        
        # Display processing logs
        if st.session_state.processing_logs:
            with st.expander("Processing Log"):
                display_processing_logs()
    
    with tab2:
        st.header("Survey Data Analysis")
        
        if st.session_state.survey_data is not None:
            # Create comprehensive dashboard
            if hasattr(st.session_state, 'csv_analysis'):
                create_survey_insights_dashboard(st.session_state.survey_data, st.session_state.csv_analysis)
            else:
                st.info("Please upload and process your CSV file first")
        else:
            st.info("Please upload a CSV file to view survey analysis")
    
    with tab3:
        st.header("Image Analysis")
        
        if st.session_state.survey_data is not None:
            if st.button("Start Image Analysis", type="primary"):
                # Step 1: Analyze Google Drive links if not done
                if not st.session_state.get('drive_analysis'):
                    with st.spinner("Analyzing Google Drive links..."):
                        temp_csv = f"temp_survey_{int(time.time())}.csv"
                        st.session_state.survey_data.to_csv(temp_csv, index=False)
                        try:
                            drive_analysis = analyze_survey_drive_links(temp_csv)
                            st.session_state.drive_analysis = drive_analysis
                            log_processing_step(f"Analyzed {drive_analysis['total_links']} Google Drive links", "success")
                        finally:
                            try:
                                os.remove(temp_csv)
                            except:
                                pass
                
                # Step 2: Download images if not done
                if not st.session_state.get('downloaded_images'):
                    drive_analysis = st.session_state.get('drive_analysis')
                    if drive_analysis and drive_analysis.get('accessible_links',0) > 0:
                        progress_container = st.empty()
                        with progress_container.container():
                            st.spinner("Downloading images from Google Drive...")
                            downloader = EnhancedDriveDownloader(st.session_state.output_directory)
                            downloaded_images = {}
                            total = len(drive_analysis['individual_analyses'])
                            prog = st.progress(0)
                            for i, item in enumerate(drive_analysis['individual_analyses']):
                                file_id = item.get('file_id')
                                if file_id and item.get('accessible'):
                                    fname = f"response_{i+1}_{file_id[:8]}.jpg"
                                    path = downloader.download_file(file_id, fname)
                                    if path:
                                        downloaded_images[i] = path
                                prog.progress((i+1)/total)
                            st.session_state.downloaded_images = downloaded_images
                            log_processing_step(f"Downloaded {len(downloaded_images)} images", "success")
                        progress_container.empty()
                
                # Step 3: Run AI analysis if not done
                if not st.session_state.get('ai_analysis') or not st.session_state.ai_analysis.get('analysis_results'):
                    if not st.session_state.openai_api_key:
                        st.error("Please enter a valid OpenAI API key in the sidebar to enable AI analysis")
                    else:
                        analysis_container = st.empty()
                        with analysis_container.container():
                            st.spinner("Running AI analysis on images...")
                            tmp_csv = "temp_survey_data.csv"
                            st.session_state.survey_data.to_csv(tmp_csv, index=False)
                            analyzer = AIVisionAnalyzer(tmp_csv, st.session_state.output_directory, st.session_state.openai_api_key)
                            analyzer.survey_data = st.session_state.survey_data
                            analyzer.downloaded_images = st.session_state.downloaded_images
                            progress = st.progress(0.0)
                            status = st.empty()
                            analyses = {}
                            total = len(st.session_state.downloaded_images)
                            for i, (idx, img_path) in enumerate(st.session_state.downloaded_images.items()):
                                status.write(f"Analyzing image {i+1}/{total}…")
                                survey_ctx = st.session_state.survey_data.iloc[idx].to_dict()
                                result = analyzer.analyze_image_with_ai(img_path, survey_ctx)
                                analyses[idx] = result
                                progress.progress((i+1)/total)
                            summary = {
                                "total_responses": len(st.session_state.survey_data),
                                "responses_with_images": total,
                                "download_success_rate": (total/len(st.session_state.survey_data))*100,
                                "analysis_timestamp": datetime.now().isoformat(),
                                "individual_analyses": analyses
                            }
                            insights = analyzer.generate_ai_insights_summary(summary)
                            st.session_state.ai_analysis = {
                                'analysis_results': summary,
                                'insights': insights
                            }
                            log_processing_step("AI analysis completed", "success")
                            try:
                                os.remove(tmp_csv)
                            except:
                                pass
                        analysis_container.empty()
                        st.success("Full image analysis complete!")
                        st.session_state.analysis_complete = True
        else:
            st.info("Please upload and process your survey data first")
    
    with tab4:
        st.header("Reports and Downloads")
        
        if st.session_state.analysis_complete:
            st.subheader("Generate Comprehensive Report")
            
            if st.button("Generate Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    # Generate report
                    report_content = create_comprehensive_report(
                        st.session_state.survey_data,
                        st.session_state.csv_analysis,
                        st.session_state.drive_analysis or {},
                        st.session_state.ai_analysis.get('analysis_results', {}) if st.session_state.ai_analysis else {},
                        st.session_state.downloaded_images
                    )
                    
                    # Display report
                    st.markdown(report_content)
                    
                    # Download options
                    st.subheader("Download Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="Download Report (Markdown)",
                            data=report_content,
                            file_name=f"survey_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    with col2:
                        if st.session_state.survey_data is not None:
                            csv_data = st.session_state.survey_data.to_csv(index=False)
                            st.download_button(
                                label="Download Survey Data (CSV)",
                                data=csv_data,
                                file_name=f"survey_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        if st.session_state.ai_analysis:
                            analysis_json = safe_json_dumps(st.session_state.ai_analysis, indent=2)
                            st.download_button(
                                label="Download AI Analysis (JSON)",
                                data=analysis_json,
                                file_name=f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    
                    # Complete package download
                    if st.session_state.downloaded_images:
                        st.subheader("Complete Analysis Package")
                        st.info("Download everything in one ZIP file including images, analysis, and reports")
                        
                        if st.button("Create Download Package"):
                            with st.spinner("Creating download package..."):
                                try:
                                    analysis_results = {
                                        'csv_analysis': st.session_state.csv_analysis,
                                        'drive_analysis': st.session_state.drive_analysis,
                                        'ai_analysis': st.session_state.ai_analysis,
                                        'ai_text_insights': st.session_state.ai_text_insights,
                                        'vc_trend_insights': st.session_state.vc_trend_insights
                                    }
                                    
                                    zip_buffer = create_download_package(
                                        st.session_state.survey_data,
                                        analysis_results,
                                        st.session_state.downloaded_images,
                                        report_content
                                    )
                                    
                                    st.download_button(
                                        label="Download Complete Package (ZIP)",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"survey_analysis_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip"
                                    )
                                    
                                    st.success("Package ready for download!")
                                    
                                except Exception as e:
                                    st.error(f"Error creating package: {str(e)}")
        else:
            st.info("Please upload and process your survey data first to generate reports")

    with tab5:
        st.header("Location Map")
        
        if st.session_state.survey_data is not None:
            # Detect possible location columns
            location_columns = [col for col in st.session_state.survey_data.columns 
                              if any(term in col.lower() for term in ['location', 'city', 'state', 'address', 'geo'])]
            
            if location_columns:
                # Let user select the location column
                location_column = st.selectbox(
                    "Select the column containing city/state locations:",
                    location_columns
                )
                
                if st.button("Generate Location Map"):
                    with st.spinner("Creating location map... This may take a moment."):
                        try:
                            # Create and display the map
                            m = create_city_state_map(st.session_state.survey_data.copy(), location_column)
                            if m:
                                # Increased dimensions for a larger map
                                folium_static(m, width=1600, height=800)
                            
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.warning("No location columns detected in the data. Please ensure your CSV contains a column with city/state information.")
                st.write("Available columns:", st.session_state.survey_data.columns.tolist())
        else:
            st.info("Please upload and process your survey data first to view the location map")

    # Always show results if analysis is complete
    if st.session_state.get('analysis_complete') and st.session_state.get('downloaded_images'):
        ai_data = st.session_state.ai_analysis.get('analysis_results', {}) if st.session_state.ai_analysis else {}
        display_image_analysis_results(st.session_state.downloaded_images, ai_data)

if __name__ == "__main__":
    main() 