## To Run insert the following cd ~/Documents && streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import json
import os
import openai
from typing import List, Dict
import folium
from streamlit_folium import folium_static
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
import requests

# Set page config
st.set_page_config(
    page_title="Survey Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS styling (simplified since config.toml handles main theme)
st.markdown("""
    <style>
        /* Additional styling for specific elements not covered by theme */
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            background-color: #8fa051 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Set OpenAI API key directly
openai.api_key = "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"

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'survey_data_global' not in st.session_state:
    st.session_state.survey_data_global = None

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

def display_styled_dataframe(df, title=None):
    """Display a dataframe with custom styling"""
    if title:
        st.subheader(title)
    styled_df = style_dataframe(df)
    st.write(styled_df.to_html(), unsafe_allow_html=True)

# Load ZIP code database
@st.cache_data
def load_zip_database():
    try:
        zip_db = pd.read_csv('us_zip_codes.csv')
        # Ensure the ZIP code is treated as a string with leading zeros
        zip_db['ZIP'] = zip_db['ZIP'].astype(str).str.zfill(5)
        return zip_db
    except Exception as e:
        st.error(f"Error loading ZIP code database: {str(e)}")
        return None

ZIP_DB = load_zip_database()

def clean_zip_code(zip_code):
    """Clean and standardize ZIP code format."""
    try:
        # Convert to string and remove any quotes or spaces
        zip_str = str(zip_code).strip('"').strip("'").strip()
        # Keep only digits
        zip_str = ''.join(filter(str.isdigit, zip_str))
        # Ensure 5-digit format
        if len(zip_str) >= 5:
            return zip_str[:5]  # Take first 5 digits
        elif len(zip_str) < 5:
            return zip_str.zfill(5)  # Pad with leading zeros
        return None
    except:
        return None

def get_coordinates_simple(zip_code: str) -> tuple:
    """Get coordinates using a simple geocoding API."""
    try:
        # Using OpenStreetMap's Nominatim API directly
        url = f"https://nominatim.openstreetmap.org/search?q={zip_code},USA&format=json&limit=1"
        headers = {'User-Agent': 'survey_analysis_app'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                st.write(f"✅ Found coordinates for ZIP {zip_code}: {lat}, {lon}")
                return lat, lon
        
        st.write(f"❌ Could not find coordinates for ZIP {zip_code}")
        return None, None
    except Exception as e:
        st.write(f"❌ Error getting coordinates for ZIP {zip_code}: {str(e)}")
        return None, None

def create_map(data: pd.DataFrame, zip_column: str) -> folium.Map:
    """Creates a map with markers for each ZIP code location."""
    try:
        # Create base map centered on US
        m = folium.Map(
            location=[39.8283, -98.5795],  # Center of US
            zoom_start=4,
            tiles='CartoDB positron'
        )
        
        # Clean ZIP codes
        st.write("### Cleaning ZIP Codes")
        data[zip_column] = data[zip_column].apply(clean_zip_code)
        
        # Show cleaned data sample with styling
        st.write("Sample of cleaned ZIP codes:")
        zip_sample = pd.DataFrame({
            'Original': data[zip_column].head(),
            'Cleaned ZIP': data[zip_column].head()
        })
        display_styled_dataframe(zip_sample)
        
        # Group by ZIP code to get counts
        zip_counts = data[zip_column].value_counts()
        st.write(f"Number of unique ZIP codes: {len(zip_counts)}")
        
        # Process each ZIP code
        success_count = 0
        error_count = 0
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (zip_code, count) in enumerate(zip_counts.items()):
            if pd.isna(zip_code) or not zip_code:
                continue
                
            try:
                # Update progress
                progress = (idx + 1) / len(zip_counts)
                progress_bar.progress(progress)
                status_text.text(f"Processing ZIP code {idx + 1} of {len(zip_counts)}")
                
                # Get coordinates
                lat, lon = get_coordinates_simple(zip_code)
                
                if lat and lon:
                    # Create marker
                    radius = min(max(count * 2, 8), 20)  # Size between 8 and 20 pixels
                    color = '#ff4b4b' if count < 5 else '#ff8c42' if count < 10 else '#dc3545'
                    
                    # Add marker
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=radius,
                        popup=f"ZIP: {zip_code}<br>Responses: {count}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        weight=2,
                        tooltip=f"ZIP: {zip_code} ({count} responses)"
                    ).add_to(m)
                    
                    success_count += 1
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                error_count += 1
                st.write(f"Error with ZIP {zip_code}: {str(e)}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show processing summary
        st.write(f"Successfully mapped: {success_count} ZIP codes")
        st.write(f"Failed to map: {error_count} ZIP codes")
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999;
                    background-color: white;
                    padding: 10px;
                    font-size:12px;
                    border-radius: 5px;">
            <p><b>Response Count</b></p>
            <p><i class="fa fa-circle" style="color:#ff4b4b"></i> 1-4 responses</p>
            <p><i class="fa fa-circle" style="color:#ff8c42"></i> 5-9 responses</p>
            <p><i class="fa fa-circle" style="color:#dc3545"></i> 10+ responses</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

# Load sample data
def load_sample_data():
    try:
        sample_path = os.path.join('Sample_CSV', 'mockgeo.csv')
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
        else:
            st.error(f"Sample file not found at {sample_path}")
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

# Main Streamlit UI
st.title("Survey Analysis Dashboard")
st.write("Upload your survey CSV file to analyze responses.")

# Add option to use sample data
use_sample = st.checkbox("Use sample data (mockgeo.csv)")

if use_sample:
    survey_data = load_sample_data()
else:
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        survey_data = pd.read_csv(uploaded_file)
    else:
        survey_data = None

if survey_data is not None:
    try:
        # Show data preview with styling
        display_styled_dataframe(survey_data.head(), "Data Preview")
        
        # Add tabs
        tab1, tab2, tab3 = st.tabs(["Analysis", "Chat with AI", "Geographic Distribution"])
        
        with tab3:
            st.subheader("Geographic Distribution of Respondents")
            
            # Detect possible ZIP code columns
            zip_columns = [col for col in survey_data.columns if any(term in col.lower() for term in ['zip', 'postal', 'zipcode', 'zip_code', 'geography', 'geo'])]
            
            if zip_columns:
                # Let user select the ZIP code column
                zip_column = st.selectbox(
                    "Select the column containing ZIP codes:",
                    zip_columns
                )
                
                if st.button("Generate Map"):
                    with st.spinner("Creating map... This may take a moment."):
                        try:
                            # Create and display the map
                            m = create_map(survey_data.copy(), zip_column)
                            if m:
                                folium_static(m)
                            
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.warning("No ZIP code columns detected in the data. Please ensure your CSV contains a column with ZIP codes.")
                st.write("Available columns:", survey_data.columns.tolist())
        
        # Add your existing Analysis and Chat tabs code here...
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    if not use_sample:
        st.info("Please upload a CSV file to begin analysis.") 