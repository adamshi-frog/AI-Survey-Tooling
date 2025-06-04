## To Run: cd AI-Survey-Tooling && streamlit run unified_analysis_app.py

import streamlit as st
import pandas as pd
import json
import os
from openai import OpenAI
from typing import List, Dict
import folium
from streamlit_folium import folium_static
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
import requests
import base64
from PIL import Image
import io
from datetime import datetime
import re

# Set page config
st.set_page_config(
    page_title="Unified Data Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# Apply theme styling
st.markdown("""
    <style>
        /* Additional styling for specific elements */
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            background-color: #8fa051 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
client = OpenAI(
    api_key="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"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'survey_data_global' not in st.session_state:
    st.session_state.survey_data_global = None

# Shared utility functions
def style_dataframe(df):
    """Apply custom styling to dataframes"""
    return df.style.set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#f6f6f3'),
                ('color', '#2c2c2c'),
                ('font-weight', 'bold'),
                ('border', '1px solid #ddd'),
                ('padding', '10px')
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('background-color', '#f6f6f3'),
                ('color', '#2c2c2c'),
                ('border', '1px solid #ddd'),
                ('padding', '10px')
            ]
        },
        {
            'selector': 'table',
            'props': [
                ('border-collapse', 'collapse'),
                ('border-radius', '5px'),
                ('overflow', 'hidden'),
                ('width', '100%')
            ]
        }
    ])

def display_styled_dataframe(df, title=None):
    """Display a dataframe with custom styling"""
    if title:
        st.subheader(title)
    styled_df = style_dataframe(df)
    st.write(styled_df.to_html(), unsafe_allow_html=True)

# Survey Analysis Functions
@st.cache_data
def load_zip_database():
    try:
        zip_db = pd.read_csv('ai_survey_tool/us_zip_codes.csv')
        zip_db['ZIP'] = zip_db['ZIP'].astype(str).str.zfill(5)
        return zip_db
    except Exception as e:
        st.error(f"Error loading ZIP code database: {str(e)}")
        return None

def clean_zip_code(zip_code):
    """Clean and standardize ZIP code format."""
    try:
        zip_str = str(zip_code).strip('"').strip("'").strip()
        zip_str = ''.join(filter(str.isdigit, zip_str))
        if len(zip_str) >= 5:
            return zip_str[:5]
        elif len(zip_str) < 5:
            return zip_str.zfill(5)
        return None
    except:
        return None

def get_coordinates_simple(zip_code: str) -> tuple:
    """Get coordinates using a simple geocoding API."""
    try:
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
        m = folium.Map(
            location=[39.8283, -98.5795],
            zoom_start=4,
            tiles='CartoDB positron'
        )
        
        st.write("### Cleaning ZIP Codes")
        data[zip_column] = data[zip_column].apply(clean_zip_code)
        
        st.write("Sample of cleaned ZIP codes:")
        zip_sample = pd.DataFrame({
            'Original': data[zip_column].head(),
            'Cleaned ZIP': data[zip_column].head()
        })
        display_styled_dataframe(zip_sample)
        
        zip_counts = data[zip_column].value_counts()
        st.write(f"Number of unique ZIP codes: {len(zip_counts)}")
        
        success_count = 0
        error_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (zip_code, count) in enumerate(zip_counts.items()):
            if pd.isna(zip_code) or not zip_code:
                continue
                
            try:
                progress = (idx + 1) / len(zip_counts)
                progress_bar.progress(progress)
                status_text.text(f"Processing ZIP code {idx + 1} of {len(zip_counts)}")
                
                lat, lon = get_coordinates_simple(zip_code)
                
                if lat and lon:
                    radius = min(max(count * 2, 8), 20)
                    color = '#ff4b4b' if count < 5 else '#ff8c42' if count < 10 else '#dc3545'
                    
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
                
                time.sleep(1)
                
            except Exception as e:
                error_count += 1
                st.write(f"Error with ZIP {zip_code}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.write(f"Successfully mapped: {success_count} ZIP codes")
        st.write(f"Failed to map: {error_count} ZIP codes")
        
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

def load_sample_data():
    try:
        sample_path = os.path.join('ai_survey_tool', 'Sample_CSV', 'mockgeo.csv')
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
        else:
            st.error(f"Sample file not found at {sample_path}")
            return None
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        return None

# App Store Parser Functions
def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def parse_app_store_screenshot(image):
    """Use OpenAI Vision API to parse the screenshot and extract app data"""
    try:
        base64_image = encode_image_to_base64(image)
        
        prompt = """
        Analyze this iOS App Store screenshot and extract the app information. 
        Look for each app entry and extract:
        1. App name (full name as shown)
        2. Download/purchase date (in the format shown)
        
        Return the data as a JSON array with this exact structure:
        [
            {
                "app_name": "App Name Here",
                "date": "Date as shown (e.g., Jan 14, 2025)"
            }
        ]
        
        Only include apps that are clearly visible and readable. If you can't read the full app name or date clearly, skip that entry.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
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
        
        try:
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                apps_data = json.loads(json_str)
                return apps_data, result
            else:
                return [], result
        except json.JSONDecodeError:
            return [], result
            
    except Exception as e:
        st.error(f"Error parsing image: {str(e)}")
        return [], str(e)

def generate_consumer_insights(apps_data):
    """Generate VC-style consumer insights from extracted app data"""
    try:
        app_list = []
        for app in apps_data:
            app_list.append(f"- {app['app_name']} (Downloaded: {app['date']})")
        
        apps_text = "\n".join(app_list)
        
        prompt = f"""
        As a top-tier consumer venture capital firm investor, analyze this consumer app download data to capture insights for investable decisions.

        App Download Data:
        {apps_text}

        Provide a comprehensive analysis covering:

        1. **EMERGING CONSUMER TRENDS**: What shifts in consumer behavior and preferences do you observe?

        2. **INVESTMENT OPPORTUNITIES**: Which app categories or business models show strong potential for growth and investment?

        3. **CUSTOMER ACQUISITION INSIGHTS**: What patterns suggest scalable paths to acquire new customers?

        4. **ENGAGEMENT & RETENTION FACTORS**: What keeps consumers engaged with these types of services?

        5. **MARKET TIMING**: Based on download dates, what trends are accelerating or declining?

        6. **DEMOGRAPHIC INSIGHTS**: What can we infer about the consumer profile and their priorities?

        7. **MONETIZATION POTENTIAL**: Which categories show strongest revenue generation opportunities?

        Focus on actionable insights that could inform investment decisions in the consumer technology space. Be specific about trends that have scalable business models and sustainable competitive advantages.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Main UI
st.title("📊 Unified Data Analysis Platform")
st.write("Comprehensive platform for survey analysis and consumer app insights.")

# Create main tabs
tab1, tab2 = st.tabs(["📋 Survey Analysis", "📱 App Store Parser"])

# Tab 1: Survey Analysis Dashboard
with tab1:
    st.header("Survey Analysis Dashboard")
    st.write("Upload your survey CSV file to analyze responses and geographic distribution.")
    
    # Add option to use sample data
    use_sample = st.checkbox("Use sample data (mockgeo.csv)")
    
    if use_sample:
        survey_data = load_sample_data()
    else:
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="survey_upload")
        if uploaded_file is not None:
            survey_data = pd.read_csv(uploaded_file)
        else:
            survey_data = None
    
    if survey_data is not None:
        try:
            # Show data preview with styling
            display_styled_dataframe(survey_data.head(), "Data Preview")
            
            # Add sub-tabs for survey analysis
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Analysis", "Chat with AI", "Geographic Distribution"])
            
            with sub_tab3:
                st.subheader("Geographic Distribution of Respondents")
                
                # Detect possible ZIP code columns
                zip_columns = [col for col in survey_data.columns if any(term in col.lower() for term in ['zip', 'postal', 'zipcode', 'zip_code', 'geography', 'geo'])]
                
                if zip_columns:
                    # Let user select the ZIP code column
                    zip_column = st.selectbox(
                        "Select the column containing ZIP codes:",
                        zip_columns
                    )
                    
                    if st.button("Generate Map", key="generate_map"):
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
            
            with sub_tab1:
                st.info("Analysis features coming soon...")
            
            with sub_tab2:
                st.info("AI Chat features coming soon...")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        if not use_sample:
            st.info("Please upload a CSV file to begin analysis.")

# Tab 2: iOS App Store Screenshot Parser
with tab2:
    st.header("iOS App Store Screenshot Parser")
    st.write("Upload screenshots from your iOS App Store purchase history to extract app names and download dates.")
    
    # Instructions
    with st.expander("📋 How to use"):
        st.markdown("""
        1. **Take screenshots** of your iOS App Store purchase history:
           - Open App Store → Account (profile icon) → Purchased
           - Take multiple screenshots of the app list (scroll down for more apps)
        
        2. **Upload the screenshots** using the file uploader below:
           - Select multiple files at once (Ctrl/Cmd + click)
           - Or drag and drop multiple images
        
        3. **Click "Parse All Screenshots"** to process the batch
        
        4. **Review results** and export the combined data
        
        **💡 Batch Processing Benefits:**
        - Process multiple screenshots at once
        - Combined results in one CSV file
        - Track which apps came from which screenshot
        - Progress tracking for large batches
        """)
    
    # File uploader - Updated for batch uploads
    uploaded_files = st.file_uploader(
        "Upload your App Store screenshots",
        type=["png", "jpg", "jpeg"],
        help="Upload one or more screenshots from your iOS App Store purchase history",
        accept_multiple_files=True,
        key="app_store_upload"
    )
    
    if uploaded_files:
        # Show number of uploaded files
        st.info(f"📁 {len(uploaded_files)} screenshot(s) uploaded")
        
        # Display uploaded images in a grid
        if len(uploaded_files) <= 4:
            cols = st.columns(len(uploaded_files))
            for i, uploaded_file in enumerate(uploaded_files):
                with cols[i]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Screenshot {i+1}", use_container_width=True)
        else:
            st.write("📸 **Uploaded Screenshots:**")
            # For more than 4 images, show them in rows of 4
            for i in range(0, len(uploaded_files), 4):
                cols = st.columns(4)
                for j, uploaded_file in enumerate(uploaded_files[i:i+4]):
                    with cols[j]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=f"Screenshot {i+j+1}", use_container_width=True)
        
        st.divider()
        
        # Processing section
        st.subheader("🔍 Batch Processing Results")
        
        if st.button("Parse All Screenshots", type="primary", key="parse_screenshots"):
            all_apps_data = []
            processing_results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing screenshot {i+1} of {len(uploaded_files)}...")
                
                try:
                    image = Image.open(uploaded_file)
                    apps_data, raw_response = parse_app_store_screenshot(image)
                    
                    # Add source file info to each app entry
                    for app in apps_data:
                        app['source_file'] = uploaded_file.name
                        app['screenshot_number'] = i + 1
                    
                    all_apps_data.extend(apps_data)
                    processing_results.append({
                        'file': uploaded_file.name,
                        'apps_found': len(apps_data),
                        'status': '✅ Success' if apps_data else '⚠️ No apps found'
                    })
                    
                except Exception as e:
                    processing_results.append({
                        'file': uploaded_file.name,
                        'apps_found': 0,
                        'status': f'❌ Error: {str(e)}'
                    })
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if all_apps_data:
                # Create comprehensive DataFrame
                df = pd.DataFrame(all_apps_data)
                
                # Reorder columns for better display
                column_order = ['app_name', 'date', 'source_file', 'screenshot_number']
                df = df[column_order]
                
                # Add index starting from 1
                df.index = range(1, len(df) + 1)
                df.index.name = "Row"
                
                # Display summary
                st.success(f"🎉 Successfully extracted {len(all_apps_data)} apps from {len(uploaded_files)} screenshots!")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Apps Found", len(all_apps_data))
                with col2:
                    st.metric("Screenshots Processed", len(uploaded_files))
                with col3:
                    avg_per_screenshot = len(all_apps_data) / len(uploaded_files)
                    st.metric("Avg Apps per Screenshot", f"{avg_per_screenshot:.1f}")
                
                # Display processing summary
                st.subheader("📊 Processing Summary")
                summary_df = pd.DataFrame(processing_results)
                summary_df.index = range(1, len(summary_df) + 1)
                summary_df.index.name = "Screenshot"
                display_styled_dataframe(summary_df)
                
                # Display all extracted data
                display_styled_dataframe(df, "📊 All Extracted App Data")
                
                # Generate and display consumer insights
                st.subheader("🔍 Consumer Insights & Investment Analysis")
                with st.spinner("🧠 Generating VC-style consumer insights..."):
                    insights = generate_consumer_insights(all_apps_data)
                
                with st.expander("📈 View Consumer Insights Analysis", expanded=True):
                    st.markdown(insights)
                
                # Download options
                st.subheader("📥 Download Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download all data
                    csv_all = df.to_csv(index=True)
                    st.download_button(
                        label="📥 Download All Data (CSV)",
                        data=csv_all,
                        file_name=f"app_store_batch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download summary
                    csv_summary = summary_df.to_csv(index=True)
                    st.download_button(
                        label="📊 Download Summary (CSV)",
                        data=csv_summary,
                        file_name=f"app_store_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                # Show apps by screenshot
                with st.expander("📱 View Apps by Screenshot"):
                    for i in range(1, len(uploaded_files) + 1):
                        screenshot_apps = df[df['screenshot_number'] == i]
                        if not screenshot_apps.empty:
                            st.write(f"**Screenshot {i}:** {len(screenshot_apps)} apps")
                            display_styled_dataframe(screenshot_apps.drop(['screenshot_number'], axis=1))
                        else:
                            st.write(f"**Screenshot {i}:** No apps found")
                            
            else:
                st.error("❌ No apps were extracted from any screenshot. Please check image quality and try again.")
                
                # Show processing summary even if no apps found
                st.subheader("📊 Processing Summary")
                summary_df = pd.DataFrame(processing_results)
                summary_df.index = range(1, len(summary_df) + 1)
                summary_df.index.name = "Screenshot"
                display_styled_dataframe(summary_df) 