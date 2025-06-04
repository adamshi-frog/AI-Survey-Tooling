## cd AI-Survey-Tooling && streamlit run app_store_parser.py

import streamlit as st
import pandas as pd
import base64
from PIL import Image
import io
from openai import OpenAI
import json
from datetime import datetime
import re

# Set page config
st.set_page_config(
    page_title="iOS App Store Screenshot Parser",
    page_icon="📱",
    layout="wide"
)

# Apply the same theme styling
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

def encode_image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def parse_app_store_screenshot(image):
    """Use OpenAI Vision API to parse the screenshot and extract app data"""
    try:
        # Convert image to base64
        base64_image = encode_image_to_base64(image)
        
        # Create the prompt for the AI
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
        
        # Call OpenAI Vision API using new client format
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
        
        # Extract the response
        result = response.choices[0].message.content
        
        # Try to parse the JSON
        try:
            # Find JSON in the response (in case there's additional text)
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
        # Prepare the data for analysis
        app_list = []
        for app in apps_data:
            app_list.append(f"- {app['app_name']} (Downloaded: {app['date']})")
        
        apps_text = "\n".join(app_list)
        
        # Create the VC analysis prompt
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
        
        # Call OpenAI for analysis
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

# Main UI
st.title("📱 iOS App Store Screenshot Parser")
st.write("Upload a screenshot from your iOS App Store purchase history to extract app names and download dates.")

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
    accept_multiple_files=True
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
    
    if st.button("Parse All Screenshots", type="primary"):
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
            st.subheader("�� Download Options")
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