## To Run insert the following cd ~/Documents && streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import json
import os
import openai
from typing import List, Dict
import folium
from streamlit_folium import folium_static
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

# Set page config
st.set_page_config(
    page_title="Survey Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Set OpenAI API key directly
openai.api_key = "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"

# Add this after the imports at the top
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'survey_data_global' not in st.session_state:
    st.session_state.survey_data_global = None

def identify_question_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies likely question columns in a survey DataFrame based on common patterns.
    """
    question_columns = []
    patterns = [
        'question', 'q_', 'q.', 'qn_', 'qn.', 
        'response', 'answer', 'survey_q', 
        'what', 'how', 'why', 'when', 'where', 'who',
        'rate', 'rank', 'opinion', 'feedback'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in patterns):
            question_columns.append(col)
        elif df[col].dtype == 'object' and df[col].str.len().mean() > 20:
            question_columns.append(col)
            
    return question_columns

def analyze_survey_data(survey_data: pd.DataFrame) -> Dict[str, str]:
    """
    Analyzes survey data using OpenAI's API.
    """
    question_analyses = {}
    columns_to_analyze = identify_question_columns(survey_data)
    
    for column_name in columns_to_analyze:
        if column_name in survey_data.columns:
            question_responses = survey_data[column_name].dropna().tolist()

            if not question_responses:
                question_analyses[column_name] = "No valid responses for analysis."
                continue

            responses_text = "\n".join([str(response) for response in question_responses])
            custom_prompt = f"Pretend that you are a top tier consumer venture capital firm investor, who wants to capture consumer insights to make investable decisions. Analyze the attached consumer survey data. Identify and summarize the most significant emerging consumer trends, shifts in preferences, and mentions of new products or services. Segment your findings by relevant demographics, time periods, and product categories. The firm also privatizes trends with scalable path to acquire new customers and reason for consumers to continue to stay engaged. Highlight any anomalies or unexpected patterns, and provide supporting evidence from the data for each identified trend:\n{responses_text}\n\nProvide a concise summary."
            
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that analyzes survey responses."},
                    {"role": "user", "content": custom_prompt}
                ]

                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=750,
                    temperature=0.7
                )
                
                analysis_result = response.choices[0].message.content.strip()
                question_analyses[column_name] = analysis_result
                
            except Exception as e:
                question_analyses[column_name] = f"Error analyzing this question: {str(e)}"
                
    return question_analyses

def generate_comprehensive_analysis(question_analyses: Dict[str, str]) -> str:
    """
    Generates a comprehensive analysis from individual question analyses.
    """
    if not question_analyses:
        return "No analyses available for comprehensive summary."

    combined_analysis_prompt = "Here are the analyses of different survey questions from a survey:\n\n"
    
    for question, analysis in question_analyses.items():
        if not analysis.startswith("Error analyzing") and "No valid responses" not in analysis:
            combined_analysis_prompt += f"Analysis for '{question}':\n{analysis}\n\n"

    combined_analysis_prompt += "Based on these analyses, provide a comprehensive overview of the key findings from the entire survey, highlighting major themes and insights. Summarize concisely."

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes survey analyses."},
            {"role": "user", "content": combined_analysis_prompt}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error generating comprehensive analysis: {str(e)}"

def get_chatbot_response(user_question: str, survey_data: pd.DataFrame = None) -> str:
    """
    Get response from the AI chatbot about the survey data with full context.
    """
    try:
        # Create detailed context about the survey data
        context = "You are a helpful assistant analyzing survey data. "
        if survey_data is not None:
            # Basic survey information
            context += f"The survey has {len(survey_data)} responses and the following columns: {', '.join(survey_data.columns)}. "
            
            # Add sample data context
            context += "\n\nHere is the actual survey data:\n"
            
            # Convert DataFrame to string representation
            data_str = survey_data.to_string(max_rows=None, max_cols=None)
            context += data_str
            
            # Add column summaries
            context += "\n\nColumn Summaries:\n"
            for column in survey_data.columns:
                if survey_data[column].dtype in ['object', 'string']:
                    # For text columns, add unique value counts
                    value_counts = survey_data[column].value_counts()
                    context += f"\n{column} - Top responses:\n{value_counts.head().to_string()}"
                else:
                    # For numeric columns, add statistical summary
                    stats = survey_data[column].describe()
                    context += f"\n{column} - Statistics:\n{stats.to_string()}"
        
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": "Based on the survey data provided above, " + user_question}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000,  # Increased token limit for more detailed responses
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting response: {str(e)}"

def extract_insightful_quote_with_ai(row, exclude_cols=['Respondent_ID', 'location']):
    """
    Uses GPT to identify the most insightful or interesting quote from a respondent's answers.
    """
    try:
        # Combine all responses into context
        responses = []
        for col, value in row.items():
            if col not in exclude_cols and pd.notna(value) and isinstance(value, str):
                responses.append(f"{col}: {value}")
        
        if not responses:
            return None
            
        context = "\n".join(responses)
        
        # Prompt GPT to identify the most insightful quote
        messages = [
            {"role": "system", "content": """You are an expert at identifying insightful and interesting quotes from survey responses. 
             Look for responses that are:
             1. Unique or unexpected perspectives
             2. Well-articulated thoughts or opinions
             3. Specific examples or experiences
             4. Novel insights or observations
             Return ONLY the quote itself, no explanation or additional text."""},
            {"role": "user", "content": f"From the following survey responses, identify and return the single most insightful or interesting quote. Return only the quote itself, maximum 150 characters:\n\n{context}"}
        ]

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        
        quote = response.choices[0].message.content.strip()
        
        # Clean up the quote (remove any quotation marks added by GPT)
        quote = quote.strip('"\'')
        
        # Truncate if somehow still too long
        if len(quote) > 150:
            quote = quote[:147] + "..."
            
        return quote if quote else None
        
    except Exception as e:
        st.warning(f"Error extracting quote: {str(e)}")
        return None

def create_location_map(survey_data: pd.DataFrame, location_column: str) -> folium.Map:
    """
    Creates an interactive map with pin-style markers for each location in the survey data.
    Includes AI-selected insightful quotes on hover.
    """
    # Initialize the geocoder with rate limiting
    geolocator = Nominatim(user_agent="survey_analysis_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # Create a map centered on the US with a dark theme
    m = folium.Map(
        location=[39.8283, -98.5795],
        zoom_start=4,
        tiles='CartoDB dark_matter'
    )
    
    # Keep track of locations we've already processed
    location_cache = {}
    
    # Add cache for quotes to avoid repeated API calls
    if 'quote_cache' not in st.session_state:
        st.session_state.quote_cache = {}

    # Define columns to exclude from display
    exclude_cols = ['Respondent_ID', location_column, 'index']

    # Function to determine marker size and color based on respondent count
    def get_marker_style(count):
        # Base size scaled by respondent count, but with reasonable limits
        size = min(max(count * 5, 20), 50)  # Min 20px, max 50px
        
        # Color gradient based on count (red -> orange -> yellow)
        if count < 5:
            color = '#ff4b4b'  # Red for small groups
        elif count < 10:
            color = '#ff8c42'  # Orange for medium groups
        else:
            color = '#ffd700'  # Yellow for large groups
            
        return size, color

    # Custom pin-style icon HTML template
    PIN_TEMPLATE = """
        <div style="
            width: {size}px;
            height: {size}px;
            background-color: {color};
            border-radius: 50% 50% 50% 0;
            transform: rotate(-45deg);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            border: 2px solid white;
            position: relative;
        ">
            <div style="
                transform: rotate(45deg);
                color: white;
                font-weight: bold;
                font-size: {font_size}px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            ">{count}</div>
        </div>
        <div style="
            width: 0;
            height: 0;
            border-left: {shadow_size}px solid transparent;
            border-right: {shadow_size}px solid transparent;
            border-top: {shadow_size}px solid rgba(0,0,0,0.2);
            margin-top: -{shadow_size}px;
            margin-left: {shadow_offset}px;
        "></div>
    """

    # Clean and process location data
    def clean_location(loc):
        if pd.isna(loc) or not isinstance(loc, str):
            return None
        loc = loc.strip()
        if ',' in loc:
            city, state = map(str.strip, loc.split(',', 1))
            return f"{city}, {state}"
        return loc

    # Format respondent data for popup
    def format_respondent_data(row):
        html = "<div style='font-family: Arial, sans-serif; max-height: 300px; overflow-y: auto;'>"
        html += f"<h4 style='color: #2c3e50;'>Respondent Details</h4>"
        
        for col in row.index:
            if col != location_column:  # Skip location column as it's already shown
                value = str(row[col])
                if pd.notna(value) and value.strip():  # Only show non-empty values
                    html += f"<p><strong>{col}:</strong><br>{value}</p>"
        
        html += "</div>"
        return html

    # Clean locations
    survey_data[location_column] = survey_data[location_column].apply(clean_location)
    
    # Group data by location
    location_groups = survey_data.groupby(location_column)
    
    # Progress bar for geocoding
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_locations = len(location_groups)
    for idx, (location, group) in enumerate(location_groups):
        if location is None:
            continue
            
        try:
            status_text.text(f"Processing location: {location}")
            progress_bar.progress((idx + 1) / total_locations)
            
            if location in location_cache:
                lat, lon = location_cache[location]
            else:
                search_location = location if 'usa' in location.lower() else f"{location}, USA"
                location_data = geocode(search_location)
                
                if location_data:
                    lat, lon = location_data.latitude, location_data.longitude
                    location_cache[location] = (lat, lon)
                else:
                    st.warning(f"Could not find coordinates for: {location}")
                    continue

            # Get marker style based on group size
            size, color = get_marker_style(len(group))
            font_size = size * 0.4  # Scale font size relative to marker size
            shadow_size = size * 0.3  # Scale shadow size
            shadow_offset = size * 0.2  # Scale shadow offset

            # Create custom pin HTML
            pin_html = PIN_TEMPLATE.format(
                size=size,
                color=color,
                count=len(group),
                font_size=font_size,
                shadow_size=shadow_size,
                shadow_offset=shadow_offset
            )

            # Create tooltip content with AI-selected quotes
            tooltip_content = f"""
                <div class="custom-tooltip" style="
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    max-width: 300px;
                    background-color: rgba(33, 37, 41, 0.95);
                    color: white;
                    border-radius: 8px;
                    padding: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    border: 1px solid rgba(255,255,255,0.1);
                ">
                    <div style="
                        font-weight: 600;
                        color: #ffffff;
                        margin-bottom: 10px;
                        font-size: 14px;
                        border-bottom: 1px solid rgba(255,255,255,0.2);
                        padding-bottom: 8px;
                    ">
                        {location} <span style="
                            background-color: rgba(255,255,255,0.2);
                            padding: 2px 6px;
                            border-radius: 12px;
                            font-size: 12px;
                            margin-left: 6px;
                        ">{len(group)} respondents</span>
                    </div>
                    <div style="
                        max-height: 300px;
                        overflow-y: auto;
                        overflow-x: hidden;
                        word-wrap: break-word;
                    ">
            """
            
            # Add quotes from each respondent
            quote_count = 0
            for _, respondent in group.iterrows():
                respondent_id = respondent.get('Respondent_ID', 'Anonymous')
                cache_key = f"{respondent_id}_{location}"
                
                # Check cache first
                if cache_key in st.session_state.quote_cache:
                    quote = st.session_state.quote_cache[cache_key]
                else:
                    quote = extract_insightful_quote_with_ai(respondent, exclude_cols=exclude_cols)
                    st.session_state.quote_cache[cache_key] = quote
                
                if quote:
                    quote_count += 1
                    if quote_count <= 3:  # Limit to 3 quotes per location
                        tooltip_content += f"""
                            <div style="
                                margin-top: 8px;
                                padding: 8px;
                                background-color: rgba(255,255,255,0.05);
                                border-radius: 6px;
                                border-left: 3px solid {color};
                            ">
                                <div style="
                                    font-style: italic;
                                    color: #e9ecef;
                                    margin-bottom: 4px;
                                    font-size: 13px;
                                    line-height: 1.4;
                                    word-wrap: break-word;
                                ">"{quote}"</div>
                                <div style="
                                    font-size: 11px;
                                    color: #adb5bd;
                                    text-align: right;
                                ">- {respondent_id}</div>
                            </div>
                        """
            
            if quote_count == 0:
                tooltip_content += """
                    <div style="
                        color: #adb5bd;
                        font-style: italic;
                        text-align: center;
                        padding: 10px;
                    ">
                        Hover and double-click to see responses
                    </div>
                """
            
            tooltip_content += """
                    </div>
                    <div style="
                        margin-top: 8px;
                        font-size: 11px;
                        color: #adb5bd;
                        text-align: center;
                        border-top: 1px solid rgba(255,255,255,0.1);
                        padding-top: 8px;
                    ">
                        Double-click for full responses
                    </div>
                </div>
            """

            # Create popup content with enhanced styling
            popup_content = f"""
                <div style="
                    font-family: 'Helvetica Neue', Arial, sans-serif;
                    max-width: 300px;
                    max-height: 400px;
                    overflow-y: auto;
                    padding: 12px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                ">
                    <style>
                        .popup-content {{
                            width: 100%;
                            box-sizing: border-box;
                        }}
                        .popup-content * {{
                            max-width: 100%;
                            word-wrap: break-word;
                            overflow-wrap: break-word;
                            white-space: pre-wrap;
                        }}
                        .response-card {{
                            margin-bottom: 12px;
                            padding: 8px;
                            background-color: #f8f9fa;
                            border-radius: 6px;
                            border-left: 3px solid {color};
                            width: 100%;
                            box-sizing: border-box;
                        }}
                        .response-field {{
                            margin-bottom: 6px;
                            width: 100%;
                        }}
                        .field-label {{
                            font-weight: 600;
                            color: #495057;
                            font-size: 12px;
                            margin-bottom: 2px;
                        }}
                        .field-value {{
                            color: #212529;
                            font-size: 13px;
                            line-height: 1.4;
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            overflow-wrap: break-word;
                            width: 100%;
                            box-sizing: border-box;
                        }}
                    </style>
                    <div class="popup-content">
                        <h3 style="
                            color: #2c3e50;
                            margin: 0 0 8px 0;
                            padding-bottom: 6px;
                            border-bottom: 2px solid {color};
                            word-wrap: break-word;
                            font-size: 16px;
                        ">{location}</h3>
                        <div style="
                            background-color: #f8f9fa;
                            padding: 6px 8px;
                            border-radius: 6px;
                            margin-bottom: 12px;
                            font-size: 13px;
                        ">
                            <b>Number of Respondents:</b> {len(group)}
                        </div>
                        <div style="
                            max-height: 300px;
                            overflow-y: auto;
                            padding-right: 8px;
                            width: 100%;
                            box-sizing: border-box;
                        ">
            """
            
            # Add individual respondent sections to popup
            for _, respondent in group.iterrows():
                popup_content += '<div class="response-card">'
                
                for col in respondent.index:
                    if col not in exclude_cols and pd.notna(respondent[col]):
                        value = str(respondent[col])
                        if value.strip():
                            popup_content += f"""
                                <div class="response-field">
                                    <div class="field-label">{col}</div>
                                    <div class="field-value">{value}</div>
                                </div>
                            """
                
                popup_content += "</div>"
            
            popup_content += """
                        </div>
                    </div>
                </div>
            """

            # Create the marker with custom icon
            icon = folium.DivIcon(
                html=pin_html,
                icon_size=(size, size + shadow_size),
                icon_anchor=(size/2, size)
            )
            
            # Create marker with the custom icon and add popup/tooltip
            folium.Marker(
                location=[lat, lon],
                icon=icon,
                popup=folium.Popup(popup_content, max_width=300),  # Restore max_width
                tooltip=folium.Tooltip(tooltip_content, permanent=False, sticky=True)
            ).add_to(m)
            
        except Exception as e:
            st.warning(f"Error processing location '{location}': {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Update legend to reflect new pin style
    legend_html = """
        <div style='position: fixed; 
                    bottom: 50px; right: 50px; 
                    border:2px solid grey; z-index:9999;
                    background-color:rgba(255, 255, 255, 0.9);
                    padding: 10px;
                    font-size:12px;
                    border-radius: 5px;'>
            <p><b>Map Information</b></p>
            <p>📍 Pin size and color indicate number of respondents</p>
            <p style='color: #ff4b4b'>● Small groups (1-4)</p>
            <p style='color: #ff8c42'>● Medium groups (5-9)</p>
            <p style='color: #ffd700'>● Large groups (10+)</p>
            <p>👆 Hover to see AI-selected insightful quotes</p>
            <p>👆👆 Double-click for full responses</p>
            <p>🤖 Quotes selected by AI for relevance</p>
        </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
            
    return m

# Main Streamlit UI
st.title("Survey Analysis Dashboard")
st.write("Upload your survey CSV file to analyze responses using AI.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load the data
        survey_data = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(survey_data.head())
        
        # Add tabs for Analysis, Chat, and Map
        tab1, tab2, tab3 = st.tabs(["Analysis", "Chat with AI", "Geographic Distribution"])
        
        with tab1:
            # Move all the existing analysis code here
            if st.button("Analyze Survey Data"):
                with st.spinner("Analyzing survey data..."):
                    # Perform individual analyses
                    question_analyses = analyze_survey_data(survey_data)
                    
                    # Display individual analyses
                    st.subheader("Individual Question Analyses")
                    for question, analysis in question_analyses.items():
                        st.write(f"**{question}**")
                        st.write(analysis)
                        st.markdown("---")
                    
                    # Generate and display comprehensive analysis
                    st.subheader("Comprehensive Analysis")
                    comprehensive_analysis = generate_comprehensive_analysis(question_analyses)
                    st.write(comprehensive_analysis)
                    
                    # Option to download results
                    combined_results = {
                        "individual_analyses": question_analyses,
                        "comprehensive_analysis": comprehensive_analysis
                    }
                    
                    st.download_button(
                        label="Download Analysis Results",
                        data=json.dumps(combined_results, indent=2),
                        file_name="survey_analysis_results.json",
                        mime="application/json"
                    )
        
        with tab2:
            st.subheader("Chat with AI about your Survey Data")
            st.write("Ask questions about your survey data and get AI-powered insights.")
            
            # Store survey data globally for chat access
            st.session_state.survey_data_global = survey_data
            
            # Chat interface
            user_question = st.text_input("Ask a question about your survey data:", key="user_question")
            if st.button("Send", key="send_button"):
                if user_question:
                    # Add user question to chat history
                    st.session_state.chat_history.append(("user", user_question))
                    
                    # Get AI response
                    ai_response = get_chatbot_response(user_question, st.session_state.survey_data_global)
                    st.session_state.chat_history.append(("assistant", ai_response))
            
            # Display chat history
            st.subheader("Chat History")
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.write("You: " + message)
                else:
                    st.write("AI: " + message)
                st.markdown("---")
                
        with tab3:
            st.subheader("Geographic Distribution of Respondents")
            
            # Detect possible location columns
            location_columns = [col for col in survey_data.columns if any(term in col.lower() for term in ['location', 'city', 'state', 'geography', 'region'])]
            
            if location_columns:
                # Let user select the location column
                location_column = st.selectbox(
                    "Select the column containing location data (should be in 'city, state' format):",
                    location_columns
                )
                
                # Debug information
                st.write("### Data Preview")
                st.write("First few rows of location data:")
                st.write(survey_data[location_column].head())
                
                if st.button("Generate Map"):
                    with st.spinner("Creating map... This may take a moment."):
                        try:
                            # Create and display the map
                            st.write("Respondent Distribution Map")
                            m = create_location_map(survey_data, location_column)
                            folium_static(m)
                            
                            # Display location statistics
                            st.subheader("Location Statistics")
                            location_stats = survey_data[location_column].value_counts()
                            st.write("Top 10 locations by number of respondents:")
                            st.dataframe(location_stats.head(10))
                            
                        except Exception as e:
                            st.error(f"Error creating map: {str(e)}")
                            st.write("### Debug Information")
                            st.write("Unique locations found:")
                            st.write(survey_data[location_column].unique())
            else:
                st.warning("No location columns detected in the data. Please ensure your CSV contains a column with geographical information in 'city, state' format.")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.") 