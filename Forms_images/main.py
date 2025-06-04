"""
Main application entry point for the Comprehensive Survey Analyzer
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from config import (
    APP_TITLE, APP_DESCRIPTION, DEFAULT_OUTPUT_DIR,
    OPENAI_API_KEY, ENABLE_AI_BY_DEFAULT, SHOW_API_KEY_INPUT
)
from ui.styling import get_custom_css
from ui.components import (
    display_styled_dataframe, display_metrics, display_quality_score,
    display_processing_status, display_column_analysis, display_image_analysis,
    display_download_options
)
from core.analyzer import SurveyAnalyzer
from core.image_processor import ImageProcessor
from core.report_generator import ReportGenerator
from utils.logging import logger
from utils.openai_utils import openai_client
from utils.file_utils import (
    ensure_directory, get_temp_file_path, cleanup_temp_files,
    save_dataframe_to_csv, load_csv_to_dataframe, create_download_package
)

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
        'openai_api_key': OPENAI_API_KEY,
        'output_directory': DEFAULT_OUTPUT_DIR,
        'csv_analysis': None,
        'use_sample_data': False,
        'ai_text_insights': {},
        'vc_trend_insights': None
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

def create_sample_data() -> pd.DataFrame:
    """Create sample survey data for testing"""
    import random
    from datetime import datetime, timedelta
    
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

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Set page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Header
    st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
    
    if OPENAI_API_KEY:
        st.markdown(f"**🚀 {APP_DESCRIPTION}**")
        st.markdown('<div class="success-banner">✨ AI Analysis Ready - No API Key Required! ✨</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"**{APP_DESCRIPTION}**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # OpenAI API Key section
        st.subheader("🤖 AI Analysis")
        
        if OPENAI_API_KEY:
            st.success("✅ AI Analysis Enabled")
            st.info("Built-in OpenAI API key is configured for seamless AI image analysis")
            
            if SHOW_API_KEY_INPUT:
                st.markdown("**Override API Key (Optional):**")
                custom_key = st.text_input(
                    "Custom OpenAI API Key", 
                    value="",
                    type="password",
                    help="Leave empty to use built-in key, or enter your own to override"
                )
                if custom_key.strip():
                    st.session_state.openai_api_key = custom_key
                    st.info("Using custom API key")
                else:
                    st.session_state.openai_api_key = OPENAI_API_KEY
            else:
                st.session_state.openai_api_key = OPENAI_API_KEY
        else:
            st.warning("⚠️ No AI Analysis Available")
            openai_key = st.text_input(
                "OpenAI API Key", 
                value="",
                type="password",
                help="Enter your OpenAI API key for AI image analysis"
            )
            if openai_key:
                st.session_state.openai_api_key = openai_key
        
        # Output directory
        st.subheader("📁 Output Settings")
        st.session_state.output_directory = st.text_input(
            "Output Directory", 
            value=st.session_state.output_directory,
            help="Directory to save downloaded images and analysis results"
        )
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        download_images = st.checkbox("Download Google Drive Images", value=True)
        run_ai_analysis = st.checkbox("Run AI Image Analysis", value=ENABLE_AI_BY_DEFAULT)
        auto_generate_report = st.checkbox("Auto Generate Report", value=True)
        generate_insights = st.checkbox("Generate Survey Insights", value=True)
        
        if run_ai_analysis and not st.session_state.openai_api_key:
            st.error("❌ AI analysis requires an OpenAI API key")
            run_ai_analysis = False
        
        # Cost information
        if run_ai_analysis and st.session_state.openai_api_key:
            st.subheader("💰 AI Analysis Cost")
            st.info("""
            **Estimated costs:**
            - $0.01-0.02 per image
            - Very affordable for most surveys
            - Provides detailed insights worth the cost
            """)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📊 Survey Analysis", "🖼️ Image Analysis", "📋 Reports"])
    
    with tab1:
        st.header("📤 Upload and Process Survey Data")
        
        # Sample data option
        st.subheader("📊 Data Source")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            use_sample = st.checkbox("🎯 Use Sample Data", 
                                   value=st.session_state.use_sample_data,
                                   help="Load pre-built sample data for testing the application")
            st.session_state.use_sample_data = use_sample
        
        with col2:
            if use_sample:
                st.info("✨ Sample data will be loaded automatically - perfect for testing all features!")
            else:
                st.info("📤 Upload your own Google Forms CSV file to analyze your survey data")
        
        # File upload or sample data loading
        uploaded_file = None
        survey_data = None
        
        if use_sample:
            if st.button("🎯 Load Sample Data", type="primary"):
                with st.spinner("Loading sample data..."):
                    survey_data = create_sample_data()
                    st.session_state.survey_data = survey_data
                    logger.log(f"Loaded sample data with {len(survey_data)} responses", "success")
        else:
            uploaded_file = st.file_uploader(
                "Upload your Google Forms CSV file",
                type=['csv'],
                help="Upload the CSV file downloaded from Google Forms"
            )
        
        # Process uploaded file
        if uploaded_file is not None and not use_sample:
            try:
                survey_data = pd.read_csv(uploaded_file)
                st.session_state.survey_data = survey_data
                logger.log(f"Loaded CSV with {len(survey_data)} responses", "success")
                st.success(f"✅ Successfully loaded {len(survey_data)} survey responses")
            except Exception as e:
                st.error(f"❌ Error loading CSV file: {str(e)}")
                logger.log(f"Failed to load CSV: {str(e)}", "error")
        
        # If we have survey data, analyze it
        if st.session_state.survey_data is not None:
            survey_data = st.session_state.survey_data
            
            # Analyze CSV structure
            analyzer = SurveyAnalyzer(survey_data)
            csv_analysis = analyzer.analyze()
            st.session_state.csv_analysis = csv_analysis
            
            # Display analysis results
            display_styled_dataframe(survey_data.head(10), "📋 Data Preview")
            
            # Show structure analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Data Structure")
                st.write(f"**Total Responses:** {csv_analysis['total_responses']}")
                st.write(f"**Total Questions:** {csv_analysis['total_columns']}")
                st.write(f"**Google Drive Columns:** {len(csv_analysis['google_drive_columns'])}")
                st.write(f"**Text Columns:** {len(csv_analysis['text_columns'])}")
                st.write(f"**Multiple Choice Columns:** {len(csv_analysis['choice_columns'])}")
                st.write(f"**Email Columns:** {len(csv_analysis['email_columns'])}")
                st.write(f"**ZIP Code Columns:** {len(csv_analysis['zip_columns'])}")
            
            with col2:
                st.subheader("📊 Data Quality")
                missing_pct = csv_analysis['data_quality']['missing_data_percentage']
                completion_rate = csv_analysis['data_quality']['completion_rate']
                duplicates = csv_analysis['data_quality']['duplicate_responses']
                
                st.write(f"**Missing Data:** {missing_pct:.1f}%")
                st.write(f"**Completion Rate:** {completion_rate:.1f}%")
                st.write(f"**Complete Responses:** {csv_analysis['data_quality']['complete_responses']}")
                st.write(f"**Duplicate Responses:** {duplicates}")
                
                # Overall quality score
                quality_score = completion_rate - (missing_pct * 0.5) - (duplicates * 2)
                display_quality_score(quality_score)
            
            # Mark as ready for analysis
            st.session_state.analysis_complete = True
        
        elif not use_sample:
            st.info("ℹ️ Please upload a CSV file or use sample data to begin analysis")
        
        # Display processing logs
        if st.session_state.processing_logs:
            with st.expander("📋 Processing Log"):
                logger.display_logs()
    
    with tab2:
        st.header("📊 Survey Data Analysis")
        
        if st.session_state.survey_data is not None:
            # Create comprehensive dashboard
            if hasattr(st.session_state, 'csv_analysis'):
                analyzer = SurveyAnalyzer(st.session_state.survey_data)
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(get_metric_card("📋 Total Responses", str(analyzer.analysis['total_responses'])), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(get_metric_card("📝 Questions", str(analyzer.analysis['total_columns'])), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(get_metric_card("🖼️ Image Columns", str(len(analyzer.analysis['google_drive_columns']))), unsafe_allow_html=True)
                
                with col4:
                    complete_pct = (analyzer.analysis['data_quality']['complete_responses'] / analyzer.analysis['total_responses']) * 100
                    st.markdown(get_metric_card("✅ Completion Rate", f"{complete_pct:.1f}%"), unsafe_allow_html=True)
                
                # Column type breakdown
                st.subheader("📊 Column Type Distribution")
                col_type_data = {
                    "🔗 Google Drive": len(analyzer.analysis['google_drive_columns']),
                    "📝 Text": len(analyzer.analysis['text_columns']),
                    "🔢 Numeric": len(analyzer.analysis['numeric_columns']),
                    "☑️ Multiple Choice": len(analyzer.analysis['choice_columns']),
                    "📧 Email": len(analyzer.analysis['email_columns']),
                    "📍 ZIP Code": len(analyzer.analysis['zip_columns']),
                    "⏰ Timestamp": len(analyzer.analysis['response_timestamps'])
                }
                
                # Filter out empty categories
                col_type_data = {k: v for k, v in col_type_data.items() if v > 0}
                
                if col_type_data:
                    try:
                        import plotly.express as px
                        fig = px.pie(values=list(col_type_data.values()), 
                                   names=list(col_type_data.keys()), 
                                   title="Column Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        # Fallback display
                        for col_type, count in col_type_data.items():
                            st.write(f"{col_type}: {count}")
                
                # Response timeline
                timeline_df = analyzer.get_response_timeline()
                if timeline_df is not None:
                    st.subheader("⏰ Response Timeline")
                    try:
                        import plotly.express as px
                        fig = px.line(timeline_df, x='response_date', y='responses', 
                                    title='Responses Over Time', markers=True)
                        fig.update_layout(xaxis_title='Date', yaxis_title='Number of Responses')
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.line_chart(timeline_df.set_index('response_date'))
                
                # Text analysis for open-ended questions
                text_columns = [col for col in analyzer.analysis['text_columns'] 
                              if col not in analyzer.analysis['google_drive_columns']]
                
                if text_columns:
                    st.subheader("📝 Text Response Analysis")
                    
                    selected_text_col = st.selectbox("Select a text column to analyze:", text_columns)
                    
                    if selected_text_col:
                        text_analysis = analyzer.get_text_analysis(selected_text_col)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Top words
                            word_counts = text_analysis['word_frequency']
                            try:
                                import plotly.express as px
                                fig = px.bar(x=list(word_counts.values()), 
                                           y=list(word_counts.keys()), 
                                           orientation='h', 
                                           title='Most Common Words')
                                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                            except ImportError:
                                # Fallback to simple bar chart
                                word_df = pd.DataFrame(list(word_counts.items()), 
                                                     columns=['word', 'count'])
                                st.bar_chart(word_df.set_index('word'))
                        
                        with col2:
                            # Response length distribution
                            try:
                                import plotly.express as px
                                fig = px.histogram(x=text_analysis['response_length_distribution'], 
                                                 title='Response Length Distribution')
                                fig.update_layout(xaxis_title='Characters', 
                                                yaxis_title='Frequency')
                                st.plotly_chart(fig, use_container_width=True)
                            except ImportError:
                                # Fallback to simple histogram
                                st.bar_chart(text_analysis['response_length_distribution'])
                        
                        # AI insights
                        if st.session_state.openai_api_key:
                            with st.expander("🤖 Generate AI Insights"):
                                if st.button("Generate AI Insights", key=f"ai_{selected_text_col}"):
                                    with st.spinner("Generating AI insights..."):
                                        insights = openai_client.generate_text_insights(
                                            selected_text_col,
                                            st.session_state.survey_data[selected_text_col].dropna().astype(str).tolist()
                                        )
                                        st.session_state.ai_text_insights[selected_text_col] = insights
                                        st.success("AI insights generated!")
                                
                                if selected_text_col in st.session_state.ai_text_insights:
                                    insight_data = st.session_state.ai_text_insights[selected_text_col]
                                    st.subheader("AI Summary")
                                    st.write(insight_data.get("summary", ""))
                                    if insight_data.get("actionable_insights"):
                                        st.subheader("Actionable Insights")
                                        for rec in insight_data["actionable_insights"]:
                                            st.info(rec)
                        else:
                            st.info("Provide an OpenAI API key to enable AI-driven insights.")
                
                # VC Trend Insights
                if st.session_state.openai_api_key and text_columns:
                    st.markdown("---")
                    st.subheader("📈 Venture Capital Trend Insights (Beta)")
                    default_prompt = (
                        "Pretend that you are a top tier consumer venture capital firm investor, who wants to capture consumer "
                        "insights to make investable decisions. Analyze the attached consumer survey data. Identify and summarize "
                        "the most significant emerging consumer trends, shifts in preferences, and mentions of new products or "
                        "services. Segment your findings by relevant demographics, time periods, and product categories. The firm "
                        "also privatizes trends with scalable path to acquire new customers and reason for consumers to continue "
                        "to stay engaged. Highlight any anomalies or unexpected patterns, and provide supporting evidence from the "
                        "data for each identified trend."
                    )
                    custom_prompt = st.text_area("Custom Prompt", value=default_prompt, height=200)
                    if st.button("Generate VC Trend Insights"):
                        with st.spinner("Analyzing survey for VC insights..."):
                            result_md = openai_client.generate_vc_trend_insights(
                                st.session_state.survey_data,
                                text_columns,
                                custom_prompt
                            )
                            st.session_state.vc_trend_insights = result_md
                            st.success("VC trend insights generated!")
                    if st.session_state.vc_trend_insights:
                        st.markdown("### 🔍 VC Trend Insights Report")
                        st.markdown(st.session_state.vc_trend_insights, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Please upload and process your CSV file first")
        else:
            st.info("ℹ️ Please upload a CSV file to view survey analysis")
    
    with tab3:
        st.header("🖼️ Image Analysis")
        
        if st.session_state.survey_data is not None:
            if st.button('🚀 Start Image Analysis', type='primary'):
                # Initialize image processor
                image_processor = ImageProcessor(st.session_state.output_directory)
                
                # Process images
                with st.spinner('Processing images...'):
                    # TODO: Implement image processing logic
                    pass
        else:
            st.info("ℹ️ Please upload a CSV file to begin image analysis")
    
    with tab4:
        st.header("📋 Reports and Downloads")
        
        if st.session_state.analysis_complete:
            st.subheader("📊 Generate Comprehensive Report")
            
            if st.button("📝 Generate Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    # Generate report
                    report_generator = ReportGenerator(
                        st.session_state.survey_data,
                        st.session_state.csv_analysis
                    )
                    
                    report_content = report_generator.generate_report(
                        st.session_state.drive_analysis,
                        st.session_state.ai_analysis.get('analysis_results', {}) if st.session_state.ai_analysis else None,
                        st.session_state.downloaded_images,
                        st.session_state.ai_text_insights,
                        st.session_state.vc_trend_insights
                    )
                    
                    # Display report
                    st.markdown(report_content)
                    
                    # Display download options
                    display_download_options(
                        report_content,
                        st.session_state.survey_data,
                        st.session_state.ai_analysis or {}
                    )
                    
                    # Complete package download
                    if st.session_state.downloaded_images:
                        st.subheader("📦 Complete Analysis Package")
                        st.info("Download everything in one ZIP file including images, analysis, and reports")
                        
                        if st.button("📦 Create Download Package"):
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
                                        label="📦 Download Complete Package (ZIP)",
                                        data=zip_buffer.getvalue(),
                                        file_name=f"survey_analysis_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip"
                                    )
                                    
                                    st.success("✅ Package ready for download!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error creating package: {str(e)}")
        else:
            st.info("ℹ️ Please upload and process your survey data first to generate reports")

if __name__ == "__main__":
    main() 