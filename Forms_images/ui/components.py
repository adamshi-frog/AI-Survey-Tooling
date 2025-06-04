"""
Reusable UI components for the Comprehensive Survey Analyzer
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional
from ..utils.logging import logger
from ..ui.styling import get_metric_card, get_banner

def display_styled_dataframe(df: pd.DataFrame, title: Optional[str] = None, use_container_width: bool = True) -> None:
    """Display a dataframe with custom styling"""
    if title:
        st.subheader(title)
    
    # Check if we have a small enough dataframe to apply styling
    if len(df) <= 100 and len(df.columns) <= 20:
        try:
            styled_df = df.style.set_table_styles([
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
            st.write(styled_df.to_html(escape=False, table_id="styled-table"), unsafe_allow_html=True)
        except Exception as e:
            # Fallback to regular dataframe display
            st.dataframe(df, use_container_width=use_container_width)
    else:
        # For large dataframes, use regular display
        st.dataframe(df, use_container_width=use_container_width)

def display_metrics(metrics: Dict[str, float]) -> None:
    """Display metrics in a grid layout"""
    cols = st.columns(len(metrics))
    for col, (title, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(get_metric_card(title, f"{value:.1f}"), unsafe_allow_html=True)

def display_quality_score(score: float) -> None:
    """Display data quality score with appropriate styling"""
    if score >= 90:
        st.success(f"🟢 Excellent quality ({score:.1f}/100)")
    elif score >= 70:
        st.warning(f"🟡 Good quality ({score:.1f}/100)")
    else:
        st.error(f"🔴 Needs improvement ({score:.1f}/100)")

def display_processing_status(message: str, status_type: str = "info") -> None:
    """Display a processing status message"""
    st.markdown(get_banner(message, status_type), unsafe_allow_html=True)
    logger.log(message, status_type)

def display_column_analysis(analysis: Dict) -> None:
    """Display detailed column analysis"""
    st.subheader(f"📊 Analysis for {analysis['column_name']}")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Values", analysis['total_values'])
    with col2:
        st.metric("Non-Null Values", analysis['non_null_values'])
    with col3:
        st.metric("Unique Values", analysis['unique_values'])
    
    # Type-specific analysis
    if 'min' in analysis:  # Numeric column
        st.subheader("📈 Numeric Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{analysis['min']:.2f}")
        with col2:
            st.metric("Max", f"{analysis['max']:.2f}")
        with col3:
            st.metric("Mean", f"{analysis['mean']:.2f}")
        with col4:
            st.metric("Median", f"{analysis['median']:.2f}")
    
    elif 'value_distribution' in analysis:  # Choice column
        st.subheader("📊 Value Distribution")
        df = pd.DataFrame(list(analysis['value_distribution'].items()), 
                         columns=['Value', 'Count'])
        df['Percentage'] = (df['Count'] / df['Count'].sum() * 100).round(1)
        display_styled_dataframe(df)
    
    # Sample values
    st.subheader("📝 Sample Values")
    st.write(analysis['sample_values'])

def display_image_analysis(image_path: str, analysis: Dict) -> None:
    """Display image analysis results"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image_path, use_column_width=True)
    
    with col2:
        st.subheader("📊 Image Analysis")
        
        # Basic info
        st.write(f"**Format:** {analysis.get('format', 'N/A')}")
        st.write(f"**Dimensions:** {analysis.get('dimensions', 'N/A')}")
        st.write(f"**Size:** {analysis.get('file_size', 0) / 1024:.1f} KB")
        
        # AI analysis if available
        if 'ai_analysis' in analysis:
            ai_info = analysis['ai_analysis']
            
            if ai_info.get('description'):
                st.write("**Description:**")
                st.write(ai_info['description'])
            
            if ai_info.get('sentiment'):
                sentiment_color = {
                    "positive": "🟢",
                    "negative": "🔴",
                    "neutral": "🟡"
                }
                st.write(f"**Sentiment:** {sentiment_color.get(ai_info['sentiment'], '⚪')} {ai_info['sentiment'].title()}")
            
            if ai_info.get('categories'):
                st.write(f"**Categories:** {', '.join(ai_info['categories'])}")

def display_download_options(report_content: str, survey_data: pd.DataFrame, 
                           analysis_results: Dict) -> None:
    """Display download options for analysis results"""
    st.subheader("📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📄 Download Report (Markdown)",
            data=report_content,
            file_name=f"survey_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    with col2:
        csv_data = survey_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Survey Data (CSV)",
            data=csv_data,
            file_name=f"survey_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        analysis_json = pd.io.json.dumps(analysis_results, indent=2)
        st.download_button(
            label="🤖 Download AI Analysis (JSON)",
            data=analysis_json,
            file_name=f"ai_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ) 