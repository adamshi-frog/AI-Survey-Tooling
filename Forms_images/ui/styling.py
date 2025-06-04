"""
UI styling and CSS for the Comprehensive Survey Analyzer
"""

from config import DEFAULT_THEME

def get_custom_css():
    """Returns custom CSS for the application"""
    return f"""
    <style>
        .main-header {{
            font-size: 3rem;
            color: {DEFAULT_THEME['primary_color']};
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .feature-card {{
            background: linear-gradient(135deg, {DEFAULT_THEME['secondary_color']} 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 0.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        .success-banner {{
            background: linear-gradient(90deg, {DEFAULT_THEME['success_color']}, #45a049);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
            font-weight: bold;
        }}
        .warning-banner {{
            background: linear-gradient(90deg, {DEFAULT_THEME['warning_color']}, #f57c00);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
            font-weight: bold;
        }}
        .info-banner {{
            background: linear-gradient(90deg, {DEFAULT_THEME['primary_color']}, #1976D2);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
            font-weight: bold;
        }}
        .stButton > button {{
            background: linear-gradient(90deg, {DEFAULT_THEME['secondary_color']}, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {{
            background-color: #8fa051 !important;
            color: white !important;
        }}
        .styled-table th {{
            background-color: #f6f6f3 !important;
            color: #2c2c2c !important;
            font-weight: bold !important;
            border: 1px solid #ddd !important;
        }}
        .styled-table td {{
            background-color: #f6f6f3 !important;
            color: #2c2c2c !important;
            border: 1px solid #ddd !important;
        }}
        .styled-table table {{
            border-collapse: collapse !important;
            border-radius: 5px !important;
            overflow: hidden !important;
        }}
    </style>
    """

def get_metric_card(title: str, value: str) -> str:
    """Returns HTML for a metric card"""
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2>{value}</h2>
    </div>
    """

def get_banner(message: str, banner_type: str = "info") -> str:
    """Returns HTML for a banner"""
    banner_class = {
        "success": "success-banner",
        "warning": "warning-banner",
        "error": "warning-banner",
        "info": "info-banner"
    }.get(banner_type, "info-banner")
    
    icon = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }.get(banner_type, "ℹ️")
    
    return f'<div class="{banner_class}">{icon} {message}</div>' 