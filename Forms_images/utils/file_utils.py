"""
File handling utilities for the Comprehensive Survey Analyzer
"""

import os
import json
import zipfile
from datetime import datetime
from io import BytesIO, StringIO
from typing import Dict, Any
import pandas as pd

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """JSON dumps that converts non-serialisable objects to strings"""
    return json.dumps(obj, default=lambda o: str(o), **kwargs)

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

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_temp_file_path(prefix: str = "temp_") -> str:
    """Generate a temporary file path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}.csv"

def cleanup_temp_files(directory: str, prefix: str = "temp_") -> None:
    """Clean up temporary files in a directory"""
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            try:
                os.remove(os.path.join(directory, filename))
            except:
                pass

def save_dataframe_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save a DataFrame to CSV with error handling"""
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save DataFrame to CSV: {e}")

def load_csv_to_dataframe(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with error handling"""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file: {e}") 