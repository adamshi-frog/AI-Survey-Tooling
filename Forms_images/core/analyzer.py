"""
Core survey analysis functionality for the Comprehensive Survey Analyzer
"""

import pandas as pd
from typing import Dict, List, Optional
from collections import Counter
from datetime import datetime

class SurveyAnalyzer:
    """Handles survey data analysis"""
    
    def __init__(self, survey_data: pd.DataFrame):
        """Initialize the analyzer with survey data"""
        self.survey_data = survey_data
        self.analysis = None
    
    def analyze(self) -> Dict:
        """Analyze the survey data structure and content"""
        analysis = {
            "total_responses": len(self.survey_data),
            "total_columns": len(self.survey_data.columns),
            "columns": list(self.survey_data.columns),
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
        for col in self.survey_data.columns:
            sample_values = self.survey_data[col].dropna().astype(str)
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
            elif self.survey_data[col].dtype in ['int64', 'float64'] or col.lower() in ['age', 'score', 'rating', 'count']:
                try:
                    pd.to_numeric(self.survey_data[col], errors='raise')
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
        total_cells = len(self.survey_data) * len(self.survey_data.columns)
        missing_cells = self.survey_data.isnull().sum().sum()
        
        analysis["data_quality"] = {
            "missing_data_percentage": (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
            "complete_responses": len(self.survey_data.dropna()),
            "completion_rate": (len(self.survey_data.dropna()) / len(self.survey_data)) * 100 if len(self.survey_data) > 0 else 0,
            "average_response_length": self.survey_data.astype(str).apply(lambda x: x.str.len()).mean().mean(),
            "columns_with_missing_data": self.survey_data.isnull().sum()[self.survey_data.isnull().sum() > 0].to_dict(),
            "duplicate_responses": self.survey_data.duplicated().sum()
        }
        
        self.analysis = analysis
        return analysis
    
    def get_column_analysis(self, column: str) -> Dict:
        """Get detailed analysis for a specific column"""
        if column not in self.survey_data.columns:
            raise ValueError(f"Column {column} not found in survey data")
        
        data = self.survey_data[column]
        analysis = {
            "column_name": column,
            "data_type": str(data.dtype),
            "total_values": len(data),
            "non_null_values": data.count(),
            "null_values": data.isnull().sum(),
            "unique_values": data.nunique(),
            "sample_values": data.dropna().head().tolist()
        }
        
        # Add type-specific analysis
        if column in self.analysis["numeric_columns"]:
            analysis.update({
                "min": data.min(),
                "max": data.max(),
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std()
            })
        elif column in self.analysis["choice_columns"]:
            value_counts = data.value_counts()
            analysis.update({
                "value_distribution": value_counts.to_dict(),
                "top_values": value_counts.head().to_dict()
            })
        
        return analysis
    
    def get_response_timeline(self) -> Optional[pd.DataFrame]:
        """Get response timeline if timestamp column exists"""
        if not self.analysis["response_timestamps"]:
            return None
        
        timestamp_col = self.analysis["response_timestamps"][0]
        try:
            df = self.survey_data.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True).dt.tz_localize(None)
            df['response_date'] = df[timestamp_col].dt.date
            return df.groupby('response_date').size().reset_index(name='responses')
        except:
            return None
    
    def get_text_analysis(self, column: str) -> Dict:
        """Get text analysis for a text column"""
        if column not in self.analysis["text_columns"]:
            raise ValueError(f"Column {column} is not a text column")
        
        responses = self.survey_data[column].dropna().astype(str)
        
        # Word frequency analysis
        all_words = ' '.join(responses).lower().split()
        word_counts = Counter([word for word in all_words if len(word) > 3])
        
        return {
            "total_responses": len(responses),
            "average_length": responses.str.len().mean(),
            "word_frequency": dict(word_counts.most_common(20)),
            "response_length_distribution": responses.str.len().describe().to_dict()
        } 