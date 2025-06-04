#!/usr/bin/env python3
"""
Google Drive Image Analyzer for Survey Data

This script processes Google Forms survey data that includes Google Drive image links,
downloads the images, and provides AI-powered analysis.

Features:
- Parse CSV survey data
- Extract Google Drive file IDs from URLs
- Download images from Google Drive (with multiple authentication methods)
- AI-powered image analysis using OpenAI Vision API
- Generate comprehensive survey reports
"""

import pandas as pd
import requests
import os
import re
from pathlib import Path
import json
from datetime import datetime
import base64
from typing import List, Dict, Optional
import urllib.parse
from io import BytesIO
from PIL import Image
import streamlit as st

class GoogleDriveImageAnalyzer:
    def __init__(self, csv_file_path: str, output_dir: str = "survey_analysis"):
        """
        Initialize the analyzer
        
        Args:
            csv_file_path: Path to the CSV file with survey data
            output_dir: Directory to save downloaded images and analysis
        """
        self.csv_file_path = csv_file_path
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.analysis_dir = self.output_dir / "analysis"
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        
        self.survey_data = None
        self.downloaded_images = {}
        
    def load_survey_data(self) -> pd.DataFrame:
        """Load and parse the CSV survey data"""
        try:
            self.survey_data = pd.read_csv(self.csv_file_path)
            print(f"✅ Loaded survey data: {len(self.survey_data)} responses")
            print(f"📊 Columns: {list(self.survey_data.columns)}")
            return self.survey_data
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None
    
    def extract_drive_file_id(self, drive_url: str) -> Optional[str]:
        """
        Extract Google Drive file ID from various URL formats
        
        Args:
            drive_url: Google Drive URL
            
        Returns:
            File ID if found, None otherwise
        """
        if not drive_url or pd.isna(drive_url):
            return None
            
        # Pattern for Google Drive file IDs
        patterns = [
            r'id=([a-zA-Z0-9-_]+)',  # id= parameter
            r'/file/d/([a-zA-Z0-9-_]+)',  # /file/d/ format
            r'/open\?id=([a-zA-Z0-9-_]+)',  # /open?id= format
            r'id=([a-zA-Z0-9-_]+)&',  # id= with additional parameters
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(drive_url))
            if match:
                return match.group(1)
        
        print(f"⚠️ Could not extract file ID from: {drive_url}")
        return None
    
    def get_direct_download_url(self, file_id: str) -> str:
        """Convert Google Drive file ID to direct download URL"""
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    def download_image_from_drive(self, file_id: str, filename: str) -> Optional[str]:
        """
        Download image from Google Drive using file ID
        
        Args:
            file_id: Google Drive file ID
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        if not file_id:
            return None
            
        try:
            # Method 1: Direct download URL
            download_url = self.get_direct_download_url(file_id)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(download_url, headers=headers, allow_redirects=True)
            
            # Check if we got a download confirmation page
            if 'download_warning' in response.text:
                # Extract the actual download link
                confirm_pattern = r'href="(/uc\?export=download[^"]+)"'
                match = re.search(confirm_pattern, response.text)
                if match:
                    confirm_url = f"https://drive.google.com{match.group(1)}"
                    response = requests.get(confirm_url, headers=headers)
            
            if response.status_code == 200 and len(response.content) > 1000:  # Basic check for actual image data
                file_path = self.images_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify it's a valid image
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    print(f"✅ Downloaded: {filename}")
                    return str(file_path)
                except Exception:
                    print(f"❌ Downloaded file is not a valid image: {filename}")
                    os.remove(file_path)
                    return None
            else:
                print(f"❌ Failed to download {filename}: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return None
    
    def download_all_images(self) -> Dict[int, str]:
        """
        Download all images from the survey data
        
        Returns:
            Dictionary mapping response index to local image path
        """
        if self.survey_data is None:
            print("❌ No survey data loaded")
            return {}
        
        # Find the column with Google Drive links
        drive_url_column = None
        for col in self.survey_data.columns:
            if any(sample_url in str(self.survey_data[col].iloc[0]) for sample_url in ['drive.google.com', 'docs.google.com'] if pd.notna(self.survey_data[col].iloc[0])):
                drive_url_column = col
                break
        
        if not drive_url_column:
            print("❌ No Google Drive URL column found")
            return {}
        
        print(f"📁 Found Drive URLs in column: '{drive_url_column}'")
        downloaded_images = {}
        
        for idx, row in self.survey_data.iterrows():
            drive_url = row[drive_url_column]
            file_id = self.extract_drive_file_id(drive_url)
            
            if file_id:
                filename = f"response_{idx}_{file_id}.jpg"
                local_path = self.download_image_from_drive(file_id, filename)
                if local_path:
                    downloaded_images[idx] = local_path
                    
        print(f"🎉 Downloaded {len(downloaded_images)} images successfully")
        self.downloaded_images = downloaded_images
        return downloaded_images
    
    def analyze_image_with_ai(self, image_path: str, survey_context: Dict) -> Dict:
        """
        Analyze image using AI (placeholder for OpenAI Vision API)
        
        Args:
            image_path: Path to the image file
            survey_context: Context from the survey response
            
        Returns:
            Analysis results
        """
        try:
            # Convert image to base64 for API calls
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            # Get image info
            with Image.open(image_path) as img:
                width, height = img.size
                format = img.format
            
            # Basic analysis (can be enhanced with actual AI API calls)
            analysis = {
                "file_path": image_path,
                "file_size": os.path.getsize(image_path),
                "dimensions": f"{width}x{height}",
                "format": format,
                "survey_context": survey_context,
                "timestamp": datetime.now().isoformat(),
                "ai_analysis": {
                    "description": "Image analysis placeholder - integrate with OpenAI Vision API",
                    "objects_detected": [],
                    "text_detected": "",
                    "sentiment": "neutral",
                    "categories": [],
                    "confidence": 0.0
                }
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing image {image_path}: {e}")
            return {"error": str(e)}
    
    def generate_comprehensive_analysis(self) -> Dict:
        """Generate comprehensive analysis of all survey responses and images"""
        if not self.downloaded_images:
            print("❌ No images downloaded for analysis")
            return {}
        
        all_analyses = {}
        
        for idx, image_path in self.downloaded_images.items():
            # Get survey context for this response
            survey_context = self.survey_data.iloc[idx].to_dict()
            
            # Analyze the image
            analysis = self.analyze_image_with_ai(image_path, survey_context)
            all_analyses[idx] = analysis
        
        # Generate summary statistics
        summary = {
            "total_responses": len(self.survey_data),
            "responses_with_images": len(self.downloaded_images),
            "download_success_rate": len(self.downloaded_images) / len(self.survey_data) * 100,
            "analysis_timestamp": datetime.now().isoformat(),
            "individual_analyses": all_analyses
        }
        
        # Save analysis to file
        analysis_file = self.analysis_dir / f"survey_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Analysis saved to: {analysis_file}")
        return summary
    
    def create_analysis_report(self) -> str:
        """Create a formatted analysis report"""
        analysis = self.generate_comprehensive_analysis()
        
        report = f"""
# Survey Image Analysis Report
Generated: {analysis.get('analysis_timestamp', 'Unknown')}

## Summary Statistics
- **Total Survey Responses**: {analysis.get('total_responses', 0)}
- **Responses with Images**: {analysis.get('responses_with_images', 0)}
- **Download Success Rate**: {analysis.get('download_success_rate', 0):.1f}%

## Individual Response Analysis
"""
        
        for idx, analysis_data in analysis.get('individual_analyses', {}).items():
            survey_context = analysis_data.get('survey_context', {})
            report += f"""
### Response {idx + 1}
- **Timestamp**: {survey_context.get('Timestamp', 'N/A')}
- **Image Dimensions**: {analysis_data.get('dimensions', 'N/A')}
- **File Format**: {analysis_data.get('format', 'N/A')}
- **Survey Data**: {survey_context}

"""
        
        # Save report
        report_file = self.analysis_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_file}")
        return str(report_file)

def main():
    """Main function to run the analyzer"""
    print("🤖 Google Drive Image Analyzer for Survey Data")
    print("=" * 50)
    
    # Initialize analyzer
    csv_file = "googleFormExample.csv"  # Update this path
    analyzer = GoogleDriveImageAnalyzer(csv_file)
    
    # Load survey data
    data = analyzer.load_survey_data()
    if data is None:
        return
    
    print(f"\n📋 Survey Data Preview:")
    print(data.head())
    
    # Download images
    print(f"\n🔄 Downloading images from Google Drive...")
    downloaded_images = analyzer.download_all_images()
    
    if downloaded_images:
        print(f"\n🔍 Analyzing downloaded images...")
        report_path = analyzer.create_analysis_report()
        print(f"\n🎉 Analysis complete! Report saved to: {report_path}")
    else:
        print("\n❌ No images were downloaded successfully")

if __name__ == "__main__":
    main() 