#!/usr/bin/env python3
"""
Enhanced Google Drive Downloader

Handles various Google Drive scenarios including:
- Public vs private files
- Different sharing permissions
- Multiple download methods
- Fallback mechanisms
"""

import requests
import re
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import time
from typing import Optional, Dict, List
import json

class EnhancedDriveDownloader:
    """Enhanced Google Drive file downloader with multiple strategies"""
    
    def __init__(self, output_dir: str = "downloaded_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
    def extract_file_id(self, url: str) -> Optional[str]:
        """Extract Google Drive file ID from various URL formats"""
        if not url:
            return None
            
        # Clean up the URL
        url = url.strip()
        
        # Multiple patterns for different Google Drive URL formats
        patterns = [
            r'id=([a-zA-Z0-9-_]+)',
            r'/file/d/([a-zA-Z0-9-_]+)',
            r'/open\?id=([a-zA-Z0-9-_]+)',
            r'folders/([a-zA-Z0-9-_]+)',
            r'drive\.google\.com/.*[?&]id=([a-zA-Z0-9-_]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def try_direct_download(self, file_id: str, filename: str) -> Optional[str]:
        """Method 1: Try direct download URL"""
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = self.session.get(download_url, timeout=30)
            
            # Check for download warning page
            if 'download_warning' in response.text or 'virus-scan-warning' in response.text:
                # Try to find the confirm link
                confirm_pattern = r'href="(/uc\?export=download[^"]*)"'
                match = re.search(confirm_pattern, response.text)
                if match:
                    confirm_url = f"https://drive.google.com{match.group(1)}"
                    response = self.session.get(confirm_url, timeout=30)
            
            # Check if we got actual file content
            if response.status_code == 200 and len(response.content) > 1000:
                content_type = response.headers.get('content-type', '').lower()
                if 'image' in content_type or self._is_image_content(response.content):
                    file_path = self.output_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    return str(file_path)
            
        except Exception as e:
            print(f"Direct download failed: {e}")
        
        return None
    
    def try_thumbnail_download(self, file_id: str, filename: str) -> Optional[str]:
        """Method 2: Try thumbnail URL (works for some restricted files)"""
        try:
            thumbnail_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"
            response = self.session.get(thumbnail_url, timeout=30)
            
            if response.status_code == 200 and len(response.content) > 1000:
                content_type = response.headers.get('content-type', '').lower()
                if 'image' in content_type:
                    file_path = self.output_dir / f"thumb_{filename}"
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    return str(file_path)
                    
        except Exception as e:
            print(f"Thumbnail download failed: {e}")
        
        return None
    
    def try_preview_download(self, file_id: str, filename: str) -> Optional[str]:
        """Method 3: Try preview URL"""
        try:
            preview_url = f"https://drive.google.com/file/d/{file_id}/preview"
            response = self.session.get(preview_url, timeout=30)
            
            if response.status_code == 200:
                # Look for image URLs in the preview page
                img_patterns = [
                    r'https://lh[0-9]+\.googleusercontent\.com/[^"\']+',
                    r'https://drive\.google\.com/[^"\']*thumbnail[^"\']*',
                ]
                
                for pattern in img_patterns:
                    matches = re.findall(pattern, response.text)
                    for img_url in matches:
                        if self._download_from_url(img_url, f"preview_{filename}"):
                            return str(self.output_dir / f"preview_{filename}")
                            
        except Exception as e:
            print(f"Preview download failed: {e}")
        
        return None
    
    def _download_from_url(self, url: str, filename: str) -> bool:
        """Helper method to download from a specific URL"""
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 1000:
                content_type = response.headers.get('content-type', '').lower()
                if 'image' in content_type or self._is_image_content(response.content):
                    file_path = self.output_dir / filename
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    return True
        except:
            pass
        return False
    
    def _is_image_content(self, content: bytes) -> bool:
        """Check if content appears to be an image based on headers"""
        if len(content) < 10:
            return False
            
        # Check for common image file signatures
        image_signatures = [
            b'\xFF\xD8\xFF',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF87a
            b'GIF89a',  # GIF89a
            b'RIFF',  # WebP (starts with RIFF)
            b'\x00\x00\x01\x00',  # ICO
        ]
        
        for sig in image_signatures:
            if content.startswith(sig):
                return True
        
        return False
    
    def download_file(self, file_id: str, filename: str) -> Optional[str]:
        """
        Download file using multiple strategies
        
        Args:
            file_id: Google Drive file ID
            filename: Local filename to save as
            
        Returns:
            Path to downloaded file if successful, None otherwise
        """
        if not file_id:
            return None
        
        print(f"🔄 Attempting to download file ID: {file_id}")
        
        # Strategy 1: Direct download
        result = self.try_direct_download(file_id, filename)
        if result:
            print(f"✅ Direct download successful: {filename}")
            return result
        
        # Strategy 2: Thumbnail download
        result = self.try_thumbnail_download(file_id, filename)
        if result:
            print(f"✅ Thumbnail download successful: {filename}")
            return result
        
        # Strategy 3: Preview download
        result = self.try_preview_download(file_id, filename)
        if result:
            print(f"✅ Preview download successful: {filename}")
            return result
        
        print(f"❌ All download methods failed for: {file_id}")
        return None
    
    def analyze_drive_url(self, url: str) -> Dict:
        """Analyze a Google Drive URL and provide detailed information"""
        file_id = self.extract_file_id(url)
        
        if not file_id:
            return {"error": "Could not extract file ID", "url": url}
        
        # Try to get file metadata
        info_url = f"https://drive.google.com/file/d/{file_id}/view"
        
        try:
            response = self.session.get(info_url, timeout=10)
            
            analysis = {
                "original_url": url,
                "file_id": file_id,
                "status_code": response.status_code,
                "accessible": response.status_code == 200,
                "response_size": len(response.content),
                "content_type": response.headers.get('content-type', 'unknown'),
            }
            
            if response.status_code == 200:
                # Look for sharing indicators
                if 'ViewerMode' in response.text:
                    analysis["sharing_status"] = "public_viewer"
                elif 'restricted' in response.text.lower():
                    analysis["sharing_status"] = "restricted"
                else:
                    analysis["sharing_status"] = "unknown"
                
                # Look for file type indicators
                if 'image' in response.text.lower():
                    analysis["likely_image"] = True
                else:
                    analysis["likely_image"] = False
            
            return analysis
            
        except Exception as e:
            return {
                "original_url": url,
                "file_id": file_id,
                "error": str(e),
                "accessible": False
            }

def analyze_survey_drive_links(csv_file: str) -> Dict:
    """Analyze all Google Drive links in a survey CSV file"""
    import pandas as pd
    
    # Load survey data
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        return {"error": f"Could not load CSV: {e}"}
    
    # Find column with Google Drive links
    drive_column = None
    for col in data.columns:
        sample_value = str(data[col].iloc[0]) if len(data) > 0 else ""
        if 'drive.google.com' in sample_value:
            drive_column = col
            break
    
    if not drive_column:
        return {"error": "No Google Drive links found in CSV"}
    
    # Analyze each link
    downloader = EnhancedDriveDownloader()
    analyses = []
    
    for idx, row in data.iterrows():
        url = row[drive_column]
        analysis = downloader.analyze_drive_url(url)
        analysis["response_index"] = idx
        analyses.append(analysis)
    
    # Summary statistics
    total_links = len(analyses)
    accessible_links = sum(1 for a in analyses if a.get('accessible', False))
    likely_images = sum(1 for a in analyses if a.get('likely_image', False))
    
    return {
        "total_links": total_links,
        "accessible_links": accessible_links,
        "likely_images": likely_images,
        "accessibility_rate": f"{(accessible_links/total_links)*100:.1f}%" if total_links > 0 else "0%",
        "individual_analyses": analyses,
        "recommendations": generate_recommendations(analyses)
    }

def generate_recommendations(analyses: List[Dict]) -> List[str]:
    """Generate recommendations based on link analysis"""
    recommendations = []
    
    inaccessible_count = sum(1 for a in analyses if not a.get('accessible', False))
    total_count = len(analyses)
    
    if inaccessible_count > 0:
        recommendations.append(f"{inaccessible_count}/{total_count} files are not publicly accessible")
        recommendations.append("Consider asking survey respondents to make their Google Drive files publicly viewable")
    
    restricted_count = sum(1 for a in analyses if a.get('sharing_status') == 'restricted')
    if restricted_count > 0:
        recommendations.append(f"{restricted_count} files appear to have restricted sharing settings")
    
    non_image_count = sum(1 for a in analyses if a.get('likely_image') == False)
    if non_image_count > 0:
        recommendations.append(f"{non_image_count} files may not be images")
    
    if not recommendations:
        recommendations.append("All links appear to be accessible and likely contain images")
    
    return recommendations

def main():
    """Main function to analyze and download from sample CSV"""
    print("🔍 Enhanced Google Drive Downloader")
    print("=" * 40)
    
    # Analyze the survey links first
    csv_file = "googleFormExample.csv"
    print(f"\n📊 Analyzing Google Drive links in {csv_file}...")
    
    analysis = analyze_survey_drive_links(csv_file)
    
    if "error" in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print(f"\n📈 Analysis Results:")
    print(f"  Total links: {analysis['total_links']}")
    print(f"  Accessible: {analysis['accessible_links']}")
    print(f"  Likely images: {analysis['likely_images']}")
    print(f"  Success rate: {analysis['accessibility_rate']}")
    
    print(f"\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    # Try downloading anyway
    print(f"\n🔄 Attempting downloads with enhanced methods...")
    
    downloader = EnhancedDriveDownloader("enhanced_downloads")
    successful_downloads = 0
    
    for i, analysis_item in enumerate(analysis['individual_analyses']):
        file_id = analysis_item.get('file_id')
        if file_id:
            filename = f"response_{i}_{file_id[:8]}.jpg"
            result = downloader.download_file(file_id, filename)
            if result:
                successful_downloads += 1
    
    print(f"\n🎉 Enhanced download complete!")
    print(f"  Successfully downloaded: {successful_downloads}/{analysis['total_links']} files")
    
    # Save analysis report
    with open("drive_analysis_report.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"📄 Analysis report saved to: drive_analysis_report.json")

if __name__ == "__main__":
    main() 