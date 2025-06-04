#!/usr/bin/env python3
"""
Demo Script - Full AI Survey Analysis Pipeline

Demonstrates the complete workflow:
1. Load survey data
2. Download images from Google Drive
3. Perform AI analysis (with fallback)
4. Generate comprehensive reports

This script uses the enhanced downloader and shows the complete functionality.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from enhanced_drive_downloader import EnhancedDriveDownloader, analyze_survey_drive_links
from google_drive_image_analyzer import GoogleDriveImageAnalyzer
import pandas as pd

def run_complete_analysis_demo():
    """Run the complete analysis pipeline demo"""
    print("🚀 AI Survey Analysis Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Load and analyze the CSV
    csv_file = "googleFormExample.csv"
    print(f"\n📊 Step 1: Loading survey data from {csv_file}")
    
    try:
        survey_data = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(survey_data)} survey responses")
        print(f"📋 Columns: {list(survey_data.columns)}")
        
        # Show a preview
        print(f"\n🔍 Data Preview:")
        for idx, row in survey_data.iterrows():
            print(f"  Response {idx + 1}:")
            print(f"    Timestamp: {row['Timestamp']}")
            print(f"    Favorite products: {row['What are your favorite products?'][:50]}...")
            print(f"    Used products: {row['Which products have you used?']}")
            print(f"    School year: {row['Which year are you in School ']}")
            print()
    
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Step 2: Analyze Google Drive links
    print(f"\n🔗 Step 2: Analyzing Google Drive links")
    
    drive_analysis = analyze_survey_drive_links(csv_file)
    if "error" in drive_analysis:
        print(f"❌ Error: {drive_analysis['error']}")
        return
    
    print(f"📈 Link Analysis Results:")
    print(f"  Total links: {drive_analysis['total_links']}")
    print(f"  Accessible: {drive_analysis['accessible_links']}")
    print(f"  Success rate: {drive_analysis['accessibility_rate']}")
    
    # Step 3: Download images with enhanced method
    print(f"\n📥 Step 3: Downloading images from Google Drive")
    
    downloader = EnhancedDriveDownloader("demo_images")
    downloaded_images = {}
    
    for i, analysis_item in enumerate(drive_analysis['individual_analyses']):
        file_id = analysis_item.get('file_id')
        if file_id:
            filename = f"survey_response_{i + 1}_{file_id[:8]}.jpg"
            result = downloader.download_file(file_id, filename)
            if result:
                downloaded_images[i] = result
    
    print(f"🎉 Successfully downloaded {len(downloaded_images)} images")
    
    # Step 4: Basic image analysis
    print(f"\n🔍 Step 4: Performing image analysis")
    
    analyses = {}
    
    for response_idx, image_path in downloaded_images.items():
        try:
            # Get basic image info
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                format = img.format
                
            file_size = os.path.getsize(image_path)
            
            # Get survey context
            survey_context = survey_data.iloc[response_idx].to_dict()
            
            analysis = {
                "response_index": response_idx + 1,
                "image_path": image_path,
                "image_info": {
                    "dimensions": f"{width}x{height}",
                    "format": format,
                    "file_size_kb": round(file_size / 1024, 1)
                },
                "survey_context": {
                    "timestamp": survey_context['Timestamp'],
                    "favorite_products": survey_context['What are your favorite products?'],
                    "used_products": survey_context['Which products have you used?'],
                    "school_year": survey_context['Which year are you in School ']
                },
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            analyses[response_idx] = analysis
            
            print(f"  ✅ Analyzed Response {response_idx + 1}: {width}x{height} {format} ({analysis['image_info']['file_size_kb']} KB)")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {image_path}: {e}")
    
    # Step 5: Generate insights
    print(f"\n📊 Step 5: Generating insights and summary")
    
    # Basic insights
    total_responses = len(survey_data)
    images_downloaded = len(downloaded_images)
    success_rate = (images_downloaded / total_responses) * 100
    
    # Analyze survey content
    all_favorite_products = []
    all_used_products = []
    school_years = []
    
    for _, row in survey_data.iterrows():
        # Parse favorite products
        fav_products = str(row['What are your favorite products?']).split(',')
        all_favorite_products.extend([p.strip() for p in fav_products])
        
        # Parse used products
        used_products = str(row['Which products have you used?']).split(';')
        all_used_products.extend([p.strip() for p in used_products])
        
        # School years
        school_years.append(row['Which year are you in School '])
    
    # Count frequencies
    from collections import Counter
    fav_counter = Counter(all_favorite_products)
    used_counter = Counter(all_used_products)
    year_counter = Counter(school_years)
    
    insights = {
        "summary_stats": {
            "total_survey_responses": total_responses,
            "images_successfully_downloaded": images_downloaded,
            "download_success_rate": f"{success_rate:.1f}%"
        },
        "survey_insights": {
            "top_favorite_products": fav_counter.most_common(5),
            "top_used_products": used_counter.most_common(5),
            "school_year_distribution": dict(year_counter)
        },
        "image_analysis": analyses,
        "generated_at": datetime.now().isoformat()
    }
    
    # Step 6: Save results
    print(f"\n💾 Step 6: Saving results")
    
    # Save detailed analysis
    analysis_file = f"demo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, 'w') as f:
        json.dump(insights, f, indent=2)
    
    print(f"📄 Analysis saved to: {analysis_file}")
    
    # Create summary report
    report_content = f"""
# AI Survey Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Survey Responses**: {total_responses}
- **Images Downloaded**: {images_downloaded}
- **Success Rate**: {success_rate:.1f}%

## Survey Insights

### Top Favorite Products
"""
    
    for product, count in fav_counter.most_common(3):
        if product and product != 'nan':
            report_content += f"- **{product}**: {count} mentions\n"
    
    report_content += f"""
### Most Used Products
"""
    
    for product, count in used_counter.most_common(3):
        if product and product != 'nan':
            report_content += f"- **{product}**: {count} users\n"
    
    report_content += f"""
### School Year Distribution
"""
    
    for year, count in year_counter.items():
        if year and year != 'nan':
            report_content += f"- **{year}**: {count} responses\n"
    
    report_content += f"""
## Individual Response Analysis
"""
    
    for response_idx, analysis in analyses.items():
        info = analysis['image_info']
        context = analysis['survey_context']
        
        report_content += f"""
### Response {response_idx + 1}
- **Timestamp**: {context['timestamp']}
- **Image**: {info['dimensions']} {info['format']} ({info['file_size_kb']} KB)
- **Favorite Products**: {context['favorite_products']}
- **Used Products**: {context['used_products']}
- **School Year**: {context['school_year']}

"""
    
    # Save report
    report_file = f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"📋 Report saved to: {report_file}")
    
    # Step 7: Display key findings
    print(f"\n🎯 Step 7: Key Findings")
    print(f"=" * 30)
    
    print(f"📊 Survey Analysis:")
    print(f"  • {total_responses} total responses analyzed")
    print(f"  • {images_downloaded} images successfully downloaded ({success_rate:.1f}% success rate)")
    
    if fav_counter.most_common(1):
        top_product = fav_counter.most_common(1)[0]
        print(f"  • Most mentioned favorite product: {top_product[0]} ({top_product[1]} mentions)")
    
    if used_counter.most_common(1):
        top_used = used_counter.most_common(1)[0]
        print(f"  • Most used product: {top_used[0]} ({top_used[1]} users)")
    
    print(f"\n🖼️ Image Analysis:")
    for response_idx, analysis in analyses.items():
        info = analysis['image_info']
        print(f"  • Response {response_idx + 1}: {info['dimensions']} {info['format']} image ({info['file_size_kb']} KB)")
    
    print(f"\n✅ Demo Complete!")
    print(f"📁 Files created:")
    print(f"  • Images: demo_images/ directory")
    print(f"  • Analysis: {analysis_file}")
    print(f"  • Report: {report_file}")
    
    return insights

if __name__ == "__main__":
    run_complete_analysis_demo() 