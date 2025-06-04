#!/usr/bin/env python3
"""
Quick Start Script for AI Survey Image Analyzer

This script provides an easy-to-use interface for analyzing your own Google Forms survey data.
Simply run this script and follow the prompts to analyze your CSV file with Google Drive images.
"""

import os
import sys
from pathlib import Path
from enhanced_drive_downloader import EnhancedDriveDownloader, analyze_survey_drive_links

def main():
    """Main quick start interface"""
    print("🚀 AI Survey Image Analyzer - Quick Start")
    print("=" * 50)
    print("Welcome! This tool will help you analyze Google Forms survey data with Google Drive images.")
    print()
    
    # Step 1: Get CSV file
    while True:
        csv_file = input("📁 Enter the path to your CSV file (or 'demo' for example): ").strip()
        
        if csv_file.lower() == 'demo':
            csv_file = "googleFormExample.csv"
            if os.path.exists(csv_file):
                print(f"✅ Using demo file: {csv_file}")
                break
            else:
                print("❌ Demo file not found. Please specify a valid CSV file.")
                continue
        
        if os.path.exists(csv_file):
            print(f"✅ Found CSV file: {csv_file}")
            break
        else:
            print("❌ File not found. Please check the path and try again.")
    
    # Step 2: Choose output directory
    output_dir = input("\n📂 Output directory (press Enter for 'analysis_results'): ").strip()
    if not output_dir:
        output_dir = "analysis_results"
    
    print(f"📁 Using output directory: {output_dir}")
    
    # Step 3: Analyze Google Drive links
    print(f"\n🔍 Analyzing Google Drive links...")
    
    try:
        analysis = analyze_survey_drive_links(csv_file)
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        print(f"📊 Analysis Results:")
        print(f"  • Total links found: {analysis['total_links']}")
        print(f"  • Accessible links: {analysis['accessible_links']}")
        print(f"  • Success rate: {analysis['accessibility_rate']}")
        
        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  • {rec}")
        
    except Exception as e:
        print(f"❌ Error analyzing links: {e}")
        return
    
    # Step 4: Download confirmation
    if analysis['accessible_links'] == 0:
        print("\n⚠️ No accessible images found. Please check your Google Drive sharing settings.")
        return
    
    proceed = input(f"\n🔄 Download {analysis['accessible_links']} images? (y/N): ").strip().lower()
    if proceed != 'y':
        print("Operation cancelled.")
        return
    
    # Step 5: Download images
    print(f"\n📥 Downloading images...")
    
    try:
        downloader = EnhancedDriveDownloader(output_dir)
        downloaded_count = 0
        
        for i, analysis_item in enumerate(analysis['individual_analyses']):
            file_id = analysis_item.get('file_id')
            if file_id and analysis_item.get('accessible', False):
                filename = f"response_{i + 1}_{file_id[:8]}.jpg"
                result = downloader.download_file(file_id, filename)
                if result:
                    downloaded_count += 1
        
        print(f"\n🎉 Successfully downloaded {downloaded_count} images!")
        
    except Exception as e:
        print(f"❌ Error downloading images: {e}")
        return
    
    # Step 6: Optional AI analysis
    ai_analysis = input(f"\n🤖 Run AI analysis? (requires OpenAI API key) (y/N): ").strip().lower()
    
    if ai_analysis == 'y':
        api_key = input("🔑 Enter your OpenAI API key: ").strip()
        if api_key:
            print("🔄 Running AI analysis...")
            try:
                from ai_vision_analyzer import AIVisionAnalyzer
                analyzer = AIVisionAnalyzer(csv_file, output_dir, api_key)
                
                # Load data and run analysis
                data = analyzer.load_survey_data()
                if data is not None:
                    downloaded_images = analyzer.download_all_images()
                    if downloaded_images:
                        analysis_data = analyzer.generate_comprehensive_analysis()
                        insights = analyzer.generate_ai_insights_summary(analysis_data)
                        
                        print(f"✅ AI analysis complete!")
                        print(f"📄 Results saved in: {output_dir}/analysis/")
                        
                        # Show key insights
                        if insights.get('key_insights'):
                            print(f"\n🔍 Key AI Insights:")
                            for insight in insights['key_insights']:
                                print(f"  • {insight}")
            except Exception as e:
                print(f"❌ Error in AI analysis: {e}")
        else:
            print("❌ No API key provided, skipping AI analysis")
    
    # Step 7: Summary
    print(f"\n✅ Analysis Complete!")
    print(f"📁 Results saved in: {output_dir}/")
    print(f"🖼️ Images: {output_dir}/ directory")
    
    # Suggest next steps
    print(f"\n🚀 Next Steps:")
    print(f"  • View downloaded images in {output_dir}/")
    print(f"  • Run: streamlit run streamlit_survey_analyzer.py")
    print(f"  • For advanced analysis: python ai_vision_analyzer.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your input and try again.") 