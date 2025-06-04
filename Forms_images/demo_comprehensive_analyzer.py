#!/usr/bin/env python3
"""
Demo Script for Comprehensive Survey Analyzer

This script demonstrates the comprehensive survey analyzer using the included
sample Google Forms data (googleFormExample.csv).
"""

import os
import sys
import pandas as pd
from datetime import datetime
import json

def print_banner():
    """Print welcome banner"""
    print("🤖 Comprehensive Survey Analyzer - Demo")
    print("=" * 50)
    print("This demo showcases the analyzer using sample Google Forms data")
    print("with real Google Drive image links.\n")

def check_sample_data():
    """Check if sample data exists"""
    sample_file = "googleFormExample.csv"
    if not os.path.exists(sample_file):
        print(f"❌ Sample data file '{sample_file}' not found!")
        print("Please ensure you're running this from the Forms_images directory.")
        return False
    return True

def analyze_sample_data():
    """Analyze the sample CSV data"""
    print("📊 Analyzing Sample Data")
    print("-" * 30)
    
    # Load sample data
    df = pd.read_csv("googleFormExample.csv")
    
    print(f"📋 Total Responses: {len(df)}")
    print(f"📝 Total Questions: {len(df.columns)}")
    print(f"🔗 Columns: {list(df.columns)}")
    
    # Find Google Drive columns
    drive_columns = []
    for col in df.columns:
        sample_values = df[col].dropna().astype(str)
        if any('drive.google.com' in str(val) for val in sample_values.head()):
            drive_columns.append(col)
    
    print(f"🖼️ Google Drive Image Columns: {len(drive_columns)}")
    if drive_columns:
        print(f"   Column(s): {drive_columns}")
    
    # Show sample responses
    print("\n📝 Sample Responses:")
    for i, row in df.iterrows():
        print(f"\n🔸 Response {i + 1}:")
        for col in df.columns:
            value = str(row[col])
            if len(value) > 80:
                value = value[:77] + "..."
            print(f"   {col}: {value}")
    
    return df, drive_columns

def show_demo_instructions():
    """Show instructions for running the full demo"""
    print("\n🚀 Ready to Try the Full Application?")
    print("-" * 40)
    
    print("\n📖 Instructions:")
    print("1. Run the comprehensive analyzer:")
    print("   python run_comprehensive_analyzer.py")
    print("   OR")
    print("   streamlit run comprehensive_survey_analyzer.py")
    
    print("\n2. In the web interface:")
    print("   • Upload 'googleFormExample.csv'")
    print("   • Click 'Analyze Google Drive Links'")
    print("   • Click 'Download Images' to fetch the 3 sample images")
    print("   • (Optional) Add OpenAI API key for AI analysis")
    print("   • Explore the different tabs for analysis results")
    
    print("\n💡 Expected Results:")
    print("   • 3 survey responses with personal app screenshots")
    print("   • All 3 Google Drive links should be accessible")
    print("   • Images will download successfully (PNG format, ~600KB each)")
    print("   • Survey analysis will show favorite products and usage patterns")
    
    print("\n🔧 Troubleshooting:")
    print("   • If images don't download, the sharing permissions may have changed")
    print("   • The sample data uses real Google Drive links that were working at creation time")
    print("   • For AI analysis, you'll need an OpenAI API key (costs ~$0.03 total for all 3 images)")

def show_sample_insights():
    """Show some basic insights from the sample data"""
    print("\n🔍 Sample Data Insights")
    print("-" * 30)
    
    df = pd.read_csv("googleFormExample.csv")
    
    # Analyze favorite products
    if "What are your favorite products?" in df.columns:
        products_col = "What are your favorite products?"
        all_products = " ".join(df[products_col].dropna()).lower()
        
        # Simple word counting
        words = all_products.split()
        product_mentions = {}
        key_products = ['chatgpt', 'instagram', 'doordash', 'spotify', 'youtube', 'discord', 'notion', 'pinterest']
        
        for product in key_products:
            count = all_products.count(product)
            if count > 0:
                product_mentions[product] = count
        
        print("🏆 Most Mentioned Products:")
        for product, count in sorted(product_mentions.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {product.title()}: {count} mention(s)")
    
    # Analyze school years
    if "Which year are you in School " in df.columns:
        school_col = "Which year are you in School "
        school_years = df[school_col].value_counts()
        print(f"\n🎓 School Year Distribution:")
        for year, count in school_years.items():
            print(f"   • {year}: {count} student(s)")
    
    # Analyze used products
    if "Which products have you used?" in df.columns:
        used_col = "Which products have you used?"
        used_products = []
        for products in df[used_col].dropna():
            used_products.extend(products.split(';'))
        
        from collections import Counter
        usage_counts = Counter(used_products)
        print(f"\n📱 Most Used Products:")
        for product, count in usage_counts.most_common():
            print(f"   • {product.strip()}: {count} user(s)")

def main():
    """Main demo function"""
    print_banner()
    
    # Check if we have the required files
    if not check_sample_data():
        return
    
    # Check if the main application exists
    if not os.path.exists("comprehensive_survey_analyzer.py"):
        print("❌ Main application 'comprehensive_survey_analyzer.py' not found!")
        print("Please ensure you're in the correct directory.")
        return
    
    # Analyze sample data
    try:
        df, drive_columns = analyze_sample_data()
        show_sample_insights()
        
        print(f"\n✅ Sample data analysis complete!")
        print(f"   Ready to process {len(df)} responses with {len(drive_columns)} image column(s)")
        
    except Exception as e:
        print(f"❌ Error analyzing sample data: {e}")
        return
    
    # Show full demo instructions
    show_demo_instructions()
    
    # Ask if user wants to start the app
    print("\n" + "=" * 50)
    response = input("🚀 Start the Comprehensive Survey Analyzer now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🌐 Starting the application...")
        print("The web interface will open in your browser!")
        print("Upload 'googleFormExample.csv' to begin the demo.\n")
        
        try:
            import subprocess
            subprocess.run([sys.executable, "run_comprehensive_analyzer.py"])
        except KeyboardInterrupt:
            print("\n👋 Demo ended. Thank you for trying the Comprehensive Survey Analyzer!")
        except Exception as e:
            print(f"\n❌ Error starting application: {e}")
            print("Try running manually: python run_comprehensive_analyzer.py")
    else:
        print("\n👋 Demo completed! Run 'python run_comprehensive_analyzer.py' when ready.")

if __name__ == "__main__":
    main() 