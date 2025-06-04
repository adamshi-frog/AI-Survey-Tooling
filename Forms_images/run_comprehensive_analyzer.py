#!/usr/bin/env python3
"""
Quick Start Script for Comprehensive Survey Analyzer

This script provides an easy way to launch the comprehensive survey analyzer
with automatic dependency checking and helpful instructions.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'Pillow', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    print("📦 Installing missing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install manually:")
        print(f"pip install {' '.join(packages)}")
        return False

def main():
    """Main function to start the comprehensive survey analyzer"""
    
    print("🤖 Comprehensive Survey Analyzer - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("comprehensive_survey_analyzer.py"):
        print("❌ Error: comprehensive_survey_analyzer.py not found!")
        print("Please run this script from the Forms_images directory.")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"⚠️ Missing packages: {', '.join(missing)}")
        
        response = input("Would you like to install them now? (y/n): ").lower().strip()
        if response == 'y' or response == 'yes':
            if not install_dependencies(missing):
                return
        else:
            print("Please install the missing packages and try again.")
            return
    else:
        print("✅ All dependencies are installed!")
    
    # Create output directory
    output_dir = "survey_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    # Display usage instructions
    print("\n🚀 Starting Comprehensive Survey Analyzer...")
    print("\n📝 How to use:")
    print("1. Upload your Google Forms CSV file")
    print("2. The app will automatically detect Google Drive image links")
    print("3. Configure your OpenAI API key for AI analysis (optional)")
    print("4. Click 'Analyze Google Drive Links' to process images")
    print("5. Download images and run AI analysis")
    print("6. View results in the Survey Analysis and Image Analysis tabs")
    print("7. Generate comprehensive reports in the Reports tab")
    
    print("\n💡 Tips:")
    print("- Ensure Google Drive images are shared publicly or with 'Anyone with the link'")
    print("- OpenAI API key is optional but provides detailed image insights")
    print("- All downloaded images and results are saved in 'survey_analysis_output/'")
    
    print("\n🌐 The app will open in your default web browser...")
    print("Press Ctrl+C to stop the application\n")
    
    try:
        # Start the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "comprehensive_survey_analyzer.py",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6"
        ])
    
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Thank you for using Comprehensive Survey Analyzer!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("Try running manually: streamlit run comprehensive_survey_analyzer.py")

if __name__ == "__main__":
    main() 