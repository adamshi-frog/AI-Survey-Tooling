#!/usr/bin/env python3
"""
Installation Test Script for Forms Image Analyzer

Run this script to verify that the installation is working correctly.
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing dependencies...")
    
    required_modules = [
        'pandas',
        'requests', 
        'PIL',
        'streamlit',
        'plotly',
        'pathlib',
        'datetime',
        'json',
        'base64',
        'typing'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - Not found")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ Missing modules: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_core_files():
    """Test that core files exist"""
    print("\n📁 Testing core files...")
    
    required_files = [
        'google_drive_image_analyzer.py',
        'enhanced_drive_downloader.py',
        'streamlit_survey_analyzer.py',
        'quick_start.py',
        'requirements.txt',
        'googleFormExample.csv'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_imports():
    """Test importing core modules"""
    print("\n🔧 Testing module imports...")
    
    try:
        from google_drive_image_analyzer import GoogleDriveImageAnalyzer
        print("✅ GoogleDriveImageAnalyzer")
    except ImportError as e:
        print(f"❌ GoogleDriveImageAnalyzer - {e}")
        return False
    
    try:
        from enhanced_drive_downloader import EnhancedDriveDownloader
        print("✅ EnhancedDriveDownloader")
    except ImportError as e:
        print(f"❌ EnhancedDriveDownloader - {e}")
        return False
    
    return True

def test_sample_data():
    """Test sample data loading"""
    print("\n📊 Testing sample data...")
    
    try:
        import pandas as pd
        data = pd.read_csv('googleFormExample.csv')
        print(f"✅ Sample data loaded: {len(data)} responses")
        
        # Check for Google Drive links
        has_drive_links = False
        for col in data.columns:
            if any('drive.google.com' in str(data[col].iloc[i]) for i in range(len(data))):
                has_drive_links = True
                print(f"✅ Found Google Drive links in column: '{col}'")
                break
        
        if not has_drive_links:
            print("⚠️ No Google Drive links found in sample data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False

def test_optional_features():
    """Test optional features"""
    print("\n🎯 Testing optional features...")
    
    # Test OpenAI
    try:
        import openai
        print("✅ OpenAI library available")
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            print("✅ OpenAI API key found in environment")
        else:
            print("⚠️ OpenAI API key not found (optional for basic features)")
    except ImportError:
        print("⚠️ OpenAI library not available (optional)")
    
    return True

def run_all_tests():
    """Run all installation tests"""
    print("🧪 Forms Image Analyzer - Installation Test")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_dependencies,
        test_core_files,
        test_imports,
        test_sample_data,
        test_optional_features
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📋 Test Results")
    print("=" * 20)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\n🚀 Next steps:")
        print("  • Run: python quick_start.py")
        print("  • Or: streamlit run streamlit_survey_analyzer.py")
        print("  • Or: python demo_full_pipeline.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("  • Run: pip install -r requirements.txt")
        print("  • Check Python version (3.8+ required)")
        print("  • Verify all files are present")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 