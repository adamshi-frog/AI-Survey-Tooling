#!/usr/bin/env python3
"""
Comprehensive Dependency Installer for Survey Analyzer

This script ensures all required dependencies are properly installed
and handles any potential import issues.
"""

import subprocess
import sys
import importlib
import os

# Required packages with their pip install names
REQUIRED_PACKAGES = {
    'streamlit': 'streamlit>=1.28.0',
    'pandas': 'pandas>=2.0.0',
    'numpy': 'numpy>=1.24.0',
    'plotly': 'plotly>=5.15.0',
    'PIL': 'Pillow>=9.5.0',
    'requests': 'requests>=2.31.0',
    'openai': 'openai>=1.0.0',
}

# Optional packages
OPTIONAL_PACKAGES = {
    'openpyxl': 'openpyxl>=3.1.0',
}

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_spec):
    """Install a package using pip"""
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_spec, "--upgrade"
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_spec}: {e}")
        return False

def main():
    """Main installation function"""
    print("🔧 Comprehensive Survey Analyzer - Dependency Installer")
    print("=" * 60)
    
    missing_packages = []
    failed_installs = []
    
    # Check required packages
    print("\n📦 Checking required packages...")
    for import_name, package_spec in REQUIRED_PACKAGES.items():
        package_name = package_spec.split('>=')[0]
        
        if check_package(import_name):
            print(f"✅ {package_name}: Already installed")
        else:
            print(f"❌ {package_name}: Missing")
            missing_packages.append(package_spec)
    
    # Check optional packages
    print("\n📦 Checking optional packages...")
    for import_name, package_spec in OPTIONAL_PACKAGES.items():
        package_name = package_spec.split('>=')[0]
        
        if check_package(import_name):
            print(f"✅ {package_name}: Already installed")
        else:
            print(f"⚠️ {package_name}: Missing (optional)")
            missing_packages.append(package_spec)
    
    # Install missing packages
    if missing_packages:
        print(f"\n🚀 Installing {len(missing_packages)} missing packages...")
        
        for package_spec in missing_packages:
            package_name = package_spec.split('>=')[0]
            print(f"\n📥 Installing {package_name}...")
            
            if install_package(package_spec):
                print(f"✅ Successfully installed {package_name}")
            else:
                failed_installs.append(package_name)
    
    # Final verification
    print("\n🔍 Final verification...")
    all_good = True
    
    for import_name, package_spec in REQUIRED_PACKAGES.items():
        package_name = package_spec.split('>=')[0]
        
        if check_package(import_name):
            print(f"✅ {package_name}: Ready")
        else:
            print(f"❌ {package_name}: Still missing!")
            all_good = False
    
    # Results
    print("\n" + "=" * 60)
    if all_good and not failed_installs:
        print("🎉 SUCCESS! All dependencies are installed and ready!")
        print("\n🚀 You can now run the Comprehensive Survey Analyzer:")
        print("   streamlit run comprehensive_survey_analyzer.py")
        
    elif failed_installs:
        print("⚠️ PARTIAL SUCCESS - Some packages failed to install:")
        for package in failed_installs:
            print(f"   • {package}")
        print("\n🔧 Try installing manually:")
        for package in failed_installs:
            print(f"   pip install {package}")
    
    else:
        print("❌ INSTALLATION ISSUES - Please check the errors above")
        print("\n🔧 Try running manually:")
        print("   pip install -r requirements.txt")
    
    # Create/update requirements.txt
    print("\n📝 Updating requirements.txt...")
    requirements_content = "\n".join(
        list(REQUIRED_PACKAGES.values()) + list(OPTIONAL_PACKAGES.values())
    )
    
    try:
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print("✅ requirements.txt updated successfully")
    except Exception as e:
        print(f"⚠️ Could not update requirements.txt: {e}")

if __name__ == "__main__":
    main() 