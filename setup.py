#!/usr/bin/env python3
"""
Setup script for Female CEO ROA Analysis
This script helps users set up the environment and test their WRDS connection.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is 3.9 or higher."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   Please install Python 3.9 or higher")
        return False

def setup_virtual_environment():
    """Set up virtual environment."""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python -m venv .venv", "Creating virtual environment")

def install_dependencies():
    """Install required packages."""
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = ".venv\\Scripts\\pip"
    else:  # macOS/Linux
        pip_path = ".venv/bin/pip"
    
    return run_command(f"{pip_path} install -r requirements.txt", "Installing dependencies")

def test_wrds_connection():
    """Test WRDS connection."""
    print("🔗 Testing WRDS connection...")
    
    # Determine the correct python path based on OS
    if os.name == 'nt':  # Windows
        python_path = ".venv\\Scripts\\python"
    else:  # macOS/Linux
        python_path = ".venv/bin/python"
    
    test_script = """
import wrds
import os

try:
    # Try to connect to WRDS
    conn = wrds.Connection()
    print("✅ WRDS connection successful!")
    print("✅ You can proceed with the analysis")
    conn.close()
except Exception as e:
    print(f"❌ WRDS connection failed: {e}")
    print("")
    print("Please check:")
    print("1. Your WRDS username and password")
    print("2. Your WRDS account has access to required databases")
    print("3. Your internet connection")
    print("")
    print("You can set up WRDS credentials by:")
    print("- Setting environment variables: WRDS_USERNAME and WRDS_PASSWORD")
    print("- Creating a .pgpass file")
    print("- The script will prompt you when you run the analysis")
"""
    
    try:
        result = subprocess.run([python_path, "-c", test_script], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ WRDS connection test timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing WRDS connection: {e}")
        return False

def create_output_directory():
    """Create output directory if it doesn't exist."""
    out_dir = Path("out")
    if not out_dir.exists():
        out_dir.mkdir()
        print("✅ Created output directory")
    else:
        print("✅ Output directory already exists")

def main():
    """Main setup function."""
    print("🚀 Setting up Female CEO ROA Analysis")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Set up virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create output directory
    create_output_directory()
    
    # Test WRDS connection
    wrds_ok = test_wrds_connection()
    
    print("\n" + "=" * 50)
    if wrds_ok:
        print("🎉 Setup completed successfully!")
        print("")
        print("You can now run the analysis with:")
        if os.name == 'nt':  # Windows
            print("  .venv\\Scripts\\python female_ceo_roa.py")
        else:  # macOS/Linux
            print("  .venv/bin/python female_ceo_roa.py")
    else:
        print("⚠️  Setup completed with warnings")
        print("")
        print("The environment is ready, but WRDS connection failed.")
        print("You can still run the analysis - it will prompt for WRDS credentials.")
        print("")
        print("To run the analysis:")
        if os.name == 'nt':  # Windows
            print("  .venv\\Scripts\\python female_ceo_roa.py")
        else:  # macOS/Linux
            print("  .venv/bin/python female_ceo_roa.py")
    
    print("")
    print("📖 See README.md for detailed instructions and troubleshooting")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
