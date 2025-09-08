#!/bin/bash
# Setup script for macOS/Linux users

echo "🚀 Setting up Female CEO ROA Analysis"
echo "====================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.9+ from https://python.org"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version is not compatible (requires 3.9+)"
    exit 1
fi

echo "✅ Python $python_version is compatible"

# Run the setup script
python3 setup.py

echo ""
echo "Setup complete!"
