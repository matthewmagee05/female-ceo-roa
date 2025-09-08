#!/bin/bash
# Run the Female CEO ROA Analysis on macOS/Linux

echo "🚀 Running Female CEO ROA Analysis"
echo "=================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Run the analysis
echo "🔄 Starting analysis..."
.venv/bin/python female_ceo_roa.py

echo ""
echo "✅ Analysis complete! Check the 'out' folder for results."
echo ""
echo "Generated files:"
echo "- reg_femaleCEO_roa.html (beautiful regression table)"
echo "- reg_femaleCEO_roa.txt (text results)"
echo "- sp1500_femaleCEO_2000_2010.csv (dataset)"
echo "- sp1500_femaleCEO_2000_2010.dta (Stata dataset)"
echo ""
