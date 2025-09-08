@echo off
REM Run the Female CEO ROA Analysis on Windows

echo 🚀 Running Female CEO ROA Analysis
echo ==================================

REM Check if virtual environment exists
if not exist ".venv" (
    echo ❌ Virtual environment not found
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Run the analysis
echo 🔄 Starting analysis...
.venv\Scripts\python female_ceo_roa.py

echo.
echo ✅ Analysis complete! Check the 'out' folder for results.
echo.
echo Generated files:
echo - reg_femaleCEO_roa.html (beautiful regression table)
echo - reg_femaleCEO_roa.txt (text results)
echo - sp1500_femaleCEO_2000_2010.csv (dataset)
echo - sp1500_femaleCEO_2000_2010.dta (Stata dataset)
echo.
pause
