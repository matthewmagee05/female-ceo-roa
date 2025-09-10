# Female CEO and ROA Analysis

This project analyzes the relationship between female CEOs and firm performance (ROA) using S&P 1500 firms from 2000-2010, with comprehensive controls including board characteristics and CEO tenure.

## 📊 Analysis Overview

- **Research Question**: Do female CEOs affect firm performance (ROA)?
- **Sample**: S&P 1500 firms, 2000-2010
- **Methodology**: OLS regression with robust standard errors (HC3)
- **Controls**: Firm size, leverage, board characteristics, CEO tenure, industry FE, year FE
- **Data Sources**: Compustat, ExecuComp, BoardEx, S&P index constituents

## 🚀 Quick Start

### Super Quick Start (3 steps)

1. **Download this repository**
2. **Run setup script:**
   - Windows: Double-click `setup.bat`
   - macOS/Linux: Run `./setup.sh`
3. **Run analysis:** `python female_ceo_roa.py`

### Prerequisites

1. **Python 3.9+** installed on your system
2. **WRDS Account** with access to:
   - Compustat (comp)
   - ExecuComp (comp_execucomp) 
   - BoardEx (boardex_na)
   - CRSP (crsp_a_indexes, crsp_a_ccm)

### Installation

#### Option 1: Automated Setup (Recommended)

**On Windows:**
```bash
# Download/clone the repository, then:
setup.bat
```

**On macOS/Linux:**
```bash
# Download/clone the repository, then:
./setup.sh
```

**Or run the Python setup script directly:**
```bash
python setup.py
```

#### Option 2: Manual Setup

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd female-ceo-roa
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   **On Windows:**
   ```bash
   .venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### WRDS Setup

1. **Set up WRDS credentials** (choose one method):

   **Method A: Environment Variables**
   ```bash
   export WRDS_USERNAME="your_username"
   export WRDS_PASSWORD="your_password"
   ```

   **Method B: .pgpass file** (recommended)
   ```bash
   # The script will prompt you to create this automatically
   # Or create manually: ~/.pgpass with format:
   # hostname:port:database:username:password
   ```

2. **Test WRDS connection**
   ```bash
   python -c "import wrds; conn = wrds.Connection(); print('WRDS connection successful!')"
   ```

### Run the Analysis

#### Option 1: Using Run Scripts (Easiest)

**On Windows:**
```bash
run_analysis.bat
```

**On macOS/Linux:**
```bash
./run_analysis.sh
```

#### Option 2: Manual Run

**On Windows:**
```bash
.venv\Scripts\python female_ceo_roa.py
```

**On macOS/Linux:**
```bash
.venv/bin/python female_ceo_roa.py
```

#### Option 3: With Virtual Environment Activated

```bash
# Activate virtual environment first
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python female_ceo_roa.py
```

## 📁 Project Structure

```
female-ceo-roa/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── female_ceo_roa.py                  # Main analysis script
├── setup.py                           # Automated setup script
├── setup.bat                          # Windows setup script
├── setup.sh                           # macOS/Linux setup script
├── run_analysis.bat                   # Windows run script
├── run_analysis.sh                    # macOS/Linux run script
├── data/                              # Local data storage (if needed)
└── out/                               # Analysis outputs
    ├── reg_femaleCEO_roa.html         # Beautiful HTML regression table
    ├── reg_femaleCEO_roa.txt          # Text regression results
    ├── sp1500_femaleCEO_2000_2010.csv # Final dataset (CSV)
    └── sp1500_femaleCEO_2000_2010.dta # Final dataset (Stata)
```

## 📊 Expected Outputs

### 1. Console Output
- Data pull progress and diagnostics
- Sample size information
- Missing data analysis
- Regression results summary

### 2. Files Generated
- **HTML Table**: Professional regression results with formatting
- **Text Results**: Standard regression output
- **CSV Dataset**: Final analysis dataset
- **Stata Dataset**: For use in Stata/econometric software

### 3. Key Results
- **Sample Size**: ~331 firm-year observations
- **Female CEO Effect**: Coefficient and significance
- **Controls**: All requested board and firm characteristics
- **Model Fit**: R-squared and diagnostic statistics

## 🔧 Troubleshooting

### Common Issues

**1. WRDS Connection Failed**
```
Error: PAM authentication failed
```
**Solution**: 
- Verify your WRDS username/password
- Check if your WRDS account has access to required databases
- Try creating a .pgpass file manually

**2. Module Not Found**
```
ModuleNotFoundError: No module named 'pandas'
```
**Solution**:
```bash
pip install -r requirements.txt
```

**3. Permission Denied**
```
PermissionError: [Errno 13] Permission denied
```
**Solution**:
- Ensure you have write permissions in the project directory
- Try running with administrator/sudo privileges if needed

**4. Sample Size Too Small**
```
DEBUG: Running regression with 0 observations
```
**Solution**:
- Check WRDS data availability
- Verify your account has access to all required databases
- The script includes fallback mechanisms for missing data

### Data Availability Issues

**Limited BoardEx Coverage**
- BoardEx data linking is complex and may have low coverage
- The analysis handles this gracefully by including controls where available
- This is normal and expected for BoardEx data

**S&P 1500 Filtering**
- Uses S&P 500 as proxy when S&P 1500 data is limited
- Includes comprehensive diagnostics to show what data is available

## 📈 Understanding the Results

### Regression Output
- **female_ceo**: Main variable of interest (coefficient and p-value)
- **ln_assets**: Firm size control (log of total assets)
- **leverage**: Financial leverage control (debt/assets)
- **board_size**: Board size control (where available)
- **ceo_tenure**: CEO tenure in years (where available)
- **sic1_***: Industry fixed effects (SIC 1-digit)
- **year_***: Year fixed effects (2001-2010)

### Key Diagnostics
- **R-squared**: Model explanatory power
- **F-statistic**: Overall model significance
- **Skewness/Kurtosis**: ROA distribution characteristics
- **Condition Number**: Multicollinearity indicator

## 🎯 Interpretation Guidelines

### Female CEO Coefficient
- **Positive**: Female CEOs associated with higher ROA
- **Negative**: Female CEOs associated with lower ROA
- **Not Significant**: No detectable effect (still a valid result)

### Sample Limitations
- **Small Sample**: Due to data availability constraints
- **BoardEx Coverage**: Low but included where available
- **Time Period**: 2000-2010 (historical data)

## 🔬 Methodology Notes

### Data Processing
- **Winsorization**: ROA values capped at 1st/99th percentiles
- **Robust Standard Errors**: HC3 to handle heteroscedasticity
- **Missing Data**: Handled through appropriate imputation and exclusion

### Variable Construction
- **ROA**: Net Income / Total Assets
- **Female CEO**: Binary indicator from ExecuComp
- **CEO Tenure**: Current year - CEO start year
- **Board Controls**: From BoardEx (where available)

## 📚 Dependencies

See `requirements.txt` for complete list. Key packages:
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `statsmodels`: Statistical modeling
- `wrds`: WRDS database access
- `psycopg2`: PostgreSQL connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for academic/research purposes. Please cite appropriately if used in research.

## 🆘 Support

For issues related to:
- **WRDS Access**: Contact WRDS support
- **Data Questions**: Check WRDS documentation
- **Code Issues**: Open an issue in this repository

## 📖 References

- WRDS Documentation: https://wrds-www.wharton.upenn.edu/
- Compustat Manual: Available through WRDS
- ExecuComp Manual: Available through WRDS
- BoardEx Documentation: Available through WRDS

---

**Note**: This analysis requires active WRDS subscription and appropriate database access. Results may vary based on data availability and access permissions.
