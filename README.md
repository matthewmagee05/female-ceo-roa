# Female CEO ROA Analysis

A modular, well-tested analysis of the relationship between female CEOs and firm performance (ROA) using S&P 1500 firms from 2000-2010, with comprehensive controls including board characteristics and CEO tenure.

## 🏗️ Architecture

The code has been refactored into modular components:

```
src/
├── __init__.py          # Package initialization
├── data_loader.py       # WRDS and local data loading
├── analysis.py          # Data processing and regression analysis
└── utils.py            # Utility functions and configuration

tests/
├── __init__.py
├── test_data_loader.py  # Tests for data loading
├── test_analysis.py     # Tests for analysis functions
├── test_utils.py        # Tests for utility functions
└── test_integration.py  # Integration tests

female_ceo_roa.py  # Main analysis script
```

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
make setup

# Or manually:
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 2. Run Analysis

```bash
# Run the analysis
make run

# Or directly:
python female_ceo_roa.py
```

### 3. Run Tests

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage
make test-cov
```

## 📊 Key Improvements

### 1. **Modular Design**
- **Data Loading**: Separated WRDS and local data loading logic
- **Analysis**: Isolated data processing and regression analysis
- **Utilities**: Centralized configuration and helper functions

### 2. **Comprehensive Testing**
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test complete pipeline end-to-end
- **Mocking**: Proper mocking of external dependencies (WRDS)
- **Coverage**: Track test coverage for quality assurance

### 3. **Better Error Handling**
- Graceful handling of missing data
- Clear error messages and warnings
- Robust fallback mechanisms

### 4. **Configuration Management**
- Environment variable support
- Centralized configuration
- Easy customization of parameters

### 5. **Code Quality**
- Type hints for better IDE support
- Docstrings for all functions and classes
- Consistent naming conventions
- Separation of concerns

## 🔧 Configuration

The refactored version supports configuration through environment variables:

```bash
# WRDS Configuration
export USE_WRDS=True
export WRDS_USERNAME=your_username

# Email Configuration (optional)
export EMAIL_ENABLE=True
export EMAIL_SMTP_HOST=smtp.gmail.com
export EMAIL_SMTP_PORT=587
export EMAIL_FROM=your_email@example.com
export EMAIL_TO=professor@example.edu
export EMAIL_USER=your_email@example.com
export EMAIL_APP_PASSWORD=your_app_password
```

## 📈 Usage Examples

### Basic Analysis

```python
from src import load_all_data, DataProcessor, RegressionAnalyzer

# Load data
data = load_all_data(use_wrds=True, years=(2000, 2010))

# Process data
processor = DataProcessor()
comp = processor.enrich_compustat(data['comp'], data['company'])
df = processor.merge_datasets(comp, data['execu'], ...)
df = processor.construct_variables(df)
df = processor.apply_filters(df, data['sp1500'])

# Run regression
analyzer = RegressionAnalyzer()
model, used_data, X, y = analyzer.run_firm_fixed_effects_regression(df)
```

### Custom Configuration

```python
from src.utils import get_config, setup_output_directory

# Get configuration
config = get_config()
config['years'] = (2005, 2015)  # Custom years

# Setup custom output directory
outdir = setup_output_directory("custom_output")
```

## 🧪 Testing

The test suite includes:

### Unit Tests
- **Data Loading**: Test WRDS connection, local file loading
- **Data Processing**: Test variable construction, filtering
- **Analysis**: Test regression pipeline, within-transformation
- **Utilities**: Test configuration, export functions

### Integration Tests
- **Complete Pipeline**: End-to-end analysis with mock data
- **Error Handling**: Test graceful failure modes
- **Data Export**: Test CSV and Stata export functionality

### Running Tests

```bash
# Run specific test file
pytest tests/test_data_loader.py -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow"
```

## 📁 Output Files

The refactored version generates the same output files as the original:

- `out/sp1500_femaleCEO_2000_2010.csv` - Analysis dataset
- `out/sp1500_femaleCEO_2000_2010.dta` - Stata format
- `out/reg_femaleCEO_roa.txt` - Regression results (text)
- `out/reg_femaleCEO_roa.html` - Regression results (HTML)
- `out/female_ceo_roa.py` - Script copy

## 🔄 Migration from Original

The refactored version maintains the same functionality as the original script but with improved:

1. **Code Organization**: Modular structure for easier maintenance
2. **Testing**: Comprehensive test coverage for reliability
3. **Error Handling**: Better error messages and recovery
4. **Documentation**: Clear docstrings and type hints
5. **Configuration**: Environment variable support

## 🛠️ Development

### Adding New Features

1. **Data Sources**: Add new data loaders in `src/data_loader.py`
2. **Analysis Methods**: Add new analysis functions in `src/analysis.py`
3. **Utilities**: Add helper functions in `src/utils.py`
4. **Tests**: Add corresponding tests in `tests/`

### Code Style

- Use type hints for all function parameters and returns
- Add docstrings for all public functions and classes
- Follow PEP 8 style guidelines
- Write tests for new functionality

## 📋 Available Commands

```bash
make help              # Show all available commands
make setup             # Install all dependencies
make test              # Run all tests
make test-cov          # Run tests with coverage
make run              # Run the analysis
make clean             # Clean up generated files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

Same as the original project.
