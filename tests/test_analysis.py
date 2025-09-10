"""
Tests for analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.analysis import DataProcessor, RegressionAnalyzer


class TestDataProcessor:
    """Test DataProcessor class."""
    
    def test_enrich_compustat_success(self):
        """Test successful Compustat enrichment."""
        comp = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'at': [1000, 2000],
            'ni': [100, 200],
            'ib': [100, 200]
        })
        
        company = pd.DataFrame({
            'gvkey': ['001', '002'],
            'sic': [1000, 2000]
        })
        
        result = DataProcessor.enrich_compustat(comp, company)
        
        assert 'gvkey' in result.columns
        assert 'fyear' in result.columns
        assert 'at' in result.columns
        assert 'ni' in result.columns
        assert 'sic' in result.columns
    
    def test_enrich_compustat_missing_required_columns(self):
        """Test Compustat enrichment with missing required columns."""
        comp = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001]
            # Missing 'at' column
        })
        
        with pytest.raises(RuntimeError, match="missing required columns"):
            DataProcessor.enrich_compustat(comp, pd.DataFrame())
    
    def test_enrich_compustat_ni_from_ib(self):
        """Test NI fallback from IB."""
        comp = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'at': [1000, 2000],
            'ib': [100, 200]  # No 'ni' column
        })
        
        result = DataProcessor.enrich_compustat(comp, pd.DataFrame())
        
        assert 'ni' in result.columns
        assert result['ni'].equals(result['ib'])
    
    def test_merge_datasets_success(self):
        """Test successful dataset merging."""
        comp = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'at': [1000, 2000]
        })
        
        execu = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'ceoann': ['CEO', 'CEO'],
            'gender': ['MALE', 'FEMALE']
        })
        
        result = DataProcessor.merge_datasets(
            comp, execu, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        
        assert 'gvkey' in result.columns
        assert 'fyear' in result.columns
        assert 'at' in result.columns
        assert 'female_ceo' in result.columns
    
    def test_construct_variables_success(self):
        """Test variable construction."""
        df = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'ni': [100, 200],
            'at': [1000, 2000],
            'dltt': [300, 400],
            'sic': [1000, 2000],
            'becameceo_year': ['1995-01-01', '1996-01-01']
        })
        
        result = DataProcessor.construct_variables(df)
        
        assert 'roa' in result.columns
        assert 'ln_assets' in result.columns
        assert 'leverage' in result.columns
        assert 'ceo_tenure' in result.columns
        assert 'sic1' in result.columns
        
        # Check ROA calculation
        expected_roa = df['ni'] / df['at']
        assert result['roa'].equals(expected_roa)
    
    def test_apply_filters_success(self):
        """Test filter application."""
        df = pd.DataFrame({
            'gvkey': ['001', '002', '003'],
            'fyear': [1999, 2000, 2001],
            'at': [1000, 2000, 3000]
        })
        
        sp1500 = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001]
        })
        
        result = DataProcessor.apply_filters(df, sp1500, (2000, 2010))
        
        assert len(result) == 2  # Only 2000 and 2001, and only gvkey 001 and 002
        assert result['fyear'].min() >= 2000
        assert result['fyear'].max() <= 2010
        assert 'sp1500' in result.columns
    
    def test_apply_filters_no_sp1500(self):
        """Test filter application without S&P 1500 data."""
        df = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'at': [1000, 2000]
        })
        
        with pytest.raises(RuntimeError, match="S&P 1500 membership missing"):
            DataProcessor.apply_filters(df, pd.DataFrame(), (2000, 2010))


class TestRegressionAnalyzer:
    """Test RegressionAnalyzer class."""
    
    def test_init(self):
        """Test initialization."""
        outdir = Path("test_out")
        analyzer = RegressionAnalyzer(outdir)
        assert analyzer.outdir == outdir
    
    def test_prepare_regression_data(self):
        """Test regression data preparation."""
        df = pd.DataFrame({
            'gvkey': ['001', '001', '002', '002'],
            'fyear': [2000, 2001, 2000, 2001],
            'roa': [0.1, 0.2, 0.15, 0.25],
            'female_ceo': [0, 0, 1, 1],
            'ln_assets': [6.0, 6.1, 7.0, 7.1],
            'leverage': [0.3, 0.4, 0.2, 0.3],
            'ceo_tenure': [5, 6, 3, 4],
            'board_size': [10, 10, 12, 12]  # No within-firm variation
        })
        
        analyzer = RegressionAnalyzer()
        result = analyzer._prepare_regression_data(df)
        
        assert 'size' in result.columns  # ln_assets renamed
        assert 'board_gender_ratio' not in result.columns  # Not in original data
        assert len(result) == 4
    
    def test_get_within_variables(self):
        """Test within-variable identification."""
        df = pd.DataFrame({
            'gvkey': ['001', '001', '002', '002'],
            'female_ceo': [0, 1, 0, 1],  # Varies within firm
            'size': [6.0, 6.1, 7.0, 7.1],  # Varies within firm
            'leverage': [0.3, 0.4, 0.2, 0.3],  # Varies within firm
            'board_size': [10, 10, 12, 12],  # No within-firm variation
            'ceo_tenure': [5, 6, 3, 4]  # Varies within firm
        })
        
        analyzer = RegressionAnalyzer()
        within_vars = analyzer._get_within_variables(df)
        
        assert 'female_ceo' in within_vars
        assert 'size' in within_vars
        assert 'leverage' in within_vars
        assert 'ceo_tenure' in within_vars
        assert 'board_size' not in within_vars  # No within-firm variation
    
    def test_apply_within_transformation(self):
        """Test within-transformation."""
        df = pd.DataFrame({
            'gvkey': ['001', '001', '002', '002'],
            'roa': [0.1, 0.2, 0.15, 0.25],
            'female_ceo': [0, 1, 0, 1],
            'size': [6.0, 6.1, 7.0, 7.1]
        })
        
        within_vars = ['female_ceo', 'size']
        
        analyzer = RegressionAnalyzer()
        result = analyzer._apply_within_transformation(df, within_vars)
        
        assert 'roa_demeaned' in result.columns
        assert 'female_ceo_demeaned' in result.columns
        assert 'size_demeaned' in result.columns
        
        # Check that within-transformed variables have mean zero by firm
        for gvkey in ['001', '002']:
            firm_data = result[result['gvkey'] == gvkey]
            assert abs(firm_data['female_ceo_demeaned'].mean()) < 1e-10
            assert abs(firm_data['size_demeaned'].mean()) < 1e-10
    
    def test_prepare_regression_matrices(self):
        """Test regression matrix preparation."""
        df = pd.DataFrame({
            'gvkey': ['001', '001', '002', '002'],
            'roa_demeaned': [0.05, -0.05, 0.05, -0.05],
            'female_ceo_demeaned': [0.5, -0.5, 0.5, -0.5],
            'size_demeaned': [0.05, -0.05, 0.05, -0.05]
        })
        
        within_vars = ['female_ceo', 'size']
        
        analyzer = RegressionAnalyzer()
        X, y, clusters = analyzer._prepare_regression_matrices(df, within_vars)
        
        assert X.shape == (4, 2)  # 4 observations, 2 variables
        assert y.shape == (4,)  # 4 observations
        assert clusters.shape == (4,)  # 4 cluster identifiers
        assert 'female_ceo_demeaned' in X.columns
        assert 'size_demeaned' in X.columns
    
    @patch('statsmodels.api.OLS')
    def test_run_firm_fixed_effects_regression(self, mock_ols):
        """Test full regression pipeline."""
        # Mock regression results
        mock_model = Mock()
        mock_model.summary.return_value.as_text.return_value = "Mock Summary"
        mock_model.rsquared = 0.05
        mock_model.rsquared_adj = 0.04
        mock_model.fvalue = 10.0
        mock_model.f_pvalue = 0.001
        mock_model.params = pd.Series({'female_ceo_demeaned': -0.01})
        mock_model.pvalues = pd.Series({'female_ceo_demeaned': 0.5})
        mock_ols.return_value.fit.return_value = mock_model
        
        df = pd.DataFrame({
            'gvkey': ['001', '001', '002', '002'],
            'fyear': [2000, 2001, 2000, 2001],
            'roa': [0.1, 0.2, 0.15, 0.25],
            'female_ceo': [0, 1, 0, 1],
            'ln_assets': [6.0, 6.1, 7.0, 7.1],
            'leverage': [0.3, 0.4, 0.2, 0.3]
        })
        
        with patch('builtins.open', mock_open()):
            analyzer = RegressionAnalyzer(Path("test_out"))
            model, used_data, X, y = analyzer.run_firm_fixed_effects_regression(df)
        
        assert model == mock_model
        assert len(used_data) == 4
        assert X.shape[0] == 4
        assert y.shape[0] == 4
    
    def test_print_data_quality_diagnostics(self, capsys):
        """Test data quality diagnostics printing."""
        df = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'roa': [0.1, 0.2],
            'female_ceo': [0, 1],
            'ln_assets': [6.0, 7.0],
            'leverage': [0.3, 0.4],
            'ceo_tenure': [5, 6],
            'board_size': [10, 12],
            'nationality_mix': [0.5, 0.6],
            'gender_ratio': [0.3, 0.4],
            'sic1': ['1', '2']
        })
        
        analyzer = RegressionAnalyzer()
        analyzer.print_data_quality_diagnostics(df)
        
        captured = capsys.readouterr()
        assert "DATA QUALITY DIAGNOSTICS" in captured.out
        assert "Total observations: 2" in captured.out
        assert "Female CEOs: 1 (50.0%)" in captured.out
