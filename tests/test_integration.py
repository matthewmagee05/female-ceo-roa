"""
Integration tests for the complete analysis pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src import (
    load_all_data, DataProcessor, RegressionAnalyzer,
    setup_output_directory, export_data, validate_data_quality
)


class TestIntegrationPipeline:
    """Test the complete analysis pipeline."""
    
    def create_mock_data(self):
        """Create mock data for testing."""
        np.random.seed(42)  # For reproducible results
        
        # Create firm-year panel data
        n_firms = 200  # Increase to meet minimum sample size requirement
        n_years = 5
        years = list(range(2000, 2005))
        
        data = []
        for i in range(n_firms):
            gvkey = f"{i:03d}"
            for year in years:
                # Simulate some firms having female CEOs
                female_ceo = 1 if (i + year) % 20 == 0 else 0
                
                # Simulate ROA with some relationship to female CEO
                roa_base = 0.1 + np.random.normal(0, 0.05)
                roa_effect = -0.02 if female_ceo else 0  # Small negative effect
                roa = roa_base + roa_effect
                
                data.append({
                    'gvkey': gvkey,
                    'fyear': year,
                    'ni': roa * 1000,  # Net income
                    'at': 1000 + np.random.normal(0, 200),  # Assets
                    'dltt': 300 + np.random.normal(0, 100),  # Debt
                    'tic': f"TIC{i:03d}",
                    'datadate': f"{year}-12-31",
                    'ceoann': 'CEO',
                    'gender': 'FEMALE' if female_ceo else 'MALE',
                    'becameceo_year': f"{year-5}-01-01",
                    'execid': f"EXEC{i:03d}",
                    'coname': f"Company {i}",
                    'sic': 1000 + (i % 9) * 100  # SIC codes
                })
        
        return pd.DataFrame(data)
    
    @patch('src.data_loader.WRDSDataLoader')
    def test_complete_pipeline_with_mock_data(self, mock_loader_class):
        """Test complete pipeline with mock WRDS data."""
        # Mock WRDS loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_conn = Mock()
        mock_loader.connect.return_value = mock_conn
        
        # Create mock data
        mock_data = self.create_mock_data()
        
        # Mock WRDS responses
        mock_loader.build_sp1500_membership.return_value = mock_data[['gvkey', 'fyear']].drop_duplicates()
        mock_loader.load_compustat_data.return_value = mock_data
        mock_loader.load_execucomp_data.return_value = mock_data
        mock_loader.load_company_data.return_value = mock_data[['gvkey', 'sic']].drop_duplicates()
        
        # Run the pipeline
        with patch('builtins.open', mock_open()):
            # Load data
            data = load_all_data(use_wrds=True, wrds_username="test_user")
            
            # Process data
            processor = DataProcessor()
            comp = processor.enrich_compustat(data['comp'], data['company'])
            df = processor.merge_datasets(
                comp, data['execu'], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )
            df = processor.construct_variables(df)
            df = processor.apply_filters(df, data['sp1500'])
            
            # Validate data quality
            assert validate_data_quality(df)
            
            # Run regression
            analyzer = RegressionAnalyzer(Path("test_out"))
            model, used_data, X, y = analyzer.run_firm_fixed_effects_regression(df)
            
            # Check results
            assert model is not None
            assert len(used_data) > 0
            assert X.shape[0] == len(used_data)
            assert y.shape[0] == len(used_data)
            
            # Check that we have some female CEOs
            female_ceo_count = (used_data['female_ceo'] == 1).sum()
            assert female_ceo_count > 0
    
    def test_pipeline_with_local_data(self, tmp_path):
        """Test pipeline with local data files."""
        # Create mock data files
        mock_data = self.create_mock_data()
        
        # Save to temporary files
        comp_file = tmp_path / "comp_funda.parquet"
        execu_file = tmp_path / "execucomp_annual.parquet"
        sp1500_file = tmp_path / "sp1500_membership.parquet"
        
        mock_data.to_parquet(comp_file)
        mock_data.to_parquet(execu_file)
        mock_data[['gvkey', 'fyear']].drop_duplicates().to_parquet(sp1500_file)
        
        # Mock the local data paths
        with patch('src.utils.get_config') as mock_config:
            mock_config.return_value = {
                'years': (2000, 2010),
                'stata_version': 118,
                'outdir': 'out',
                'local_data_paths': {
                    'comp': str(comp_file),
                    'execu': str(execu_file),
                    'link': 'nonexistent.parquet',
                    'bx': 'nonexistent.parquet',
                    'sp1500': str(sp1500_file)
                }
            }
            
            # Run pipeline
            with patch('builtins.open', mock_open()):
                data = load_all_data(use_wrds=False)
                
                processor = DataProcessor()
                comp = processor.enrich_compustat(data['comp'], pd.DataFrame())
                df = processor.merge_datasets(
                    comp, data['execu'], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                )
                df = processor.construct_variables(df)
                df = processor.apply_filters(df, data['sp1500'])
                
                # Should have valid data
                assert len(df) > 0
                assert 'roa' in df.columns
                assert 'female_ceo' in df.columns
    
    def test_data_export_integration(self, tmp_path):
        """Test data export integration."""
        # Create sample data
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
        
        # Export data
        csv_path, dta_path = export_data(df, tmp_path)
        
        # Verify files exist and have correct content
        assert csv_path.exists()
        assert dta_path.exists()
        
        # Check CSV content
        csv_data = pd.read_csv(csv_path)
        assert len(csv_data) == 2
        assert 'gvkey' in csv_data.columns
        
        # Check that all numeric columns are present
        expected_numeric_cols = ['roa', 'female_ceo', 'ln_assets', 'leverage', 'ceo_tenure', 'board_size']
        for col in expected_numeric_cols:
            assert col in csv_data.columns
    
    def test_error_handling_missing_data(self):
        """Test error handling with missing data."""
        # Test with empty datasets
        empty_df = pd.DataFrame()
        
        processor = DataProcessor()
        
        # Should handle empty data gracefully
        with pytest.raises(RuntimeError, match="missing required columns"):
            processor.enrich_compustat(empty_df, pd.DataFrame())
    
    def test_error_handling_invalid_filters(self):
        """Test error handling with invalid filters."""
        df = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'at': [1000, 2000]
        })
        
        processor = DataProcessor()
        
        # Should raise error when S&P 1500 data is missing
        with pytest.raises(RuntimeError, match="S&P 1500 membership missing"):
            processor.apply_filters(df, pd.DataFrame(), (2000, 2010))
