"""
Tests for utils module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.utils import (
    setup_output_directory, load_environment_variables, export_data,
    send_email_notification, print_analysis_summary, validate_data_quality,
    get_config
)


class TestSetupOutputDirectory:
    """Test setup_output_directory function."""
    
    def test_setup_output_directory_default(self):
        """Test default output directory setup."""
        result = setup_output_directory()
        assert result == Path("out")
        assert result.exists()
    
    def test_setup_output_directory_custom(self):
        """Test custom output directory setup."""
        result = setup_output_directory("custom_out")
        assert result == Path("custom_out")
        assert result.exists()
    
    def test_setup_output_directory_existing(self):
        """Test setup with existing directory."""
        # Directory should already exist from previous test
        result = setup_output_directory("out")
        assert result.exists()


class TestLoadEnvironmentVariables:
    """Test load_environment_variables function."""
    
    @patch('src.utils.load_dotenv')
    @patch.dict('os.environ', {
        'USE_WRDS': 'True',
        'WRDS_USERNAME': 'test_user',
        'EMAIL_ENABLE': 'True',
        'EMAIL_SMTP_HOST': 'smtp.test.com',
        'EMAIL_SMTP_PORT': '465',
        'EMAIL_FROM': 'test@example.com',
        'EMAIL_TO': 'prof@example.com',
        'EMAIL_USER': 'test@example.com',
        'EMAIL_APP_PASSWORD': 'test_pass'
    }, clear=True)
    def test_load_environment_variables_all_set(self, mock_load_dotenv):
        """Test loading all environment variables."""
        result = load_environment_variables()
        
        assert result['use_wrds'] is True
        assert result['wrds_username'] == 'test_user'
        assert result['email_enable'] is True
        assert result['email_smtp_host'] == 'smtp.test.com'
        assert result['email_smtp_port'] == 465
        assert result['email_from'] == 'test@example.com'
        assert result['email_to'] == 'prof@example.com'
        assert result['email_user'] == 'test@example.com'
        assert result['email_pass'] == 'test_pass'
    
    @patch('src.utils.load_dotenv')
    @patch.dict('os.environ', {}, clear=True)
    def test_load_environment_variables_defaults(self, mock_load_dotenv):
        """Test loading with default values."""
        result = load_environment_variables()
        
        assert result['use_wrds'] is True  # Default from get()
        assert result['wrds_username'] is None
        assert result['email_enable'] is False
        assert result['email_smtp_host'] == 'smtp.gmail.com'
        assert result['email_smtp_port'] == 587
        assert result['email_from'] == 'you@example.com'
        assert result['email_to'] == 'professor@example.edu'
        assert result['email_user'] == 'you@example.com'
        assert result['email_pass'] is None


class TestExportData:
    """Test export_data function."""
    
    def test_export_data_success(self, tmp_path):
        """Test successful data export."""
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
        
        csv_path, dta_path = export_data(df, tmp_path)
        
        assert csv_path.exists()
        assert dta_path.exists()
        
        # Check CSV content
        csv_data = pd.read_csv(csv_path)
        assert len(csv_data) == 2
        assert 'gvkey' in csv_data.columns
        
        # Check that column names are stataized
        assert all(len(col) <= 32 for col in csv_data.columns)
    
    def test_export_data_with_inf_values(self, tmp_path):
        """Test export with infinite values."""
        df = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'roa': [0.1, float('inf')],
            'female_ceo': [0, 1]
        })
        
        csv_path, dta_path = export_data(df, tmp_path)
        
        assert csv_path.exists()
        assert dta_path.exists()
        
        # Check that inf values are handled
        csv_data = pd.read_csv(csv_path)
        assert not csv_data['roa'].isna().all()  # Should have some valid values


class TestSendEmailNotification:
    """Test send_email_notification function."""
    
    def test_send_email_notification_disabled(self):
        """Test email notification when disabled."""
        config = {'email_enable': False}
        files = ['test1.txt', 'test2.txt']
        
        # Should not raise any errors
        send_email_notification(files, config)
    
    @patch('smtplib.SMTP')
    def test_send_email_notification_success(self, mock_smtp):
        """Test successful email sending."""
        config = {
            'email_enable': True,
            'email_smtp_host': 'smtp.test.com',
            'email_smtp_port': 587,
            'email_from': 'test@example.com',
            'email_to': 'prof@example.com',
            'email_user': 'test@example.com',
            'email_pass': 'test_pass'
        }
        
        files = ['test1.txt', 'test2.txt']
        
        # Mock file reading
        with patch('builtins.open', mock_open(read_data=b'test content')):
            send_email_notification(files, config)
        
        # Check that SMTP was called
        mock_smtp.assert_called_once_with('smtp.test.com', 587)
    
    @patch('smtplib.SMTP')
    def test_send_email_notification_failure(self, mock_smtp):
        """Test email sending failure."""
        config = {
            'email_enable': True,
            'email_smtp_host': 'smtp.test.com',
            'email_smtp_port': 587,
            'email_from': 'test@example.com',
            'email_to': 'prof@example.com',
            'email_user': 'test@example.com',
            'email_pass': 'test_pass'
        }
        
        files = ['test1.txt']
        
        # Mock SMTP failure
        mock_smtp.side_effect = Exception("SMTP Error")
        
        with patch('builtins.open', mock_open(read_data=b'test content')):
            # Should not raise exception, just warn
            send_email_notification(files, config)


class TestPrintAnalysisSummary:
    """Test print_analysis_summary function."""
    
    def test_print_analysis_summary(self, capsys):
        """Test analysis summary printing."""
        # Mock model
        mock_model = Mock()
        mock_model.rsquared = 0.05
        mock_model.fvalue = 10.0
        mock_model.f_pvalue = 0.001
        mock_model.params = pd.Series({'female_ceo_demeaned': -0.01})
        mock_model.pvalues = pd.Series({'female_ceo_demeaned': 0.5})
        
        # Mock data
        data = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001]
        })
        
        files = ['test1.txt', 'test2.txt']
        
        print_analysis_summary(mock_model, data, files)
        
        captured = capsys.readouterr()
        assert "ANALYSIS SUMMARY" in captured.out
        assert "Sample: 2 observations from 2 firms" in captured.out
        assert "R-squared: 0.050" in captured.out
        assert "Female CEO coefficient: -0.0100 (p = 0.500)" in captured.out


class TestValidateDataQuality:
    """Test validate_data_quality function."""
    
    def test_validate_data_quality_success(self):
        """Test successful data quality validation."""
        df = pd.DataFrame({
            'gvkey': ['001'] * 1000 + ['002'] * 1000,  # 2000 observations
            'fyear': [2000] * 1000 + [2001] * 1000,
            'roa': np.random.normal(0.1, 0.05, 2000),
            'female_ceo': [0] * 1900 + [1] * 100  # 100 female CEOs
        })
        
        result = validate_data_quality(df)
        assert result is True
    
    def test_validate_data_quality_small_sample(self):
        """Test data quality validation with small sample."""
        df = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'roa': [0.1, 0.2],
            'female_ceo': [0, 1]
        })
        
        result = validate_data_quality(df)
        assert result is False
    
    def test_validate_data_quality_missing_columns(self):
        """Test data quality validation with missing columns."""
        df = pd.DataFrame({
            'gvkey': ['001'] * 1000,
            'fyear': [2000] * 1000,
            'roa': [0.1] * 1000
            # Missing 'female_ceo' column
        })
        
        result = validate_data_quality(df)
        assert result is False
    
    def test_validate_data_quality_few_female_ceos(self):
        """Test data quality validation with few female CEOs."""
        df = pd.DataFrame({
            'gvkey': ['001'] * 1000,
            'fyear': [2000] * 1000,
            'roa': [0.1] * 1000,
            'female_ceo': [0] * 995 + [1] * 5  # Only 5 female CEOs
        })
        
        result = validate_data_quality(df)
        assert result is False
    
    def test_validate_data_quality_extreme_roa(self):
        """Test data quality validation with extreme ROA values."""
        df = pd.DataFrame({
            'gvkey': ['001'] * 1000,
            'fyear': [2000] * 1000,
            'roa': [0.1] * 999 + [100.0],  # One extreme value
            'female_ceo': [0] * 900 + [1] * 100
        })
        
        result = validate_data_quality(df)
        assert result is False


class TestGetConfig:
    """Test get_config function."""
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = get_config()
        
        assert 'years' in config
        assert 'stata_version' in config
        assert 'outdir' in config
        assert 'local_data_paths' in config
        
        assert config['years'] == (2000, 2010)
        assert config['stata_version'] == 118
        assert config['outdir'] == 'out'
        assert isinstance(config['local_data_paths'], dict)
