"""
Tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.data_loader import WRDSDataLoader, LocalDataLoader, load_all_data


class TestWRDSDataLoader:
    """Test WRDSDataLoader class."""
    
    def test_init(self):
        """Test initialization."""
        loader = WRDSDataLoader(use_wrds=True, wrds_username="test_user")
        assert loader.use_wrds is True
        assert loader.wrds_username == "test_user"
        assert loader.conn is None
    
    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict('os.environ', {'WRDS_USERNAME': 'env_user'}):
            loader = WRDSDataLoader()
            assert loader.wrds_username == "env_user"
    
    @patch('src.data_loader.wrds')
    def test_connect_success(self, mock_wrds):
        """Test successful WRDS connection."""
        mock_conn = Mock()
        mock_wrds.Connection.return_value = mock_conn
        
        loader = WRDSDataLoader(use_wrds=True, wrds_username="test_user")
        result = loader.connect()
        
        assert result == mock_conn
        assert loader.conn == mock_conn
        mock_wrds.Connection.assert_called_once_with(wrds_username="test_user")
    
    @patch('src.data_loader.wrds')
    def test_connect_failure(self, mock_wrds):
        """Test WRDS connection failure."""
        mock_wrds.Connection.side_effect = Exception("Connection failed")
        
        loader = WRDSDataLoader(use_wrds=True)
        result = loader.connect()
        
        assert result is None
        assert loader.conn is None
    
    def test_connect_no_wrds(self):
        """Test connection when WRDS is disabled."""
        loader = WRDSDataLoader(use_wrds=False)
        result = loader.connect()
        assert result is None
    
    def test_disconnect(self):
        """Test disconnection."""
        mock_conn = Mock()
        loader = WRDSDataLoader()
        loader.conn = mock_conn
        
        loader.disconnect()
        
        mock_conn.close.assert_called_once()
        assert loader.conn is None
    
    def test_build_sp1500_membership_no_connection(self):
        """Test S&P 1500 membership building without connection."""
        loader = WRDSDataLoader()
        result = loader.build_sp1500_membership()
        assert result.empty
    
    @patch('src.data_loader.wrds')
    def test_build_sp1500_membership_success(self, mock_wrds):
        """Test successful S&P 1500 membership building."""
        # Mock connection and data
        mock_conn = Mock()
        mock_wrds.Connection.return_value = mock_conn
        
        # Mock S&P 1500 list data
        sp1500_data = pd.DataFrame({
            'gvkey': ['001', '002', '003'],
            'fyear': [2000, 2001, 2002]
        })
        mock_conn.raw_sql.return_value = sp1500_data
        
        loader = WRDSDataLoader()
        loader.connect()
        
        result = loader.build_sp1500_membership()
        
        assert not result.empty
        assert len(result) == 3
        assert 'gvkey' in result.columns
        assert 'fyear' in result.columns
    
    def test_load_compustat_data_no_connection(self):
        """Test Compustat data loading without connection."""
        loader = WRDSDataLoader()
        result = loader.load_compustat_data((2000, 2010), pd.DataFrame())
        assert result.empty
    
    @patch('src.data_loader.wrds')
    def test_load_compustat_data_success(self, mock_wrds):
        """Test successful Compustat data loading."""
        mock_conn = Mock()
        mock_wrds.Connection.return_value = mock_conn
        
        # Mock Compustat data
        comp_data = pd.DataFrame({
            'gvkey': ['001', '002'],
            'fyear': [2000, 2001],
            'ni': [100, 200],
            'at': [1000, 2000],
            'dltt': [300, 400],
            'tic': ['ABC', 'DEF']
        })
        mock_conn.raw_sql.return_value = comp_data
        
        loader = WRDSDataLoader()
        loader.connect()
        
        result = loader.load_compustat_data((2000, 2010), pd.DataFrame())
        
        assert not result.empty
        assert len(result) == 2
        assert 'gvkey' in result.columns


class TestLocalDataLoader:
    """Test LocalDataLoader class."""
    
    def test_load_file_nonexistent(self):
        """Test loading non-existent file."""
        result = LocalDataLoader.load_file("nonexistent.csv")
        assert result.empty
    
    @patch('pandas.read_csv')
    def test_load_file_csv(self, mock_read_csv):
        """Test loading CSV file."""
        mock_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_data
        
        with patch('pathlib.Path.exists', return_value=True):
            result = LocalDataLoader.load_file("test.csv")
        
        assert not result.empty
        mock_read_csv.assert_called_once()
    
    @patch('pandas.read_parquet')
    def test_load_file_parquet(self, mock_read_parquet):
        """Test loading Parquet file."""
        mock_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_parquet.return_value = mock_data
        
        with patch('pathlib.Path.exists', return_value=True):
            result = LocalDataLoader.load_file("test.parquet")
        
        assert not result.empty
        mock_read_parquet.assert_called_once()
    
    @patch('pandas.read_excel')
    def test_load_file_excel(self, mock_read_excel):
        """Test loading Excel file."""
        mock_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_excel.return_value = mock_data
        
        with patch('pathlib.Path.exists', return_value=True):
            result = LocalDataLoader.load_file("test.xlsx")
        
        assert not result.empty
        mock_read_excel.assert_called_once()


class TestLoadAllData:
    """Test load_all_data function."""
    
    @patch('src.data_loader.WRDSDataLoader')
    def test_load_all_data_wrds_success(self, mock_loader_class):
        """Test loading all data with WRDS."""
        # Mock WRDS loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_conn = Mock()
        mock_loader.connect.return_value = mock_conn
        
        # Mock data
        mock_loader.build_sp1500_membership.return_value = pd.DataFrame({'gvkey': ['001'], 'fyear': [2000]})
        mock_loader.load_compustat_data.return_value = pd.DataFrame({'gvkey': ['001'], 'fyear': [2000]})
        mock_loader.load_execucomp_data.return_value = pd.DataFrame({'gvkey': ['001'], 'fyear': [2000]})
        mock_loader.load_company_data.return_value = pd.DataFrame({'gvkey': ['001'], 'sic': [1000]})
        
        result = load_all_data(use_wrds=True, wrds_username="test_user")
        
        assert 'sp1500' in result
        assert 'comp' in result
        assert 'execu' in result
        assert 'company' in result
        mock_loader.disconnect.assert_called_once()
    
    @patch('src.data_loader.LocalDataLoader')
    def test_load_all_data_local_fallback(self, mock_local_loader):
        """Test loading all data with local fallback."""
        mock_local_loader.load_file.return_value = pd.DataFrame()
        
        result = load_all_data(use_wrds=False)
        
        assert 'sp1500' in result
        assert 'comp' in result
        assert 'execu' in result
        assert 'company' in result
        # All should be empty DataFrames
        for key, df in result.items():
            assert df.empty
