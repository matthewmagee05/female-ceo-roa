"""
Female CEO ROA Analysis Package

A modular package for analyzing the relationship between female CEOs and firm performance.
"""

from .data_loader import WRDSDataLoader, LocalDataLoader, load_all_data
from .analysis import DataProcessor, RegressionAnalyzer
from .utils import (
    setup_output_directory, load_environment_variables, export_data,
    send_email_notification, print_analysis_summary, validate_data_quality,
    get_config
)

__version__ = "1.0.0"
__author__ = "Female CEO ROA Analysis Team"

__all__ = [
    "WRDSDataLoader",
    "LocalDataLoader", 
    "load_all_data",
    "DataProcessor",
    "RegressionAnalyzer",
    "setup_output_directory",
    "load_environment_variables",
    "export_data",
    "send_email_notification",
    "print_analysis_summary",
    "validate_data_quality",
    "get_config"
]
