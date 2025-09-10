#!/usr/bin/env python3
"""
Female CEO ROA Analysis - Refactored Main Script

End-to-end pipeline for analyzing the relationship between female CEOs and firm performance.
Uses modular components for better maintainability and testability.
"""

import warnings
from pathlib import Path

import pandas as pd

from src import (
    load_all_data, DataProcessor, RegressionAnalyzer,
    setup_output_directory, load_environment_variables, export_data,
    send_email_notification, print_analysis_summary, validate_data_quality,
    get_config
)


def main():
    """Main analysis pipeline."""
    print("🚀 Female CEO ROA Analysis - Refactored Version")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    env_config = load_environment_variables()
    
    # Setup output directory
    outdir = setup_output_directory(config['outdir'])
    
    # Load data
    print("\n📊 Loading data...")
    data = load_all_data(
        use_wrds=env_config['use_wrds'],
        wrds_username=env_config['wrds_username'],
        years=config['years']
    )
    
    # Process data
    print("\n🔧 Processing data...")
    processor = DataProcessor()
    
    # Enrich Compustat data
    comp = processor.enrich_compustat(data['comp'], data['company'])
    
    # Merge datasets
    df = processor.merge_datasets(
        comp, data['execu'], pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        data['link'], data['bx'], pd.DataFrame(), pd.DataFrame()
    )
    
    # Construct variables
    df = processor.construct_variables(df)
    
    # Apply filters
    df = processor.apply_filters(df, data['sp1500'], config['years'])
    
    # Validate data quality
    if not validate_data_quality(df):
        print("⚠️  Proceeding with data quality issues...")
    
    # Run analysis
    print("\n📈 Running regression analysis...")
    analyzer = RegressionAnalyzer(outdir)
    analyzer.print_data_quality_diagnostics(df)
    
    model, used_data, X, y = analyzer.run_firm_fixed_effects_regression(df)
    
    # Print results
    print(f"\n📊 REGRESSION RESULTS")
    print("=" * 50)
    print(model.summary())
    
    # Export data
    print("\n💾 Exporting data...")
    csv_path, dta_path = export_data(df, outdir, config['stata_version'])
    
    # Prepare files for email
    files_to_send = [csv_path, dta_path]
    txt_path = outdir / "reg_femaleCEO_roa.txt"
    html_path = outdir / "reg_femaleCEO_roa.html"
    
    if txt_path.exists():
        files_to_send.append(txt_path)
    if html_path.exists():
        files_to_send.append(html_path)
    
    # Send email notification (if enabled)
    if env_config['email_enable']:
        send_email_notification(files_to_send, env_config)
    
    # Print summary
    print_analysis_summary(model, used_data, files_to_send)
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"📁 Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
