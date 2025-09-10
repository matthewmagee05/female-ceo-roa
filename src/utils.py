"""
Utility functions for Female CEO ROA analysis.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd


def setup_output_directory(outdir: str = "out") -> Path:
    """Create and return output directory path."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def load_environment_variables() -> dict:
    """Load environment variables for configuration."""
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    return {
        'use_wrds': os.environ.get('USE_WRDS', 'True').lower() == 'true',
        'wrds_username': os.environ.get('WRDS_USERNAME'),
        'email_enable': os.environ.get('EMAIL_ENABLE', 'False').lower() == 'true',
        'email_smtp_host': os.environ.get('EMAIL_SMTP_HOST', 'smtp.gmail.com'),
        'email_smtp_port': int(os.environ.get('EMAIL_SMTP_PORT', '587')),
        'email_from': os.environ.get('EMAIL_FROM', 'you@example.com'),
        'email_to': os.environ.get('EMAIL_TO', 'professor@example.edu'),
        'email_user': os.environ.get('EMAIL_USER', 'you@example.com'),
        'email_pass': os.environ.get('EMAIL_APP_PASSWORD'),
    }


def export_data(df: pd.DataFrame, outdir: Path, stata_version: int = 118) -> tuple:
    """
    Export data to CSV and Stata formats.
    
    Args:
        df: DataFrame to export
        outdir: Output directory
        stata_version: Stata version for .dta export
        
    Returns:
        Tuple of (csv_path, dta_path)
    """
    def stataize(col):
        """Convert column name to Stata-compatible format."""
        col = str(col).strip().lower().replace(' ', '_')
        col = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in col)
        return col[:32]
    
    out = df.copy()
    out.columns = [stataize(c) for c in out.columns]
    
    # Clean for export
    for col in out.columns:
        try:
            col_series = out[col]
            if hasattr(col_series, 'dtype'):
                if col_series.dtype == 'object':
                    out[col] = col_series.astype(str)
                    out[col] = out[col].replace(['nan', '<NA>'], '')
                elif 'Int64' in str(col_series.dtype):
                    out[col] = col_series.astype('Int64')
                elif col_series.dtype == 'boolean':
                    out[col] = col_series.astype(int)
                elif col_series.dtype in ['float64', 'Float64']:
                    out[col] = col_series.replace([float('inf'), float('-inf')], pd.NA)
                    # Convert to numeric to handle inf values properly
                    out[col] = pd.to_numeric(out[col], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not process column {col}: {e}")
            continue
    
    # Export CSV
    csv_path = outdir / "sp1500_femaleCEO_2000_2010.csv"
    out.to_csv(csv_path, index=False)
    
    # Export Stata
    stata_cols = []
    for col in out.columns:
        try:
            col_series = out[col]
            if hasattr(col_series, 'dtype'):
                if col_series.dtype in ['int64', 'float64', 'Int64', 'Float64']:
                    stata_cols.append(col)
                elif col_series.dtype == 'object' and col_series.astype(str).str.len().max() <= 244:
                    stata_cols.append(col)
        except Exception as e:
            print(f"Warning: Could not process column {col} for Stata export: {e}")
            continue
    
    dta_path = outdir / "sp1500_femaleCEO_2000_2010.dta"
    out[stata_cols].to_stata(dta_path, write_index=False, version=stata_version,
                            data_label="S&P1500 Female CEO & ROA 2000–2010")
    
    return csv_path, dta_path


def send_email_notification(files: list, config: dict):
    """Send email notification with analysis results."""
    if not config.get('email_enable', False):
        return
    
    import smtplib
    import ssl
    from email.message import EmailMessage
    
    msg = EmailMessage()
    msg["From"] = config['email_from']
    msg["To"] = config['email_to']
    msg["Subject"] = "Female CEO and ROA (S&P1500 2000–2010) – Data & Code"
    
    body = (
        "Professor,\n\n"
        "Attached are the data export, regression output, and the script for the assignment:\n"
        "- Data (.csv and .dta)\n"
        "- Regression summary (.txt and .html)\n"
        "- Script (female_ceo_roa.py)\n\n"
        "Best,\nStudent"
    )
    msg.set_content(body)
    
    for file_path in files:
        try:
            with open(file_path, "rb") as fh:
                data = fh.read()
            msg.add_attachment(data, maintype="application", subtype="octet-stream", 
                             filename=Path(file_path).name)
        except Exception as e:
            warnings.warn(f"Could not attach {file_path}: {e}")
    
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(config['email_smtp_host'], config['email_smtp_port']) as server:
            server.starttls(context=context)
            server.login(config['email_user'], config['email_pass'])
            server.send_message(msg)
        print(f"✅ Emailed results to {config['email_to']}")
    except Exception as e:
        warnings.warn(f"Email sending failed: {e}")


def copy_script_to_output(outdir: Path):
    """Copy the main script to output directory."""
    script_copy = outdir / "female_ceo_roa.py"
    try:
        import shutil
        import sys
        if hasattr(sys, 'argv') and len(sys.argv) > 0 and Path(sys.argv[0]).exists():
            shutil.copyfile(Path(sys.argv[0]), script_copy)
    except Exception:
        pass


def print_analysis_summary(model, data: pd.DataFrame, files: list):
    """Print summary of analysis results."""
    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"=" * 50)
    print(f"Sample: {len(data):,} observations from {data['gvkey'].nunique():,} firms")
    print(f"R-squared: {model.rsquared:.3f}")
    print(f"F-statistic: {model.fvalue:.2f} (p = {model.f_pvalue:.3f})")
    
    # Main coefficient
    if 'female_ceo_demeaned' in model.params:
        coef = model.params['female_ceo_demeaned']
        pval = model.pvalues['female_ceo_demeaned']
        print(f"Female CEO coefficient: {coef:.4f} (p = {pval:.3f})")
    
    # Check BoardEx availability
    if 'boardex_available' in data.columns and data['boardex_available'].iloc[0] == 0:
        print(f"\n⚠️  Note: BoardEx controls absorbed by firm fixed effects (no within-firm variation)")
    
    print(f"\n📁 Output files:")
    for file_path in files:
        print(f"   - {file_path}")


def validate_data_quality(df: pd.DataFrame) -> bool:
    """
    Validate data quality and return True if acceptable.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if data quality is acceptable
    """
    issues = []
    
    # Check minimum sample size
    if len(df) < 1000:
        issues.append(f"Sample size too small: {len(df)} observations")
    
    # Check for required columns
    required_cols = ['gvkey', 'fyear', 'roa', 'female_ceo']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for sufficient female CEOs
    if 'female_ceo' in df.columns:
        female_count = (df['female_ceo'] == 1).sum()
        if female_count < 10:
            issues.append(f"Too few female CEOs: {female_count}")
    
    # Check for reasonable ROA values
    if 'roa' in df.columns:
        roa_data = df['roa'].dropna()
        if len(roa_data) > 0:
            roa_range = roa_data.max() - roa_data.min()
            if roa_range > 10:  # Unreasonably large range
                issues.append(f"ROA range too large: {roa_range:.2f}")
    
    if issues:
        print("⚠️  Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Data quality validation passed")
    return True


def get_config() -> dict:
    """Get configuration settings."""
    return {
        'years': (2000, 2010),
        'stata_version': 118,
        'outdir': 'out',
        'local_data_paths': {
            'comp': 'data/comp_funda.parquet',
            'execu': 'data/execucomp_annual.parquet',
            'link': 'data/boardex_compustat_link.parquet',
            'bx': 'data/boardex_company.parquet',
            'sp1500': 'data/sp1500_membership.parquet'
        }
    }
