"""
Analysis module for Female CEO ROA regression and statistical operations.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import statsmodels.api as sm


class DataProcessor:
    """Handles data processing and variable construction."""
    
    @staticmethod
    def enrich_compustat(comp: pd.DataFrame, company: pd.DataFrame) -> pd.DataFrame:
        """Enrich Compustat data with SIC codes and clean variables."""
        comp = comp.copy()
        comp.columns = [c.lower() for c in comp.columns]
        
        # Check required columns
        must = {'gvkey', 'fyear', 'at'}
        missing = must - set(comp.columns)
        if missing:
            raise RuntimeError(f"Compustat missing required columns: {missing}")
        
        # Handle NI from IB fallback
        if 'ni' not in comp.columns or comp['ni'].isna().all():
            if 'ib' in comp.columns:
                comp['ni'] = pd.to_numeric(comp['ib'], errors='coerce')
            else:
                comp['ni'] = pd.NA
        else:
            comp['ni'] = pd.to_numeric(comp['ni'], errors='coerce')
        
        # Add SIC from company table if missing
        if 'sic' not in comp.columns or comp['sic'].isna().all():
            if company is not None and not company.empty:
                company = company.copy()
                company.columns = [c.lower() for c in company.columns]
                if 'gvkey' in company.columns and 'sic' in company.columns:
                    comp = comp.merge(company[['gvkey', 'sic']].drop_duplicates('gvkey'), 
                                    on='gvkey', how='left')
                else:
                    comp['sic'] = pd.NA
            else:
                comp['sic'] = pd.NA
        
        # Coerce numeric columns
        for col in ['fyear', 'sic', 'ni', 'at', 'dltt']:
            if col in comp.columns:
                comp[col] = pd.to_numeric(comp[col], errors='coerce')
        
        return comp
    
    @staticmethod
    def merge_datasets(comp: pd.DataFrame, execu: pd.DataFrame, 
                      execu_person: pd.DataFrame, execu_roles: pd.DataFrame,
                      execu_company: pd.DataFrame, link: pd.DataFrame,
                      bx: pd.DataFrame, prof: pd.DataFrame, 
                      bx_stocks: pd.DataFrame) -> pd.DataFrame:
        """Merge all datasets into a single analysis dataset."""
        comp = comp.copy()
        execu = execu.copy()
        
        # Ensure consistent data types for merging
        if 'gvkey' in comp.columns:
            comp['gvkey'] = comp['gvkey'].astype(str)
        if 'gvkey' in execu.columns:
            execu['gvkey'] = execu['gvkey'].astype(str)
        if 'fyear' in comp.columns:
            comp['fyear'] = comp['fyear'].astype(int)
        if 'fyear' in execu.columns:
            execu['fyear'] = execu['fyear'].astype(int)
        
        # Process ExecuComp data
        if not execu.empty:
            execu = DataProcessor._process_execucomp(execu)
        
        # Merge Compustat + ExecuComp
        on_cols = [c for c in ['gvkey', 'fyear'] if c in comp.columns and c in execu.columns]
        if on_cols:
            df = comp.merge(execu, on=on_cols, how='left', suffixes=('', '_exec'))
        else:
            df = comp.copy()
            warnings.warn("ExecuComp merge failed - incompatible columns")
        
        # Add BoardEx data (simplified for now)
        df = DataProcessor._add_boardex_data(df, link, bx, prof, bx_stocks)
        
        # Remove duplicates
        df = df.sort_values(['gvkey', 'fyear']).drop_duplicates(['gvkey', 'fyear'], keep='first')
        
        return df
    
    @staticmethod
    def _process_execucomp(execu: pd.DataFrame) -> pd.DataFrame:
        """Process ExecuComp data to extract CEO information."""
        # Convert year to fyear if needed
        if 'fyear' not in execu.columns and 'year' in execu.columns:
            execu.rename(columns={'year': 'fyear'}, inplace=True)
        
        # Filter to CEO observations
        if 'ceoann' in execu.columns:
            print(f"DEBUG: Before CEO filter: {len(execu)} ExecuComp observations")
            ceo_mask = execu['ceoann'].astype(str).str.strip().str.upper().isin(['1', 'CEO', 'CEOANN', 'Y', 'TRUE'])
            execu = execu[ceo_mask].copy()
            print(f"DEBUG: After CEO filter: {len(execu)} CEO observations")
            
            # Handle duplicate CEOs per firm-year
            ceo_duplicates = execu.groupby(['gvkey', 'fyear']).size()
            duplicate_firms = ceo_duplicates[ceo_duplicates > 1]
            if len(duplicate_firms) > 0:
                print(f"WARNING: {len(duplicate_firms)} firm-years have multiple CEOs")
                execu = execu.drop_duplicates(subset=['gvkey', 'fyear'], keep='first')
        
        # Create female_ceo variable
        if 'gender' in execu.columns:
            execu['female_ceo'] = (execu['gender'].astype(str).str.upper().str[0] == 'F').astype('Int64')
        elif 'ceo_gender' in execu.columns:
            execu['female_ceo'] = (execu['ceo_gender'].astype(str).str.upper().str[0] == 'F').astype('Int64')
        else:
            execu['female_ceo'] = pd.NA
        
        # Handle becameceo_year
        if 'becameceo_year' not in execu.columns and 'becameceo' in execu.columns:
            execu.rename(columns={'becameceo': 'becameceo_year'}, inplace=True)
        
        return execu
    
    @staticmethod
    def _add_boardex_data(df: pd.DataFrame, link: pd.DataFrame, bx: pd.DataFrame,
                         prof: pd.DataFrame, bx_stocks: pd.DataFrame) -> pd.DataFrame:
        """Add BoardEx data to the main dataset."""
        # BoardEx data is not available in this WRDS instance
        # Set explicit flags so the report explains omission
        df = df.copy()
        df['board_size'] = pd.NA
        df['nationality_mix'] = pd.NA
        df['gender_ratio'] = pd.NA
        df['boardex_available'] = 0  # Flag indicating BoardEx not available
        print("DEBUG: BoardEx not available in WRDS instance – controls omitted.")
        return df
    
    @staticmethod
    def construct_variables(df: pd.DataFrame) -> pd.DataFrame:
        """Construct analysis variables from raw data."""
        df = df.copy()
        
        # Core accounting variables
        df['roa'] = pd.to_numeric(df.get('ni'), errors='coerce') / pd.to_numeric(df.get('at'), errors='coerce')
        
        # Log assets (with safety check)
        at_clean = pd.to_numeric(df.get('at'), errors='coerce')
        df['ln_assets'] = at_clean.where(at_clean > 0).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
        
        df['leverage'] = pd.to_numeric(df.get('dltt'), errors='coerce') / pd.to_numeric(df.get('at'), errors='coerce')
        
        # Winsorize ROA
        roa_data = df['roa'].dropna()
        if len(roa_data) > 0:
            roa_1pct = roa_data.quantile(0.01)
            roa_99pct = roa_data.quantile(0.99)
            print(f"DEBUG: ROA winsorization - 1st percentile: {roa_1pct:.4f}, 99th percentile: {roa_99pct:.4f}")
            df['roa'] = df['roa'].clip(lower=roa_1pct, upper=roa_99pct)
        
        # CEO tenure
        df = DataProcessor._compute_ceo_tenure(df)
        
        # SIC handling
        df['sic'] = pd.to_numeric(df.get('sic'), errors='coerce')
        df['sic1'] = df['sic'].astype('Int64').astype('string').str.slice(0, 1)
        
        return df
    
    @staticmethod
    def _compute_ceo_tenure(df: pd.DataFrame) -> pd.DataFrame:
        """Compute CEO tenure from becameceo_year."""
        if 'becameceo_year' not in df.columns or df['becameceo_year'].isna().all():
            df['ceo_tenure'] = pd.NA
            return df
        
        # Convert becameceo_year to numeric
        becameceo_clean = df['becameceo_year'].copy()
        becameceo_clean = becameceo_clean.replace(['<NA>', 'nan', 'NaN', '', 'None'], pd.NA)
        
        # Handle date format
        becameceo_dates = pd.to_datetime(becameceo_clean, errors='coerce')
        becameceo_numeric = becameceo_dates.dt.year
        
        # Compute tenure
        fyear_numeric = pd.to_numeric(df['fyear'], errors='coerce')
        df['ceo_tenure'] = fyear_numeric - becameceo_numeric
        df['ceo_tenure'] = df['ceo_tenure'].where(df['ceo_tenure'] >= 0, pd.NA)
        
        print(f"DEBUG: ceo_tenure computed - non-null count: {df['ceo_tenure'].notna().sum()}")
        return df
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, sp1500: pd.DataFrame, 
                     years: Tuple[int, int] = (2000, 2010)) -> pd.DataFrame:
        """Apply sample filters (years and S&P 1500 membership)."""
        print(f"DEBUG: Before filters: {len(df)} observations")
        df = df.copy()
        
        # Year filter
        df = df[(df['fyear'] >= years[0]) & (df['fyear'] <= years[1])]
        print(f"DEBUG: After year filter ({years[0]}-{years[1]}): {len(df)} observations")
        
        # S&P 1500 filter
        if not sp1500.empty:
            sp1500_fy = sp1500[['gvkey', 'fyear']].drop_duplicates().copy()
            sp1500_fy['sp1500_member'] = 1  # Use different column name to avoid conflicts
            
            # Ensure consistent data types for merging
            sp1500_fy['gvkey'] = sp1500_fy['gvkey'].astype(str)
            sp1500_fy['fyear'] = sp1500_fy['fyear'].astype(int)
            df['gvkey'] = df['gvkey'].astype(str)
            df['fyear'] = df['fyear'].astype(int)
            
            print(f"DEBUG: Before S&P 1500 filter - rows: {len(df)}")
            df = df.merge(sp1500_fy, on=['gvkey', 'fyear'], how='inner')
            print(f"DEBUG: After S&P 1500 filter - rows: {len(df)}")
            
            # Validation check
            print(f"[CHECK] After SP1500 join: rows={len(df)}, firms={df['gvkey'].nunique()}, years={df['fyear'].min()}–{df['fyear'].max()}")
        else:
            raise RuntimeError("S&P 1500 membership missing—sample must be restricted per assignment.")
        
        return df


class RegressionAnalyzer:
    """Handles regression analysis and model estimation."""
    
    def __init__(self, outdir: Path = Path("out")):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
    
    def run_firm_fixed_effects_regression(self, df: pd.DataFrame) -> Tuple[sm.regression.linear_model.RegressionResults, 
                                                                          pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Run firm fixed effects regression with within-transformation.
        
        Returns:
            Tuple of (model, used_data, X, y)
        """
        # Prepare data
        use_data = self._prepare_regression_data(df)
        
        # Within-transformation
        within_vars = self._get_within_variables(use_data)
        use_within = self._apply_within_transformation(use_data, within_vars)
        
        # Prepare regression matrices
        X, y, clusters = self._prepare_regression_matrices(use_within, within_vars)
        
        # Run regression
        model = sm.OLS(y, X, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': clusters})
        
        # Save results
        self._save_results(model, use_within)
        
        return model, use_within, X, y
    
    def _prepare_regression_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for regression analysis."""
        core_cols = ['roa', 'female_ceo', 'ln_assets', 'leverage', 'gvkey', 'fyear']
        optional_cols = ['board_size', 'nationality_mix', 'gender_ratio', 'ceo_tenure']
        
        # Check available columns
        available_core = [c for c in core_cols if c in df.columns]
        available_optional = [c for c in optional_cols if c in df.columns and df[c].notna().sum() > 0]
        
        print(f"DEBUG: Available core columns: {available_core}")
        print(f"DEBUG: Available optional columns: {available_optional}")
        
        # Drop missing core variables
        use = df.dropna(subset=available_core).copy()
        print(f"DEBUG: After dropna (core only): {use.shape}")
        
        # Fill missing optional variables
        for col in available_optional:
            if col in use.columns and use[col].notna().sum() > 0:
                unique_vals = use[col].nunique()
                if unique_vals <= 1:
                    continue
                
                if col in ['board_size', 'ceo_tenure']:
                    use[col] = use[col].fillna(use[col].median())
                elif col in ['nationality_mix', 'gender_ratio']:
                    use[col] = use[col].fillna(use[col].mean())
        
        # Type hygiene
        for col in ['female_ceo', 'ln_assets', 'leverage', 'board_size', 'nationality_mix', 'gender_ratio', 'ceo_tenure', 'roa']:
            if col in use.columns:
                use[col] = pd.to_numeric(use[col], errors='coerce')
        
        # Rename variables
        use = use.rename(columns={'ln_assets': 'size', 'gender_ratio': 'board_gender_ratio'})
        
        return use
    
    def _get_within_variables(self, df: pd.DataFrame) -> List[str]:
        """Determine which variables have within-firm variation."""
        within_vars = ['female_ceo', 'size', 'leverage']  # Core variables
        
        candidate_vars = ['board_size', 'nationality_mix', 'board_gender_ratio', 'ceo_tenure']
        for var in candidate_vars:
            if var in df.columns:
                varies = df.groupby('gvkey')[var].nunique(dropna=True).gt(1).any()
                if varies:
                    within_vars.append(var)
                    print(f"DEBUG: Including {var} (has within-firm variation)")
                else:
                    print(f"DEBUG: {var} has no within-firm variation; FE absorbs it")
        
        print(f"DEBUG: Variables for within-transformation: {within_vars}")
        return within_vars
    
    def _apply_within_transformation(self, df: pd.DataFrame, within_vars: List[str]) -> pd.DataFrame:
        """Apply within-transformation (firm fixed effects)."""
        use_within = df.copy()
        
        # Calculate firm means
        firm_means = df.groupby('gvkey')[within_vars + ['roa']].mean()
        
        # Subtract firm means
        for var in within_vars + ['roa']:
            if var in df.columns:
                use_within[f'{var}_demeaned'] = df[var] - df['gvkey'].map(firm_means[var])
        
        # Drop rows with NaN after transformation
        demeaned_vars = [f'{var}_demeaned' for var in within_vars]
        use_final = use_within.dropna(subset=demeaned_vars + ['roa_demeaned']).copy()
        
        print(f"DEBUG: After within-transformation: {use_final.shape}")
        return use_final
    
    def _prepare_regression_matrices(self, df: pd.DataFrame, within_vars: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare regression matrices."""
        demeaned_vars = [f'{var}_demeaned' for var in within_vars]
        
        X = df[demeaned_vars].copy().astype(float)
        y = df['roa_demeaned'].copy().astype(float)
        clusters = df['gvkey'].astype(str)
        
        n_obs = len(X)
        n_clusters = clusters.nunique()
        print(f"[Firm FE] N (obs) = {n_obs} | clusters (gvkey) = {n_clusters}")
        
        return X, y, clusters
    
    def _save_results(self, model: sm.regression.linear_model.RegressionResults, data: pd.DataFrame):
        """Save regression results to files."""
        # Text summary
        txt_path = self.outdir / "reg_femaleCEO_roa.txt"
        with open(txt_path, "w") as f:
            f.write(model.summary().as_text())
        
        # HTML summary
        html_path = self.outdir / "reg_femaleCEO_roa.html"
        html_content = self._create_html_summary(model, data)
        with open(html_path, "w") as f:
            f.write(html_content)
        
        print(f"✅ Results saved to {txt_path} and {html_path}")
    
    def _create_html_summary(self, model: sm.regression.linear_model.RegressionResults, 
                           data: pd.DataFrame) -> str:
        """Create HTML summary of regression results."""
        n_obs = len(data)
        n_firms = data['gvkey'].nunique()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Female CEO and ROA Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Female CEO and Return on Assets Analysis</h1>
            <p><strong>Sample:</strong> S&P 1500 firms (2000-2010)</p>
            <p><strong>Observations:</strong> {n_obs:,}</p>
            <p><strong>Firms:</strong> {n_firms:,}</p>
            <p><strong>R-squared:</strong> {model.rsquared:.3f}</p>
            <p><strong>Adjusted R-squared:</strong> {model.rsquared_adj:.3f}</p>
            
            <h2>Regression Results</h2>
            <pre>{model.summary().as_text()}</pre>
        </body>
        </html>
        """
    
    def print_data_quality_diagnostics(self, df: pd.DataFrame):
        """Print comprehensive data quality diagnostics."""
        print(f"\n🔍 DATA QUALITY DIAGNOSTICS")
        print(f"=" * 50)
        print(f"Total observations: {len(df)}")
        
        key_vars = ['roa', 'female_ceo', 'ln_assets', 'leverage', 'ceo_tenure', 
                    'board_size', 'nationality_mix', 'gender_ratio']
        
        for var in key_vars:
            if var in df.columns:
                non_null = df[var].notna().sum()
                pct = (non_null / len(df)) * 100
                print(f"{var:15}: {non_null:4d}/{len(df)} ({pct:5.1f}%)")
                
                if var == 'female_ceo' and non_null > 0:
                    female_count = (df[var] == 1).sum()
                    print(f"{'':15}  Female CEOs: {female_count} ({female_count/non_null*100:.1f}%)")
            else:
                print(f"{var:15}: NOT FOUND")
        
        # SIC1 distribution
        if 'sic1' in df.columns:
            print(f"\nSIC1 Distribution:")
            sic_counts = df['sic1'].value_counts().head(10)
            for sic, count in sic_counts.items():
                pct = (count / len(df)) * 100
                print(f"  SIC1 {sic}: {count} ({pct:.1f}%)")
        
        print(f"=" * 50)
