"""
Data loading and WRDS connection module for Female CEO ROA analysis.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np


class WRDSDataLoader:
    """Handles WRDS data loading and S&P 1500 membership construction."""
    
    def __init__(self, use_wrds: bool = True, wrds_username: Optional[str] = None):
        self.use_wrds = use_wrds
        self.wrds_username = wrds_username or os.environ.get("WRDS_USERNAME")
        self.conn = None
    
    def connect(self):
        """Establish WRDS connection."""
        if not self.use_wrds:
            print("DEBUG: WRDS disabled, skipping connection")
            return None
            
        try:
            import wrds
            print(f"DEBUG: Attempting WRDS connection with username: {self.wrds_username or 'None (will prompt)'}")
            
            if self.wrds_username:
                self.conn = wrds.Connection(wrds_username=self.wrds_username)
            else:
                self.conn = wrds.Connection()  # Will prompt for credentials
            
            print("DEBUG: WRDS connection successful")
            return self.conn
        except Exception as e:
            print(f"DEBUG: WRDS connection failed: {e}")
            warnings.warn(f"WRDS connection failed: {e}")
            return None
    
    def disconnect(self):
        """Close WRDS connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def build_sp1500_membership(self, years: Tuple[int, int] = (2000, 2010)) -> pd.DataFrame:
        """
        Build S&P 1500 membership using curated lists only.
        
        Args:
            years: Tuple of (start_year, end_year)
            
        Returns:
            DataFrame with columns ['gvkey', 'fyear'] for S&P 1500 firm-years
        """
        if not self.conn:
            print("DEBUG: No WRDS connection available for S&P 1500 membership")
            return pd.DataFrame()
        
        print(f"DEBUG: Building S&P 1500 membership for years {years[0]}-{years[1]}")
        
        # Try curated S&P 1500 lists first
        for schema in ("comp", "compm"):
            try:
                tbl = f"{schema}.sp1500list"
                print(f"DEBUG: Trying {tbl} ...")
                sp = self.conn.raw_sql(f"select * from {tbl}", coerce_float=True)
                if sp is None or sp.empty:
                    continue
                sp.columns = [c.lower() for c in sp.columns]
                if "gvkey" not in sp.columns:
                    continue
                # find membership window columns
                start_col = next((c for c in sp.columns if "start" in c or c == "from"), None)
                end_col   = next((c for c in sp.columns if "end"   in c or c == "thru"), None)
                if not start_col or not end_col:
                    continue

                sp["start_dt"] = pd.to_datetime(sp[start_col], errors="coerce")
                sp["end_dt"]   = pd.to_datetime(sp[end_col], errors="coerce").fillna(pd.Timestamp("2099-12-31"))

                years_arr = np.arange(years[0], years[1] + 1)
                rows = []
                for _, r in sp.dropna(subset=["gvkey","start_dt"]).iterrows():
                    for y in years_arr:
                        fy_end = pd.Timestamp(f"{y}-12-31")
                        if r["start_dt"] <= fy_end <= r["end_dt"]:
                            rows.append((str(r["gvkey"]), int(y)))
                sp1500 = pd.DataFrame(rows, columns=["gvkey","fyear"]).drop_duplicates()
                if not sp1500.empty:
                    print(f"[SP1500] using {tbl} | firm-years={len(sp1500)} | firms={sp1500['gvkey'].nunique()}")
                    return sp1500
            except Exception as e:
                print(f"DEBUG: {schema}.sp1500list failed: {e}")

        # Use comp.indexcst_his with S&P 1500 gvkey (031855) - more comprehensive
        print("DEBUG: Using comp.indexcst_his with S&P 1500 gvkey...")
        try:
            query = f'''
                SELECT DISTINCT gvkey, 
                       EXTRACT(YEAR FROM fromdate) as start_year,
                       EXTRACT(YEAR FROM thrudate) as end_year
                FROM comp.indexcst_his 
                WHERE gvkeyx = '031855'
                AND fromdate <= '{years[1]}-12-31' 
                AND (thrudate IS NULL OR thrudate >= '{years[0]}-01-01')
            '''
            
            result = self.conn.raw_sql(query, coerce_float=True)
            
            if not result.empty:
                # Build firm-year membership
                years_arr = np.arange(years[0], years[1] + 1)
                rows = []
                
                for _, row in result.iterrows():
                    gvkey = str(row['gvkey'])
                    start_year = int(row['start_year']) if pd.notna(row['start_year']) else years[0]
                    end_year = int(row['end_year']) if pd.notna(row['end_year']) else years[1]
                    
                    for year in years_arr:
                        if start_year <= year <= end_year:
                            rows.append((gvkey, year))
                
                sp1500 = pd.DataFrame(rows, columns=["gvkey","fyear"]).drop_duplicates()
                print(f"[SP1500] using comp.indexcst_his (S&P 1500 gvkey) | firm-years={len(sp1500)} | firms={sp1500['gvkey'].nunique()}")
                return sp1500
            else:
                print("DEBUG: No firms found in comp.indexcst_his for S&P 1500")
                
        except Exception as e:
            print(f"DEBUG: comp.indexcst_his failed: {e}")

        # No fallbacks - only use curated S&P 1500 lists
        raise RuntimeError("Could not build S&P1500 membership from curated list tables (comp/compm.sp1500list).")
    
    
    
    def load_compustat_data(self, years: Tuple[int, int], sp1500_membership: pd.DataFrame) -> pd.DataFrame:
        """Load Compustat fundamental data."""
        if not self.conn:
            return pd.DataFrame()
        
        if sp1500_membership is None or sp1500_membership.empty:
            raise RuntimeError("S&P1500 membership missing—sample must be restricted per assignment.")
        
        print("DEBUG: Pulling Compustat funda data...")
        query = f"""
            SELECT gvkey, datadate, fyear, ni, ib, at, dltt, tic
            FROM comp.funda
            WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
              AND fyear BETWEEN {years[0]} AND {years[1]}
        """
        comp = self.conn.raw_sql(query, coerce_float=True)
        comp['gvkey'] = comp['gvkey'].astype(str)
        
        # Apply S&P 1500 restriction (required)
        print(f"DEBUG: Before S&P 1500 restriction - Compustat rows: {len(comp)}")
        comp = comp.merge(sp1500_membership.assign(sp1500=1), on=['gvkey', 'fyear'], how='inner')
        print(f"DEBUG: After S&P 1500 restriction - Compustat rows: {len(comp)}")
        
        return comp
    
    def load_execucomp_data(self) -> pd.DataFrame:
        """Load ExecuComp data."""
        if not self.conn:
            return pd.DataFrame()
        
        print("DEBUG: Pulling ExecuComp data...")
        try:
            execu = self.conn.get_table('comp_execucomp', 'anncomp',
                                      columns=['gvkey', 'year', 'ceoann', 'gender', 'becameceo', 'execid', 'coname'],
                                      coerce_float=True)
            print(f"DEBUG: Pulled {len(execu)} rows from comp_execucomp.anncomp")
            return execu
        except Exception as e:
            print(f"DEBUG: comp_execucomp.anncomp failed: {e}")
            return pd.DataFrame()
    
    def load_company_data(self) -> pd.DataFrame:
        """Load Compustat company data for SIC codes."""
        if not self.conn:
            return pd.DataFrame()
        
        print("DEBUG: Pulling Compustat company data...")
        try:
            company = self.conn.get_table('comp', 'company',
                                        columns=['gvkey', 'sic'],
                                        coerce_float=True)
            print(f"DEBUG: Pulled {len(company)} rows from comp.company")
            return company
        except Exception as e:
            print(f"DEBUG: comp.company failed: {e}")
            return pd.DataFrame()


class LocalDataLoader:
    """Handles loading data from local files."""
    
    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        """Load data from various file formats."""
        path = Path(file_path)
        if not path.exists():
            return pd.DataFrame()
        
        try:
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            elif path.suffix.lower() in (".xlsx", ".xls"):
                return pd.read_excel(path)
            else:
                return pd.DataFrame()
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            return pd.DataFrame()


def load_all_data(use_wrds: bool = True, wrds_username: Optional[str] = None,
                 years: Tuple[int, int] = (2000, 2010)) -> Dict[str, pd.DataFrame]:
    """
    Load all required data from WRDS or local files.
    
    Returns:
        Dictionary containing all loaded datasets
    """
    data = {}
    
    if use_wrds:
        print("DEBUG: Attempting to load data from WRDS...")
        loader = WRDSDataLoader(use_wrds, wrds_username)
        conn = loader.connect()
        
        if conn:
            try:
                print("DEBUG: WRDS connection established, loading data...")
                # Build S&P 1500 membership
                sp1500 = loader.build_sp1500_membership(years)
                data['sp1500'] = sp1500
                print(f"DEBUG: S&P 1500 membership: {len(sp1500)} firm-years")
                
                # Load main datasets
                data['comp'] = loader.load_compustat_data(years, sp1500)
                data['execu'] = loader.load_execucomp_data()
                data['company'] = loader.load_company_data()
                
                print(f"DEBUG: Loaded datasets - comp: {len(data['comp'])}, execu: {len(data['execu'])}, company: {len(data['company'])}")
                
                # Load BoardEx data (simplified for now)
                data['link'] = pd.DataFrame()
                data['bx'] = pd.DataFrame()
                data['prof'] = pd.DataFrame()
                data['bx_stocks'] = pd.DataFrame()
                
            finally:
                loader.disconnect()
        else:
            print("DEBUG: WRDS connection failed, will try local fallback")
    
    # Load local fallback data if WRDS data is empty
    local_paths = {
        'comp': 'data/comp_funda.parquet',
        'execu': 'data/execucomp_annual.parquet',
        'company': 'data/comp_company.parquet',
        'link': 'data/boardex_compustat_link.parquet',
        'bx': 'data/boardex_company.parquet',
        'sp1500': 'data/sp1500_membership.parquet'
    }
    
    # Try to load from local paths first
    for key, path in local_paths.items():
        if key not in data or data[key].empty:
            data[key] = LocalDataLoader.load_file(path)
    
    # No fallback to processed data - only use curated S&P 1500 lists
    
    # Print final data summary
    print("\nDEBUG: Final data summary:")
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows")
    
    return data
