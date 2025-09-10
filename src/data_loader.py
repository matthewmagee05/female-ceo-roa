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

        # Try comp.index_members for S&P 500, 400, 600 constituents with robust name matching
        print("DEBUG: Trying comp.index_members for S&P 500+400+600 with robust name matching...")
        try:
            query = f'''
                SELECT DISTINCT gvkey,
                       fromdate,
                       thrudate
                FROM comp.index_members
                WHERE upper(indexname) SIMILAR TO
                      '(S&P.*500)|(S&P.*MID.*400)|(S&P.*SMALL.*600)'
                  AND fromdate <= '{years[1]}-12-31'
                  AND (thrudate IS NULL OR thrudate >= '{years[0]}-01-01')
            '''
            
            result = self.conn.raw_sql(query, coerce_float=True)
            
            if not result.empty:
                # Expand by fiscal year (FY end on Dec 31 is fine for 2000–2010)
                result['fromdate'] = pd.to_datetime(result['fromdate'], errors='coerce')
                result['thrudate'] = pd.to_datetime(result['thrudate'], errors='coerce').fillna(pd.Timestamp('2099-12-31'))
                years_arr = np.arange(years[0], years[1] + 1)
                rows = []
                
                for _, r in result.iterrows():
                    for y in years_arr:
                        fy_end = pd.Timestamp(f"{y}-12-31")
                        if r['fromdate'] <= fy_end <= r['thrudate']:
                            rows.append((str(r['gvkey']), int(y)))
                
                sp1500 = pd.DataFrame(rows, columns=["gvkey","fyear"]).drop_duplicates()
                print(f"[SP1500] using comp.index_members (robust names) | firm-years={len(sp1500)} | firms={sp1500['gvkey'].nunique()}")
                return sp1500
            else:
                print("DEBUG: No firms found in comp.index_members for S&P 500+400+600")
                
        except Exception as e:
            print(f"DEBUG: comp.index_members failed: {e}")
        
        # Fallback: Use robust S&P 1500 builder from comp.idxcst_his
        print("DEBUG: Using robust S&P 1500 builder from comp.idxcst_his...")
        try:
            sp1500 = self._build_sp1500_firmyears_from_idxcst(years)
            return sp1500
        except Exception as e:
            print(f"DEBUG: Robust S&P 1500 builder failed: {e}")
        
        # No more fallbacks - only use curated S&P 1500 lists
        raise RuntimeError("Could not build S&P1500 membership from curated list tables (comp/compm.sp1500list).")
    
    def _build_sp1500_firmyears_from_idxcst(self, years: Tuple[int, int] = (2000, 2010)) -> pd.DataFrame:
        """
        Robust SP1500 firm-year builder using comp.idxcst_his and index gvkeys.
        Works even when indexid/indexname tables aren't available.
        Returns columns: gvkey (str), fyear (int)
        """
        SP_IDS = ['031855','000003','000011','000012']  # SP1500, SP500, SP400, SP600
        
        # 1) Pull a tiny sample to detect column names
        samp = self.conn.raw_sql("select * from comp.idxcst_his limit 1", coerce_float=True)
        cols = {c.lower(): c for c in samp.columns}

        # Detect index key and date columns
        idx_key = None
        for cand in ['indexgvkey', 'gvkeyx', 'index_gvkey', 'igvkey', 'idx_gvkey']:
            if cand in cols:
                idx_key = cols[cand]
                break
        if not idx_key:
            raise RuntimeError("comp.idxcst_his has no recognizable index key (tried indexgvkey/gvkeyx/index_gvkey/igvkey/idx_gvkey).")

        from_col = cols.get('from') or cols.get('fromdate')
        thru_col = cols.get('thru') or cols.get('thrudate')
        if not from_col or not thru_col:
            # Load with explicit aliases to be safe
            q = f'''
                select gvkey,
                       {idx_key} as idxkey,
                       "from" as from_date,
                       "thru"  as thru_date
                from comp.idxcst_his
                where {idx_key} in ({",".join("'" + i + "'" for i in SP_IDS)})
            '''
            mem = self.conn.raw_sql(q, coerce_float=True)
            from_name, thru_name = 'from_date', 'thru_date'
        else:
            q = f'''
                select gvkey,
                       {idx_key} as idxkey,
                       "{from_col}" as from_date,
                       "{thru_col}" as thru_date
                from comp.idxcst_his
                where {idx_key} in ({",".join("'" + i + "'" for i in SP_IDS)})
            '''
            mem = self.conn.raw_sql(q, coerce_float=True)
            from_name, thru_name = 'from_date', 'thru_date'

        if mem is None or mem.empty:
            raise RuntimeError("idxcst_his returned 0 rows for S&P index gvkeys; check schema or permissions.")

        # 2) Clean & expand to firm-years
        mem['gvkey'] = mem['gvkey'].astype(str)
        mem[from_name] = pd.to_datetime(mem[from_name], errors='coerce')
        mem[thru_name] = pd.to_datetime(mem[thru_name], errors='coerce').fillna(pd.Timestamp('2099-12-31'))
        mem = mem.dropna(subset=['gvkey', from_name])

        y0, y1 = years
        years_arr = np.arange(y0, y1 + 1)
        rows = []
        for _, r in mem.iterrows():
            for y in years_arr:
                fy_end = pd.Timestamp(f"{y}-12-31")
                if r[from_name] <= fy_end <= r[thru_name]:
                    rows.append((r['gvkey'], int(y)))
        sp1500_fy = pd.DataFrame(rows, columns=['gvkey', 'fyear']).drop_duplicates()

        # 3) Sanity prints
        print(f"[SP1500] firm-years={len(sp1500_fy)} | firms={sp1500_fy['gvkey'].nunique()} | years {sp1500_fy['fyear'].min()}–{sp1500_fy['fyear'].max()}")
        if sp1500_fy.empty:
            raise RuntimeError("Expanded SP1500 firm-year table is empty; check date columns and index gvkeys.")

        return sp1500_fy
    
    
    
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
            try:
                execu = self.conn.get_table('execucomp', 'anncomp',
                                          columns=['gvkey', 'year', 'ceoann', 'gender', 'becameceo', 'execid', 'coname'],
                                          coerce_float=True)
                print(f"DEBUG: Pulled {len(execu)} rows from execucomp.anncomp")
                return execu
            except Exception as e2:
                print(f"DEBUG: execucomp.anncomp also failed: {e2}")
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
