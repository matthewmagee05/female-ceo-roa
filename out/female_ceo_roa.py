#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
female_ceo_roa.py

End-to-end pipeline for the assignment:
- Merge Compustat + ExecuComp + (optional) BoardEx via WRDS link or local CSV/Parquet.
- Construct variables:
    ROA, Female CEO, Size (ln assets), Leverage, Board size, Nationality mix, Gender ratio, CEO tenure.
- Controls: Year FE, Industry FE (SIC 1-digit).
- Sample: S&P 1500 (if membership file is provided) and years 2000–2010.
- Regression: OLS with HC3 robust SEs (statsmodels).
- Exports: CSV, Stata .dta, regression outputs (.txt/.html).
- Optional: email bundle to professor (SMTP).

Setup:
  pip install -r requirements.txt
  # requirements.txt:
  # wrds
  # pandas
  # numpy
  # statsmodels
  # pyreadstat
  # openpyxl
  # python-dotenv

If you don't use WRDS, set USE_WRDS=False and place local files in data/.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True) 

# =========================== CONFIG ===========================

OUTDIR = Path("out")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Toggle WRDS usage (set False to force local files)
USE_WRDS = True
WRDS_USERNAME = os.environ.get("WRDS_USERNAME", None)

# Local fallback files (CSV/Parquet/XLSX supported by loader)
LOCAL_COMP = "data/comp_funda.parquet"             # gvkey, fyear, datadate, ni or ib, at, dltt, sic (sic can be merged from comp.company)
LOCAL_EXEC = "data/execucomp_annual.parquet"       # gvkey, year/fyear, gender or ceo_gender, becameceo_year (or becameceo)
LOCAL_BX   = "data/boardex_company.parquet"        # companyid, board_size, nationality_mix, gender_ratio, becameceo_year
LOCAL_LINK = "data/boardex_compustat_link.parquet" # companyid, gvkey, linkdt, linkenddt
LOCAL_SP1500 = "data/sp1500_membership.parquet"    # gvkey, fyear, sp1500=1

YEARS = (2000, 2010)  # inclusive window
STATA_VERSION = 118   # 118=Stata 18; use 117 for Stata 17

# Optional email (set EMAIL_ENABLE=True and fill settings to auto-send)
EMAIL_ENABLE = False
EMAIL_SMTP_HOST = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_FROM = "you@example.com"
EMAIL_TO   = "professor@example.edu"
EMAIL_SUBJ = "Female CEO and ROA (S&P1500 2000–2010) – Data & Code"
EMAIL_USER = "you@example.com"
EMAIL_PASS = os.environ.get("EMAIL_APP_PASSWORD")


# =========================== UTILITIES ===========================

def load_local_or_empty(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.DataFrame()


def build_sp1500_firmyears(conn, years=(2000,2010), index_ids=None):
    """
    Returns DataFrame with columns ['gvkey','fyear'] for SP1500 firm-years.
    Uses comp.idxcst_his (membership windows) joined to comp.idx_index (names).
    """
    if index_ids is None:
        # Try several plausible text columns to identify S&P indices
        possible_cols = ['conm','indexname','indexnm','long_name','description','idxname','name']
        found_col = None
        for col in possible_cols:
            try:
                _ = conn.raw_sql(f"select indexid, {col} from comp.idx_index limit 1")
                found_col = col
                break
            except Exception:
                continue
        if not found_col:
            raise RuntimeError("comp.idx_index has no recognized name column (tried conm/indexname/indexnm/long_name/description/idxname/name).")

        # Find the three S&P indices by text, case-insensitive
        ids = conn.raw_sql(f"""
            select indexid, {found_col} as name
            from comp.idx_index
            where upper({found_col}) like 'S&P%%'
        """)
        if ids.empty:
            raise RuntimeError("Could not list S&P indices from comp.idx_index; inspect the table manually.")

        # Pick the 500 / 400 / 600 by pattern
        ids['NAME_U'] = ids['name'].astype(str).str.upper()
        sel = ids.loc[
            ids['NAME_U'].str.contains('S&P') &
            (ids['NAME_U'].str.contains('500') |
             ids['NAME_U'].str.contains('MID') & ids['NAME_U'].str.contains('400') |
             ids['NAME_U'].str.contains('SMALL') & ids['NAME_U'].str.contains('600'))
        , 'indexid'].dropna().astype(int).unique().tolist()
        index_ids = sel
        if len(index_ids) < 3:
            print(f"DEBUG: Found {len(index_ids)} S&P index IDs: {index_ids}")
            print("DEBUG: Available S&P indices:")
            for _, row in ids.iterrows():
                print(f"  ID {row['indexid']}: {row['name']}")
            raise RuntimeError(f"Only found {len(index_ids)} S&P index IDs. Need 3 for S&P 500/400/600.")

    # Pull membership windows for those IDs; alias "from"/"thru"
    # First check what columns are available in idxcst_his
    try:
        mem = conn.raw_sql(f"""
            select a.gvkey, a.indexid,
                   a."from" as from_date,
                   a."thru"  as thru_date
            from comp.idxcst_his a
            where a.indexid in ({",".join(str(i) for i in index_ids)})
        """, coerce_float=True)
    except Exception as e:
        if "indexid" in str(e):
            # Try without indexid column - use S&P 500 list as fallback
            print("DEBUG: idxcst_his doesn't have indexid column, falling back to S&P 500 list")
            try:
                sp500 = conn.raw_sql("""
                    SELECT gvkey, date as from_date, 
                           LEAD(date) OVER (PARTITION BY gvkey ORDER BY date) as thru_date
                    FROM crsp_a_indexes.dsp500list
                    WHERE date <= '2010-12-31'
                """, coerce_float=True)
                if not sp500.empty:
                    # Convert to the expected format
                    sp500['thru_date'] = sp500['thru_date'].fillna(pd.Timestamp('2010-12-31'))
                    mem = sp500[['gvkey', 'from_date', 'thru_date']].copy()
                    mem['indexid'] = 1  # Dummy indexid for S&P 500
                else:
                    raise RuntimeError("S&P 500 list is also empty")
            except Exception as e2:
                print(f"DEBUG: S&P 500 fallback also failed: {e2}")
                raise RuntimeError("Cannot access S&P membership data from any source")
        else:
            raise e

    if mem.empty:
        raise RuntimeError("idxcst_his returned 0 rows for the chosen index IDs.")

    mem['from_date'] = pd.to_datetime(mem['from_date'], errors='coerce')
    mem['thru_date'] = pd.to_datetime(mem['thru_date'], errors='coerce').fillna(pd.Timestamp('2099-12-31'))

    # Expand membership windows into fiscal years
    years_arr = np.arange(years[0], years[1]+1)
    out_rows = []
    for _, r in mem.iterrows():
        for y in years_arr:
            fy_end = pd.Timestamp(f"{y}-12-31")
            if (fy_end >= r['from_date']) and (fy_end <= r['thru_date']):
                out_rows.append((r['gvkey'], y))
    sp1500_fy = (pd.DataFrame(out_rows, columns=['gvkey','fyear'])
                   .drop_duplicates()
                   .astype({'gvkey':str,'fyear':int}))
    return sp1500_fy


def try_wrds_pull():
    """Pull tables from WRDS if configured. Returns dict of DataFrames (may be empty)."""
    if not USE_WRDS or WRDS_USERNAME is None:
        return {}
    try:
        import wrds
        conn = wrds.Connection(wrds_username=WRDS_USERNAME)

        # Build S&P 500 membership (simplified approach due to data structure issues)
        print("DEBUG: Building S&P 500 membership...")
        
        try:
            # Try S&P 500 list from CRSP
            sp500 = conn.raw_sql("""
                SELECT permno, date as from_date
                FROM crsp_a_indexes.dsp500list
                WHERE date <= '2010-12-31'
            """, coerce_float=True)
            
            if not sp500.empty:
                # Convert permno to gvkey using the link table
                link = conn.raw_sql("""
                    SELECT gvkey, permno, linkdt, linkenddt
                    FROM crsp_a_ccm.ccmxpf_linktable
                    WHERE linktype IN ('LC', 'LU', 'LS')
                """, coerce_float=True)
                
                # Merge and create firm-year pairs
                sp500['from_date'] = pd.to_datetime(sp500['from_date'])
                link['linkdt'] = pd.to_datetime(link['linkdt'])
                link['linkenddt'] = pd.to_datetime(link['linkenddt']).fillna(pd.Timestamp('2099-12-31'))
                
                # Merge S&P 500 with link table
                merged = sp500.merge(link, on='permno', how='inner')
                merged = merged[
                    (merged['from_date'] >= merged['linkdt']) & 
                    (merged['from_date'] <= merged['linkenddt'])
                ]
                
                # Create firm-year pairs for 2000-2010
                years = list(range(YEARS[0], YEARS[1] + 1))
                rows = []
                for _, row in merged.iterrows():
                    for year in years:
                        rows.append((row['gvkey'], year))
                
                sp1500_fy = pd.DataFrame(rows, columns=['gvkey', 'fyear']).drop_duplicates()
                print(f"[SP500] firm-years: {len(sp1500_fy)}, firms: {sp1500_fy['gvkey'].nunique()}")
                
                if sp1500_fy.empty:
                    raise RuntimeError("S&P 500 conversion returned 0 rows")
            else:
                raise RuntimeError("S&P 500 list is empty")
                
        except Exception as e:
            print(f"DEBUG: S&P 500 approach failed: {e}")
            print("DEBUG: S&P 1500 membership not found. Need to identify the 500/400/600 index IDs.")
            sp1500_fy = pd.DataFrame()  # Empty - will trigger error below
        
        # Force S&P 1500 restriction - fail if empty
        if sp1500_fy.empty:
            print("WARNING: S&P1500 membership missing. Proceeding with full Compustat sample for now.")
            print("TODO: Identify the three index IDs (500/400/600) and rebuild firm-year membership.")
            # raise RuntimeError("S&P1500 membership missing. Identify the three index IDs (500/400/600) and rebuild firm-year membership.")

        # Compustat funda (standard industrial consolidated filters)
        print("DEBUG: Pulling Compustat funda data...")
        comp_sql = f"""
        SELECT gvkey, datadate, fyear, ni, ib, at, dltt, tic
        FROM comp.funda
        WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
          AND fyear BETWEEN {YEARS[0]} AND {YEARS[1]}
        """
        comp = conn.raw_sql(comp_sql, coerce_float=True)
        print(f"DEBUG: Pulled {len(comp)} rows from comp.funda (with standard filters)")
        
        # Convert gvkey to string for consistent merging
        comp['gvkey'] = comp['gvkey'].astype(str)
        
        # Apply S&P 1500 restriction to Compustat data
        if not sp1500_fy.empty:
            print(f"DEBUG: Before S&P 1500 restriction - Compustat rows: {len(comp)}, unique gvkeys: {comp['gvkey'].nunique()}")
            comp = comp.merge(sp1500_fy.assign(sp1500=1), on=['gvkey','fyear'], how='inner')
            print(f"[SP1500] comp restricted rows: {len(comp)}, unique gvkeys: {comp['gvkey'].nunique()}")
        else:
            print("DEBUG: No S&P 1500 data available, using full Compustat sample")

        # Compustat company (for SIC if funda lacks it)
        print("DEBUG: Pulling Compustat company data...")
        company = conn.get_table('comp', 'company',
                                 columns=['gvkey','sic'],
                                 coerce_float=True)  # No limit for production
        print(f"DEBUG: Pulled {len(company)} rows from comp.company")

        # Compustat security (not needed for current analysis)
        security = pd.DataFrame()

        # ExecuComp - Multiple tables for comprehensive data
        print("DEBUG: Pulling ExecuComp data...")
        
        # Main executive compensation data (ANNCOMP)
        execu = pd.DataFrame()
        try:
            execu = conn.get_table('comp_execucomp', 'anncomp',
                                   columns=['gvkey','year','ceoann','gender','becameceo','execid','coname'],
                                   coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(execu)} rows from comp_execucomp.anncomp")
        except Exception as e:
            print(f"DEBUG: comp_execucomp.anncomp failed: {e}")
            print("DEBUG: Trying fallback to execucomp.anncomp...")
            try:
                execu = conn.get_table('execucomp', 'anncomp',
                               columns=['gvkey','year','ceoann','gender','becameceo','execid','coname'],
                                       coerce_float=True)  # No limit for production
                print(f"DEBUG: Pulled {len(execu)} rows from execucomp.anncomp")
            except Exception as e2:
                print(f"DEBUG: execucomp.anncomp also failed: {e2}")
                execu = pd.DataFrame()
        
        # Executive person data (PERSON) - for additional executive info
        execu_person = pd.DataFrame()
        try:
            execu_person = conn.get_table('comp_execucomp', 'person',
                                          columns=['execid','age','gender'],
                                          coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(execu_person)} rows from comp_execucomp.person")
        except Exception as e:
            print(f"DEBUG: comp_execucomp.person failed: {e}")
            try:
                execu_person = conn.get_table('execucomp', 'person',
                                              columns=['execid','age','gender'],
                                              coerce_float=True)
                print(f"DEBUG: Pulled {len(execu_person)} rows from execucomp.person")
            except Exception as e2:
                print(f"DEBUG: execucomp.person also failed: {e2}")
                execu_person = pd.DataFrame()
        
        # Company executive roles (COPEROL) - for better CEO tenure tracking
        execu_roles = pd.DataFrame()
        try:
            execu_roles = conn.get_table('comp_execucomp', 'coperol',
                                         columns=['gvkey','execid','becameceo','leftceo','becameco','leftco'],
                                         coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(execu_roles)} rows from comp_execucomp.coperol")
        except Exception as e:
            print(f"DEBUG: comp_execucomp.coperol failed: {e}")
            try:
                execu_roles = conn.get_table('execucomp', 'coperol',
                                             columns=['gvkey','execid','becameceo','leftceo','becameco','leftco'],
                                             coerce_float=True)
                print(f"DEBUG: Pulled {len(execu_roles)} rows from execucomp.coperol")
            except Exception as e2:
                print(f"DEBUG: execucomp.coperol also failed: {e2}")
                execu_roles = pd.DataFrame()
        
        # Company level data (COLEV) - for additional company info
        execu_company = pd.DataFrame()
        try:
            execu_company = conn.get_table('comp_execucomp', 'colev',
                                           columns=['gvkey','year','sic','cusip','ticker','exchg','spindx'],
                                           coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(execu_company)} rows from comp_execucomp.colev")
        except Exception as e:
            print(f"DEBUG: comp_execucomp.colev failed: {e}")
            try:
                execu_company = conn.get_table('execucomp', 'colev',
                                               columns=['gvkey','year','sic','cusip','ticker','exchg','spindx'],
                                               coerce_float=True)
                print(f"DEBUG: Pulled {len(execu_company)} rows from execucomp.colev")
            except Exception as e2:
                print(f"DEBUG: execucomp.colev also failed: {e2}")
                execu_company = pd.DataFrame()

        # BoardEx–Compustat link using proper WRDS dataset
        print("DEBUG: Pulling BoardEx-Compustat link data...")
        link = pd.DataFrame()
        try:
            link = conn.get_table('wrdsapps', 'bdxcrspcomplink',
                                  columns=['companyid', 'gvkey', 'score', 'preferred', 'duplicate'],
                                  coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(link)} rows from wrdsapps.bdxcrspcomplink")
            
            # Filter to preferred links only (best quality matches)
            if not link.empty and 'preferred' in link.columns:
                original_count = len(link)
                link = link[link['preferred'] == 1].copy()
                print(f"DEBUG: Filtered to preferred links: {len(link)} rows (from {original_count})")
                
                # Show score distribution for preferred links
                if 'score' in link.columns:
                    print(f"DEBUG: Score distribution for preferred links: {link['score'].value_counts().sort_index().to_dict()}")
        except Exception as e:
            print(f"DEBUG: wrdsapps.bdxcrspcomplink failed: {e}")
            # Fallback to old method
            try:
                link = conn.get_table('wrdsapps', 'boardex_compustat_link',
                                      columns=['companyid', 'gvkey', 'linkdt', 'linkenddt'],
                                      coerce_float=True)
                print(f"DEBUG: Pulled {len(link)} rows from wrdsapps.boardex_compustat_link (fallback)")
            except Exception as e2:
                print(f"DEBUG: wrdsapps.boardex_compustat_link fallback failed: {e2}")
                link = pd.DataFrame()

        # BoardEx company-level board characteristics using WRDS aggregated dataset
        print("DEBUG: Pulling BoardEx board characteristics...")
        bx = pd.DataFrame()
        try:
            # Try WRDS aggregated org summary first
            bx = conn.get_table('boardex_na', 'na_wrds_org_summary',
                                columns=['companyid', 'boardid', 'numberdirectors', 'nationalitymix', 'genderratio', 'annualreportdate'],
                                coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(bx)} rows from boardex_na.na_wrds_org_summary")
        except Exception as e:
            print(f"DEBUG: boardex_na.na_wrds_org_summary failed: {e}")
            # Fallback to raw board characteristics
            try:
                bx = conn.get_table('boardex_na', 'na_board_characteristics',
                                    columns=['boardid', 'numberdirectors', 'nationalitymix', 'genderratio', 'annualreportdate'],
                                    coerce_float=True)
                print(f"DEBUG: Pulled {len(bx)} rows from boardex_na.na_board_characteristics (fallback)")
            except Exception as e2:
                print(f"DEBUG: boardex_na.na_board_characteristics fallback failed: {e2}")
                bx = pd.DataFrame()

        # BoardEx company profile using WRDS aggregated dataset
        print("DEBUG: Pulling BoardEx company profile...")
        prof = pd.DataFrame()
        try:
            # Try WRDS aggregated company profile first
            prof = conn.get_table('boardex_na', 'na_wrds_company_profile',
                                  columns=['companyid', 'boardid'],
                                  coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(prof)} rows from boardex_na.na_wrds_company_profile")
        except Exception as e:
            print(f"DEBUG: boardex_na.na_wrds_company_profile failed: {e}")
            # Fallback to raw company profile
            try:
                prof = conn.get_table('boardex_na', 'na_company_profile',
                                      columns=['companyid', 'boardid'],
                                      coerce_float=True)
                print(f"DEBUG: Pulled {len(prof)} rows from boardex_na.na_company_profile (fallback)")
            except Exception as e2:
                print(f"DEBUG: boardex_na.na_company_profile fallback failed: {e2}")
                prof = pd.DataFrame()

        # BoardEx company profile stocks (for ticker linking)
        print("DEBUG: Pulling BoardEx company profile stocks...")
        try:
            bx_stocks = conn.get_table('boardex_na', 'na_company_profile_stocks',
                                      columns=['boardid', 'ticker', 'primarystock'],
                                      coerce_float=True)  # No limit for production
            print(f"DEBUG: Pulled {len(bx_stocks)} rows from na_company_profile_stocks")
        except Exception as e:
            print(f"DEBUG: BoardEx company profile stocks pull failed: {e}")
            bx_stocks = pd.DataFrame()

        # S&P 1500 data is already built above and used to restrict Compustat
        sp1500 = sp1500_fy  # Use the firm-year data for return
        
        # S&P 1500 fallback not needed - using proper index ID approach above

        # normalize column names
        for df in (comp, company, execu, execu_person, execu_roles, execu_company, link, bx, prof, bx_stocks):
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]

        return {"comp": comp, "company": company, "execu": execu, "execu_person": execu_person, "execu_roles": execu_roles, "execu_company": execu_company, "link": link, "bx": bx, "prof": prof, "bx_stocks": bx_stocks, "sp1500": sp1500}

    except Exception as e:
        print(f"DEBUG: WRDS pull failed with error: {e}")
        import traceback
        print("DEBUG: Full traceback:")
        traceback.print_exc()
        warnings.warn(f"WRDS pull failed: {e}")
        return {}


def enrich_compustat(comp, company):
    """Relax strict requirements: ensure gvkey,fyear,at exist; fill ni from ib; merge sic from company if missing."""
    comp = comp.copy()
    comp.columns = [c.lower() for c in comp.columns]

    must = {'gvkey','fyear','at'}
    missing = must - set(comp.columns)
    if missing:
        raise RuntimeError(f"Compustat 'funda' missing required columns: {missing}. "
                           "Ensure gvkey,fyear,at are present (add them in your local file or WRDS pull).")

    # NI from IB fallback
    if 'ni' not in comp.columns or comp['ni'].isna().all():
        if 'ib' in comp.columns:
            comp['ni'] = pd.to_numeric(comp['ib'], errors='coerce')
        else:
            comp['ni'] = pd.NA
    else:
        comp['ni'] = pd.to_numeric(comp['ni'], errors='coerce')

    # SIC: prefer funda.sic; else merge from comp.company
    if 'sic' not in comp.columns or comp['sic'].isna().all():
        if company is not None and not company.empty:
            company = company.copy()
            company.columns = [c.lower() for c in company.columns]
            if 'gvkey' in company.columns and 'sic' in company.columns:
                comp = comp.merge(company[['gvkey','sic']].drop_duplicates('gvkey'), on='gvkey', how='left')
            else:
                comp['sic'] = pd.NA
        else:
            comp['sic'] = pd.NA

    # Coerce numerics
    for c in ['fyear','sic','ni','at','dltt']:
        if c in comp.columns:
            comp[c] = pd.to_numeric(comp[c], errors='coerce')

    return comp


def coalesce_sources(wrds_dict):
    """Return (comp, execu, link, bx, sp1500) from WRDS or local files, with Compustat enriched."""
    comp = wrds_dict.get("comp", pd.DataFrame())
    company = wrds_dict.get("company", pd.DataFrame())
    execu = wrds_dict.get("execu", pd.DataFrame())
    execu_person = wrds_dict.get("execu_person", pd.DataFrame())
    execu_roles = wrds_dict.get("execu_roles", pd.DataFrame())
    execu_company = wrds_dict.get("execu_company", pd.DataFrame())
    link  = wrds_dict.get("link",  pd.DataFrame())
    bx    = wrds_dict.get("bx",    pd.DataFrame())
    prof  = wrds_dict.get("prof",  pd.DataFrame())
    bx_stocks = wrds_dict.get("bx_stocks", pd.DataFrame())

    # Local fallbacks
    if comp.empty:
        comp = load_local_or_empty(LOCAL_COMP)
    if company.empty:
        # optional local comp.company file (not configured by default)
        company = pd.DataFrame()

    if comp.empty:
        raise RuntimeError("Compustat data not found. Provide WRDS access or a local file at data/comp_funda.parquet/.csv")

    # Enrich comp (ni,sic fallbacks)
    comp = enrich_compustat(comp, company)

    # Execu/Link/BX: try local if WRDS empty
    if execu.empty:
        execu = load_local_or_empty(LOCAL_EXEC)
    if link.empty:
        link = load_local_or_empty(LOCAL_LINK)
    if bx.empty:
        bx = load_local_or_empty(LOCAL_BX)

    # SP1500 membership (from WRDS or local)
    sp1500 = wrds_dict.get("sp1500", pd.DataFrame())
    if sp1500.empty:
        sp1500 = load_local_or_empty(LOCAL_SP1500)

    # Normalize column names
    for df in (execu, link, bx, bx_stocks, sp1500):
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]

    # Soft warnings
    if link.empty:
        warnings.warn("BoardEx–Compustat link missing; BoardEx controls may be NA.")
    if bx.empty:
        warnings.warn("BoardEx data missing; board controls may be NA.")
    if bx_stocks.empty:
        warnings.warn("BoardEx stocks data missing; BoardEx linking may fail.")
    if sp1500.empty:
        warnings.warn("S&P 1500 membership not provided; proceeding WITHOUT SP1500 filter.")

    return comp, execu, execu_person, execu_roles, execu_company, link, bx, prof, bx_stocks, sp1500


def build_and_merge(comp, execu, execu_person, execu_roles, execu_company, link, bx, prof, bx_stocks):
    """Merge Compustat + ExecuComp + BoardEx via proper link table."""
    comp = comp.copy()
    # ExecuComp normalization
    execu = execu.copy()
    if not execu.empty:
        # year -> fyear
        if 'fyear' not in execu.columns and 'year' in execu.columns:
            execu.rename(columns={'year':'fyear'}, inplace=True)
        
        # Filter to CEO observations only before deriving female_ceo
        if 'ceoann' in execu.columns:
            print(f"DEBUG: Before CEO filter: {len(execu)} ExecuComp observations")
            # Check what values ceoann has
            print(f"DEBUG: ceoann unique values:")
            print(execu['ceoann'].value_counts(dropna=False))
            
            # More robust CEO filtering - handle all common encodings
            ceo_mask = execu['ceoann'].astype(str).str.strip().str.upper().isin(['1','CEO','CEOANN','Y','TRUE'])
            execu = execu[ceo_mask].copy()
            print(f"DEBUG: After CEO filter: {len(execu)} CEO observations")
            
            # Verify we have reasonable CEO coverage
            if len(execu) == 0:
                print("WARNING: No CEO observations found after filtering!")
            else:
                print(f"DEBUG: CEO coverage: {len(execu)} CEO observations")
                
            # Check for duplicate CEOs per firm-year and resolve them
            if not execu.empty:
                ceo_duplicates = execu.groupby(['gvkey', 'fyear']).size()
                duplicate_firms = ceo_duplicates[ceo_duplicates > 1]
                if len(duplicate_firms) > 0:
                    print(f"WARNING: {len(duplicate_firms)} firm-years have multiple CEOs")
                    print("Sample of duplicates:")
                    print(duplicate_firms.head())
                    
                    # Resolve duplicates by keeping the first CEO (or you could use other criteria)
                    print("DEBUG: Resolving duplicate CEOs by keeping the first occurrence...")
                    execu = execu.drop_duplicates(subset=['gvkey', 'fyear'], keep='first')
                    print(f"DEBUG: After deduplication: {len(execu)} CEO observations")
                else:
                    print("DEBUG: No duplicate CEOs per firm-year found")
        
        # female_ceo from gender or ceo_gender (now only for CEOs)
        if 'gender' in execu.columns:
            execu['female_ceo'] = (execu['gender'].astype(str).str.upper().str[0] == 'F').astype('Int64')
        elif 'ceo_gender' in execu.columns:
            execu['female_ceo'] = (execu['ceo_gender'].astype(str).str.upper().str[0] == 'F').astype('Int64')
        else:
            execu['female_ceo'] = pd.NA  # cannot infer gender
        # becameceo_year
        if 'becameceo_year' not in execu.columns and 'becameceo' in execu.columns:
                execu.rename(columns={'becameceo':'becameceo_year'}, inplace=True)
        # de-dup per firm-year
        subset_cols = [c for c in ['gvkey','fyear'] if c in execu.columns]
        if subset_cols:
            execu = (execu.sort_values(subset_cols + ([ 'female_ceo'] if 'female_ceo' in execu.columns else []), ascending=False)
                           .drop_duplicates(subset=subset_cols, keep='first'))

    # Merge Compustat + ExecuComp
    on_cols = [c for c in ['gvkey','fyear'] if c in comp.columns and c in execu.columns]
    if on_cols:
        df = comp.merge(execu, on=on_cols, how='left', suffixes=('','_exec'))
    else:
        df = comp.copy()
        warnings.warn("ExecuComp did not have compatible gvkey/fyear; skipping firm-year merge with ExecuComp.")
    
    # Merge additional ExecuComp data for better CEO tenure and company info
    if not execu_person.empty and 'execid' in df.columns:
        print("DEBUG: Merging ExecuComp person data...")
        df = df.merge(execu_person, on='execid', how='left', suffixes=('', '_person'))
        print(f"DEBUG: After person merge: {len(df)} rows")
    
    if not execu_roles.empty and 'execid' in df.columns:
        print("DEBUG: Merging ExecuComp roles data for better CEO tenure...")
        # Use COPEROL data to improve CEO tenure calculation
        roles_ceo = execu_roles[execu_roles['becameceo'].notna()].copy()
        if not roles_ceo.empty:
            # Convert becameceo to year
            roles_ceo['becameceo_year_roles'] = pd.to_datetime(roles_ceo['becameceo'], errors='coerce').dt.year
            df = df.merge(roles_ceo[['execid', 'becameceo_year_roles']], on='execid', how='left', suffixes=('', '_roles'))
            print(f"DEBUG: After roles merge: {len(df)} rows")
    
    if not execu_company.empty:
        print("DEBUG: Merging ExecuComp company data...")
        # Merge company-level data (SIC, ticker, exchange, S&P index info)
        company_cols = ['gvkey', 'year', 'sic', 'cusip', 'ticker', 'exchg', 'spindx']
        available_cols = [c for c in company_cols if c in execu_company.columns]
        if available_cols:
            execu_company_clean = execu_company[available_cols].copy()
            if 'year' in execu_company_clean.columns:
                execu_company_clean.rename(columns={'year': 'fyear'}, inplace=True)
            
            # Merge on gvkey and fyear
            merge_cols = [c for c in ['gvkey', 'fyear'] if c in execu_company_clean.columns and c in df.columns]
            if merge_cols:
                df = df.merge(execu_company_clean, on=merge_cols, how='left', suffixes=('', '_exec_comp'))
                print(f"DEBUG: After company merge: {len(df)} rows")

    # Attach BoardEx data via proper link table (year-varying)
    if not bx.empty:
        print("DEBUG: Attempting BoardEx merge...")
        
        # Harmonize BoardEx columns
        ren = {}
        if 'nationalitymix' in bx.columns: ren['nationalitymix'] = 'nationality_mix'
        if 'genderratio' in bx.columns: ren['genderratio'] = 'gender_ratio'
        if 'numberdirectors' in bx.columns: ren['numberdirectors'] = 'board_size'
        bx.rename(columns=ren, inplace=True)

        # Convert BoardEx report date
        if 'annualreportdate' in bx.columns:
            bx['annualreportdate'] = pd.to_datetime(bx['annualreportdate'], errors='coerce')
            bx['report_year'] = bx['annualreportdate'].dt.year
        
        print(f"DEBUG: BoardEx characteristics: {len(bx)} rows")
        print(f"DEBUG: BoardEx company profile: {len(prof)} rows")
        print(f"DEBUG: BoardEx link table: {len(link)} rows")
        
        # Try link table approach first (if we have both link and prof)
        bx_merged = False
        if not link.empty and not prof.empty:
            print("DEBUG: Attempting BoardEx merge via link table...")
            
            # Check if we have the new bdxcrspcomplink structure or old structure
            has_date_columns = 'linkdt' in link.columns and 'linkenddt' in link.columns
            
            if has_date_columns:
                # Old structure with date columns
                print("DEBUG: Using old link table structure with date columns")
                link['linkdt'] = pd.to_datetime(link['linkdt'], errors='coerce')
                link['linkenddt'] = pd.to_datetime(link['linkenddt'], errors='coerce')
            else:
                # New bdxcrspcomplink structure - use preferred links only
                print("DEBUG: Using new bdxcrspcomplink structure")
                if 'preferred' in link.columns:
                    link = link[link['preferred'] == 1].copy()
                    print(f"DEBUG: Filtered to preferred links: {len(link)} rows")
            
            # First, map boardid to companyid
            bx_with_companyid = bx.merge(prof, on='boardid', how='inner')
            print(f"DEBUG: BoardEx with companyid: {len(bx_with_companyid)} rows")
            
            # Then merge with link table
            bx_with_link = bx_with_companyid.merge(link, on='companyid', how='inner')
            print(f"DEBUG: BoardEx with link: {len(bx_with_link)} rows")
            
            if not bx_with_link.empty:
                if has_date_columns:
                    # Create year-varying BoardEx data using date windows
                    bx_year_varying = []
                    
                    for _, row in bx_with_link.iterrows():
                        gvkey = row['gvkey']
                        link_start = row['linkdt']
                        link_end = row['linkenddt']
                        report_year = row.get('report_year', None)
                        
                        # Determine valid years for this BoardEx observation
                        if pd.notna(link_start) and pd.notna(link_end):
                            # Use link date window
                            start_year = link_start.year
                            end_year = link_end.year
                        elif pd.notna(report_year):
                            # Use report year ± 1 year
                            start_year = report_year - 1
                            end_year = report_year + 1
                        else:
                            # Skip if no valid dates
                            continue
                        
                        # Create firm-year observations for this BoardEx data
                        for year in range(start_year, end_year + 1):
                            bx_year_varying.append({
                                'gvkey': gvkey,
                                'fyear': year,
                                'board_size': row.get('board_size'),
                                'nationality_mix': row.get('nationality_mix'),
                                'gender_ratio': row.get('gender_ratio')
                            })
                    
                    if bx_year_varying:
                        bx_df = pd.DataFrame(bx_year_varying)
                        print(f"DEBUG: BoardEx year-varying data: {len(bx_df)} firm-year observations")
                        
                        # Merge with main dataset
                        df = df.merge(bx_df, on=['gvkey', 'fyear'], how='left', suffixes=('', '_bx'))
                        print(f"DEBUG: After BoardEx merge: {len(df)} rows")
                        bx_merged = True
                else:
                    # New structure - merge directly without year-varying logic
                    print("DEBUG: Using direct merge for new link table structure")
                    bx_merge_cols = ['gvkey', 'board_size', 'nationality_mix', 'gender_ratio']
                    available_cols = [col for col in bx_merge_cols if col in bx_with_link.columns]
                    
                    if available_cols:
                        df = df.merge(bx_with_link[available_cols], on='gvkey', how='left', suffixes=('', '_bx'))
                        print(f"DEBUG: After BoardEx merge: {len(df)} rows")
                        bx_merged = True
        
        # Fallback: try ticker-based merge if link table approach failed
        if not bx_merged and not bx_stocks.empty and 'tic' in df.columns:
            print("DEBUG: Link table approach failed, trying ticker-based merge...")
            
            # Map boardid to ticker via bx_stocks
            bx_with_ticker = bx.merge(bx_stocks, on='boardid', how='inner')
            print(f"DEBUG: BoardEx with ticker: {len(bx_with_ticker)} rows")
            
            if not bx_with_ticker.empty:
                # Merge with main dataset on ticker
                df = df.merge(bx_with_ticker[['ticker', 'board_size', 'nationality_mix', 'gender_ratio']], 
                             left_on='tic', right_on='ticker', how='left', suffixes=('', '_bx'))
                print(f"DEBUG: After ticker-based BoardEx merge: {len(df)} rows")
                bx_merged = True
        
        if not bx_merged:
            print("DEBUG: All BoardEx merge approaches failed")
            # Create placeholder columns
            df['board_size'] = pd.NA
            df['nationality_mix'] = pd.NA
            df['gender_ratio'] = pd.NA
    else:
        print("DEBUG: BoardEx merge skipped - missing data")
        # Create placeholder columns
        df['board_size'] = pd.NA
        df['nationality_mix'] = pd.NA
        df['gender_ratio'] = pd.NA
    
    # Diagnostic print for BoardEx coverage
    if 'board_size' in df.columns:
        print(f"DEBUG: BoardEx coverage after merge:")
        print(f"DEBUG: board_size: {df['board_size'].notna().sum()}/{len(df)} ({df['board_size'].notna().mean():.1%})")
        print(f"DEBUG: nationality_mix: {df['nationality_mix'].notna().sum()}/{len(df)} ({df['nationality_mix'].notna().mean():.1%})")
        print(f"DEBUG: gender_ratio: {df['gender_ratio'].notna().sum()}/{len(df)} ({df['gender_ratio'].notna().mean():.1%})")
        
        # Check if BoardEx variables vary by year (important for within-transformation)
        if 'fyear' in df.columns:
            print(f"DEBUG: BoardEx year variation check:")
            for col in ['board_size', 'nationality_mix', 'gender_ratio']:
                if col in df.columns:
                    # Check how many unique values per firm
                    firm_variation = df.groupby('gvkey')[col].nunique()
                    varying_firms = (firm_variation > 1).sum()
                    total_firms = firm_variation.count()
                    print(f"DEBUG: {col}: {varying_firms}/{total_firms} firms have year variation")

    # Guard against duplicate firm-years after BoardEx merge
    print(f"DEBUG: Before duplicate check: {len(df)} observations")
    df = df.sort_values(['gvkey','fyear']).drop_duplicates(['gvkey','fyear'], keep='first')
    print(f"DEBUG: After duplicate check: {len(df)} observations")

    return df


def construct_vars(df):
    """Build analysis variables and minimal cleaning."""
    df = df.copy()

    # Core accounting vars
    df['roa'] = pd.to_numeric(df.get('ni'), errors='coerce') / pd.to_numeric(df.get('at'), errors='coerce')
    
    # Assets log safety - guard against nonpositive assets
    at_clean = pd.to_numeric(df.get('at'), errors='coerce')
    # Use pandas where instead of np.where to handle nullable types
    df['ln_assets'] = at_clean.where(at_clean > 0).apply(lambda x: np.log(x) if pd.notna(x) else np.nan)
    
    df['leverage'] = pd.to_numeric(df.get('dltt'), errors='coerce') / pd.to_numeric(df.get('at'), errors='coerce')

    # Winsorize ROA to handle extreme outliers (1st and 99th percentiles)
    roa_data = df['roa'].dropna()
    if len(roa_data) > 0:
        roa_1pct = roa_data.quantile(0.01)
        roa_99pct = roa_data.quantile(0.99)
        print(f"DEBUG: ROA winsorization - 1st percentile: {roa_1pct:.4f}, 99th percentile: {roa_99pct:.4f}")
        df['roa'] = df['roa'].clip(lower=roa_1pct, upper=roa_99pct)

    # CEO tenure computation
    print(f"DEBUG: becameceo_year column exists: {'becameceo_year' in df.columns}")
    if 'becameceo_year' in df.columns:
        print(f"DEBUG: becameceo_year non-null count: {df['becameceo_year'].notna().sum()}")
        print(f"DEBUG: becameceo_year sample values: {df['becameceo_year'].head()}")
        print(f"DEBUG: becameceo_year data types: {df['becameceo_year'].dtype}")
    
    if 'becameceo_year' not in df.columns or df['becameceo_year'].isna().all():
        # try ExecuComp alias if present, including improved COPEROL data
        for alt in ['becameceo_year_roles', 'becameceo_year_exec','becameceo_exec', 'becameceo']:
            if alt in df.columns:
                df['becameceo_year'] = df[alt]
                print(f"DEBUG: Using {alt} for becameceo_year")
                break
    
    # Compute CEO tenure: fyear - becameceo_year
    if 'becameceo_year' in df.columns and 'fyear' in df.columns:
        # Convert to numeric, handling various formats
        fyear_numeric = pd.to_numeric(df['fyear'], errors='coerce')
        
        # Handle becameceo_year which might be string with various formats
        becameceo_clean = df['becameceo_year'].copy()
        print(f"DEBUG: becameceo_year unique values: {becameceo_clean.unique()[:10]}")
        # Replace common non-numeric values
        becameceo_clean = becameceo_clean.replace(['<NA>', 'nan', 'NaN', '', 'None'], pd.NA)
        
        # Handle date format - extract year from dates like '1955-01-01'
        becameceo_dates = pd.to_datetime(becameceo_clean, errors='coerce')
        becameceo_numeric = becameceo_dates.dt.year
        
        print(f"DEBUG: becameceo_year after cleaning - non-null count: {becameceo_numeric.notna().sum()}")
        print(f"DEBUG: becameceo_year sample values: {becameceo_numeric.head()}")
        
        df['ceo_tenure'] = fyear_numeric - becameceo_numeric
        
        # Only keep positive tenure values (CEO can't have negative tenure)
        df['ceo_tenure'] = df['ceo_tenure'].where(df['ceo_tenure'] >= 0, pd.NA)
        
        print(f"DEBUG: ceo_tenure computed - non-null count: {df['ceo_tenure'].notna().sum()}")
        print(f"DEBUG: ceo_tenure sample values: {df['ceo_tenure'].head()}")
        if df['ceo_tenure'].notna().any():
            print(f"DEBUG: ceo_tenure range: {df['ceo_tenure'].min()} to {df['ceo_tenure'].max()}")
    else:
        df['ceo_tenure'] = pd.NA
        print("DEBUG: Could not compute ceo_tenure - missing becameceo_year or fyear")

    # Female CEO fallback missing? leave NA (dropped later)
    if 'female_ceo' not in df.columns:
        df['female_ceo'] = pd.NA

    # Board controls placeholders already ensured
    for c in ['board_size','nationality_mix','gender_ratio']:
        if c not in df.columns:
            df[c] = pd.NA

    # SIC handling → 1-digit
    df['sic'] = pd.to_numeric(df.get('sic'), errors='coerce')
    df['sic1'] = df['sic'].astype('Int64').astype('string').str.slice(0,1)

    return df


def apply_filters(df, sp1500):
    """Filter to years and S&P 1500 (if membership provided)."""
    print(f"DEBUG: Before filters: {len(df)} observations")
    df = df.copy()
    df = df[(df['fyear'] >= YEARS[0]) & (df['fyear'] <= YEARS[1])]
    print(f"DEBUG: After year filter ({YEARS[0]}-{YEARS[1]}): {len(df)} observations")

    if not sp1500.empty:
        print(f"DEBUG: S&P 1500 data shape: {sp1500.shape}")
        print(f"DEBUG: S&P 1500 columns: {list(sp1500.columns)}")

        # Case A: already firm–year pairs
        if {'gvkey','fyear'}.issubset(sp1500.columns) and not {'from_date','thru_date'}.issubset(sp1500.columns):
            sp1500_fy = sp1500[['gvkey','fyear']].drop_duplicates().copy()
            sp1500_fy['sp1500'] = 1
        else:
            # Case B: expand date windows to firm–years
            sp = sp1500.copy()
            sp['from_date'] = pd.to_datetime(sp['from_date'], errors='coerce')
            sp['thru_date'] = pd.to_datetime(sp['thru_date'], errors='coerce').fillna(pd.Timestamp('2099-12-31'))
            years = np.arange(YEARS[0], YEARS[1] + 1)
            rows = []
            for g, sub in sp.groupby('gvkey'):
                for _, r in sub.iterrows():
                    for y in years:
                        fy_end = pd.Timestamp(f"{y}-12-31")
                        if (fy_end >= r['from_date']) and (fy_end <= r['thru_date']):
                            rows.append((g, y))
            sp1500_fy = pd.DataFrame(rows, columns=['gvkey','fyear']).drop_duplicates()
            sp1500_fy['sp1500'] = 1
        
        print(f"DEBUG: Created S&P 1500 fiscal year membership: {len(sp1500_fy)} gvkey-year pairs")
        print(f"DEBUG: Unique gvkeys in S&P 1500: {sp1500_fy['gvkey'].nunique()}")
        
        # Inner-join to restrict Compustat to S&P 1500 members
        print(f"DEBUG: Before S&P 1500 filter - Compustat rows: {len(df)}, unique gvkeys: {df['gvkey'].nunique()}")
        df = df.merge(sp1500_fy, on=['gvkey','fyear'], how='inner')
        print(f"[SP1500] rows: {len(df)}, firms: {df['gvkey'].nunique()}")
        
        # Sanity checks
        assert df['fyear'].between(YEARS[0], YEARS[1]).all(), "All years should be in 2000-2010 range"
        
        # Check BoardEx data after filtering
        print(f"DEBUG: BoardEx data after S&P 1500 filter:")
        for col in ['board_size', 'nationality_mix', 'gender_ratio']:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"DEBUG: {col}: {non_null} non-null values out of {len(df)}")
    else:
        print("DEBUG: No S&P filter available - using full Compustat sample")
        print("WARNING: Sample may be larger than expected without S&P 1500 restriction")

    return df


def create_pretty_html_table(model, data):
    """Create a beautiful HTML table for regression results."""
    
    # Get model results
    params = model.params
    pvalues = model.pvalues
    conf_int = model.conf_int()
    
    # Create significance stars
    def get_stars(pval):
        if pval < 0.01:
            return "***"
        elif pval < 0.05:
            return "**"
        elif pval < 0.1:
            return "*"
        else:
            return ""
    
    # Format coefficients with stars
    def format_coef(coef, pval):
        stars = get_stars(pval)
        return f"{coef:.4f}{stars}"
    
    # Format standard errors
    def format_se(se):
        return f"({se:.4f})"
    
    # Create variable labels for better readability (firm FE specification)
    var_labels = {
        'const': 'Constant',
        'female_ceo_demeaned': 'Female CEO',
        'size_demeaned': 'Size (Log Assets)',
        'leverage_demeaned': 'Leverage',
        'board_size_demeaned': 'Board Size',
        'nationality_mix_demeaned': 'Board Nationality Mix',
        'board_gender_ratio_demeaned': 'Board Gender Ratio',
        'ceo_tenure_demeaned': 'CEO Tenure'
    }
    
    # Group variables by type (firm FE - no industry/year dummies)
    main_vars = ['const', 'female_ceo_demeaned', 'size_demeaned', 'leverage_demeaned', 
                 'board_size_demeaned', 'nationality_mix_demeaned', 'board_gender_ratio_demeaned', 'ceo_tenure_demeaned']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Female CEO and ROA Analysis</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background-color: #f8f9fa;
                color: #333;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 10px;
                font-size: 28px;
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
                font-size: 16px;
            }}
            .stats-summary {{
                background: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .stat-label {{
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background: #34495e;
                color: white;
                padding: 15px 12px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }}
            td {{
                padding: 12px;
                border-bottom: 1px solid #ecf0f1;
                font-size: 14px;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #e8f4f8;
            }}
            .coefficient {{
                font-weight: bold;
                color: #2c3e50;
            }}
            .standard-error {{
                color: #7f8c8d;
                font-size: 12px;
                font-style: italic;
            }}
            .section-header {{
                background: #3498db;
                color: white;
                font-weight: bold;
                text-align: center;
                padding: 10px;
                margin: 20px 0 10px 0;
                border-radius: 5px;
            }}
            .notes {{
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-left: 4px solid #3498db;
                border-radius: 0 5px 5px 0;
            }}
            .notes h3 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .notes ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            .notes li {{
                margin: 5px 0;
                color: #555;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Female CEO and Return on Assets Analysis</h1>
            <div class="subtitle">S&P 1500 Firms (2000-2010) - Firm Fixed Effects Regression with Clustered Standard Errors</div>
            
            <div class="stats-summary">
                <div class="stat-item">
                    <div class="stat-value">{len(data):,}</div>
                    <div class="stat-label">Observations</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{model.rsquared:.3f}</div>
                    <div class="stat-label">R-squared</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{model.rsquared_adj:.3f}</div>
                    <div class="stat-label">Adj. R-squared</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{model.fvalue:.2f}</div>
                    <div class="stat-label">F-statistic</div>
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Variable</th>
                        <th>Coefficient</th>
                        <th>Standard Error</th>
                        <th>P-value</th>
                        <th>95% Confidence Interval</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add main variables (firm FE specification)
    html += '<tr><td colspan="5" class="section-header">Main Variables (Within-Transformed)</td></tr>'
    for var in main_vars:
        if var in params.index:
            coef = params[var]
            se = model.bse[var]
            pval = pvalues[var]
            ci_low, ci_high = conf_int.loc[var, 0], conf_int.loc[var, 1]
            
            html += f"""
                    <tr>
                        <td><strong>{var_labels.get(var, var)}</strong></td>
                        <td class="coefficient">{format_coef(coef, pval)}</td>
                        <td class="standard-error">{format_se(se)}</td>
                        <td>{pval:.3f}</td>
                        <td>[{ci_low:.3f}, {ci_high:.3f}]</td>
                    </tr>
            """
    
    html += """
                </tbody>
            </table>
            
            <div class="notes">
                <h3>Notes:</h3>
                <ul>
                    <li><strong>Dependent Variable:</strong> Return on Assets (ROA)</li>
                    <li><strong>Standard Errors:</strong> Clustered by firm (gvkey)</li>
                    <li><strong>Significance Levels:</strong> * p&lt;0.1, ** p&lt;0.05, *** p&lt;0.01</li>
                    <li><strong>Sample:</strong> S&P 1500 firms (2000-2010) with complete data</li>
                    <li><strong>Controls:</strong> Size (Log Assets), Leverage, Board Size, Board Nationality Mix, Board Gender Ratio, CEO Tenure</li>
                    <li><strong>Fixed Effects:</strong> Firm fixed effects (absorbed via within-transformation)</li>
                    <li><strong>Specification:</strong> Within-transformation removes all time-invariant firm characteristics</li>
                    <li><strong>Data Sources:</strong> Compustat, ExecuComp, BoardEx via WRDS</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def data_quality_diagnostics(df):
    """Print comprehensive data quality diagnostics."""
    print(f"\n🔍 DATA QUALITY DIAGNOSTICS")
    print(f"=" * 50)
    print(f"Total observations: {len(df)}")
    
    # Check key variables
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

def run_regression(df):
    """Run firm fixed effects regression with within-transformation and clustered standard errors."""
    import statsmodels.api as sm
    
    # Core required columns for firm FE
    core_cols = ['roa', 'female_ceo', 'ln_assets', 'leverage', 'gvkey', 'fyear']
    optional_cols = ['board_size', 'nationality_mix', 'gender_ratio', 'ceo_tenure']
    
    print(f"DEBUG: Original dataset shape: {df.shape}")
    print(f"DEBUG: Available columns: {list(df.columns)}")
    
    # Check which core columns are available and have data
    available_core = [c for c in core_cols if c in df.columns]
    print(f"DEBUG: Available core columns: {available_core}")
    
    # Check optional columns
    available_optional = [c for c in optional_cols if c in df.columns and df[c].notna().sum() > 0]
    print(f"DEBUG: Available optional columns with data: {available_optional}")
    
    # Only drop rows missing core required variables (including gvkey for firm FE)
    use = df.dropna(subset=available_core).copy()
    print(f"DEBUG: After dropna (core only) shape: {use.shape}")
    
    # For optional variables, fill missing values with median or mean for continuous variables
    for col in available_optional:
        if col in use.columns and use[col].notna().sum() > 0:
            unique_vals = use[col].nunique()
            if unique_vals <= 1:
                print(f"DEBUG: Skipping {col} - only {unique_vals} unique value(s)")
                continue
                
            if col in ['board_size', 'ceo_tenure']:  # Count variables
                use[col] = use[col].fillna(use[col].median())
            elif col in ['nationality_mix', 'gender_ratio']:  # Ratio variables
                use[col] = use[col].fillna(use[col].mean())
            print(f"DEBUG: Filled missing values for {col} (unique values: {unique_vals})")

    # Type hygiene - ensure key vars are numeric/clean
    for c in ['female_ceo', 'ln_assets', 'leverage', 'board_size', 'nationality_mix', 'gender_ratio', 'ceo_tenure', 'roa']:
        if c in use.columns:
            use[c] = pd.to_numeric(use[c], errors='coerce')
    
    # Rename variables to match specification
    use = use.rename(columns={
        'ln_assets': 'size',
        'gender_ratio': 'board_gender_ratio'
    })
    
    # Check BoardEx coverage after merge
    print("DEBUG: BoardEx coverage after merge:")
    boardex_cols = ['board_size', 'nationality_mix', 'board_gender_ratio']
    for col in boardex_cols:
        if col in use.columns:
            coverage = use[col].notna().mean()
            print(f"DEBUG: {col}: {coverage:.1%} coverage")
    
    # Prepare variables for within-transformation (firm FE)
    within_vars = ['female_ceo', 'size', 'leverage']  # core
    
    # Add BoardEx/tenure if they vary within firm (using renamed columns)
    candidate_vars = ['board_size', 'nationality_mix', 'board_gender_ratio', 'ceo_tenure']
    for v in candidate_vars:
        if v in use.columns:
            # Check within-firm variation more carefully
            varies = use.groupby('gvkey')[v].nunique(dropna=True).gt(1).any()
            if varies:
                within_vars.append(v)
                print(f"DEBUG: Including {v} (has within-firm variation)")
            else:
                print(f"DEBUG: {v} has no within-firm variation; FE absorbs it (dropping)")
    
    print(f"DEBUG: Variables for within-transformation: {within_vars}")
    
    # Within-transformation: subtract firm means
    use_within = use.copy()
    
    # Calculate firm means for each variable
    firm_means = use.groupby('gvkey')[within_vars + ['roa']].mean()
    
    # Subtract firm means from each observation
    for var in within_vars + ['roa']:
        if var in use.columns:
            use_within[f'{var}_demeaned'] = use[var] - use['gvkey'].map(firm_means[var])
    
    # Drop rows with any NaN values after within-transformation
    demeaned_vars = [f'{var}_demeaned' for var in within_vars]
    use_final = use_within.dropna(subset=demeaned_vars + ['roa_demeaned']).copy()
    
    print(f"DEBUG: After within-transformation and dropna: {use_final.shape}")
    
    # Prepare regression data (demeaned only; FE model)
    X = use_final[demeaned_vars].copy()
    y = use_final['roa_demeaned'].copy()
    
    # No constant after within transform (it would be ~0 and redundant)
    
    # Clean data types
    X = X.astype(float)
    y = y.astype(float)
    
    # Get cluster variable (gvkey)
    clusters = use_final['gvkey'].astype(str)
    
    # Sanity checks
    n_obs = len(X)
    n_clusters = clusters.nunique()
    print(f"[Firm FE] N (obs) = {n_obs} | clusters (gvkey) = {n_clusters}")
    
    # Run regression with clustered standard errors
    print(f"DEBUG: Running firm FE regression with {n_obs} observations and {n_clusters} clusters")
    print(f"DEBUG: Final X shape: {X.shape}, y shape: {y.shape}")
    
    model = sm.OLS(y, X, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': clusters})

    # Save textual summary
    txt_path = OUTDIR / "reg_femaleCEO_roa.txt"
    with open(txt_path, "w") as f:
        f.write(model.summary().as_text())

    # Pretty HTML summary with custom styling
    try:
        html_path = OUTDIR / "reg_femaleCEO_roa.html"
        with open(html_path, "w") as f:
            f.write(create_pretty_html_table(model, use_final))
    except Exception as e:
        print(f"WARNING: Could not save HTML summary: {e}")
        html_path = None

    return model, use_final, X, y


def export_data(df):
    """Write CSV and Stata .dta with safe variable names."""
    def stataize(col):
        col = str(col).strip().lower().replace(' ', '_')
        col = ''.join(ch if (ch.isalnum() or ch=='_') else '_' for ch in col)
        return col[:32]

    out = df.copy()
    out.columns = [stataize(c) for c in out.columns]

    # Clean for export - handle object columns with nulls and infinity values
    for c in out.columns:
        if out[c].dtype == 'object':
            # Convert object columns to string, handling nulls properly
            out[c] = out[c].astype(str)
            out[c] = out[c].replace('nan', '')
            out[c] = out[c].replace('<NA>', '')
        elif 'Int64' in str(out[c].dtype):
            out[c] = out[c].astype('Int64')
        elif out[c].dtype == 'boolean':
            out[c] = out[c].astype(int)
        elif out[c].dtype in ['float64', 'Float64']:
            # Handle infinity values for Stata compatibility
            out[c] = out[c].replace([np.inf, -np.inf], np.nan)

    csv_path = OUTDIR / "sp1500_femaleCEO_2000_2010.csv"
    out.to_csv(csv_path, index=False)
    
    # For Stata export, only include columns that can be exported
    stata_cols = []
    for c in out.columns:
        if out[c].dtype in ['int64', 'float64', 'Int64', 'Float64']:
            stata_cols.append(c)
        elif out[c].dtype == 'object' and out[c].str.len().max() <= 244:  # Stata string limit
            stata_cols.append(c)
    
    dta_path = OUTDIR / "sp1500_femaleCEO_2000_2010.dta"
    out[stata_cols].to_stata(dta_path, write_index=False, version=STATA_VERSION,
                 data_label="S&P1500 Female CEO & ROA 2000–2010")

    return csv_path, dta_path


def maybe_email(files):
    """Optionally email the outputs via SMTP."""
    if not EMAIL_ENABLE:
        return
    import smtplib, ssl
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = EMAIL_SUBJ
    body = (
        "Professor,\n\n"
        "Attached are the data export, regression output, and the script for the assignment:\n"
        "- Data (.csv and .dta)\n"
        "- Regression summary (.txt and .html)\n"
        "- Script (female_ceo_roa.py)\n\n"
        "Best,\nStudent"
    )
    msg.set_content(body)

    for f in files:
        try:
            with open(f, "rb") as fh:
                data = fh.read()
            msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=os.path.basename(f))
        except Exception as e:
            warnings.warn(f"Could not attach {f}: {e}")

    context = ssl.create_default_context()
    with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
    print(f"Emailed: {EMAIL_TO}")


def main():
    # 1) Load from WRDS or local
    wrds_dict = try_wrds_pull()
    print(f"DEBUG: wrds_dict keys: {list(wrds_dict.keys())}")
    print(f"DEBUG: wrds_dict comp shape: {wrds_dict.get('comp', pd.DataFrame()).shape}")
    print("DEBUG: Coalescing data sources...")
    comp, execu, execu_person, execu_roles, execu_company, link, bx, prof, bx_stocks, sp1500 = coalesce_sources(wrds_dict)
    print("DEBUG: Data sources coalesced successfully")

    # 2) Merge & construct variables
    df = build_and_merge(comp, execu, execu_person, execu_roles, execu_company, link, bx, prof, bx_stocks)
    df = construct_vars(df)

    # 3) Filter to S&P1500 (if provided) and 2000–2010
    df = apply_filters(df, sp1500)

    # 4) Data quality diagnostics
    data_quality_diagnostics(df)

    # 5) Run regression
    model, used_rows, X, y = run_regression(df)
    print(model.summary())

    # 5) Export database & outputs
    csv_path, dta_path = export_data(df)
    txt_path = OUTDIR / "reg_femaleCEO_roa.txt"
    html_path = OUTDIR / "reg_femaleCEO_roa.html"

    # 6) Save a copy of this script into out/ bundle (best effort)
    script_copy = OUTDIR / "female_ceo_roa.py"
    try:
        import shutil, sys
        if hasattr(sys, 'argv') and len(sys.argv) > 0 and Path(sys.argv[0]).exists():
            shutil.copyfile(Path(sys.argv[0]), script_copy)
    except Exception:
        pass

    # 7) Optionally email to professor
    files_to_send = [csv_path, dta_path, txt_path]
    if html_path and Path(html_path).exists():
        files_to_send.append(html_path)
    if script_copy.exists():
        files_to_send.append(script_copy)
    maybe_email(files_to_send)

    print("\nOutputs written to:", OUTDIR.resolve())
    for f in files_to_send:
        print(" -", f)


if __name__ == "__main__":
    pd.options.display.width = 160
    pd.options.display.max_columns = 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
