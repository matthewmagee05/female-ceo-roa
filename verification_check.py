#!/usr/bin/env python3
"""
Comprehensive verification script to check potential issues in the Female CEO ROA analysis.
This script addresses the user's concerns about:
1. WRDS pull limits
2. S&P 1500 membership filtering
3. BoardEx merge (static vs year-varying)
4. CEO tenure data availability
5. ExecuComp gender variable consistency
6. Industry FE reference category
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def check_wrds_limits():
    """Check for hardcoded obs limits in the main script."""
    print("🔍 CHECKING WRDS PULL LIMITS")
    print("=" * 50)
    
    script_path = Path("female_ceo_roa.py")
    if not script_path.exists():
        print("❌ Main script not found")
        return
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find all obs= limits
    import re
    obs_limits = re.findall(r'obs=(\d+)', content)
    
    if obs_limits:
        print(f"⚠️  Found {len(obs_limits)} hardcoded obs limits:")
        for limit in set(obs_limits):
            count = obs_limits.count(limit)
            print(f"   - obs={limit}: {count} occurrence(s)")
        print("\n💡 RECOMMENDATION: Remove or increase these limits for production runs")
        print("   Current limits will restrict your sample size significantly")
    else:
        print("✅ No hardcoded obs limits found")

def check_sp1500_filtering():
    """Check S&P 1500 filtering implementation."""
    print("\n🔍 CHECKING S&P 1500 MEMBERSHIP FILTERING")
    print("=" * 50)
    
    script_path = Path("female_ceo_roa.py")
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for idxcst_his usage
    if 'idxcst_his' in content:
        print("✅ Using idxcst_his table (correct)")
        
        # Check for specific S&P 1500 filtering
        if 'sp1500' in content.lower() and 'proxy' in content.lower():
            print("⚠️  WARNING: Code uses 'all idxcst_his data as S&P 1500 proxy'")
            print("   This may include other indices beyond S&P 1500")
            print("   💡 RECOMMENDATION: Filter by specific index type if available")
        
        # Check for spindx filtering
        if 'spindx' in content:
            print("✅ Code attempts to filter by spindx")
        else:
            print("⚠️  No spindx filtering found - may include non-S&P indices")
    else:
        print("❌ Not using idxcst_his table")

def check_boardex_merge():
    """Check BoardEx merge implementation for year-varying vs static data."""
    print("\n🔍 CHECKING BOARDEX MERGE IMPLEMENTATION")
    print("=" * 50)
    
    script_path = Path("female_ceo_roa.py")
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check for aggregation by gvkey
    if 'groupby.*gvkey' in content or 'groupby(\'gvkey\')' in content:
        print("⚠️  WARNING: BoardEx data is aggregated by gvkey")
        print("   This makes board characteristics static across all firm-years")
        print("   💡 RECOMMENDATION: If assignment expects year-varying board data,")
        print("      consider merging with date ranges to preserve time variation")
    
    # Check for left merge
    if 'how=\'left\'' in content:
        print("✅ Using left merge (preserves all observations)")
    
    # Check for ticker-based linking
    if 'ticker' in content and 'boardex' in content.lower():
        print("✅ Using ticker-based BoardEx linking")

def check_ceo_tenure():
    """Check CEO tenure data availability and cleaning."""
    print("\n🔍 CHECKING CEO TENURE DATA")
    print("=" * 50)
    
    # Check if we have output data to analyze
    csv_path = Path("out/sp1500_femaleCEO_2000_2010.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            
            # Check CEO tenure availability
            if 'ceo_tenure' in df.columns:
                non_null_tenure = df['ceo_tenure'].notna().sum()
                total_obs = len(df)
                tenure_pct = (non_null_tenure / total_obs) * 100
                
                print(f"📊 CEO Tenure Statistics:")
                print(f"   - Total observations: {total_obs}")
                print(f"   - Non-null tenure: {non_null_tenure}")
                print(f"   - Coverage: {tenure_pct:.1f}%")
                
                if tenure_pct < 50:
                    print("⚠️  WARNING: Low CEO tenure coverage")
                    print("   💡 RECOMMENDATION: Check becameceo_year data quality")
                
                # Check tenure distribution
                tenure_data = df['ceo_tenure'].dropna()
                if len(tenure_data) > 0:
                    print(f"   - Mean tenure: {tenure_data.mean():.1f} years")
                    print(f"   - Median tenure: {tenure_data.median():.1f} years")
                    print(f"   - Range: {tenure_data.min():.0f} - {tenure_data.max():.0f} years")
            else:
                print("❌ CEO tenure column not found in output")
        except Exception as e:
            print(f"❌ Error reading output data: {e}")
    else:
        print("⚠️  No output data found - run analysis first to check tenure")

def check_gender_variable():
    """Check ExecuComp gender variable consistency."""
    print("\n🔍 CHECKING EXECUCOMP GENDER VARIABLE")
    print("=" * 50)
    
    script_path = Path("female_ceo_roa.py")
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check gender variable handling
    if 'gender.*str.upper' in content:
        print("✅ Gender variable properly converted to uppercase")
    
    if 'ceo_gender' in content:
        print("✅ Code handles ceo_gender as fallback")
    
    if 'ceoann' in content and 'pd.NA' in content:
        print("⚠️  WARNING: Falls back to pd.NA when only ceoann available")
        print("   This may reduce usable observations")
    
    # Check if we have output data to analyze
    csv_path = Path("out/sp1500_femaleCEO_2000_2010.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            
            if 'female_ceo' in df.columns:
                non_null_gender = df['female_ceo'].notna().sum()
                total_obs = len(df)
                gender_pct = (non_null_gender / total_obs) * 100
                
                print(f"📊 Female CEO Variable Statistics:")
                print(f"   - Total observations: {total_obs}")
                print(f"   - Non-null gender: {non_null_gender}")
                print(f"   - Coverage: {gender_pct:.1f}%")
                
                if gender_pct < 80:
                    print("⚠️  WARNING: Low gender variable coverage")
                    print("   💡 RECOMMENDATION: Check ExecuComp gender data quality")
                
                # Check female CEO count
                female_ceos = (df['female_ceo'] == 1).sum()
                print(f"   - Female CEOs: {female_ceos} ({female_ceos/non_null_gender*100:.1f}% of non-null)")
            else:
                print("❌ female_ceo column not found in output")
        except Exception as e:
            print(f"❌ Error reading output data: {e}")

def check_industry_fe():
    """Check industry fixed effects implementation."""
    print("\n🔍 CHECKING INDUSTRY FIXED EFFECTS")
    print("=" * 50)
    
    script_path = Path("female_ceo_roa.py")
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Check SIC1 implementation
    if 'sic1.*str.slice' in content:
        print("✅ SIC1 created from first digit of SIC")
    
    # Check drop_first usage
    if 'drop_first=True' in content:
        print("⚠️  WARNING: Using drop_first=True for industry dummies")
        print("   This drops the first category as reference")
        print("   💡 RECOMMENDATION: Verify which SIC1 category is dropped")
        print("      (likely SIC1=0 or blank values)")
    
    # Check if we have output data to analyze
    csv_path = Path("out/sp1500_femaleCEO_2000_2010.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            
            if 'sic1' in df.columns:
                sic1_counts = df['sic1'].value_counts()
                print(f"📊 SIC1 Distribution:")
                for sic, count in sic1_counts.head(10).items():
                    pct = (count / len(df)) * 100
                    print(f"   - SIC1 {sic}: {count} ({pct:.1f}%)")
                
                # Check for missing SIC1
                missing_sic1 = df['sic1'].isna().sum()
                if missing_sic1 > 0:
                    print(f"   - Missing SIC1: {missing_sic1} ({(missing_sic1/len(df)*100):.1f}%)")
            else:
                print("❌ sic1 column not found in output")
        except Exception as e:
            print(f"❌ Error reading output data: {e}")

def generate_recommendations():
    """Generate actionable recommendations."""
    print("\n🎯 ACTIONABLE RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = [
        "1. WRDS LIMITS: Remove or increase obs= limits for production runs",
        "2. S&P 1500: Consider filtering idxcst_his by specific index type if available",
        "3. BOARDEX: Verify if assignment expects year-varying board data",
        "4. CEO TENURE: Check becameceo_year data quality if coverage is low",
        "5. GENDER: Verify ExecuComp gender variable consistency across years",
        "6. INDUSTRY FE: Confirm which SIC1 category serves as reference"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Run this verification after each code change")
    print(f"   2. Test with full data (remove obs limits)")
    print(f"   3. Document any data limitations in your analysis")
    print(f"   4. Consider robustness checks with different specifications")

def main():
    """Run all verification checks."""
    print("🔍 FEMALE CEO ROA ANALYSIS - VERIFICATION CHECK")
    print("=" * 60)
    print("This script checks for potential issues identified by the user.")
    print("=" * 60)
    
    check_wrds_limits()
    check_sp1500_filtering()
    check_boardex_merge()
    check_ceo_tenure()
    check_gender_variable()
    check_industry_fe()
    generate_recommendations()
    
    print(f"\n✅ Verification complete!")
    print(f"Review the warnings and recommendations above.")

if __name__ == "__main__":
    main()
