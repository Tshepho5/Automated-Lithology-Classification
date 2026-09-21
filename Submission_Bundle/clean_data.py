"""
Data Cleaning & Preprocessing Pipeline for Well Log Lithology Classification
Project: Automated Lithology Classification from Subsurface Well Logs
Dataset: FORCE 2020 Well 15/9-23 (15_9-23.csv)

Tasks Handled:
1. Ingestion of raw well log CSV.
2. Removal of 100% empty/unrecorded sensor columns (e.g., RSHA, SGR, SP, etc.).
3. Elimination of redundant/collinear columns and non-geological drilling noise.
4. Row-level deduplication (exact duplicates and depth collision checks).
5. Target lithology mapping adhering to user-defined class grouping and labels:
     30000: 'Sandstone'
     65000: 'Shale'
     65030: 'Shale'
     70000: 'Limestone'
     70032: 'Chalk'
     80000: 'Marl'
     90000: 'Tuff'
     99000: 'Coal'
6. Quality filtering for physical sensor bounds and missing core interval removal.
7. Target label encoding (integer-indexed 0 to N-1 for ML algorithms).
8. Export of cleaned dataset and detailed audit report.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DataCleaner")

# --- User-Defined Lithology Facies Mapping ---
LITHOLOGY_MAPPING = {
    30000: 'Sandstone',
    65000: 'Shale',
    65030: 'Shale',
    70000: 'Limestone',
    70032: 'Chalk',
    80000: 'Marl',
    90000: 'Tuff',
    99000: 'Coal'
}

# Core physical petrophysical features per Project Proposal & Pipeline
CORE_FEATURES = ['GR', 'RHOB', 'NPHI', 'DTC', 'CALI']

# Optional high-value geological logs (non-redundant)
EXTENDED_FEATURES = ['RDEP', 'PEF']

# Stratigraphic and spatial metadata
METADATA_COLS = ['WELL', 'DEPTH_MD', 'GROUP', 'FORMATION']

# Raw target column name
TARGET_RAW = 'FORCE_2020_LITHOFACIES_LITHOLOGY'


def audit_raw_data(df: pd.DataFrame) -> dict:
    """Inspects and logs raw data health, shape, and null counts."""
    report = {
        "raw_rows": len(df),
        "raw_cols": len(df.columns),
        "raw_columns": df.columns.tolist(),
        "all_null_cols": [col for col in df.columns if df[col].isnull().all()],
        "exact_duplicates": int(df.duplicated().sum()),
        "depth_duplicates": int(df.duplicated(subset=['DEPTH_MD']).sum()),
        "null_counts_by_col": df.isnull().sum().to_dict()
    }
    return report


def clean_well_data(
    input_csv: str = "15_9-23.csv",
    output_csv: str = "15_9-23_cleaned.csv",
    include_rdep: bool = True
) -> pd.DataFrame:
    """
    Cleans raw borehole well log data by removing redundant features,
    deduplicating rows, handling null sensor intervals, and encoding labels.
    """
    logger.info(f"Loading raw dataset from {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    raw_rows, raw_cols = df.shape
    logger.info(f"Raw shape: {raw_rows} rows x {raw_cols} columns")

    # Step 1: Deduplication of rows
    df_dedup = df.drop_duplicates()
    df_dedup = df_dedup.drop_duplicates(subset=['DEPTH_MD'])
    num_dropped_rows = raw_rows - len(df_dedup)
    if num_dropped_rows > 0:
        logger.info(f"Removed {num_dropped_rows} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")

    # Step 2: Drop 100% empty columns (redundant sensor slots)
    empty_cols = [c for c in df_dedup.columns if df_dedup[c].isnull().all()]
    logger.info(f"Dropping {len(empty_cols)} completely empty columns: {empty_cols}")
    df_clean = df_dedup.drop(columns=empty_cols).copy()

    # Step 3: Target Lithology Mapping & Validation
    logger.info("Mapping raw facies codes to user lithology definitions...")
    df_clean['LITHOLOGY'] = df_clean[TARGET_RAW].map(LITHOLOGY_MAPPING)

    unmapped = df_clean['LITHOLOGY'].isnull().sum()
    if unmapped > 0:
        unmapped_codes = df_clean[df_clean['LITHOLOGY'].isnull()][TARGET_RAW].unique()
        logger.warning(f"Found {unmapped} rows with unmapped codes: {unmapped_codes}")
        df_clean = df_clean.dropna(subset=['LITHOLOGY'])

    # Step 4: Core Physical Features Filtering (Handling Missing Intervals)
    required_features = list(CORE_FEATURES)
    if include_rdep and 'RDEP' in df_clean.columns:
        required_features.append('RDEP')
        logger.info("Including Deep Resistivity (RDEP) as standard non-redundant feature.")

    rows_before_null_filter = len(df_clean)
    df_clean = df_clean.dropna(subset=required_features).copy()
    rows_after_null_filter = len(df_clean)
    logger.info(
        f"Filtered missing core sensor records: {rows_before_null_filter} -> {rows_after_null_filter} rows "
        f"({(rows_after_null_filter / raw_rows) * 100:.2f}% data retention)"
    )

    # Step 5: Physical Bounds & Integrity Validation
    # Ensure physical limits are respected without discarding genuine geological signatures (e.g. Coal)
    valid_mask = (
        (df_clean['GR'] >= 0) & (df_clean['GR'] <= 600) &
        (df_clean['RHOB'] >= 1.0) & (df_clean['RHOB'] <= 3.8) &
        (df_clean['NPHI'] >= -0.05) & (df_clean['NPHI'] <= 1.0) &
        (df_clean['DTC'] >= 30.0) & (df_clean['DTC'] <= 250.0) &
        (df_clean['CALI'] >= 4.0) & (df_clean['CALI'] <= 26.0)
    )
    if include_rdep and 'RDEP' in df_clean.columns:
        valid_mask = valid_mask & (df_clean['RDEP'] > 0)

    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Dropping {invalid_count} physically impossible records.")
        df_clean = df_clean[valid_mask].copy()

    # Step 6: Borehole Washout Indicator (Quality Flag)
    # Washout: Caliper significantly exceeds bit size (DCAL = CALI - BS > 2.5 inches)
    if 'BS' in df_clean.columns and 'CALI' in df_clean.columns:
        df_clean['WASHOUT_FLAG'] = ((df_clean['CALI'] - df_clean['BS']) > 2.5).astype(int)
        washout_count = df_clean['WASHOUT_FLAG'].sum()
        logger.info(f"Borehole washout flag created: {washout_count} intervals marked with washout flag.")

    # Step 7: Encode Target Lithology Labels
    # Create sorted mapping for consistent, reproducible 0 to K-1 integer encoding
    unique_classes = sorted(df_clean['LITHOLOGY'].unique())
    label_to_id = {name: idx for idx, name in enumerate(unique_classes)}
    df_clean['LITHOLOGY_CODE'] = df_clean['LITHOLOGY'].map(label_to_id)
    logger.info(f"Target encoded classes ({len(unique_classes)} classes): {label_to_id}")

    # Step 8: Final Column Pruning to Eliminate Redundancy
    # Retain strictly necessary metadata, core features, washout flag, and target variables
    final_cols = [c for c in METADATA_COLS if c in df_clean.columns]
    final_cols += required_features
    if 'WASHOUT_FLAG' in df_clean.columns:
        final_cols.append('WASHOUT_FLAG')
    final_cols += [TARGET_RAW, 'LITHOLOGY', 'LITHOLOGY_CODE']

    # Sort strictly by depth index
    df_final = df_clean[final_cols].sort_values(by='DEPTH_MD').reset_index(drop=True)

    # Step 9: Save Cleaned CSV
    logger.info(f"Saving cleaned dataset to {output_csv}...")
    try:
        df_final.to_csv(output_csv, index=False)
        logger.info(f"Successfully saved {len(df_final)} rows and {len(df_final.columns)} columns to {output_csv}.")
    except PermissionError:
        fallback_csv = output_csv.replace(".csv", "_latest.csv")
        logger.warning(
            f"Notice: '{output_csv}' is currently open in another program (such as Excel). "
            f"Saved copy to '{fallback_csv}' instead."
        )
        df_final.to_csv(fallback_csv, index=False)

    return df_final


def generate_audit_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """Generates comparison summary metrics for verification."""
    summary = {
        "raw_record_count": len(df_raw),
        "cleaned_record_count": len(df_clean),
        "retention_percentage": round((len(df_clean) / len(df_raw)) * 100, 2),
        "cleaned_columns": df_clean.columns.tolist(),
        "null_counts_in_clean": df_clean.isnull().sum().to_dict(),
        "class_distribution": df_clean['LITHOLOGY'].value_counts().to_dict(),
        "class_percentages": (df_clean['LITHOLOGY'].value_counts(normalize=True) * 100).round(2).to_dict(),
        "feature_statistics": df_clean[['GR', 'RHOB', 'NPHI', 'DTC', 'CALI', 'RDEP']].describe().to_dict()
    }
    return summary


if __name__ == "__main__":
    input_file = "15_9-23.csv"
    output_file = "15_9-23_cleaned.csv"
    report_file = "cleaning_report.json"

    logger.info("=== Starting Well Log Data Cleaning Pipeline ===")
    raw_df = pd.read_csv(input_file)
    cleaned_df = clean_well_data(input_csv=input_file, output_csv=output_file)

    summary = generate_audit_summary(raw_df, cleaned_df)
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Audit report saved to {report_file}")

    print("\n" + "="*50)
    print("CLEANED DATASET CLASS DISTRIBUTION:")
    print("="*50)
    for lith, count in summary["class_distribution"].items():
        pct = summary["class_percentages"][lith]
        print(f"  {lith:<15}: {count:>5} samples ({pct:>5.2f}%)")
    print("="*50)
    print(f"Total Cleaned Samples: {len(cleaned_df)} (Retained: {summary['retention_percentage']}%)")
    print("="*50 + "\n")
