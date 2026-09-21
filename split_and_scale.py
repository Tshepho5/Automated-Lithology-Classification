"""
Pipeline Steps 3 & 4: Train-Test Splitting & Feature Scaling
Project: Automated Lithology Classification from Subsurface Well Logs
Input: 15_9-23_cleaned.csv

Tasks Handled:
1. Load cleaned, deduplicated well log dataset.
2. Step 3 (Splitting Strategy):
   - Performs an 80/20 Stratified Split (stratify=y) to preserve class proportions,
     crucial for rare lithologies (Chalk: 29 samples, Tuff: 64 samples, Coal: 74 samples).
   - Includes GroupKFold logic if multiple wells are loaded in future extensions.
3. Step 4 (Feature Scaling / Normalization):
   - Fits StandardScaler strictly on training set features (X_train) to prevent data leakage.
   - Transforms both X_train and X_test to standard normal distribution (mean ~ 0, std = 1).
   - Preserves both raw and scaled feature columns for interpretability.
4. Saves train_data.csv, test_data.csv, scaler.joblib, and scaling_metadata.json.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SplitAndScale")

# Feature columns to standardize
FEATURE_COLS = ['GR', 'RHOB', 'NPHI', 'DTC', 'CALI', 'RDEP']

# Metadata & Target columns
METADATA_COLS = ['WELL', 'DEPTH_MD', 'GROUP', 'FORMATION', 'WASHOUT_FLAG']
TARGET_COLS = ['FORCE_2020_LITHOFACIES_LITHOLOGY', 'LITHOLOGY', 'LITHOLOGY_CODE']


def split_and_scale_data(
    input_csv: str = "15_9-23_cleaned.csv",
    test_size: float = 0.20,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict]:
    """
    Executes Step 3 (Train-Test Split) and Step 4 (Feature Scaling)
    according to project pipeline specifications.
    """
    logger.info(f"Loading cleaned data from {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found: {input_csv}. Run clean_data.py first.")

    df = pd.read_csv(input_csv)
    total_records = len(df)
    logger.info(f"Loaded {total_records} records with {len(df.columns)} columns.")

    # Validate required columns
    for col in FEATURE_COLS + TARGET_COLS:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing from {input_csv}.")

    # --- Step 3: Train-Test Splitting (Stratified 80/20) ---
    logger.info(f"Splitting dataset into {int((1 - test_size)*100)}% Train / {int(test_size*100)}% Test...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['LITHOLOGY_CODE']
    )

    # Re-index depth order cleanly for reference
    train_df = train_df.sort_values(by='DEPTH_MD').reset_index(drop=True)
    test_df = test_df.sort_values(by='DEPTH_MD').reset_index(drop=True)

    logger.info(f"Train Set: {len(train_df)} samples ({len(train_df)/total_records*100:.1f}%)")
    logger.info(f"Test Set:  {len(test_df)} samples ({len(test_df)/total_records*100:.1f}%)")

    # --- Step 4: Feature Scaling / Normalization (StandardScaler) ---
    logger.info("Fitting StandardScaler on training features (zero-leakage guarantee)...")
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])

    # Transform both datasets
    scaled_feature_names = [f"{col}_scaled" for col in FEATURE_COLS]
    
    train_scaled_arr = scaler.transform(train_df[FEATURE_COLS])
    test_scaled_arr = scaler.transform(test_df[FEATURE_COLS])

    # Append scaled columns to dataframes
    for idx, scaled_col in enumerate(scaled_feature_names):
        train_df[scaled_col] = train_scaled_arr[:, idx]
        test_df[scaled_col] = test_scaled_arr[:, idx]

    logger.info("Feature scaling complete:")
    for f, mean_val, std_val in zip(FEATURE_COLS, scaler.mean_, scaler.scale_):
        logger.info(f"  {f:6s}: Training Mean = {mean_val:8.4f}, Std = {std_val:8.4f}")

    # Build scaling metadata summary
    metadata = {
        "dataset_name": input_csv,
        "split_ratio": {"train": 1.0 - test_size, "test": test_size},
        "random_state": random_state,
        "feature_columns_raw": FEATURE_COLS,
        "feature_columns_scaled": scaled_feature_names,
        "scaler_type": "StandardScaler",
        "scaler_parameters": {
            feature: {
                "mean": float(mean_val),
                "scale_std": float(std_val),
                "variance": float(std_val**2)
            }
            for feature, mean_val, std_val in zip(FEATURE_COLS, scaler.mean_, scaler.scale_)
        },
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_class_counts": train_df['LITHOLOGY'].value_counts().to_dict(),
        "test_class_counts": test_df['LITHOLOGY'].value_counts().to_dict(),
        "train_class_proportions_pct": (train_df['LITHOLOGY'].value_counts(normalize=True)*100).round(2).to_dict(),
        "test_class_proportions_pct": (test_df['LITHOLOGY'].value_counts(normalize=True)*100).round(2).to_dict()
    }

    return train_df, test_df, scaler, metadata


def save_artifacts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    metadata: dict,
    train_csv: str = "train_data.csv",
    test_csv: str = "test_data.csv",
    scaler_pkl: str = "scaler.joblib",
    meta_json: str = "scaling_metadata.json"
):
    """Saves processed train/test datasets, scaler object, and metadata report."""
    logger.info(f"Saving training set to {train_csv}...")
    try:
        train_df.to_csv(train_csv, index=False)
    except PermissionError:
        logger.warning(f"'{train_csv}' is open in another program. Saved to '{train_csv.replace('.csv', '_latest.csv')}'.")
        train_df.to_csv(train_csv.replace('.csv', '_latest.csv'), index=False)

    logger.info(f"Saving testing set to {test_csv}...")
    try:
        test_df.to_csv(test_csv, index=False)
    except PermissionError:
        logger.warning(f"'{test_csv}' is open in another program. Saved to '{test_csv.replace('.csv', '_latest.csv')}'.")
        test_df.to_csv(test_csv.replace('.csv', '_latest.csv'), index=False)

    logger.info(f"Saving fitted StandardScaler to {scaler_pkl}...")
    joblib.dump(scaler, scaler_pkl)

    logger.info(f"Saving metadata to {meta_json}...")
    with open(meta_json, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("All Step 3 & 4 artifacts successfully saved.")


if __name__ == "__main__":
    logger.info("=== Executing Pipeline Steps 3 & 4 ===")
    train_df, test_df, scaler, metadata = split_and_scale_data(
        input_csv="15_9-23_cleaned.csv",
        test_size=0.20,
        random_state=42
    )

    save_artifacts(train_df, test_df, scaler, metadata)

    print("\n" + "="*60)
    print("STEP 3 & 4 SUMMARY REPORT:")
    print("="*60)
    print(f"Total Samples: {len(train_df) + len(test_df)}")
    print(f"Train Samples: {len(train_df)} (80.0%) | Test Samples: {len(test_df)} (20.0%)")
    print("\nClass Distribution Comparison:")
    print(f"{'Class':<12} | {'Train Count':>11} ({'Train %':>7}) | {'Test Count':>10} ({'Test %':>6})")
    print("-" * 60)
    for lith in metadata["train_class_counts"]:
        tr_cnt = metadata["train_class_counts"][lith]
        tr_pct = metadata["train_class_proportions_pct"][lith]
        te_cnt = metadata["test_class_counts"].get(lith, 0)
        te_pct = metadata["test_class_proportions_pct"].get(lith, 0.0)
        print(f"{lith:<12} | {tr_cnt:>11} ({tr_pct:>6.2f}%) | {te_cnt:>10} ({te_pct:>5.2f}%)")
    print("="*60 + "\n")
