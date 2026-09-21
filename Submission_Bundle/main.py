"""
Item 2: Master Pipeline Orchestrator (main.py)
Project: Automated Lithology Classification from Subsurface Well Logs

Provides a single unified entry point to run the end-to-end pipeline:
1. Data Ingestion & Cleaning (clean_data.py)
2. Train-Test Splitting & Feature Scaling (split_and_scale.py)
3. Model Training & Validation (train_and_evaluate.py)
4. Down-Hole Composite Track Visualization (plot_lithology_track.py)

Usage:
  python main.py              # Runs the complete end-to-end pipeline
  python main.py --all        # Runs the complete end-to-end pipeline
  python main.py --clean      # Runs only data ingestion & cleaning
  python main.py --split      # Runs only train-test split & scaling
  python main.py --train      # Runs only model training & evaluation
  python main.py --plot       # Runs only well log track plotting
"""

import sys
import time
import argparse
import logging

from clean_data import clean_well_data, generate_audit_summary
from split_and_scale import split_and_scale_data, save_artifacts
from train_and_evaluate import train_and_evaluate_pipeline
from plot_lithology_track import load_data_and_predict, plot_composite_log

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MasterPipeline")


def run_pipeline(
    input_file: str = "15_9-23.csv",
    cleaned_file: str = "15_9-23_cleaned.csv",
    train_file: str = "train_data.csv",
    test_file: str = "test_data.csv",
    scaler_file: str = "scaler.joblib",
    model_file: str = "random_forest_model.joblib",
    run_clean: bool = True,
    run_split: bool = True,
    run_train: bool = True,
    run_plot: bool = True
):
    start_time = time.time()
    print("\n" + "="*70)
    print("      AUTOMATED LITHOLOGY CLASSIFICATION PIPELINE (WELL 15/9-23)     ")
    print("="*70)

    # Step 1 & 2: Ingestion & Cleaning
    if run_clean:
        logger.info(">>> STEP 1 & 2: Data Ingestion & Cleaning...")
        clean_df = clean_well_data(input_csv=input_file, output_csv=cleaned_file)
        logger.info(f"Data Cleaning Complete: {len(clean_df)} records saved to {cleaned_file}.")
    else:
        logger.info("Skipping Data Cleaning (using existing cleaned file).")

    # Step 3 & 4: Splitting & Scaling
    if run_split:
        logger.info(">>> STEP 3 & 4: Stratified Splitting & StandardScaler Normalization...")
        train_df, test_df, scaler, metadata = split_and_scale_data(
            input_csv=cleaned_file,
            test_size=0.20,
            random_state=42
        )
        save_artifacts(
            train_df, test_df, scaler, metadata,
            train_csv=train_file, test_csv=test_file, scaler_pkl=scaler_file
        )
        logger.info("Splitting & Scaling Complete: Datasets and scaler saved.")
    else:
        logger.info("Skipping Splitting & Scaling.")

    # Step 5 & 6: Training & Evaluation
    if run_train:
        logger.info(">>> STEP 5 & 6: Model Training & Comprehensive Evaluation...")
        eval_report = train_and_evaluate_pipeline(
            train_csv=train_file,
            test_csv=test_file,
            n_estimators=150,
            random_state=42
        )
        logger.info("Training & Evaluation Complete: Metrics and figures saved.")
    else:
        logger.info("Skipping Model Training.")

    # Visualization: Well Log Tracks
    if run_plot:
        logger.info(">>> VISUALIZATION: Generating Down-Hole Composite Well Log Plots...")
        df_pred, code_to_name = load_data_and_predict(
            csv_file=cleaned_file,
            scaler_file=scaler_file,
            model_file=model_file
        )
        # 1. Full well profile
        plot_composite_log(
            df=df_pred,
            code_to_name=code_to_name,
            output_filename="well_log_full_profile.png",
            title="Well 15/9-23: Full Borehole Lithology Classification"
        )
        # 2. Reservoir zoom
        plot_composite_log(
            df=df_pred,
            code_to_name=code_to_name,
            top_depth=2800.0,
            bottom_depth=3200.0,
            output_filename="well_log_reservoir_section.png",
            title="Well 15/9-23: Reservoir & Heterogeneous Lithology Zoom (2800m - 3200m)"
        )
        logger.info("Visualization Complete: High-resolution track plots saved.")
    else:
        logger.info("Skipping Visualization.")

    elapsed = time.time() - start_time
    print("="*70)
    print(f"PIPELINE RUN COMPLETED IN {elapsed:.2f} SECONDS")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Pipeline Runner for Lithology Classification")
    parser.add_argument("--all", action="store_true", default=False, help="Run all pipeline steps (default)")
    parser.add_argument("--clean", action="store_true", default=False, help="Run only data cleaning")
    parser.add_argument("--split", action="store_true", default=False, help="Run only splitting & scaling")
    parser.add_argument("--train", action="store_true", default=False, help="Run only model training & evaluation")
    parser.add_argument("--plot", action="store_true", default=False, help="Run only well log track plotting")

    args = parser.parse_args()

    # If no specific step flag is provided, default to running all steps
    if not (args.clean or args.split or args.train or args.plot) or args.all:
        run_pipeline(run_clean=True, run_split=True, run_train=True, run_plot=True)
    else:
        run_pipeline(
            run_clean=args.clean,
            run_split=args.split,
            run_train=args.train,
            run_plot=args.plot
        )
