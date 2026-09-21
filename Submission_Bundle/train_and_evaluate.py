"""
Pipeline Steps 5 & 6: Model Training & Evaluation
Project: Automated Lithology Classification from Subsurface Well Logs
Input: train_data.csv, test_data.csv

Tasks Handled:
1. Load scaled training and testing datasets.
2. Step 5 (Model Training):
   - Trains a high-performance supervised Random Forest Classifier
     as specified in the project proposal and pipeline.
   - Evaluates 5-fold cross-validation performance on the training set.
3. Step 6 (Evaluation & Validation):
   - Evaluates predictions on the unseen test set.
   - Computes Macro-F1 Score, Weighted-F1 Score, Overall Accuracy, Precision, and Recall.
   - Generates and plots the Confusion Matrix (normalized and counts).
   - Extracts and plots Petrophysical Feature Importances.
4. Saves:
   - random_forest_model.joblib (Trained model file)
   - evaluation_report.json (Complete evaluation metrics)
   - confusion_matrix.png (High-resolution heatmap)
   - feature_importance.png (Feature ranking plot)
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TrainAndEvaluate")

FEATURE_COLS = ['GR_scaled', 'RHOB_scaled', 'NPHI_scaled', 'DTC_scaled', 'CALI_scaled', 'RDEP_scaled']
RAW_FEATURE_LABELS = {
    'GR_scaled': 'Gamma Ray (GR)',
    'RHOB_scaled': 'Bulk Density (RHOB)',
    'NPHI_scaled': 'Neutron Porosity (NPHI)',
    'DTC_scaled': 'Sonic Slowness (DTC)',
    'CALI_scaled': 'Caliper (CALI)',
    'RDEP_scaled': 'Deep Resistivity (RDEP)'
}
TARGET_COL = 'LITHOLOGY_CODE'


def train_and_evaluate_pipeline(
    train_csv: str = "train_data.csv",
    test_csv: str = "test_data.csv",
    n_estimators: int = 150,
    random_state: int = 42
):
    """Executes model training, cross-validation, and comprehensive evaluation."""
    logger.info("=== Starting Step 5: Model Training ===")
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError("Training or testing dataset missing. Run split_and_scale.py first.")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # Map code to string label
    class_mapping = (
        train_df[['LITHOLOGY_CODE', 'LITHOLOGY']]
        .drop_duplicates()
        .sort_values(by='LITHOLOGY_CODE')
    )
    class_names = class_mapping['LITHOLOGY'].tolist()
    class_codes = class_mapping['LITHOLOGY_CODE'].tolist()

    logger.info(f"Features: {FEATURE_COLS}")
    logger.info(f"Target classes ({len(class_names)}): {class_names}")

    # 1. Initialize and Cross-Validate Baseline Random Forest
    logger.info(f"Initializing RandomForestClassifier (trees={n_estimators}, seed={random_state})...")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    logger.info("Evaluating 5-Fold Stratified Cross-Validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_macro_f1 = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
    logger.info(f"5-Fold CV Macro-F1: {cv_macro_f1.mean():.4f} (+/- {cv_macro_f1.std():.4f})")

    # 2. Fit model on full training set
    logger.info("Fitting model on complete training dataset (8,709 records)...")
    rf_model.fit(X_train, y_train)

    # 3. Step 6: Evaluation on Unseen Test Set
    logger.info("=== Starting Step 6: Evaluation & Validation ===")
    y_pred = rf_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

    logger.info(f"Test Accuracy:    {acc * 100:.2f}%")
    logger.info(f"Test Macro-F1:    {macro_f1:.4f}")
    logger.info(f"Test Weighted-F1: {weighted_f1:.4f}")

    # Classification report dict
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_codes,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=class_codes,
        target_names=class_names,
        zero_division=0
    )

    # Feature importances
    importances = rf_model.feature_importances_
    feat_imp_dict = {
        RAW_FEATURE_LABELS.get(f, f): float(imp)
        for f, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_codes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Save model
    model_path = "random_forest_model.joblib"
    joblib.dump(rf_model, model_path)
    logger.info(f"Saved trained model to {model_path}")

    # Save Evaluation Report JSON
    eval_report = {
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "random_state": random_state
        },
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "cv_macro_f1_scores_5fold": cv_macro_f1.tolist(),
        "cv_macro_f1_mean": float(cv_macro_f1.mean()),
        "cv_macro_f1_std": float(cv_macro_f1.std()),
        "test_metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec)
        },
        "per_class_metrics": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1-score": report_dict[name]["f1-score"],
                "support": report_dict[name]["support"]
            }
            for name in class_names
        },
        "feature_importances": feat_imp_dict,
        "confusion_matrix": cm.tolist()
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved evaluation report to evaluation_report.json")

    # --- Plotting Visualizations ---
    # 1. Confusion Matrix Plot
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"Confusion Matrix - Well Log Lithology Classification\nTest Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}", fontsize=12, pad=15)
    plt.xlabel("Predicted Lithology", fontsize=11)
    plt.ylabel("True Lithology", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    logger.info("Saved confusion matrix heatmap to confusion_matrix.png")

    # 2. Feature Importance Plot
    plt.figure(figsize=(8, 5))
    feat_names = list(feat_imp_dict.keys())[::-1]
    feat_vals = [feat_imp_dict[k] for k in feat_names]

    bars = plt.barh(feat_names, feat_vals, color="#2b5c8f", edgecolor="black", height=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va='center', fontsize=10)

    plt.title("Petrophysical Feature Importances (Random Forest)", fontsize=12, pad=15)
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.xlim(0, max(feat_vals) * 1.18)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.close()
    logger.info("Saved feature importance chart to feature_importance.png")

    # Display console summary
    print("\n" + "="*65)
    print("STEP 5 & 6: MODEL EVALUATION SUMMARY")
    print("="*65)
    print(f"Overall Test Accuracy : {acc*100:.2f}%")
    print(f"Test Macro-F1 Score   : {macro_f1:.4f}")
    print(f"Test Weighted-F1 Score: {weighted_f1:.4f}")
    print(f"5-Fold CV Macro-F1    : {cv_macro_f1.mean():.4f} (+/- {cv_macro_f1.std():.4f})")
    print("\nDetailed Classification Report:")
    print("-" * 65)
    print(report_text)
    print("="*65)
    print("Petrophysical Feature Importances:")
    print("-" * 65)
    for f, imp in feat_imp_dict.items():
        print(f"  {f:<28}: {imp*100:>5.2f}%")
    print("="*65 + "\n")

    return eval_report


if __name__ == "__main__":
    train_and_evaluate_pipeline()
