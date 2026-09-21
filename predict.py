"""
Item 3: Lithology Prediction & Inference Tool (predict.py)
Project: Automated Lithology Classification from Subsurface Well Logs

Loads trained scaler and Random Forest model to predict rock lithology
from raw well log measurements.

Supports:
1. Batch CSV inference:
   python predict.py --input test_data.csv --output my_predictions.csv
2. Single-point sensor query:
   python predict.py --gr 120.5 --rhob 2.15 --nphi 0.44 --dtc 150.0 --cali 12.8 --rdep 0.85
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd

# Class index to lithology name mapping
CODE_TO_LITHOLOGY = {
    0: 'Chalk',
    1: 'Coal',
    2: 'Limestone',
    3: 'Marl',
    4: 'Sandstone',
    5: 'Shale',
    6: 'Tuff'
}

FEATURES_RAW = ['GR', 'RHOB', 'NPHI', 'DTC', 'CALI', 'RDEP']
FEATURES_SCALED = [f"{f}_scaled" for f in FEATURES_RAW]


class LithologyPredictor:
    """Wrapper for lithology model inference with automated scaling."""

    def __init__(
        self,
        model_path: str = "random_forest_model.joblib",
        scaler_path: str = "scaler.joblib"
    ):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Model or Scaler artifact missing. Run main.py first.")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_single(
        self,
        gr: float,
        rhob: float,
        nphi: float,
        dtc: float,
        cali: float,
        rdep: float
    ) -> dict:
        """Predicts lithology for a single sensor depth reading."""
        raw_df = pd.DataFrame([[gr, rhob, nphi, dtc, cali, rdep]], columns=FEATURES_RAW)
        scaled_arr = self.scaler.transform(raw_df)
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURES_SCALED)

        pred_code = int(self.model.predict(scaled_df)[0])
        probabilities = self.model.predict_proba(scaled_df)[0]
        confidence = float(np.max(probabilities))

        class_probs = {
            CODE_TO_LITHOLOGY.get(i, f"Class_{i}"): round(float(p), 4)
            for i, p in enumerate(probabilities)
        }

        return {
            "predicted_code": pred_code,
            "predicted_lithology": CODE_TO_LITHOLOGY.get(pred_code, "Unknown"),
            "confidence": round(confidence, 4),
            "class_probabilities": class_probs
        }

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch predicts lithology for a pandas DataFrame."""
        for feat in FEATURES_RAW:
            if feat not in df.columns:
                raise KeyError(f"Missing required feature column '{feat}' in input DataFrame.")

        # Clean missing
        clean_mask = df[FEATURES_RAW].notnull().all(axis=1)
        res_df = df.copy()

        scaled_arr = self.scaler.transform(res_df[FEATURES_RAW])
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURES_SCALED)

        preds = self.model.predict(scaled_df)
        probs = self.model.predict_proba(scaled_df)

        res_df['PRED_LITHOLOGY_CODE'] = preds
        res_df['PRED_LITHOLOGY'] = [CODE_TO_LITHOLOGY.get(c, "Unknown") for c in preds]
        res_df['PRED_CONFIDENCE'] = np.max(probs, axis=1).round(4)

        return res_df


def main():
    parser = argparse.ArgumentParser(description="Lithology Classifier Inference Tool")
    parser.add_argument("--input", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--output", type=str, default="predicted_output.csv", help="Path to output CSV")

    # Single-point parameters
    parser.add_argument("--gr", type=float, help="Gamma Ray (API)")
    parser.add_argument("--rhob", type=float, help="Bulk Density (g/cm³)")
    parser.add_argument("--nphi", type=float, help="Neutron Porosity (v/v)")
    parser.add_argument("--dtc", type=float, help="Sonic Slowness (µs/ft)")
    parser.add_argument("--cali", type=float, help="Caliper (in)")
    parser.add_argument("--rdep", type=float, help="Deep Resistivity (ohm.m)")

    args = parser.parse_args()
    predictor = LithologyPredictor()

    if args.input:
        print(f"Loading input file: {args.input}...")
        df_in = pd.read_csv(args.input)
        df_out = predictor.predict_dataframe(df_in)
        df_out.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output} ({len(df_out)} rows processed).")
        print("\nPredicted Class Breakdown:")
        print(df_out['PRED_LITHOLOGY'].value_counts())
    elif None not in [args.gr, args.rhob, args.nphi, args.dtc, args.cali, args.rdep]:
        res = predictor.predict_single(
            gr=args.gr,
            rhob=args.rhob,
            nphi=args.nphi,
            dtc=args.dtc,
            cali=args.cali,
            rdep=args.rdep
        )
        print("\n" + "="*50)
        print("SINGLE SENSOR LOG INFERENCE RESULT:")
        print("="*50)
        print(f"  Input: GR={args.gr}, RHOB={args.rhob}, NPHI={args.nphi}, DTC={args.dtc}, CALI={args.cali}, RDEP={args.rdep}")
        print(f"  Predicted Lithology : {res['predicted_lithology']}")
        print(f"  Confidence Score    : {res['confidence']*100:.2f}%")
        print("\nClass Probabilities:")
        for lith, prob in sorted(res['class_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lith:<15}: {prob*100:>5.2f}%")
        print("="*50 + "\n")
    else:
        print("Interactive Test Demo (Example Sandstone / Reservoir log):")
        # Clean Sandstone example: Low GR (35), Low DTC (70), Normal RHOB (2.25), NPHI (0.15), RDEP (8.0)
        demo = predictor.predict_single(gr=35.0, rhob=2.25, nphi=0.15, dtc=72.0, cali=12.2, rdep=8.5)
        print(f"Sample Reading -> Predicted: {demo['predicted_lithology']} (Confidence: {demo['confidence']*100:.2f}%)")
        print("\nTip: Pass --input <file.csv> or pass sensor flags (--gr, --rhob, --nphi, --dtc, --cali, --rdep)")


if __name__ == "__main__":
    main()
