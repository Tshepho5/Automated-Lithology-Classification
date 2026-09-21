========================================================================
SCOA032 - Subsurface Characterization & Advanced Analytics
Final Project: Automated Lithology Classification from Well Logs
========================================================================

Project Deliverables Included:
1. Final_Project_Report.docx  : Formatted academic report with embedded figures
2. Project_Report.md         : Markdown version of project report
3. Lithology_Classification_Pipeline.ipynb : Pre-executed Jupyter Notebook
4. main.py                   : Master pipeline orchestrator (python main.py)
5. predict.py                : Production inference tool (batch and single-point)
6. clean_data.py             : Data cleaning and facies mapping
7. split_and_scale.py        : Stratified 80/20 splitting & StandardScaler
8. train_and_evaluate.py     : Random Forest training & evaluation
9. plot_lithology_track.py   : Down-hole 6-track composite well log plotter
10. compare_models.py        : Multi-model benchmark script
11. 15_9-23_cleaned.csv      : Cleaned, deduplicated well log dataset (0 nulls)
12. random_forest_model.joblib : Trained classifier (96.46% Accuracy, 0.8600 Macro-F1)
13. scaler.joblib            : Fitted StandardScaler object

Quick Start Commands:
---------------------
1. Run the entire end-to-end pipeline:
   python main.py

2. Predict lithology on a new well log CSV:
   python predict.py --input test_data.csv --output my_predictions.csv

3. Run single-point sensor prediction:
   python predict.py --gr 35.0 --rhob 2.25 --nphi 0.15 --dtc 72.0 --cali 12.2 --rdep 8.5

4. Benchmark multiple ML classifiers:
   python compare_models.py
========================================================================
