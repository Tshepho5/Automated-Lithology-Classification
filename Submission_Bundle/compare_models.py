"""
Multi-Model Benchmark & Comparison
Project: Automated Lithology Classification from Subsurface Well Logs

Evaluates and benchmarks multiple supervised classifiers:
1. Random Forest Classifier
2. Extra Trees Classifier
3. HistGradientBoosting Classifier
4. K-Nearest Neighbors (KNN)
5. Logistic Regression (Linear baseline)

Outputs:
- model_comparison.csv
- model_comparison.png
"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

FEATURE_COLS = ['GR_scaled', 'RHOB_scaled', 'NPHI_scaled', 'DTC_scaled', 'CALI_scaled', 'RDEP_scaled']
TARGET_COL = 'LITHOLOGY_CODE'

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

models = {
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*75)
print("BENCHMARKING MULTIPLE MACHINE LEARNING MODELS")
print("="*75)

for name, clf in models.items():
    print(f"Training {name:<25}...", end="", flush=True)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    # Quick 5-fold CV score
    cv_score = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1).mean()

    results.append({
        "Model": name,
        "Test Accuracy (%)": round(acc * 100, 2),
        "Macro-F1": round(macro_f1, 4),
        "Weighted-F1": round(weighted_f1, 4),
        "5-Fold CV Macro-F1": round(cv_score, 4),
        "Train Time (s)": round(train_time, 2)
    })
    print(f" Acc: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f} | Time: {train_time:.2f}s")

res_df = pd.DataFrame(results).sort_values(by="Macro-F1", ascending=False).reset_index(drop=True)
res_df.to_csv("model_comparison.csv", index=False)
print("\n" + "="*75)
print(res_df.to_string(index=False))
print("="*75)

# Plot Model Comparison Bar Chart
plt.figure(figsize=(10, 5))
x = np.arange(len(res_df))
width = 0.35

bars1 = plt.bar(x - width/2, res_df['Test Accuracy (%)'], width, label='Test Accuracy (%)', color='#1B365D')
bars2 = plt.bar(x + width/2, res_df['Macro-F1'] * 100, width, label='Macro-F1 Score (x100)', color='#D97706')

plt.xlabel('Supervised Classifier', fontsize=11, fontweight='bold')
plt.ylabel('Score (%)', fontsize=11, fontweight='bold')
plt.title('Multi-Model Performance Benchmark for Well Log Lithology Classification', fontsize=12, pad=12)
plt.xticks(x, res_df['Model'], rotation=15, ha='right')
plt.ylim(0, 115)
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.close()
print("Saved comparison chart to model_comparison.png and CSV to model_comparison.csv")
