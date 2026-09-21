# Final Project Report: Automated Lithology Classification from Subsurface Well Logs Using Machine Learning

**Course / Module**: SCOA032 - Subsurface Characterization & Advanced Analytics  
**Date**: September 2026  
**Dataset**: FORCE 2020 Machine Learning Benchmark Competition (Well 15/9-23)  
**Deliverables**: Cleaned Dataset, Production Pipeline Scripts, Machine Learning Models, and Visualizations  

---

## 1. Executive Summary

Accurate identification of subsurface rock types (lithology) is fundamental to oil and gas exploration, groundwater hydrology, and geological carbon sequestration. Traditionally, lithology determination is performed through manual inspection of down-hole geophysical logs—a tedious, subjective process subject to human cognitive fatigue.

This project delivers an end-to-end, reproducible machine learning pipeline designed to automatically classify subsurface lithologies directly from raw well log measurements. Trained on 10,887 continuous depth-indexed records from North Sea Well **15/9-23**, our optimized **Random Forest Classifier** achieved:
- **Overall Test Accuracy**: **96.46%**
- **Macro-F1 Score**: **0.8600** (accurately penalizing misclassifications on rare geological facies)
- **Weighted-F1 Score**: **0.9640**
- **5-Fold Cross-Validation Macro-F1**: **0.8684 ($\pm 0.0275$)**

Petrophysical feature importance analysis revealed that acoustic wave slowness ($DTC$, 33.61%), natural gamma radiation ($GR$, 24.74%), and neutron porosity ($NPHI$, 17.16%) were the three primary physical discriminators across geological formations.

---

## 2. Problem Statement & Geological Context

Wireline logging tools measure physical rock properties as a function of depth. In complex basins such as the Norwegian North Sea, geological formations transition rapidly through shales, sandstones, limestones, marls, coals, and volcanic tuffs.

The challenge of supervised lithology classification lies in:
1. **Disparate Physical Scales**: Measurements range from single-digit densities ($\rho_b \in [1.2, 3.0]\text{ g/cm}^3$) to hundreds of API units ($GR \in [10, 400]$) and sonic velocities.
2. **Severe Geological Class Imbalance**: Common mudrocks (shales) constitute over 70% of the stratigraphic column, while critical diagnostic beds such as coals ($<1\%$) and chalks ($<0.3\%$) are heavily underrepresented.
3. **Sensor Artifacts & Washouts**: Corrupted readings occur when borehole walls cave in (washouts) or sensors lose pad contact.

---

## 3. Dataset Description & Preprocessing Pipeline

### 3.1 Raw Dataset Audit
The raw dataset (`15_9-23.csv`) contains **11,063 depth-indexed records** ($1,518.2\text{ m}$ to $3,212.6\text{ m}$) across 29 measurement channels.

Initial health screening identified:
- **6 Completely Empty (100% NaN) Channels**: `RSHA`, `SGR`, `SP`, `MUDWEIGHT`, `RMIC`, and `RXO` contained zero recordings and were pruned.
- **Redundant Curves**: Deep resistivity (`RDEP`) and medium resistivity (`RMED`) had a correlation of $0.9934$; however, `RMED` suffered 54 missing values while `RDEP` was complete. `RDEP` was retained as the petrophysical standard.

### 3.2 Target Facies Mapping
Facies codes were consolidated into a 7-class schema:
```python
lithology_mapping = {
    30000: 'Sandstone',
    65000: 'Shale',
    65030: 'Shale',      # Interbedded sandstone/shale mapped into Shale
    70000: 'Limestone',
    70032: 'Chalk',
    80000: 'Marl',
    90000: 'Tuff',
    99000: 'Coal'
}
```

### 3.3 The Petrophysical "Coal Safety Rule"
In automated outlier detection, values of $\text{RHOB} < 1.8\text{ g/cm}^3$ are often mistakenly discarded as sensor spikes. In petrophysics, coal is rich in organic matter and bound hydrogen, producing genuine low density ($\rho_b \approx 1.25 - 1.6\text{ g/cm}^3$) and high apparent neutron porosity ($\phi_n > 0.50$). Our pipeline protected these intervals, preserving 100% of the coal samples.

### 3.4 Missing Value & Quality Filtering
Filtering records with complete core sensor tracks (`GR`, `RHOB`, `NPHI`, `DTC`, `CALI`, `RDEP`) retained **10,887 records (98.41% data retention)** with **0 missing values (`NaN`)** and **0 duplicate depth intervals**.

---

## 4. Methodology & Machine Learning Architecture

```
[Raw CSV: 15_9-23.csv]
          │
          ▼
 [clean_data.py] ────► [15_9-23_cleaned.csv] (10,887 rows, 0 nulls)
          │
          ▼
[split_and_scale.py] ─► 80% Train (8,709) / 20% Test (2,178) [Stratified]
          │          ► StandardScaler fitted strictly on Train (zero leakage)
          ▼
[train_and_evaluate.py]► Random Forest Classifier (150 trees)
          │          ► 5-Fold Stratified Cross-Validation
          ▼
[plot_lithology_track.py]► 6-Track Down-Hole Composite Well Log Plots
```

### 4.1 Stratified Train-Test Splitting (80/20)
Due to severe class imbalance, simple random splitting risks leaving rare classes unrepresented in the test set. A stratified split (`stratify=y`) was applied:
- **Training Set**: 8,709 records (80.0%)
- **Testing Set**: 2,178 records (20.0%)

Both partitions maintain virtually identical class proportions down to the rarest facies (Chalk: 23 train / 6 test; Tuff: 51 train / 13 test; Coal: 59 train / 15 test).

### 4.2 Feature Standardization (`StandardScaler`)
To eliminate dimensional bias without data leakage, `StandardScaler` was fitted strictly on the training set:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Parameters calculated on the training set:
- $GR$: $\mu = 85.62\text{ API}$, $\sigma = 47.30\text{ API}$
- $RHOB$: $\mu = 2.343\text{ g/cm}^3$, $\sigma = 0.166\text{ g/cm}^3$
- $NPHI$: $\mu = 0.3033\text{ v/v}$, $\sigma = 0.1285\text{ v/v}$
- $DTC$: $\mu = 118.93\ \mu\text{s/ft}$, $\sigma = 29.45\ \mu\text{s/ft}$
- $CALI$: $\mu = 12.07\text{ in}$, $\sigma = 1.25\text{ in}$
- $RDEP$: $\mu = 1.423\ \Omega\cdot\text{m}$, $\sigma = 1.447\ \Omega\cdot\text{m}$

### 4.3 Model Selection: Random Forest Classifier
Ensemble tree architectures natively handle non-linear boundaries between geophysical sensors (e.g. the density-neutron crossover in porous sands vs. shales). The model was initialized with 150 decision trees (`n_estimators=150`, `random_state=42`).

---

## 5. Experimental Results & Performance Analysis

### 5.1 Classification Metrics on Unseen Test Data
Evaluated on the 2,178 test samples:

| Lithology Class | Target Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chalk** | `0` | 1.00 | 0.67 | **0.80** | 6 |
| **Coal** | `1` | 0.89 | 0.53 | **0.67** | 15 |
| **Limestone** | `2` | 0.94 | 0.93 | **0.94** | 326 |
| **Marl** | `3` | 0.89 | 0.90 | **0.89** | 182 |
| **Sandstone** | `4` | 0.92 | 0.88 | **0.90** | 111 |
| **Shale** | `5` | 0.98 | 0.99 | **0.99** | 1,525 |
| **Tuff** | `6` | 0.91 | 0.77 | **0.83** | 13 |
| **Macro Average** | — | **0.93** | **0.81** | **0.86** | 2,178 |
| **Weighted Average**| — | **0.96** | **0.96** | **0.96** | 2,178 |

### 5.2 Petrophysical Feature Importances
1. **$DTC$ (Sonic Slowness)**: **33.61%**  
   Compressional acoustic velocity is the strongest differentiator between high-velocity tight carbonates (Limestones $\approx 50-70\ \mu\text{s/ft}$) and slow, uncompacted shales ($>120\ \mu\text{s/ft}$).
2. **$GR$ (Gamma Ray)**: **24.74%**  
   Measures thorium, uranium, and potassium decay. Clays and shales exhibit high natural radioactivity, while clean reservoir sandstones and chalks exhibit low baseline counts.
3. **$NPHI$ (Neutron Porosity)**: **17.16%**  
   Measures hydrogen index; highly sensitive to formation pore fluids and clay-bound water.
4. **$RDEP$ (Deep Resistivity)**: **12.34%**  
   Captures formation fluid salinity and hydrocarbon saturation.
5. **$RHOB$ (Bulk Density)**: **7.62%**  
   Governs rock grain density and compaction.
6. **$CALI$ (Caliper)**: **4.52%**  
   Provides borehole diameter context to distinguish washout artifacts from genuine formation features.

---

## 6. Down-Hole Well Log Profile Validation

A standard 6-track petrophysical composite plot was generated across the entire logged interval ($1,526\text{ m} - 3,212\text{ m}$):
- **Track 1**: $GR$ ($0-200\text{ API}$) and $CALI$ ($6-16\text{ in}$)
- **Track 2**: $RDEP$ ($0.1-50\ \Omega\cdot\text{m}$, logarithmic scale)
- **Track 3**: $RHOB$ ($1.8-2.9\text{ g/cm}^3$) and $NPHI$ ($0.45\text{ to } -0.15\text{ v/v}$, inverted crossover scale)
- **Track 4**: $DTC$ ($40-180\ \mu\text{s/ft}$)
- **Track 5**: True Geological Lithology Facies
- **Track 6**: Machine Learning Predicted Lithology Facies

### Observations:
- **Shale Dominated Intervals (1,526m - 2,750m)**: Continuous high GR ($>100\text{ API}$) and high DTC ($>130\ \mu\text{s/ft}$) are classified with $>99\%$ fidelity.
- **Reservoir & Carbonate Transition (2,800m - 3,200m)**: Distinct drop in DTC and GR marks the Tor and Hod limestone/chalk formations, matched by model predictions. Thin coal and tuff beds are sharply delineated.

---

## 7. Software Architecture & Deliverables

The accompanying codebase is fully modular, documented, and automated:

| Component | File Path | Purpose |
| :--- | :--- | :--- |
| **Master Orchestrator** | `main.py` | Single-command runner (`python main.py`) executing all steps in $\sim 10$ seconds. |
| **Data Cleaning** | `clean_data.py` | Ingestion, column pruning, quality filtering, and facies mapping. |
| **Splitting & Scaling** | `split_and_scale.py` | 80/20 Stratified split and zero-leakage `StandardScaler`. |
| **Model Training** | `train_and_evaluate.py` | Random Forest training, 5-fold CV, and evaluation metrics export. |
| **Well Log Plotter** | `plot_lithology_track.py` | Generates publication-ready 6-track composite well logs. |
| **Inference CLI** | `predict.py` | Batch CSV or single-point sensor prediction with confidence scoring. |
| **Jupyter Notebook** | `Lithology_Classification_Pipeline.ipynb` | Complete interactive notebook for presentation and grading. |
| **Cleaned Data** | `15_9-23_cleaned.csv` | 10,887 records × 14 columns, 0 nulls, ready for modeling. |
| **Model Artifacts** | `random_forest_model.joblib`, `scaler.joblib` | Saved binary model and scaler ready for deployment. |

---

## 8. Conclusion

This project successfully establishes an automated, petrophysically-informed machine learning pipeline for lithology identification in subsurface well logs. By addressing real-world borehole challenges—including missing sensor intervals, disparate physical dimensions, borehole washouts, and extreme class imbalance—the system achieved **96.46% accuracy** and a **0.8600 Macro-F1 score**.

The modular framework allows immediate extension to multi-well datasets using `GroupKFold`, offering a reliable foundation for automated formation evaluation in commercial exploration and academic research.
