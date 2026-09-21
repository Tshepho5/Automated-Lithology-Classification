"""
Interactive Streamlit UI Demonstration for Subsurface Lithology Classification
Project: Automated Lithology Classification from Subsurface Well Logs
Course: SCOA032 - Subsurface Characterization & Advanced Analytics
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure Streamlit page
st.set_page_config(
    page_title="Lithology AI - Well Log Facies Classifier",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color theme for lithologies
LITHOLOGY_COLORS = {
    'Shale': '#708090',        # Slate Gray
    'Limestone': '#4169E1',    # Royal Blue
    'Marl': '#BC8F8F',         # Rosy Brown
    'Sandstone': '#FFD700',    # Gold / Yellow
    'Coal': '#1C1C1C',         # Charcoal
    'Tuff': '#9370DB',         # Medium Purple
    'Chalk': '#00CED1'         # Dark Turquoise
}

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

@st.cache_resource
def load_model_and_scaler():
    """Loads and caches model and scaler."""
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_model_and_scaler()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/geology.png", width=64)
    st.title("Well Log AI")
    st.markdown("**Course**: SCOA032  \n**Dataset**: FORCE 2020 (Well 15/9-23)")
    st.markdown("---")

    st.subheader("Model Performance")
    st.metric("Test Accuracy", "96.46%")
    st.metric("Macro-F1 Score", "0.8600")
    st.metric("5-Fold CV Score", "0.8684")

    st.markdown("---")
    st.markdown("### Facies Color Key")
    for name, color in LITHOLOGY_COLORS.items():
        st.markdown(
            f'<div style="display: flex; align-items: center; margin-bottom: 4px;">'
            f'<div style="width: 14px; height: 14px; background-color: {color}; border-radius: 3px; margin-right: 8px;"></div>'
            f'<span style="font-size: 13px;">{name}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

# ----------------- MAIN INTERFACE -----------------
st.title("🪨 Automated Lithology Facies Classifier")
st.caption("A Supervised Machine Learning Pipeline for Real-Time Borehole Formation Evaluation")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Sensor Predictor",
    "📁 Batch Well CSV Inference",
    "📊 Down-Hole Well Log Profile",
    "📈 Model Diagnostics & Explainability"
])

# ==================== TAB 1: LIVE PREDICTOR ====================
with tab1:
    st.subheader("Interactive Sensor Query & Lithology Prediction")
    st.write("Adjust the continuous sensor readings below or select a geological preset to test the classifier in real-time.")

    # Geological Presets
    st.markdown("##### Quick Geological Presets:")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    
    # Initialize session state for inputs
    if "gr_val" not in st.session_state:
        st.session_state.gr_val = 35.0
        st.session_state.rhob_val = 2.25
        st.session_state.nphi_val = 0.15
        st.session_state.dtc_val = 72.0
        st.session_state.cali_val = 12.2
        st.session_state.rdep_val = 8.5

    with col_p1:
        if st.button("🟡 Sandstone Reservoir"):
            st.session_state.gr_val = 35.0
            st.session_state.rhob_val = 2.25
            st.session_state.nphi_val = 0.15
            st.session_state.dtc_val = 72.0
            st.session_state.cali_val = 12.2
            st.session_state.rdep_val = 8.5
            st.rerun()

    with col_p2:
        if st.button("⚪ Tight Limestone"):
            st.session_state.gr_val = 22.0
            st.session_state.rhob_val = 2.68
            st.session_state.nphi_val = 0.08
            st.session_state.dtc_val = 54.0
            st.session_state.cali_val = 12.0
            st.session_state.rdep_val = 18.0
            st.rerun()

    with col_p3:
        if st.button("🟤 Marine Shale"):
            st.session_state.gr_val = 135.0
            st.session_state.rhob_val = 2.38
            st.session_state.nphi_val = 0.42
            st.session_state.dtc_val = 145.0
            st.session_state.cali_val = 12.5
            st.session_state.rdep_val = 1.1
            st.rerun()

    with col_p4:
        if st.button("⚫ Coal Seam"):
            st.session_state.gr_val = 30.0
            st.session_state.rhob_val = 1.35
            st.session_state.nphi_val = 0.62
            st.session_state.dtc_val = 155.0
            st.session_state.cali_val = 9.1
            st.session_state.rdep_val = 4.2
            st.rerun()

    with col_p5:
        if st.button("🟣 Volcanic Tuff"):
            st.session_state.gr_val = 60.0
            st.session_state.rhob_val = 2.36
            st.session_state.nphi_val = 0.28
            st.session_state.dtc_val = 104.0
            st.session_state.cali_val = 12.0
            st.session_state.rdep_val = 2.5
            st.rerun()

    st.markdown("---")
    col_in1, col_in2 = st.columns([1, 1.2])

    with col_in1:
        st.markdown("#### Input Sensor Measurements")
        gr = st.slider("Gamma Ray - GR (API)", 0.0, 300.0, float(st.session_state.gr_val), step=0.5, help="Natural radioactivity: High in shales, low in clean sands/carbonates")
        rhob = st.slider("Bulk Density - RHOB (g/cm³)", 1.0, 3.2, float(st.session_state.rhob_val), step=0.01, help="Matrix & fluid density: Coal ~1.3, Sandstone ~2.3, Limestone ~2.7")
        nphi = st.slider("Neutron Porosity - NPHI (v/v)", -0.05, 0.80, float(st.session_state.nphi_val), step=0.01, help="Hydrogen concentration: High in shales & coals, low in tight rocks")
        dtc = st.slider("Sonic Slowness - DTC (µs/ft)", 40.0, 200.0, float(st.session_state.dtc_val), step=0.5, help="Acoustic wave transit time: Low in tight carbonates, high in soft shales")
        cali = st.slider("Borehole Caliper - CALI (in)", 6.0, 20.0, float(st.session_state.cali_val), step=0.1, help="Borehole diameter")
        rdep = st.slider("Deep Resistivity - RDEP (ohm.m)", 0.1, 50.0, float(st.session_state.rdep_val), step=0.1, help="Formation electrical resistivity")

    with col_in2:
        st.markdown("#### Prediction Output")
        
        # Prepare inputs
        input_raw = pd.DataFrame([[gr, rhob, nphi, dtc, cali, rdep]], columns=FEATURES_RAW)
        scaled_arr = scaler.transform(input_raw)
        scaled_df = pd.DataFrame(scaled_arr, columns=FEATURES_SCALED)

        pred_code = int(model.predict(scaled_df)[0])
        pred_facies = CODE_TO_LITHOLOGY[pred_code]
        probs = model.predict_proba(scaled_df)[0]
        confidence = float(np.max(probs))
        facies_color = LITHOLOGY_COLORS.get(pred_facies, '#333333')

        # Highlight Card
        st.markdown(
            f"""
            <div style="background-color: {facies_color}20; border: 2px solid {facies_color}; border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #555; font-size: 14px;">CLASSIFIED LITHOLOGY FACIES</h4>
                <h1 style="margin: 5px 0; color: {facies_color}; font-size: 32px; font-weight: bold;">{pred_facies}</h1>
                <p style="margin: 0; font-size: 16px; font-weight: 600; color: #333;">Classification Confidence: {confidence*100:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Probabilities chart
        st.markdown("##### Facies Probability Distribution:")
        prob_df = pd.DataFrame({
            'Facies': [CODE_TO_LITHOLOGY[i] for i in range(len(probs))],
            'Probability (%)': (probs * 100).round(1)
        }).sort_values(by='Probability (%)', ascending=True)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = [LITHOLOGY_COLORS.get(f, '#888888') for f in prob_df['Facies']]
        bars = ax.barh(prob_df['Facies'], prob_df['Probability (%)'], color=colors, edgecolor='black', height=0.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)", fontsize=10, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            w = bar.get_width()
            if w > 2:
                ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ==================== TAB 2: BATCH CSV INFERENCE ====================
with tab2:
    st.subheader("Batch Well Log CSV Classification")
    st.write("Upload an unlabelled or labelled well log CSV to run automated classification across all depth intervals.")

    upload_col1, upload_col2 = st.columns([1, 1])
    with upload_col1:
        uploaded_file = st.file_uploader("Choose a well log CSV file", type="csv")
    with upload_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_sample = st.button("📂 Load Pre-Split Test Dataset (test_data.csv)")

    df_to_predict = None
    if uploaded_file is not None:
        df_to_predict = pd.read_csv(uploaded_file)
    elif use_sample:
        if os.path.exists("test_data.csv"):
            df_to_predict = pd.read_csv("test_data.csv")
            st.success("Loaded test_data.csv (2,178 records).")

    if df_to_predict is not None:
        missing_req = [f for f in FEATURES_RAW if f not in df_to_predict.columns]
        if missing_req:
            st.error(f"Input CSV is missing required sensor curves: {missing_req}")
        else:
            with st.spinner("Running model inference..."):
                scaled_arr = scaler.transform(df_to_predict[FEATURES_RAW])
                scaled_df = pd.DataFrame(scaled_arr, columns=FEATURES_SCALED)
                preds = model.predict(scaled_df)
                probs = model.predict_proba(scaled_df)

                res_df = df_to_predict.copy()
                res_df['PRED_LITHOLOGY_CODE'] = preds
                res_df['PRED_LITHOLOGY'] = [CODE_TO_LITHOLOGY[c] for c in preds]
                res_df['PRED_CONFIDENCE'] = np.max(probs, axis=1).round(4)

            st.markdown("#### Classification Results")
            st.write(f"Processed **{len(res_df)}** depth intervals.")

            stat_col1, stat_col2 = st.columns([1, 2])
            with stat_col1:
                st.markdown("##### Facies Breakdown")
                breakdown = res_df['PRED_LITHOLOGY'].value_counts().reset_index()
                breakdown.columns = ['Lithology', 'Count']
                breakdown['Percentage'] = (breakdown['Count'] / len(res_df) * 100).round(2).astype(str) + '%'
                st.dataframe(breakdown, hide_index=True)

            with stat_col2:
                st.markdown("##### Classified Records Preview")
                preview_cols = ['DEPTH_MD'] + FEATURES_RAW + ['PRED_LITHOLOGY', 'PRED_CONFIDENCE']
                preview_cols = [c for c in preview_cols if c in res_df.columns]
                st.dataframe(res_df[preview_cols].head(10), hide_index=True)

            # Download button
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_data,
                file_name="well_log_predictions.csv",
                mime="text/csv"
            )


# ==================== TAB 3: WELL LOG TRACKS ====================
with tab3:
    st.subheader("Down-Hole Petrophysical Composite Log Viewer")
    st.write("Examine continuous down-hole sensor tracks alongside True vs. Predicted Facies.")

    plot_choice = st.radio(
        "Select Depth View:",
        ["Reservoir Zoom (2,800m - 3,200m depth)", "Full Borehole Profile (1,526m - 3,212m depth)"],
        horizontal=True
    )

    if plot_choice.startswith("Reservoir"):
        if os.path.exists("well_log_reservoir_section.png"):
            st.image("well_log_reservoir_section.png", caption="6-Track Composite Well Log Profile: Reservoir Section", use_container_width=True)
    else:
        if os.path.exists("well_log_full_profile.png"):
            st.image("well_log_full_profile.png", caption="6-Track Composite Well Log Profile: Full Well 15/9-23", use_container_width=True)


# ==================== TAB 4: MODEL DIAGNOSTICS ====================
with tab4:
    st.subheader("Model Performance Diagnostics & Benchmark Comparison")

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.markdown("#### Confusion Matrix Heatmap")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)

    with diag_col2:
        st.markdown("#### Petrophysical Feature Importances")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)

    st.markdown("---")
    st.markdown("#### Multi-Model Benchmark Comparison")
    if os.path.exists("model_comparison.png"):
        st.image("model_comparison.png", use_container_width=True)

    if os.path.exists("model_comparison.csv"):
        comp_df = pd.read_csv("model_comparison.csv")
        st.dataframe(comp_df, hide_index=True)
