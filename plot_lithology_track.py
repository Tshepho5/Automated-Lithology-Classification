"""
Item 1: Down-Hole Well Log Composite Track Visualization
Project: Automated Lithology Classification from Subsurface Well Logs
Input: 15_9-23_cleaned.csv, scaler.joblib, random_forest_model.joblib

Generates:
1. well_log_full_profile.png: Complete down-hole log (1,526m - 3,212m)
2. well_log_reservoir_section.png: Zoomed-in complex facies section (2,800m - 3,212m)
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("WellLogPlotter")

# Standard petrophysical color scheme for facies
LITHOLOGY_COLORS = {
    'Shale': '#708090',        # Slate Gray
    'Limestone': '#4169E1',    # Royal Blue
    'Marl': '#BC8F8F',         # Rosy Brown
    'Sandstone': '#FFD700',    # Gold / Yellow
    'Coal': '#1C1C1C',         # Very Dark Charcoal
    'Tuff': '#9370DB',         # Medium Purple
    'Chalk': '#00CED1'         # Dark Turquoise / Light Blue
}

FEATURES_RAW = ['GR', 'RHOB', 'NPHI', 'DTC', 'CALI', 'RDEP']
FEATURES_SCALED = [f"{col}_scaled" for col in FEATURES_RAW]


def load_data_and_predict(
    csv_file: str = "15_9-23_cleaned.csv",
    scaler_file: str = "scaler.joblib",
    model_file: str = "random_forest_model.joblib"
) -> tuple[pd.DataFrame, dict]:
    """Loads well log data and computes model predictions."""
    logger.info(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)

    X_scaled = scaler.transform(df[FEATURES_RAW])
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES_SCALED)
    df['PRED_CODE'] = model.predict(X_scaled_df)

    code_to_name = (
        df[['LITHOLOGY_CODE', 'LITHOLOGY']]
        .drop_duplicates()
        .set_index('LITHOLOGY_CODE')['LITHOLOGY']
        .to_dict()
    )
    df['PRED_LITHOLOGY'] = df['PRED_CODE'].map(code_to_name)
    df['MATCH'] = (df['LITHOLOGY_CODE'] == df['PRED_CODE']).astype(int)

    acc = df['MATCH'].mean()
    logger.info(f"Predictions generated. Overall well classification accuracy: {acc * 100:.2f}%")
    return df, code_to_name


def draw_lithology_strip(ax, series, depth_series, colors):
    """Draws colored facies blocks efficiently by grouping contiguous layers."""
    block_id = (series != series.shift()).cumsum()
    temp_df = pd.DataFrame({'lith': series, 'depth': depth_series, 'block': block_id})
    
    for _, group in temp_df.groupby('block'):
        d_top = group['depth'].min()
        d_bot = group['depth'].max()
        lith_name = group['lith'].iloc[0]
        color = colors.get(lith_name, '#CCCCCC')
        ax.axhspan(d_top, d_bot, 0, 1, facecolor=color, edgecolor=None)


def plot_composite_log(
    df: pd.DataFrame,
    code_to_name: dict,
    top_depth: float = None,
    bottom_depth: float = None,
    output_filename: str = "well_log_track.png",
    title: str = "Well 15/9-23 Down-Hole Lithology Profile"
):
    """Plots a 6-track petrophysical composite well log."""
    if top_depth is not None and bottom_depth is not None:
        plot_df = df[(df['DEPTH_MD'] >= top_depth) & (df['DEPTH_MD'] <= bottom_depth)].copy()
    else:
        plot_df = df.copy()
        top_depth = plot_df['DEPTH_MD'].min()
        bottom_depth = plot_df['DEPTH_MD'].max()

    depth = plot_df['DEPTH_MD'].values

    # Setup figure with 6 tracks: GR/CALI, RDEP, RHOB/NPHI, DTC, True Lith, Pred Lith
    fig, axes = plt.subplots(
        1, 6,
        figsize=(16, 12),
        sharey=True,
        gridspec_kw={'width_ratios': [1.2, 1.0, 1.2, 1.0, 0.7, 0.7]}
    )
    plt.subplots_adjust(wspace=0.08, top=0.90, bottom=0.08, left=0.06, right=0.92)

    # Invert Depth (depth increases downwards)
    for ax in axes:
        ax.set_ylim(bottom_depth, top_depth)
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[0].set_ylabel("Measured Depth (m)", fontsize=11, fontweight='bold')

    # ================= TRACK 1: Gamma Ray & Caliper =================
    ax1 = axes[0]
    ax1.plot(plot_df['GR'], depth, color='#2E7D32', linewidth=0.9, label='GR (API)')
    ax1.set_xlim(0, 200)
    ax1.set_xlabel("GR (API)", color='#2E7D32', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', colors='#2E7D32')

    ax1_twin = ax1.twiny()
    ax1_twin.set_ylim(bottom_depth, top_depth)
    ax1_twin.plot(plot_df['CALI'], depth, color='#212121', linestyle='--', linewidth=0.8, label='CALI (in)')
    ax1_twin.set_xlim(6, 16)
    ax1_twin.set_xlabel("CALI (in)", color='#212121', fontsize=10, fontweight='bold')
    ax1_twin.spines['top'].set_position(('outward', 0))

    # ================= TRACK 2: Deep Resistivity =================
    ax2 = axes[1]
    ax2.plot(plot_df['RDEP'], depth, color='#C62828', linewidth=0.9)
    ax2.set_xscale('log')
    ax2.set_xlim(0.1, 50)
    ax2.set_xlabel("RDEP (ohm.m)\n[Log Scale]", color='#C62828', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', colors='#C62828')

    # ================= TRACK 3: Bulk Density & Neutron Porosity =================
    ax3 = axes[2]
    ax3.plot(plot_df['RHOB'], depth, color='#1565C0', linewidth=0.9, label='RHOB (g/cm³)')
    ax3.set_xlim(1.8, 2.9)
    ax3.set_xlabel("RHOB (g/cm³)", color='#1565C0', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', colors='#1565C0')

    ax3_twin = ax3.twiny()
    ax3_twin.set_ylim(bottom_depth, top_depth)
    ax3_twin.plot(plot_df['NPHI'], depth, color='#00838F', linestyle='-', linewidth=0.8, label='NPHI (v/v)')
    ax3_twin.set_xlim(0.45, -0.15)  # Inverted standard petrophysical scale
    ax3_twin.set_xlabel("NPHI (v/v) [Inverted]", color='#00838F', fontsize=10, fontweight='bold')
    ax3_twin.tick_params(axis='x', colors='#00838F')

    # ================= TRACK 4: Sonic Transit Time (DTC) =================
    ax4 = axes[3]
    ax4.plot(plot_df['DTC'], depth, color='#E65100', linewidth=0.9)
    ax4.set_xlim(40, 180)
    ax4.set_xlabel("DTC (µs/ft)", color='#E65100', fontsize=10, fontweight='bold')
    ax4.tick_params(axis='x', colors='#E65100')

    # ================= TRACK 5: True Lithology Strip =================
    ax5 = axes[4]
    ax5.set_xlabel("True\nFacies", fontsize=10, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    draw_lithology_strip(ax5, plot_df['LITHOLOGY'], plot_df['DEPTH_MD'], LITHOLOGY_COLORS)

    # ================= TRACK 6: Predicted Lithology Strip =================
    ax6 = axes[5]
    ax6.set_xlabel("Predicted\nFacies", fontsize=10, fontweight='bold')
    ax6.set_xlim(0, 1)
    ax6.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    draw_lithology_strip(ax6, plot_df['PRED_LITHOLOGY'], plot_df['DEPTH_MD'], LITHOLOGY_COLORS)

    # Legend for Lithologies
    legend_patches = [
        mpatches.Patch(color=color, label=name)
        for name, color in LITHOLOGY_COLORS.items()
        if name in plot_df['LITHOLOGY'].values or name in plot_df['PRED_LITHOLOGY'].values
    ]
    fig.legend(
        handles=legend_patches,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.96),
        title="Lithology Legend",
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9
    )

    interval_str = f"Depth: {top_depth:.1f} m - {bottom_depth:.1f} m"
    fig.suptitle(f"{title}\n({interval_str})", fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved well log plot to {output_filename}")


if __name__ == "__main__":
    logger.info("Generating Well Log Composite Track Visualizations...")
    df, code_to_name = load_data_and_predict()

    # 1. Full Well Profile
    plot_composite_log(
        df=df,
        code_to_name=code_to_name,
        output_filename="well_log_full_profile.png",
        title="Well 15/9-23: Full Borehole Lithology Classification"
    )

    # 2. Detailed Zoom Section (Geologically diverse reservoir interval: 2800m - 3200m)
    plot_composite_log(
        df=df,
        code_to_name=code_to_name,
        top_depth=2800.0,
        bottom_depth=3200.0,
        output_filename="well_log_reservoir_section.png",
        title="Well 15/9-23: Reservoir & Heterogeneous Lithology Zoom (2800m - 3200m)"
    )

    logger.info("Item 1 Complete: Well log track plots generated successfully.")
