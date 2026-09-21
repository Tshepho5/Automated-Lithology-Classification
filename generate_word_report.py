"""
Generates the formatted Word Document Report (Final_Project_Report.docx)
for the Lithology Classification Project.
"""

import os
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def create_report_document(output_docx="Final_Project_Report.docx"):
    print(f"Creating Word Report: {output_docx}...")
    doc = Document()

    # Set standard margins (1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ------------------ TITLE & METADATA ------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(6)
    title_run = title_p.add_run("Automated Lithology Classification from Subsurface Well Logs Using Machine Learning")
    title_run.font.name = "Arial"
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(27, 54, 93)

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(18)
    sub_run = subtitle_p.add_run("Final Academic Project Report | Course: SCOA032 - Subsurface Characterization & Advanced Analytics")
    sub_run.font.name = "Arial"
    sub_run.font.size = Pt(11)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(100, 100, 100)

    # Metadata box table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.rows[0].cells[0].text = "Dataset: FORCE 2020 Well Log (Well 15/9-23)"
    meta_table.rows[0].cells[1].text = "Model: Supervised Random Forest Classifier"
    meta_table.rows[1].cells[0].text = "Evaluated Test Accuracy: 96.46%"
    meta_table.rows[1].cells[1].text = "Evaluated Test Macro-F1: 0.8600"

    for row in meta_table.rows:
        for cell in row.cells:
            set_cell_background(cell, "F0F4F8")
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.name = "Arial"
                    r.font.size = Pt(9.5)
                    r.font.bold = True
                    r.font.color.rgb = RGBColor(30, 41, 59)

    doc.add_paragraph()

    # Helper function for headings
    def add_custom_heading(text, level=1):
        h = doc.add_heading(level=level)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
        r = h.add_run(text)
        r.font.name = "Arial"
        if level == 1:
            r.font.size = Pt(14)
            r.font.bold = True
            r.font.color.rgb = RGBColor(27, 54, 93)
        elif level == 2:
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = RGBColor(51, 65, 85)
        return h

    def add_body_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(33, 37, 41)
        return p

    # ------------------ 1. EXECUTIVE SUMMARY ------------------
    add_custom_heading("1. Executive Summary", 1)
    add_body_p(
        "Accurate identification of subsurface rock formations (lithology) is a foundational objective in petroleum exploration, "
        "hydrogeology, and geological carbon sequestration. Traditionally, lithology determination is performed through manual "
        "interpretation of borehole geophysical logs—a laborious, subjective process prone to human cognitive fatigue."
    )
    add_body_p(
        "This project established a robust, automated, end-to-end machine learning pipeline that classifies discrete rock types directly "
        "from continuous down-hole sensor measurements. Using data from North Sea Well 15/9-23, our optimized Random Forest Classifier "
        "achieved an overall test accuracy of 96.46% and a Macro-F1 score of 0.8600 across 7 geological facies categories, validated "
        "via 5-fold stratified cross-validation (0.8684 ± 0.0275)."
    )

    # ------------------ 2. PROBLEM STATEMENT ------------------
    add_custom_heading("2. Problem Statement & Objectives", 1)
    add_body_p(
        "Manual well log interpretation lacks standardization across analysts and cannot scale to thousands of meters of borehole data. "
        "The core technical challenge involves constructing a supervised classification framework that effectively maps multi-modal physical sensor logs "
        "(operating on disparate physical units) to discrete lithology facies, while rigorously handling severe geological class imbalance."
    )

    # ------------------ 3. DATASET & PREPROCESSING ------------------
    add_custom_heading("3. Dataset Description & Preprocessing Pipeline", 1)
    add_body_p(
        "The project utilizes subsurface well log data from the FORCE 2020 Machine Learning Benchmark Competition (Well 15/9-23). "
        "The raw dataset comprised 11,063 depth-indexed records (1,518.2 m to 3,212.6 m) across 29 sensor tracks."
    )
    add_body_p(
        "Key Data Cleaning Operations:\n"
        "• Dead Channel Pruning: Dropped 6 completely empty (100% NaN) curves (RSHA, SGR, SP, MUDWEIGHT, RMIC, RXO).\n"
        "• Redundancy Elimination: Pruned collinear drilling noise (ROP, ROPA, DCAL, RMED), retaining standard deep resistivity (RDEP).\n"
        "• Petrophysical Coal Protection Rule: Outlier filtering strictly protected low-density intervals (RHOB < 1.8 g/cm³) from deletion, "
        "preserving genuine organic coal beds and interbedded formations.\n"
        "• Sensor Completeness: Retaining records with complete core sensor curves (GR, RHOB, NPHI, DTC, CALI, RDEP) yielded 10,887 high-quality "
        "records (98.41% data retention) with zero missing values and zero duplicate depths."
    )

    # ------------------ 4. METHODOLOGY ------------------
    add_custom_heading("4. Machine Learning Methodology", 1)
    add_body_p(
        "1. Stratified Train-Test Split (80/20): Because natural geological class imbalance exists (shale comprises ~70% of samples while "
        "chalk, coal, and tuff each comprise <1%), a stratified split was enforced to ensure identical facies proportions in both sets (8,709 train / 2,178 test).\n"
        "2. Zero-Leakage StandardScaler: Features were normalized to zero mean and unit variance strictly using training set parameters:\n"
        "   - GR: Mean = 85.62 API, Std = 47.30 API\n"
        "   - RHOB: Mean = 2.343 g/cm³, Std = 0.166 g/cm³\n"
        "   - NPHI: Mean = 0.3033 v/v, Std = 0.1285 v/v\n"
        "   - DTC: Mean = 118.93 µs/ft, Std = 29.45 µs/ft\n"
        "   - CALI: Mean = 12.07 in, Std = 1.25 in\n"
        "   - RDEP: Mean = 1.423 ohm.m, Std = 1.447 ohm.m\n"
        "3. Model Architecture: A Random Forest Classifier with 150 estimators was trained to model complex non-linear rock physics interactions."
    )

    # ------------------ 5. EVALUATION RESULTS ------------------
    add_custom_heading("5. Performance Evaluation & Validation", 1)
    add_body_p(
        "Evaluation on the unseen test set (2,178 records) demonstrated exceptional classification performance across all facies:"
    )

    # Load metrics from evaluation_report.json
    with open("evaluation_report.json") as f:
        eval_data = json.load(f)

    # Create Performance Table
    table = doc.add_table(rows=1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Lithology Facies", "Precision", "Recall", "F1-Score", "Test Samples"]
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1B365D")
        for p in hdr_cells[i].paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(9.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)

    per_class = eval_data["per_class_metrics"]
    for facies, m in per_class.items():
        row_cells = table.add_row().cells
        row_cells[0].text = facies
        row_cells[1].text = f"{m['precision']:.2f}"
        row_cells[2].text = f"{m['recall']:.2f}"
        row_cells[3].text = f"{m['f1-score']:.2f}"
        row_cells[4].text = str(m['support'])

        for i, cell in enumerate(row_cells):
            for p in cell.paragraphs:
                if i > 0:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                for r in p.runs:
                    r.font.name = "Arial"
                    r.font.size = Pt(9)

    # Add Summary Row
    summary_row = table.add_row().cells
    summary_row[0].text = "Macro Average"
    summary_row[1].text = f"{eval_data['test_metrics']['macro_precision']:.2f}"
    summary_row[2].text = f"{eval_data['test_metrics']['macro_recall']:.2f}"
    summary_row[3].text = f"{eval_data['test_metrics']['macro_f1']:.2f}"
    summary_row[4].text = str(eval_data['test_samples'])
    for cell in summary_row:
        set_cell_background(cell, "E2E8F0")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.name = "Arial"
                r.font.size = Pt(9)
                r.font.bold = True

    doc.add_paragraph()

    # Embed Confusion Matrix
    if os.path.exists("confusion_matrix.png"):
        add_custom_heading("Confusion Matrix Heatmap", 2)
        doc.add_picture("confusion_matrix.png", width=Inches(5.5))
        caption_p = doc.add_paragraph()
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = caption_p.add_run("Figure 1: Confusion Matrix of True vs. Predicted Facies (Test Set).")
        c_run.font.italic = True
        c_run.font.size = Pt(9)

    # Embed Feature Importance
    if os.path.exists("feature_importance.png"):
        add_custom_heading("Petrophysical Feature Importances", 2)
        doc.add_picture("feature_importance.png", width=Inches(5.5))
        caption_p = doc.add_paragraph()
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = caption_p.add_run("Figure 2: Petrophysical Feature Importances extracted from Random Forest.")
        c_run.font.italic = True
        c_run.font.size = Pt(9)

    # ------------------ 6. DOWN-HOLE PROFILE ------------------
    if os.path.exists("well_log_reservoir_section.png"):
        add_custom_heading("6. Down-Hole Well Log Profile Validation", 1)
        add_body_p(
            "To evaluate down-hole continuity, a 6-track composite well log was generated comparing model predictions to true geology "
            "across the complex reservoir interval (2,800 m to 3,200 m depth):"
        )
        doc.add_picture("well_log_reservoir_section.png", width=Inches(6.0))
        caption_p = doc.add_paragraph()
        caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c_run = caption_p.add_run("Figure 3: Down-Hole Petrophysical Composite Track Profile (Reservoir Section 2,800m - 3,200m).")
        c_run.font.italic = True
        c_run.font.size = Pt(9)

    # ------------------ 7. CONCLUSION ------------------
    add_custom_heading("7. Conclusion", 1)
    add_body_p(
        "The developed machine learning pipeline demonstrates that subsurface rock facies can be classified with high precision (96.46%) "
        "directly from wireline sensor logs. Acoustic wave slowness (DTC, 33.61%) and natural gamma radiation (GR, 24.74%) were revealed "
        "as the primary physical controls governing facies discrimination in the North Sea basin."
    )

    doc.save(output_docx)
    print(f"Successfully generated formatted Word Report: {output_docx}")

if __name__ == "__main__":
    create_report_document()
