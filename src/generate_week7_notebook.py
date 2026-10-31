"""
Generator for Week 7 Master Jupyter Notebook:
notebooks/Week7_HealthConnect_Rwanda_Model_Testing_Refinement.ipynb
Authors: Saad Janane (Lead Data Scientist) & Zaid Harboul (Lead ML Engineer - Collaborator)
"""

import os
import nbformat as nbf

def create_week7_notebook():
    os.makedirs("notebooks", exist_ok=True)
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell(
"""# HealthConnect Rwanda Healthcare Analytics & AI Transformation
## Track 2: Healthcare Predictive Analytics & Population Health Intelligence
### Official Week 7 Assignment: Model Testing, Refinement & End-to-End Validation
**AnalystLab Africa Experience Lab Internship Programme**

---

### 📋 Document Control & Project Metadata
| Parameter | Project Control Value |
| :--- | :--- |
| **Programme** | AnalystLab Africa Experience Lab |
| **Project Title** | Improving Patient Appointment Attendance & Healthcare Support Using Data and AI |
| **Track** | **Track 2 — Data Science Track** |
| **Author (Lead Data Scientist)** | **Saad Janane** |
| **Collaborator (Lead ML Engineer)** | **Zaid Harboul** |
| **Collaborating Track** | Machine Learning Engineering (HC-POD Cross-Track Integration) |
| **Document Stage** | Week 7 Official Capstone Notebook & Validation Package |
| **Version** | 2.0 (Testing, Refinement & Production Integration Release) |
| **Core Technology Stack** | Python, Pandas, NumPy, Scikit-Learn, SciPy, Statsmodels, Matplotlib, Seaborn, Pytest |

---

### 🎯 Week 7 Assignment Objectives & Scope
In accordance with official AnalystLab Africa Week 7 guidelines (*HealthConnect Testing, Refinement & End-to-End Validation*), this notebook demonstrates rigorous evidence-based validation:
1. **Week 6 → Week 7 Transition**: Establishing continuity from Week 6 baseline classification outputs.
2. **Systematic Candidate Model Testing**: Evaluating classification metrics on a holdout validation set ($N = 1,250$).
3. **In-Depth Error Analysis**: Investigating False Positives (over-triage bed crowding) vs. False Negatives (under-triage critical patient risk) and quantifying the clinical loss function ($L = 5 \\times FN + 1 \\times FP$).
4. **Demographic & Clinical Slice Testing**: Assessing performance disparities across patient age groups, gender, and chronic morbidities.
5. **Model Refinement**: Engineering clinical interaction acuity features (`ClinicalAcuityScore`, `Age_Chronic_Interaction`, `Comorbidity_Risk`) and class re-balancing.
6. **Probability Calibration**: Applying isotonic regression to calibrate probability outputs against population base rates, reducing Brier score loss.
7. **Decision Threshold Optimization**: Sweeping $\\tau \\in [0.25, 0.55]$ to maximize $F_2$-score and reduce dangerous missed admissions from 144 to 17.
8. **HC-POD Cross-Track Validation**: Collaborative testing with **Zaid Harboul** (ML Engineering Track) verifying inference schemas, adversarial edge cases, and achieving a **1,558x speedup** via vectorized batch inference.
"""
    ))

    # Cell 1: Imports and Environment Setup
    cells.append(nbf.v4.new_markdown_cell("## 1. Environment Setup & Core Dependencies"))
    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

print(">> Python Environment Initialized.")
print(f"   Pandas Version: {pd.__version__}")
print(f"   NumPy Version: {np.__version__}")
"""
    ))

    # Cell 2: Data Ingestion & Audit
    cells.append(nbf.v4.new_markdown_cell("## 2. Ingestion & Audit of Feature-Engineered Healthcare Cohort"))
    cells.append(nbf.v4.new_code_cell(
"""cohort_path = "../data/processed/feature_engineered_cohort.csv"
if not os.path.exists(cohort_path):
    cohort_path = "data/processed/feature_engineered_cohort.csv"

df = pd.read_csv(cohort_path)
print(f">> Loaded Cohort Matrix: {df.shape[0]:,} rows x {df.shape[1]} columns")

admit_counts = df['IsAdmitted'].value_counts()
admit_pct = df['IsAdmitted'].value_counts(normalize=True) * 100

print(f"   - Discharged Outpatients (0): {admit_counts[0]:,} ({admit_pct[0]:.2f}%)")
print(f"   - Admitted Inpatients (1)   : {admit_counts[1]:,} ({admit_pct[1]:.2f}%)")
print(f"   - Severe Class Imbalance Ratio: ~{admit_counts[0]/admit_counts[1]:.1f}:1")
"""
    ))

    # Cell 3: Loading Week 7 Model Testing Results
    cells.append(nbf.v4.new_markdown_cell("## 3. Review of Week 6 Candidate Model vs. Week 7 Refined Model"))
    cells.append(nbf.v4.new_code_cell(
"""results_path = "../data/processed/week7_model_testing_results.json"
if not os.path.exists(results_path):
    results_path = "data/processed/week7_model_testing_results.json"

with open(results_path, 'r') as f:
    results = json.load(f)

comp_csv_path = "../data/processed/week7_model_comparison.csv"
if not os.path.exists(comp_csv_path):
    comp_csv_path = "data/processed/week7_model_comparison.csv"

comp_df = pd.read_csv(comp_csv_path)
comp_df
"""
    ))

    # Cell 4: Detailed Error Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""## 4. In-Depth Error Analysis & Clinical Cost Matrix
In clinical triage, errors are fundamentally asymmetric:
- **False Negative (FN)**: An acute patient needing inpatient admission is incorrectly classified as routine outpatient and sent home. This carries severe clinical risk (morbidity, preventable mortality, malpractice liability).
- **False Positive (FP)**: A non-acute patient is flagged for admission review. This carries minor administrative observation cost (triage re-check), but zero mortality risk.
- **Clinical Loss Function**: $L = 5 \\times FN + 1 \\times FP$.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""w6_err = results["Week6_Candidate_Evaluation"]["Error_Analysis"]
w7_err = results["Week7_Refined_Evaluation"]["Error_Analysis"]

error_comparison = pd.DataFrame([
    {
        "Model Stage": "Week 6 Candidate (Random Forest, tau=0.50)",
        "True Positives (Captured)": w6_err["True_Positives"],
        "False Negatives (Missed)": w6_err["False_Negatives_MissedAdmission"],
        "Miss Rate (%)": f"{w6_err['Miss_Rate_Percent']}%",
        "False Positives (Over-Triage)": w6_err["False_Positives_OverTriage"],
        "Clinical Loss Score": w6_err["Clinical_Cost_Loss"]
    },
    {
        "Model Stage": "Week 7 Refined (Balanced RF, tau=0.38)",
        "True Positives (Captured)": w7_err["True_Positives"],
        "False Negatives (Missed)": w7_err["False_Negatives_MissedAdmission"],
        "Miss Rate (%)": f"{w7_err['Miss_Rate_Percent']}%",
        "False Positives (Over-Triage)": w7_err["False_Positives_OverTriage"],
        "Clinical Loss Score": w7_err["Clinical_Cost_Loss"]
    }
])
error_comparison
"""
    ))

    # Cell 5: Display Figure 16
    cells.append(nbf.v4.new_markdown_cell("### Figure 16: Confusion Matrix Comparison (Week 6 vs Week 7)"))
    cells.append(nbf.v4.new_code_cell(
"""from IPython.display import Image, display
fig16_path = "../reports/figures/fig16_confusion_matrix_before_after.png"
if not os.path.exists(fig16_path):
    fig16_path = "reports/figures/fig16_confusion_matrix_before_after.png"
display(Image(filename=fig16_path, width=900))
"""
    ))

    # Cell 6: Decision Threshold Optimization Curve
    cells.append(nbf.v4.new_markdown_cell(
"""## 5. Decision Threshold Optimization
We swept the decision threshold $\\tau$ from $0.25$ to $0.55$ to evaluate the trade-off between Recall, Precision, $F_1$, $F_2$, and Clinical Loss.
At $\\tau = 0.38$, clinical recall reaches **88.36%**, capturing 129 out of 146 true admissions while maintaining viable bed allocation throughput.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""sweep_path = "../data/processed/week7_threshold_sweep.csv"
if not os.path.exists(sweep_path):
    sweep_path = "data/processed/week7_threshold_sweep.csv"

sweep_df = pd.read_csv(sweep_path)
print("Key Threshold Operating Points:")
print(sweep_df[(sweep_df['Threshold'] >= 0.35) & (sweep_df['Threshold'] <= 0.42)].to_string(index=False))

fig17_path = "../reports/figures/fig17_threshold_optimization_curve.png"
if not os.path.exists(fig17_path):
    fig17_path = "reports/figures/fig17_threshold_optimization_curve.png"
display(Image(filename=fig17_path, width=900))
"""
    ))

    # Cell 7: ROC and PR Curves
    cells.append(nbf.v4.new_markdown_cell("## 6. Discrimination & Precision-Recall Trajectory"))
    cells.append(nbf.v4.new_code_cell(
"""fig18_path = "../reports/figures/fig18_roc_pr_curves_comparison.png"
if not os.path.exists(fig18_path):
    fig18_path = "reports/figures/fig18_roc_pr_curves_comparison.png"
display(Image(filename=fig18_path, width=900))
"""
    ))

    # Cell 8: Probability Calibration
    cells.append(nbf.v4.new_markdown_cell(
"""## 7. Probability Calibration & Reliability Assessment
Tree models output voting fractions that are uncalibrated probabilities. By fitting an isotonic calibrator via cross-validation, the Brier score was reduced from **0.1743 to 0.1038**, aligning predicted probabilities with true population disease prevalence.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""fig19_path = "../reports/figures/fig19_probability_calibration_curve.png"
if not os.path.exists(fig19_path):
    fig19_path = "reports/figures/fig19_probability_calibration_curve.png"
display(Image(filename=fig19_path, width=800))
"""
    ))

    # Cell 9: Subgroup Disparity Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""## 8. Subgroup Fairness & Demographic Slice Testing
In Week 6, the candidate model had **0.0% recall on Middle-Aged, Pediatric, and Young Adult cohorts**.
In Week 7, through feature interactions and cost-sensitive calibration, recall reached $\\ge 60\\%$ across all demographic cohorts, achieving 93.9% in Geriatric and 100.0% in Senior patients.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""fig20_path = "../reports/figures/fig20_subgroup_disparity_analysis.png"
if not os.path.exists(fig20_path):
    fig20_path = "reports/figures/fig20_subgroup_disparity_analysis.png"
display(Image(filename=fig20_path, width=900))
"""
    ))

    # Cell 10: HC-POD Cross-Track Validation with Zaid Harboul
    cells.append(nbf.v4.new_markdown_cell(
"""## 9. Mandatory HC-POD Cross-Track Validation (Data Science ↔ ML Engineering)
### Co-developed with Zaid Harboul (Lead ML Engineer)
- **Dependency**: Serving the Data Science serialized model package in the ML Engineering serving engine.
- **Edge-Case Stress Testing**: 10 adversarial edge cases (unseen categories, extreme ages, null values) were injected. **All 10 passed (100%)**.
- **Performance Profiling**: Unbatched row-by-row scoring took **164.1ms p50**. Zaid Harboul and Saad Janane implemented a **Vectorized Batch Engine (`predict_batch`)**, processing 1,000 patient requests in **0.105 seconds (9,496 requests/sec, 0.11ms per patient)**.
- **Speedup**: **1,558x acceleration**, satisfying the production clinical SLO ($<5$ms).
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""telemetry_path = "../data/processed/week7_cross_track_mle_telemetry.json"
if not os.path.exists(telemetry_path):
    telemetry_path = "data/processed/week7_cross_track_mle_telemetry.json"

with open(telemetry_path, 'r') as f:
    telem = json.load(f)

bench = telem["Latency_Benchmark"]
bench_summary_df = pd.DataFrame([
    {"Metric": "Total Requests Processed", "Value": f"{bench['Total_Requests_Processed']:,}"},
    {"Metric": "Initial Unbatched p50 Latency", "Value": f"{bench['Initial_Unbatched_P50_ms']} ms"},
    {"Metric": "Initial Unbatched p95 Latency", "Value": f"{bench['Initial_Unbatched_P95_ms']} ms"},
    {"Metric": "Refined Vectorized Batch Throughput", "Value": f"{bench['Refined_Vectorized_Throughput_QPS']} req/sec"},
    {"Metric": "Refined Vectorized Latency per Patient", "Value": f"{bench['Refined_Vectorized_Per_Sample_ms']} ms"},
    {"Metric": "Throughput Speedup Factor", "Value": f"{bench['Speedup_Factor']}x faster"},
    {"Metric": "Production SLO Target (<5ms)", "Value": "MET (PASSED)" if bench['SLO_Target_Met'] else "FAILED"}
])
print("Cross-Track Telemetry Summary:")
print(bench_summary_df.to_string(index=False))

fig21_path = "../reports/figures/fig21_cross_track_mle_integration_telemetry.png"
if not os.path.exists(fig21_path):
    fig21_path = "reports/figures/fig21_cross_track_mle_integration_telemetry.png"
display(Image(filename=fig21_path, width=900))
"""
    ))

    # Cell 11: Production Serialization & Verification
    cells.append(nbf.v4.new_markdown_cell(
"""## 10. Production Model Serialization & Verification
The refined, calibrated model artifact is saved to `models/refined_admission_model.pkl` with full metadata and feature dictionaries.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""model_path = "../models/refined_admission_model.pkl"
if not os.path.exists(model_path):
    model_path = "models/refined_admission_model.pkl"

pkg = joblib.load(model_path)
print(">> Loaded Production Model Package:")
print(f"   Model Version   : {pkg['version']}")
print(f"   Author          : {pkg['author']}")
print(f"   Collaborator    : {pkg['collaborator']}")
print(f"   Training Date   : {pkg['training_date']}")
print(f"   Optimal tau     : {pkg['optimal_threshold']}")
print(f"   Numerical Feats : {len(pkg['features_num'])} features")
print(f"   Categorical Feats: {len(pkg['features_cat'])} features")
"""
    ))

    # Cell 12: Executive Conclusions & Week 8 Roadmap
    cells.append(nbf.v4.new_markdown_cell(
"""## 11. Executive Summary & Week 8 Integration Roadmap

### Key Findings & Achievements in Week 7:
1. **Critical Vulnerability Remedied**: Resolved the severe under-triage flaw of Week 6 (where 98.6% of patients requiring admission were missed) by elevating Recall from **1.37% to 88.36%**.
2. **Reliable Probability Calibration**: Isotonic calibration reduced Brier score to **0.1038**, ensuring predicted risk reflects true epidemiological rates.
3. **Subgroup Equity Validated**: Eliminated zero-recall demographic blindspots across pediatric, young adult, and middle-aged cohorts.
4. **Cross-Track MLOps Success**: Partnered with **Zaid Harboul** to validate schema contracts and accelerate batch inference by **1,558x** to 9,496 requests/sec.

### Week 8 Final Integration Handoff:
- Containerize the inference engine via Docker and integrate with FastAPI endpoints.
- Connect clinical admission alerts with HealthConnect Clinic's real-time EHR dashboard.
- Deploy Prometheus telemetry for monitoring data drift and inference latency at bedside triage.

---
*HealthConnect Rwanda Healthcare Analytics & AI Transformation — AnalystLab Africa Experience Lab.*
"""
    ))

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }

    notebook_path = "notebooks/Week7_HealthConnect_Rwanda_Model_Testing_Refinement.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f">> Successfully generated Week 7 Notebook: {notebook_path}")


if __name__ == "__main__":
    create_week7_notebook()
