# HealthConnect Rwanda Healthcare Analytics & AI Transformation
## Official Week 7 Project Summary
**AnalystLab Africa Experience Lab Internship Programme — Official Executive Deliverable**

---

### 📋 Document Control
| Field | Value |
| :--- | :--- |
| **Programme** | AnalystLab Africa Experience Lab |
| **Project Title** | Improving Patient Appointment Attendance and Healthcare Support Using Data and AI |
| **Track** | **Track 2 — Data Science Track** |
| **Lead Data Scientist** | **Saad Janane** |
| **Collaborator** | **Zaid Harboul (Lead ML Engineer)** |
| **Document Stage** | Official Week 7 Project Summary (Section 14 Compliance) |
| **Version** | 1.0 (Final Validated Release) |

---

## 1. What You Planned to Test
We planned to evaluate whether the Week 6 candidate hospital admission classification model (`hospital_admission_model.pkl`) performed reliably on an independent holdout test set ($N=1,250$). Specifically, our test plan aimed to assess:
- Classification discrimination (Accuracy, Recall, Precision, $F_1$, $F_2$, ROC-AUC).
- Distribution and clinical severity of classification errors (False Positives vs. False Negatives).
- Model fairness and consistency across demographic slices (Age groups, Chronic illnesses, Provinces, Facility types).
- Probability score calibration against empirical population admission frequencies ($11.72\%$).
- Robustness of the model inference contract when integrated into the ML Engineering serving pipeline.

## 2. What You Actually Tested
We executed a 5-stage testing and refinement protocol:
1. **Holdout Evaluation**: Benchmarked the candidate Random Forest against a stratified 25% test split ($1,250$ consultations).
2. **Clinical Error Cost Quantification**: Computed the asymmetric clinical loss metric ($L = 5 \times FN + 1 \times FP$) reflecting the real-world danger of discharging acute patients.
3. **Granular Subgroup Slicing**: Evaluated true positive rates across 5 age brackets, 4 chronic disease cohorts, 4 facility tiers, and 5 provinces.
4. **Probability Calibration Assessment**: Constructed reliability diagrams and evaluated Brier score before and after isotonic calibration.
5. **Cross-Track MLOps Testing with Zaid Harboul**: Injected 10 adversarial edge cases and executed a 1,000-request latency and throughput stress test.

## 3. The Most Important Testing Results
- **Week 6 Candidate Defect Exposed**: While the candidate model exhibited an ostensibly high accuracy of **$87.52\%$**, it suffered from an unacceptably low clinical recall of **$1.37\%$** (capturing only 2 out of 146 true admissions; 144 false negatives).
- **Week 7 Refined Performance**: At the tuned clinical operating threshold ($\tau = 0.38$), recall surged to **$88.36\%$** ($+86.99\%$ gain), capturing **129 out of 146 acute admissions**.
- **Probability Calibration**: Brier score dropped from **$0.1743$ to $0.1038$**, aligning risk estimates with actual disease incidence.
- **Cross-Track Throughput**: Vectorized batch inference reached **$9,495.9\text{ requests/sec}$** ($0.11\text{ms}$ per patient), achieving a **$1,558\times$ speedup** over unbatched execution.

## 4. Issues or Weaknesses Identified
1. **Class Imbalance Blindspot**: In Week 6, the default $0.50$ probability threshold caused tree estimators to default to the negative majority class ($88.28\%$), rendering the model clinically inert.
2. **Demographic Equity Failure**: The candidate model had **$0.0\%$ recall across Pediatric, Young Adult, and Middle-Aged cohorts**, detecting admissions only in a fraction of elderly patients.
3. **Production Exception Vulnerability**: The preprocessing pipeline failed on novel categorical values (unseen diagnoses, new clinics), raising unhandled `ValueError` crashes.
4. **Unbatched Latency Bottleneck**: Row-by-row DataFrame scoring took $164.1\text{ms } p50$, breaching the ML Engineering SLO target ($< 25\text{ms}$).

## 5. Improvements / Refinements Made
1. **Clinical Feature Acuity Engineering**: Created interactive acuity features: `ClinicalAcuityScore` (combining Emergency visit type, severity $\ge 3$, abnormal labs, and chronic conditions), `Age_Chronic_Interaction`, and `Comorbidity_Risk`.
2. **Balanced Cost-Sensitive Architecture**: Retrained Random Forest with balanced class weights and leaf constraints to counter class imbalance.
3. **Isotonic Probability Calibration**: Applied cross-validated isotonic regression to align model confidence with real-world epidemiological risk.
4. **Decision Threshold Optimization**: Swept $\tau \in [0.25, 0.55]$, identifying $\tau = 0.38$ as the optimal clinical tradeoff maximizing $F_2$-score ($0.3771$).
5. **Vectorized Batch Engine**: Partnered with Zaid Harboul to implement `predict_batch()` with robust missing-value imputation and `handle_unknown='ignore'`.

## 6. Retesting Results
- **Admission Capture**: Captured admissions jumped from **$2 \longrightarrow 129$**; missed admissions plummeted from **$144 \longrightarrow 17$** (an **$88.2\%$ reduction in missed critical patients**).
- **Subgroup Parity**: Recall reached **$93.9\%$ in Geriatrics**, **$97.1\%$ in Middle-Aged**, **$100.0\%$ in Seniors**, and **$60.7\%$ in Pediatrics** (eliminating zero-recall blindspots).
- **Edge Case Robustness**: All 10 adversarial scenarios passed ($100\%$) with zero unhandled exceptions.
- **Automated Regression**: $7 / 7$ Pytest tests passed in $4.41$ seconds.

## 7. Which Track(s) You Collaborated With
- **Machine Learning Engineering Track**: Direct partnership with **Zaid Harboul** on pipeline integration, schema validation, latency profiling, and batch serving.
- **Data Analytics Track**: Integrated findings on chronic disease comorbidity prevalence and department workload pressures into feature definitions.

## 8. What Was Tested Collaboratively
- Serialization and deserialization of the Data Science pipeline artifact (`refined_admission_model.pkl`).
- API payload schema contracts and automatic fallback handling for malformed or missing EHR inputs.
- Concurrency, latency distribution ($p50, p95, p99$), and memory footprint under a 1,000-request production simulation.

## 9. What Changed as a Result
- Transformed an academic prototype with severe triage failure into a resilient, deployment-ready clinical decision engine.
- Replaced a slow row-by-row Python loop with a vectorized batch engine capable of processing nationwide consultations in milliseconds.
- Established joint ownership and reproducible CI/CD test gates between Data Science and ML Engineering.

## 10. Key Findings or Validation Outcomes
- High nominal accuracy ($87.5\%$) in imbalanced healthcare datasets frequently masks catastrophic clinical failure (98.6% false negatives).
- Tuning the decision threshold based on an asymmetric clinical cost function ($F_2$-score) is far more impactful than hyperparameter tuning alone.
- Probability calibration is essential for healthcare: uncalibrated models deceive clinicians, whereas calibrated probabilities allow transparent risk communication.

## 11. Major Challenges Encountered
- **Overcoming Class Imbalance with Noisy Clinical Indicators**: In outpatient datasets, acute admission signals are subtle. Overcoming this required combining multiple physiological signals into composite clinical indices.
- **Python Subprocess Overhead on Windows**: Addressed cross-validation latency by tuning joblib backend execution and optimizing feature transformations.
- **Latency Tradeoff in Scikit-Learn Pipelines**: Calling single-record inferences incurred heavy overhead; resolved via vectorized batching.

## 12. Important Decisions Made
- **Decision 1**: Adopt $\tau = 0.38$ as the primary clinical operating threshold, prioritizing patient safety over false positive minimization.
- **Decision 2**: Retain Random Forest as the primary classifier over Logistic Regression due to superior feature interaction handling, backed by isotonic calibration.
- **Decision 3**: Standardize all categorical encoding with `handle_unknown='ignore'` to guarantee zero runtime crashes during live hospital deployment.

## 13. Remaining Limitations
- **Synthetic Data Acuity Ceiling**: Because diagnostic features in the benchmark dataset are semi-synthetic, ROC-AUC is constrained ($0.685$), requiring reliance on threshold tuning for sensitivity.
- **Lack of Vital Sign Time-Series**: Real-time triage would benefit from longitudinal heart rate, blood pressure, and oxygen saturation telemetry.

## 14. Remaining Issues or Dependencies
- **Dependency on EHR Ingestion**: Production scoring depends on clean upstream integration with the hospital registration desk database.
- **Model Drift Monitoring**: Performance must be continuously re-validated against real clinical admission records as seasonal epidemics evolve.

## 15. Your Contribution to the Overall HealthConnect Project
As Lead Healthcare Data Scientist (Track 2), in close collaboration with Zaid Harboul (Track 4), I delivered a thoroughly tested, evidence-based, and calibrated clinical admission risk suite. This contribution ensures that HealthConnect Clinic can safely triage patient admissions, optimize bed allocation, eliminate demographic disparities, and protect patient lives.

## 16. What Must Be Completed Before Week 8
1. Wrap the validated inference pipeline inside a production FastAPI container and build Docker compose configurations.
2. Integrate clinical admission alerts directly into the Executive BI Dashboard for real-time bedside visualization.
3. Prepare the final 15-minute executive presentation deck and capstone presentation for Week 8 graduation.

---
*HealthConnect Rwanda Healthcare Analytics & AI Transformation — AnalystLab Africa Experience Lab.*
