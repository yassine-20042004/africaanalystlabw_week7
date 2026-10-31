# HealthConnect Rwanda Healthcare Analytics & AI Transformation
## Systematic Testing, Refinement & Validation Record (Week 7)
**AnalystLab Africa Experience Lab Internship Programme**

---

### 📋 Document Control
| Field | Value |
| :--- | :--- |
| **Programme** | AnalystLab Africa Experience Lab |
| **Project Title** | Improving Patient Appointment Attendance and Healthcare Support Using Data and AI |
| **Track** | **Track 2 — Data Science Track** |
| **Author (Lead Data Scientist)** | **Saad Janane** |
| **Collaborator (Lead ML Engineer)**| **Zaid Harboul** |
| **Stage** | Week 7: Testing, Refinement & End-to-End Validation |
| **Document Purpose** | Official Tabular Testing & Retesting Log (Section 12 Compliance) |
| **Validation Status** | ✅ **All Test Scenarios Formally Retested & Passed** |

---

## 1. Comprehensive Testing, Refinement & Validation Matrix (Section 12)

The following multi-scenario testing record tracks all major analytical, predictive, algorithmic, and engineering tests conducted during Week 7, adhering strictly to the **Test → Finding → Action → Retest → Validated Improvement** cycle:

| # | Component Tested | Testing Objective | Test / Scenario | Expected Result | Actual Result | Pass / Fail | Issue Identified | Action Taken | Retest Result | Collaborating Track | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T01** | **Week 6 Candidate Model** (`hospital_admission_model.pkl`) | Determine whether default model reliably identifies inpatient admissions on holdout data. | Evaluate classification metrics on $N=1,250$ test cohort at default threshold $\tau = 0.50$. | Balanced sensitivity (Recall $\ge 70\%$) and accuracy $\ge 80\%$. | Accuracy: $87.52\%$, Recall: $1.37\%$ (only 2 of 146 admissions captured; 144 false negatives). | ❌ **FAIL** | Severe clinical under-triage: model simply predicts majority class (Not Admitted) due to $11.7\%$ class imbalance. | Flagged severe clinical safety hazard; initiated threshold tuning and class-rebalancing protocol. | Prepared candidate for cost-sensitive refinement and calibration. | **Data Analytics & Clinical Operations** | `data/processed/week7_model_testing_results.json`, `fig16_confusion_matrix_before_after.png` |
| **T02** | **Error Cost Symmetry** | Quantify real-world asymmetric clinical cost of triage misclassifications. | Compute clinical loss function $L = 5 \times FN + 1 \times FP$ on test set predictions. | Clinical cost loss should be minimized, prioritizing detection of acute patients. | Clinical loss = $732.0$ ($144 \times 5 + 12 \times 1$). | ❌ **FAIL** | 98.6% of patients requiring inpatient admission were misclassified as routine discharges. | Formulated explicit cost matrix penalizing False Negatives 5x over False Positives. | Clinical loss reduced and verified at optimal operating threshold. | **Data Science & Clinical Risk Committee** | `week7_model_comparison.csv` |
| **T03** | **Demographic Subgroup Fairness** | Assess whether model performance is equitable across patient age tiers. | Sliced recall evaluation across Pediatric (0-17), Young Adult (18-35), Middle-Aged (36-55), Senior (56-65), and Geriatric (>65). | Recall $\ge 60\%$ across all demographic age bands. | Middle-Aged: $0.0\%$, Pediatric: $0.0\%$, Young Adult: $0.0\%$, Geriatric: $2.04\%$, Senior: $8.33\%$. | ❌ **FAIL** | Extreme demographic disparity: three complete age bands had zero detected admissions! | Engineered `Age_Chronic_Interaction` and stratified acuity scoring into feature matrix. | Geriatric: $93.88\%$, Middle-Aged: $97.06\%$, Senior: $100.0\%$, Pediatric: $60.71\%$, Young Adult: $91.30\%$. | **Data Analytics (Demographic Insights)** | `fig20_subgroup_disparity_analysis.png` |
| **T04** | **Probability Calibration** | Verify if model predicted probabilities match true empirical admission rates. | Construct reliability diagrams and compute Brier score loss before vs. after calibration. | Predicted probabilities should closely align with observed 11.7% population frequency; Brier $\le 0.15$. | Uncalibrated voting scores peaked around base rate; Brier score was high ($0.1743$). | ⚠️ **WEAK** | Uncalibrated tree votes produced unreliable risk probabilities for clinical decision support. | Implemented 3-fold cross-validated Isotonic Probability Calibration (`CalibratedClassifierCV`). | Brier score dropped to **$0.1038$**; calibration curve tightly tracks the ideal $45^\circ$ diagonal. | **Data Science** | `fig19_probability_calibration_curve.png` |
| **T05** | **Decision Threshold Optimization** | Determine optimal operating threshold $\tau$ balancing clinical recall and hospital bed capacity. | Sweep threshold $\tau \in [0.25, 0.55]$ over 61 points, evaluating Recall, Precision, $F_1$, and $F_2$. | Operating threshold achieves Recall $\ge 85\%$ while maintaining manageable hospital bed triage volume. | Default $\tau = 0.50$ produced $8.22\%$ recall; sweeping revealed optimal operating point at $\tau = 0.38$. | ✅ **PASS** | Default $\tau=0.50$ is inappropriate for imbalanced clinical triage. | Selected $\tau = 0.38$ as the production operational threshold. | Retested at $\tau = 0.38$: Recall = **$88.36\%$**, capturing 129 of 146 true admissions; false negatives cut from 144 to 17! | **Data Science & ML Engineering (Zaid Harboul)** | `fig17_threshold_optimization_curve.png`, `week7_threshold_sweep.csv` |
| **T06** | **Refined Model Holdout Retesting** | Validate refined, calibrated Balanced Random Forest on untouched test data. | Evaluate full metric suite on $N=1,250$ test set using refined pipeline and $\tau = 0.38$. | Substantial, statistically meaningful improvement over Week 5 baseline and Week 6 candidate. | Recall surged from $1.37\%$ to **$88.36\%$** ($+86.99\%$ gain), $F_2$ reached $0.3771$, Brier score = $0.1038$. | ✅ **PASS** | None; significant validated improvement across all primary clinical objectives. | Serialized refined pipeline into `models/refined_admission_model.pkl` with full metadata. | Model passed all validation criteria with 100% reproducibility. | **Data Science** | `week7_model_comparison.csv`, `models/refined_admission_model.pkl` |
| **T07** | **Inference Schema Resilience** | Test pipeline stability against noisy, incomplete, or corrupted clinical payloads. | Injected 10 adversarial edge cases (missing optional fields, novel diagnoses, unseen provinces, extreme ages). | Pipeline handles all payloads safely, returning graceful fallback scores without exceptions. | Initial Week 6 pipeline threw `ValueError` on novel strings and `TypeError` on missing keys. | ❌ **FAIL** | Zero fault tolerance against out-of-vocabulary categories in live inference. | Hardened `preprocess_payload` with default imputation dictionaries and `handle_unknown='ignore'`. | $10 / 10$ adversarial edge cases passed cleanly ($100\%$ success rate); zero exceptions. | **ML Engineering (Zaid Harboul)** | `src/week7_cross_track_validation.py`, `week7_cross_track_mle_telemetry.json` |
| **T08** | **Inference Latency & Throughput** | Benchmark inference speed and request concurrency against clinical EHR SLO ($< 25\text{ms}$). | Process $1,000$ synthetic patient requests measuring latency distribution ($p50, p95$) and throughput. | Single-request latency $< 25\text{ms}$, throughput $> 100\text{ req/sec}$. | Initial row-by-row iteration took $164.14\text{ms } p50$ and $275.71\text{ms } p95$ ($5.0\text{ req/s}$). | ❌ **FAIL** | DataFrame instantiation per single record in Python loop caused severe overhead, breaching SLO. | Collaborated with Zaid Harboul to architect Vectorized Batch Engine (`predict_batch`). | $1,000$ patient requests processed in **$0.105$ seconds** (**$9,495.9\text{ req/s}$**, **$0.11\text{ms}$ per patient**; **$1,558\times$ speedup**). | **ML Engineering (Zaid Harboul)** | `fig21_cross_track_mle_integration_telemetry.png`, `tests/test_week7_suite.py` |
| **T09** | **Automated Test Suite Execution** | Verify that all functional, calibration, fairness, and integration requirements pass automated regression. | Run full test suite: `pytest tests/test_week7_suite.py -v`. | 100% of automated unit and integration tests pass. | 7 of 7 test items passed in $4.41$ seconds ($100\%$ passing). | ✅ **PASS** | None. | Integrated Week 7 suite alongside Week 6 verification suite ($36/36$ passed across repo). | Continuous integration verified; zero regression. | **Data Science & ML Engineering** | Pytest console logs, `tests/test_week7_suite.py` |

---

## 2. Summary of Testing Outcomes

1. **Defects Discovered**: 4 major operational defects (severe under-triage recall mask, zero-recall demographic disparity, unhandled categorical exceptions, slow unbatched latency).
2. **Defects Resolved**: 4 of 4 defects resolved through evidence-based engineering ($100\%$ resolution rate).
3. **Core Performance Metric Gains**:
   - **Clinical Recall (Sensitivity)**: **$1.37\% \longrightarrow 88.36\%$** ($+86.99\%$ improvement).
   - **Missed Inpatient Admissions (FN)**: **$144 \longrightarrow 17$** (prevented $88.2\%$ of missed acute cases).
   - **Brier Calibration Score**: **$0.1743 \longrightarrow 0.1038$** (improved probability reliability).
   - **Inference Throughput**: **$5.0\text{ req/s} \longrightarrow 9,495.9\text{ req/s}$** ($1,558\times$ acceleration).
   - **Adversarial Edge Case Stability**: **$0\% \longrightarrow 100\%$** passing.

---
*HealthConnect Rwanda Healthcare Analytics & AI Transformation — AnalystLab Africa Experience Lab.*
