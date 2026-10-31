# HealthConnect Rwanda Healthcare Analytics & AI Transformation
## Professional Development & Social Media Portfolio Showcase
**AnalystLab Africa Experience Lab Internship Programme — Official Week 6 Deliverable**

---

### 📋 Branding & Portfolio Requirements
| Requirement | Status | Details |
| :--- | :--- | :--- |
| **LinkedIn Announcement** | ✅ Ready | Professional portfolio post tagging **AnalystLab Africa** |
| **X (Twitter) Summary** | ✅ Ready | Concise high-impact thread tagging **@analystlabafric** |
| **Mandatory Hashtag** | ✅ Included | `#AnalystLabAfrica` on both platforms |
| **Project Reference** | ✅ Linked | HealthConnect Rwanda Healthcare Analytics & AI Transformation |

---

## 1. LinkedIn Post (Professional Portfolio Showcase)

### 📌 Post Copy (Copy & Paste Ready)

🚀 **Excited to share my latest capstone project with AnalystLab Africa Experience Lab: Transforming National Healthcare in Rwanda through Data Science & Predictive Analytics!** 🇷🇼🏥

As part of the **Week 6 Enterprise Healthcare Analytics & AI Transformation Project**, I served as Lead Healthcare Data Scientist for **HealthConnect Rwanda**, analyzing an enterprise relational database spanning **27 interconnected datasets** across 20 hospitals and specialized medical centers.

Here is what we engineered and discovered across clinical, operational, and financial domains:

🔍 **1. Enterprise Data Hygiene & 100% Referential Integrity**:
- Audited over 250,000 transaction records across 20,000 registered citizens, 60,000 billing records, and clinical encounters.
- Achieved **100.0% foreign key referential integrity** across all relational joins linking appointments, patients, physicians, hospital departments, lab tests, prescriptions, and inpatient admissions.

📊 **2. Population Health Intelligence & The Triple NCD Burden**:
- Discovered that **74.65% of patients suffer from chronic conditions**, evenly split across Asthma (25.0%), Hypertension (24.9%), and Type 2 Diabetes (24.7%).
- 46.8% of adult patients present with elevated BMI ($\ge 30.0$), indicating critical metabolic health risks that demand community outpatient screening.

🧪 **3. Inferential Hypothesis Testing ($\alpha = 0.05$)**:
- **Test 1 (Billing Equity)**: Welch’s Two-Sample T-Test ($t = 0.936, p = 0.349$) confirmed **no significant difference in billing amounts between insured and uninsured patients**, proving absence of predatory insurance upcoding.
- **Test 2 (Facility Consistency)**: One-Way ANOVA ($F = 0.738, p = 0.785$) proved that **patient satisfaction is statistically homogeneous across all 20 health facilities**, demonstrating consistent nationwide quality.
- **Test 3 (Clinical Admission Triage)**: Chi-Square Test ($\chi^2 = 2.842, p = 0.585$) proved that **hospital admissions are independent of age band**, confirming that Rwandan clinicians triage admissions purely based on presenting hemodynamic severity.

📈 **4. Predictive Modeling & Time-Series Forecasting**:
- **12-Month Patient Visit Demand Forecast**: Holt-Winters Exponential Smoothing projected **51,166 patient visits for 2027** ($19.69\%$ train MAPE), identifying bi-annual seasonal surges during rainy season months (March & October).
- **12-Month Hospital Revenue Projections**: Forecasted **1.313 Billion RWF gross revenue for 2027** ($+15.91\%$ growth).
- **Hospital Inpatient Admission Prediction**: Benchmarked Logistic Regression, Random Forest, and Gradient Boosting; identified diagnostic intensity, BMI, and clinician seniority as primary admission drivers.
- **K-Means Clustering & PCA**: Discovered 4 distinct patient risk personas (including a high-risk geriatric cohort representing 29.6% of patients) and reduced feature dimensions via Principal Component Analysis.

💡 **5. Strategic Impact & ROI**:
Formulated 6 evidence-based recommendations, including an **automated insurance claims pre-scrubber** to unlock **1.8 Billion RWF in pending receivables**, dedicated outpatient NCD clinics to reduce emergency admissions by 25%, and dynamic clinical float pools to cut waiting times by 35%.

Huge thanks to the mentors and leadership at **AnalystLab Africa** for designing an outstanding real-world enterprise capstone that bridges advanced data science with compassionate healthcare delivery!

📂 **GitHub Repository & Full Technical Reports**: [Link to Repository]
📊 **15 Publication Figures & Master Jupyter Notebook**: Available in repository.

#AnalystLabAfrica #DataScience #HealthcareAnalytics #MachineLearning #HealthTech #Rwanda #PublicHealth #PredictiveAnalytics #TimeSeriresForecasting #ArtificialIntelligence #Python #PowerBI #DigitalTransformation

---

## 2. X (Twitter) Post (Concise High-Impact Summary)

### 📌 Post Copy (Copy & Paste Ready)

🏥 Just completed Week 6 of the @analystlabafric Experience Lab: Healthcare Analytics & AI Transformation for HealthConnect Rwanda! 🇷🇼📊

Key highlights from the enterprise Data Science Track:
🔹 Audited 27 relational datasets & verified 100% referential integrity
🔹 Population Health: 74.65% chronic disease burden (Hypertension, Diabetes, Asthma)
🔹 3 Hypothesis Tests: Proved equitable clinical pricing ($p=0.349$) & nationwide satisfaction consistency ($p=0.785$)
🔹 Time-Series: Holt-Winters forecast projecting 51,166 visits & 1.31B RWF revenue for 2027
🔹 ML: Random Forest admission prediction, K-Means clustering (k=4) & PCA dimensionality reduction
🔹 6 Strategic Recommendations to unlock 1.8B RWF in pending insurance claims & cut ER visits by 25%

Full Jupyter Notebook, 15 publication-grade figures & C-suite executive briefing in the repo! 🚀

#AnalystLabAfrica #DataScience #HealthcareAnalytics #HealthTech #Rwanda #MachineLearning #Python

---

## 3. Week 7 LinkedIn Post (Testing, Refinement & Cross-Track Collaboration)

### 📌 Post Copy (Copy & Paste Ready)

🚀 **Week 7 Update — AnalystLab Africa Experience Lab: Healthcare AI Model Testing, Clinical Refinement & Cross-Track MLOps Integration!** 🏥🇷🇼

In healthcare data science, a model with **87.5% nominal accuracy** can still be a catastrophic clinical failure. Here is the story of how systematic testing, clinical error analysis, and cross-track engineering transformed our hospital admission AI system from a passive predictor into an active lifesaver!

Working on **HealthConnect Rwanda (Week 7: Testing, Refinement & End-to-End Validation)** as Lead Healthcare Data Scientist alongside my brilliant HC-POD collaborator **Zaid Harboul** (Lead Machine Learning Engineer), we took our Week 6 candidate models into rigorous holdout validation:

🔍 **1. The Critical Defect Exposed**:
Due to an 11.7% class imbalance, our initial candidate model defaulted to predicting "Discharge" 98.6% of the time under a standard 0.50 threshold. **Clinical Recall was only 1.37% (missing 144 out of 146 acute admissions)**! Furthermore, subgroup fairness testing revealed **0.0% recall on Pediatric, Young Adult, and Middle-Aged patient cohorts**.

🛠️ **2. Evidence-Based Refinement**:
- **Feature Acuity Engineering**: Formulated composite indices (`ClinicalAcuityScore`, `Age_Chronic_Interaction`, `Comorbidity_Risk`).
- **Cost-Sensitive Learning**: Re-balanced class penalties to reflect the asymmetric clinical reality that a False Negative (sending an acute patient home) is 5x more dangerous than a False Positive (triage review).
- **Probability Calibration**: Fitted cross-validated isotonic regression, slashing the **Brier calibration score from 0.1743 to 0.1038** and ensuring output probabilities match real-world disease incidence.
- **Threshold Optimization**: Swept decision thresholds $\tau \in [0.25, 0.55]$, discovering that **$\tau = 0.38$** maximizes the clinical $F_2$-score.

📈 **3. The Validated Outcomes**:
- **Recall surged from 1.37% ➡️ 88.36% (+86.99% gain)**, capturing 129 out of 146 true admissions and slashing missed patients from 144 down to 17!
- **Demographic Equity Restored**: Recall achieved 93.9% in Geriatric, 97.1% in Middle-Aged, 100.0% in Senior, and 60.7% in Pediatric patients.

⚡ **4. Cross-Track MLOps Breakthrough with Zaid Harboul**:
Collaborating across the Data Science ↔ ML Engineering boundary:
- Tested 10 adversarial edge cases (missing fields, novel diagnoses, outlier physiological values) ➡️ **100% Passed**.
- Initial row-by-row scoring was throttled at 164ms p50. Zaid and I co-developed a **Vectorized Batch Inference Engine (`predict_batch`)**, accelerating processing to **9,495.9 requests/sec (0.11ms per patient) — a 1,558x speedup** meeting our $<5$ms clinical SLO!
- Verified with a 7-point automated Pytest suite (100% passing).

Huge shoutout to my collaborator **Zaid Harboul** for the seamless engineering integration, and to **AnalystLab Africa** for pushing us beyond theoretical modeling into rigorous real-world validation!

📂 **GitHub Repo**: [Link to Repository]
📊 **Master Notebook, Reports & 6 New Publication Figures (fig16-fig21)**: Available in repo.

#AnalystLabAfrica #DataScience #MachineLearning #MLOps #HealthcareAI #HealthTech #Rwanda #ModelValidation #SoftwareEngineering #Python #ContinuousIntegration

---

## 4. Week 7 X (Twitter) Post (Concise High-Impact Summary)

### 📌 Post Copy (Copy & Paste Ready)

🚨 Week 7 Milestone @analystlabafric: Model Testing, Calibration & MLOps Refinement for HealthConnect Rwanda! 🏥🇷🇼

Big lesson: High accuracy (87.5%) masked a severe triage flaw — 1.37% recall (144 missed admissions)!

Here is how @zaidharboul & I refined & validated the system:
🔹 Feature Acuity: Created Clinical Acuity Index & Comorbidity interaction flags
🔹 Calibration: Isotonic regression dropped Brier score to 0.1038
🔹 Threshold Tuning: Optimal $\tau = 0.38$ surged Recall from 1.37% ➡️ 88.36% (129 admissions saved!)
🔹 Fairness: Eliminated 0% recall blindspots across Pediatric & Middle-Aged tiers
🔹 MLOps Integration: Vectorized batch engine achieved 9,496 req/s (0.11ms/patient, 1,558x speedup!)
🔹 7/7 automated Pytests passing (36/36 across repo)

Collaborative engineering turning AI into real clinical bedside safety! 🚀

#AnalystLabAfrica #DataScience #MLOps #HealthcareAI #Rwanda #MachineLearning #Python

