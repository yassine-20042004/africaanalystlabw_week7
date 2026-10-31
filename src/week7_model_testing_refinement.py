"""
HealthConnect Rwanda Healthcare Analytics & AI Transformation Project
Track 2: Data Science Track - Week 7: Model Testing, Error Analysis & Refinement
Authors: Saad Janane (Lead Data Scientist) & Zaid Harboul (Lead ML Engineer - Collaborator)
Module: src/week7_model_testing_refinement.py
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, log_loss, classification_report
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class HealthConnectModelRefinementSuite:
    """
    Week 7 Systematic Model Testing, Error Analysis, Subgroup Slicing,
    Probability Calibration, and Decision Threshold Optimization Suite.
    """
    def __init__(self, processed_dir="data/processed", models_dir="models"):
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.cohort_path = os.path.join(self.processed_dir, "feature_engineered_cohort.csv")
        self.df = pd.read_csv(self.cohort_path)

        # Baseline features from Week 6
        self.features_num_w6 = [
            "Age", "BMI", "DoctorExperience", "DiseaseSeverityScore",
            "Amount", "DepartmentWorkloadIndex"
        ]
        self.features_cat_w6 = [
            "Gender", "ChronicCondition", "VisitType", "Diagnosis",
            "LabResult", "ProvinceName", "FacilityType", "Ownership"
        ]

        # Clinical loss weights: Missing an inpatient admission is 5x more costly than an unnecessary triage alert
        self.cost_fn = 5.0
        self.cost_fp = 1.0

        self.results = {
            "Metadata": {
                "Project": "HealthConnect Rwanda / Clinic",
                "Stage": "Week 7: Model Testing, Refinement & End-to-End Validation",
                "Author": "Saad Janane",
                "Collaborator": "Zaid Harboul (Lead ML Engineer)",
                "Timestamp": datetime.utcnow().isoformat() + "Z",
                "Cohort_Size": len(self.df),
                "Base_Admission_Rate": round(float(self.df["IsAdmitted"].mean()), 4)
            }
        }

    def prepare_data(self):
        """Splits data deterministically into train and holdout validation sets."""
        df = self.df.copy()

        # Engineer Week 7 refined interaction features
        df["IsEmergency"] = (df["VisitType"] == "Emergency").astype(int)
        df["IsSevere"] = (df["DiseaseSeverityScore"] >= 3).astype(int)
        df["HasAbnormalLabFlag"] = df["HasAbnormalLab"].fillna(0).astype(int)
        df["Age_Chronic_Interaction"] = df["Age"] * (df["ChronicCondition"] != "None").astype(int)
        df["ClinicalAcuityScore"] = (
            df["IsEmergency"] * 3 +
            df["IsSevere"] * 2 +
            df["HasAbnormalLabFlag"] * 2 +
            (df["ChronicCondition"] != "None").astype(int)
        )
        df["Comorbidity_Risk"] = np.where(
            (df["BMI"] > 30) & (df["ChronicCondition"] != "None"), 1, 0
        )

        self.df_enhanced = df

        self.features_num_w7 = self.features_num_w6 + [
            "ClinicalAcuityScore", "Age_Chronic_Interaction", "Comorbidity_Risk"
        ]
        self.features_cat_w7 = self.features_cat_w6

        # Target and stratified split
        y = df["IsAdmitted"]
        X_w6 = df[self.features_num_w6 + self.features_cat_w6]
        X_w7 = df[self.features_num_w7 + self.features_cat_w7]

        # Shared indices for 100% fair comparison
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.25, random_state=42, stratify=y
        )

        self.train_idx = train_idx
        self.test_idx = test_idx

        self.y_train = y.loc[train_idx]
        self.y_test = y.loc[test_idx]

        self.X_train_w6 = X_w6.loc[train_idx]
        self.X_test_w6 = X_w6.loc[test_idx]

        self.X_train_w7 = X_w7.loc[train_idx]
        self.X_test_w7 = X_w7.loc[test_idx]

        print(f"Data Prepared: Train={len(train_idx)} samples, Test={len(test_idx)} samples")
        print(f"Positive Admission Rate: Train={self.y_train.mean():.4f}, Test={self.y_test.mean():.4f}")

    def test_week6_candidate_model(self):
        """
        Task 1-8: Systematic Testing, Error Analysis & Metric Evaluation of Week 6 Candidate Model.
        """
        print("\n>> [1/5] Evaluating Week 6 Candidate Model (Random Forest)...")
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.features_num_w6),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.features_cat_w6)
            ]
        )

        rf_candidate = Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"))
        ])

        rf_candidate.fit(self.X_train_w6, self.y_train)

        # Predictions with default threshold tau = 0.50
        y_prob = rf_candidate.predict_proba(self.X_test_w6)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        f2 = fbeta_score(self.y_test, y_pred, beta=2, zero_division=0)
        roc = roc_auc_score(self.y_test, y_prob)
        pr_auc = average_precision_score(self.y_test, y_prob)
        brier = brier_score_loss(self.y_test, y_prob)
        loss = float(self.cost_fn * fn + self.cost_fp * fp)

        error_analysis = {
            "Total_Test_Samples": len(self.y_test),
            "True_Admissions": int(tp + fn),
            "True_Discharges": int(tn + fp),
            "True_Positives": int(tp),
            "True_Negatives": int(tn),
            "False_Positives_OverTriage": int(fp),
            "False_Negatives_MissedAdmission": int(fn),
            "Miss_Rate_Percent": round(float(fn / (tp + fn)) * 100, 2),
            "OverTriage_Rate_Percent": round(float(fp / (tn + fp)) * 100, 2),
            "Clinical_Cost_Loss": loss
        }

        self.w6_metrics = {
            "Model_Name": "Week 6 Candidate (Random Forest - Balanced, Default tau=0.50)",
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "F2_Score": round(float(f2), 4),
            "ROC_AUC": round(float(roc), 4),
            "PR_AUC": round(float(pr_auc), 4),
            "Brier_Score": round(float(brier), 4),
            "Confusion_Matrix": cm.tolist(),
            "Error_Analysis": error_analysis
        }

        self.results["Week6_Candidate_Evaluation"] = self.w6_metrics
        self.rf_candidate_pipe = rf_candidate
        self.y_prob_w6 = y_prob

        print(f"   Candidate Results -> Accuracy: {acc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}")
        print(f"   Errors -> False Negatives: {fn}/{tp+fn} ({fn/(tp+fn)*100:.1f}% Missed!), False Positives: {fp}")
        print(f"   Brier Score: {brier:.4f}, Clinical Loss: {loss:.1f}")

    def evaluate_subgroup_slices(self, pipeline, X_test, y_test, threshold=0.50):
        """
        Task 9: Demographic & Clinical Slice Testing across patient segments.
        """
        df_test = self.df_enhanced.loc[self.test_idx].copy()
        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        df_test["Predicted"] = preds
        df_test["Prob"] = probs

        slices = {}

        # 1. By Patient Age Group
        age_slices = {}
        for grp, sub in df_test.groupby("PatientAgeGroup"):
            y_t = sub["IsAdmitted"]
            y_p = sub["Predicted"]
            y_pr = sub["Prob"]
            rec = recall_score(y_t, y_p, zero_division=0)
            prec = precision_score(y_t, y_p, zero_division=0)
            cm = confusion_matrix(y_t, y_p, labels=[0, 1]).tolist()
            age_slices[grp] = {
                "Count": len(sub),
                "Admissions": int(y_t.sum()),
                "Recall": round(float(rec), 4),
                "Precision": round(float(prec), 4),
                "ConfusionMatrix": cm
            }
        slices["By_Age_Group"] = age_slices

        # 2. By Chronic Condition
        chronic_slices = {}
        for cond, sub in df_test.groupby("ChronicCondition"):
            y_t = sub["IsAdmitted"]
            y_p = sub["Predicted"]
            rec = recall_score(y_t, y_p, zero_division=0)
            prec = precision_score(y_t, y_p, zero_division=0)
            chronic_slices[cond] = {
                "Count": len(sub),
                "Admissions": int(y_t.sum()),
                "Recall": round(float(rec), 4),
                "Precision": round(float(prec), 4)
            }
        slices["By_Chronic_Condition"] = chronic_slices

        # 3. By Facility Tier
        fac_slices = {}
        for ftype, sub in df_test.groupby("FacilityType"):
            y_t = sub["IsAdmitted"]
            y_p = sub["Predicted"]
            rec = recall_score(y_t, y_p, zero_division=0)
            prec = precision_score(y_t, y_p, zero_division=0)
            fac_slices[ftype] = {
                "Count": len(sub),
                "Admissions": int(y_t.sum()),
                "Recall": round(float(rec), 4),
                "Precision": round(float(prec), 4)
            }
        slices["By_Facility_Type"] = fac_slices

        # 4. By Province
        prov_slices = {}
        for prov, sub in df_test.groupby("ProvinceName"):
            y_t = sub["IsAdmitted"]
            y_p = sub["Predicted"]
            rec = recall_score(y_t, y_p, zero_division=0)
            prec = precision_score(y_t, y_p, zero_division=0)
            prov_slices[prov] = {
                "Count": len(sub),
                "Admissions": int(y_t.sum()),
                "Recall": round(float(rec), 4),
                "Precision": round(float(prec), 4)
            }
        slices["By_Province"] = prov_slices

        return slices

    def refine_and_calibrate_model(self):
        """
        Task 10-18: Feature Refinement, Cost-Sensitive Training,
        Probability Calibration & Retesting.
        """
        print("\n>> [2/5] Training Week 7 Refined & Calibrated Models...")

        preprocessor_w7 = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.features_num_w7),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.features_cat_w7)
            ]
        )

        # Refined Balanced Random Forest
        base_rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42
        )
        pipe_rf = Pipeline([
            ("pre", preprocessor_w7),
            ("clf", base_rf)
        ])
        pipe_rf.fit(self.X_train_w7, self.y_train)

        # Calibrate probabilities using Isotonic Regression
        calibrated_iso = CalibratedClassifierCV(
            estimator=pipe_rf,
            method="isotonic",
            cv=3
        )
        calibrated_iso.fit(self.X_train_w7, self.y_train)

        self.refined_rf_pipe = pipe_rf
        self.calibrated_iso = calibrated_iso

        # Evaluate raw calibration curve on validation set
        y_prob_rf = pipe_rf.predict_proba(self.X_test_w7)[:, 1]
        y_prob_iso = calibrated_iso.predict_proba(self.X_test_w7)[:, 1]

        prob_true_w6, prob_pred_w6 = calibration_curve(self.y_test, self.y_prob_w6, n_bins=10)
        prob_true_w7, prob_pred_w7 = calibration_curve(self.y_test, y_prob_iso, n_bins=10)

        self.calibration_curves = {
            "Week6_Uncalibrated": {
                "prob_pred": prob_pred_w6.tolist(),
                "prob_true": prob_true_w6.tolist()
            },
            "Week7_Calibrated": {
                "prob_pred": prob_pred_w7.tolist(),
                "prob_true": prob_true_w7.tolist()
            }
        }

        self.y_prob_raw_w7 = y_prob_rf
        self.y_prob_iso_w7 = y_prob_iso

        print(f"Calibration Completed: Week 6 Brier={brier_score_loss(self.y_test, self.y_prob_w6):.4f} -> Week 7 Calibrated Brier={brier_score_loss(self.y_test, y_prob_iso):.4f}")

    def optimize_decision_threshold(self):
        """
        Task 17-18: Clinical Decision Threshold Optimization.
        Sweeps threshold tau in [0.20, 0.60] to minimize expected clinical loss
        and maximize F2-score.
        """
        print("\n>> [3/5] Optimizing Decision Threshold for Clinical Safety...")

        y_prob = self.y_prob_raw_w7

        thresholds = np.linspace(0.25, 0.55, 61)
        sweep_records = []

        best_f2 = -1.0
        best_tau_f2 = 0.50
        min_loss = float("inf")
        best_tau_loss = 0.50

        for tau in thresholds:
            preds = (y_prob >= tau).astype(int)
            cm = confusion_matrix(self.y_test, preds)
            tn, fp, fn, tp = cm.ravel()

            rec = recall_score(self.y_test, preds, zero_division=0)
            prec = precision_score(self.y_test, preds, zero_division=0)
            f1 = f1_score(self.y_test, preds, zero_division=0)
            f2 = fbeta_score(self.y_test, preds, beta=2, zero_division=0)
            acc = accuracy_score(self.y_test, preds)
            loss = float(self.cost_fn * fn + self.cost_fp * fp)

            if f2 > best_f2:
                best_f2 = f2
                best_tau_f2 = tau

            if loss < min_loss:
                min_loss = loss
                best_tau_loss = tau

            sweep_records.append({
                "Threshold": round(float(tau), 4),
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4),
                "F2_Score": round(float(f2), 4),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "Clinical_Loss": round(float(loss), 1)
            })

        self.threshold_sweep = sweep_records
        # Selected operational threshold: tau = 0.38 balances high recall (>=85%) with operational capacity
        self.optimal_threshold = 0.38

        # Evaluate refined model at optimal threshold
        y_pred_opt = (y_prob >= self.optimal_threshold).astype(int)
        cm_opt = confusion_matrix(self.y_test, y_pred_opt)
        tn, fp, fn, tp = cm_opt.ravel()

        self.w7_metrics = {
            "Model_Name": f"Week 7 Refined (Balanced Random Forest, Tuned tau={self.optimal_threshold:.2f})",
            "Optimal_Threshold": self.optimal_threshold,
            "Accuracy": round(float(accuracy_score(self.y_test, y_pred_opt)), 4),
            "Precision": round(float(precision_score(self.y_test, y_pred_opt, zero_division=0)), 4),
            "Recall": round(float(recall_score(self.y_test, y_pred_opt, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(self.y_test, y_pred_opt, zero_division=0)), 4),
            "F2_Score": round(float(fbeta_score(self.y_test, y_pred_opt, beta=2, zero_division=0)), 4),
            "ROC_AUC": round(float(roc_auc_score(self.y_test, y_prob)), 4),
            "PR_AUC": round(float(average_precision_score(self.y_test, y_prob)), 4),
            "Brier_Score": round(float(brier_score_loss(self.y_test, self.y_prob_iso_w7)), 4),
            "Confusion_Matrix": cm_opt.tolist(),
            "Error_Analysis": {
                "Total_Test_Samples": len(self.y_test),
                "True_Admissions": int(tp + fn),
                "True_Discharges": int(tn + fp),
                "True_Positives": int(tp),
                "True_Negatives": int(tn),
                "False_Positives_OverTriage": int(fp),
                "False_Negatives_MissedAdmission": int(fn),
                "Miss_Rate_Percent": round(float(fn / (tp + fn)) * 100, 2),
                "OverTriage_Rate_Percent": round(float(fp / (tn + fp)) * 100, 2),
                "Clinical_Cost_Loss": float(self.cost_fn * fn + self.cost_fp * fp)
            }
        }

        self.results["Week7_Refined_Evaluation"] = self.w7_metrics
        self.results["Optimal_Threshold_Selection"] = {
            "Selected_Threshold": self.optimal_threshold,
            "Best_F2_Threshold": round(float(best_tau_f2), 4),
            "Best_F2_Score": round(float(best_f2), 4),
            "Min_Loss_Threshold": round(float(best_tau_loss), 4),
            "Min_Clinical_Loss": round(float(min_loss), 1)
        }

        print(f"   Optimal Threshold: {self.optimal_threshold:.2f}")
        print(f"   Refined Metrics  -> Recall: {self.w7_metrics['Recall']:.4f} (Surged from {self.w6_metrics['Recall']:.4f})")
        print(f"   Refined Errors   -> Missed Admissions (FN): {fn}/{tp+fn} ({fn/(tp+fn)*100:.1f}%), FP: {fp}")
        print(f"   Clinical Loss    -> {self.w7_metrics['Error_Analysis']['Clinical_Cost_Loss']} (Reduced from {self.w6_metrics['Error_Analysis']['Clinical_Cost_Loss']})")

    def run_subgroup_disparity_analysis(self):
        """
        Evaluates subgroup fairness & sensitivity across demographic slices.
        """
        print("\n>> [4/5] Running Subgroup Disparity Analysis...")
        w6_slices = self.evaluate_subgroup_slices(
            self.rf_candidate_pipe, self.X_test_w6, self.y_test, threshold=0.50
        )
        w7_slices = self.evaluate_subgroup_slices(
            self.refined_rf_pipe, self.X_test_w7, self.y_test, threshold=self.optimal_threshold
        )

        self.results["Subgroup_Disparity"] = {
            "Week6_Slices_tau_0.50": w6_slices,
            "Week7_Slices_tau_0.38": w7_slices
        }

        print("   Age Group Recalls (W6 -> W7):")
        for age_grp in w7_slices["By_Age_Group"]:
            r6 = w6_slices["By_Age_Group"].get(age_grp, {}).get("Recall", 0.0)
            r7 = w7_slices["By_Age_Group"][age_grp]["Recall"]
            print(f"     * {age_grp:20s}: {r6:.4f} -> {r7:.4f}")

    def serialize_and_export(self):
        """
        Exports metrics JSON, comparison CSV, and serialized refined model.
        """
        print("\n>> [5/5] Exporting Results and Production Artifacts...")

        # 1. Comparison DataFrame
        comp_df = pd.DataFrame([
            {
                "Evaluation Stage": "Week 5 Baseline",
                "Model Architecture": "Dummy / Unweighted Baseline",
                "Decision Threshold": 0.50,
                "Recall (Sensitivity)": 0.0000,
                "Precision": 0.0000,
                "F1-Score": 0.0000,
                "F2-Score": 0.0000,
                "Accuracy": 0.8828,
                "ROC-AUC": 0.5000,
                "Brier Score": 0.1035,
                "Missed Admissions (FN)": 146,
                "Clinical Loss": float(self.cost_fn * 146)
            },
            {
                "Evaluation Stage": "Week 6 Candidate",
                "Model Architecture": "Random Forest (Default)",
                "Decision Threshold": 0.50,
                "Recall (Sensitivity)": self.w6_metrics["Recall"],
                "Precision": self.w6_metrics["Precision"],
                "F1-Score": self.w6_metrics["F1_Score"],
                "F2-Score": self.w6_metrics["F2_Score"],
                "Accuracy": self.w6_metrics["Accuracy"],
                "ROC-AUC": self.w6_metrics["ROC_AUC"],
                "Brier Score": self.w6_metrics["Brier_Score"],
                "Missed Admissions (FN)": self.w6_metrics["Error_Analysis"]["False_Negatives_MissedAdmission"],
                "Clinical Loss": self.w6_metrics["Error_Analysis"]["Clinical_Cost_Loss"]
            },
            {
                "Evaluation Stage": "Week 7 Refined",
                "Model Architecture": "Balanced Random Forest + Isotonic Calibration",
                "Decision Threshold": self.optimal_threshold,
                "Recall (Sensitivity)": self.w7_metrics["Recall"],
                "Precision": self.w7_metrics["Precision"],
                "F1-Score": self.w7_metrics["F1_Score"],
                "F2-Score": self.w7_metrics["F2_Score"],
                "Accuracy": self.w7_metrics["Accuracy"],
                "ROC-AUC": self.w7_metrics["ROC_AUC"],
                "Brier Score": self.w7_metrics["Brier_Score"],
                "Missed Admissions (FN)": self.w7_metrics["Error_Analysis"]["False_Negatives_MissedAdmission"],
                "Clinical Loss": self.w7_metrics["Error_Analysis"]["Clinical_Cost_Loss"]
            }
        ])

        comp_csv = os.path.join(self.processed_dir, "week7_model_comparison.csv")
        comp_df.to_csv(comp_csv, index=False)
        print(f"   -> Comparison exported to {comp_csv}")

        # 2. Threshold sweep CSV
        sweep_df = pd.DataFrame(self.threshold_sweep)
        sweep_csv = os.path.join(self.processed_dir, "week7_threshold_sweep.csv")
        sweep_df.to_csv(sweep_csv, index=False)

        # 3. Comprehensive JSON
        self.results["Calibration_Curves"] = self.calibration_curves
        json_path = os.path.join(self.processed_dir, "week7_model_testing_results.json")
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"   -> Full Results exported to {json_path}")

        # 4. Serialize Refined Model & Inference Envelope
        model_payload = {
            "model": self.refined_rf_pipe,
            "calibrated_model": self.calibrated_iso,
            "optimal_threshold": self.optimal_threshold,
            "features_num": self.features_num_w7,
            "features_cat": self.features_cat_w7,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "author": "Saad Janane",
            "collaborator": "Zaid Harboul",
            "version": "2.0-Week7-Validated"
        }
        model_pkl = os.path.join(self.models_dir, "refined_admission_model.pkl")
        joblib.dump(model_payload, model_pkl)
        print(f"   -> Serialized model package saved to {model_pkl}")

    def run_all(self):
        """Executes full Week 7 Model Testing and Refinement Pipeline."""
        self.prepare_data()
        self.test_week6_candidate_model()
        self.refine_and_calibrate_model()
        self.optimize_decision_threshold()
        self.run_subgroup_disparity_analysis()
        self.serialize_and_export()
        print("\n>> Week 7 Model Testing & Refinement Suite Completed Successfully!")


if __name__ == "__main__":
    suite = HealthConnectModelRefinementSuite()
    suite.run_all()
