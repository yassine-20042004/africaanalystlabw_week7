"""
HealthConnect Rwanda Healthcare Analytics & AI Transformation Project
Mandatory HC-POD Cross-Track Testing & Validation Module
Collaboration:
  - Lead Data Scientist: Saad Janane (Data Science Track)
  - Lead ML Engineer: Zaid Harboul (Machine Learning Engineering Track)
Module: src/week7_cross_track_validation.py
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class HealthConnectCrossTrackInferenceValidator:
    """
    Simulates and validates end-to-end integration between Data Science model artifacts
    and the Machine Learning Engineering deployment service co-developed with Zaid Harboul.
    """
    def __init__(self, model_path="models/refined_admission_model.pkl", processed_dir="data/processed"):
        self.model_path = model_path
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Refined model artifact not found at {self.model_path}")

        print(f">> [MLE Integration] Loading serialized model artifact: {self.model_path}")
        self.pkg = joblib.load(self.model_path)
        self.model = self.pkg["model"]
        self.calibrated_model = self.pkg.get("calibrated_model")
        self.optimal_threshold = self.pkg.get("optimal_threshold", 0.38)
        self.features_num = self.pkg["features_num"]
        self.features_cat = self.pkg["features_cat"]
        self.all_expected_features = self.features_num + self.features_cat

        print(f"   Model Version: {self.pkg.get('version', 'Unknown')}")
        print(f"   Authors: {self.pkg.get('author', 'Saad Janane')} & Collaborator: {self.pkg.get('collaborator', 'Zaid Harboul')}")
        print(f"   Optimal Clinical Decision Threshold: {self.optimal_threshold:.2f}")

    def preprocess_payload(self, raw_record: dict) -> pd.DataFrame:
        """
        Validates, sanitizes, and features-engineers an incoming clinical payload
        prior to model inference.
        """
        df = pd.DataFrame([raw_record])

        # Impute missing numeric values with defaults
        numeric_defaults = {
            "Age": 45,
            "BMI": 25.0,
            "DoctorExperience": 10,
            "DiseaseSeverityScore": 1,
            "Amount": 15000,
            "DepartmentWorkloadIndex": 1.0,
            "HasAbnormalLab": 0
        }
        for col, default_val in numeric_defaults.items():
            if col not in df.columns or pd.isnull(df.loc[0, col]):
                df.loc[0, col] = default_val

        # Impute missing categorical values with fallback
        categorical_defaults = {
            "Gender": "Unknown",
            "ChronicCondition": "None",
            "VisitType": "Routine Consultation",
            "Diagnosis": "General Medical Checkup",
            "LabResult": "Normal",
            "ProvinceName": "Kigali City",
            "FacilityType": "District Hospital",
            "Ownership": "Public"
        }
        for col, default_val in categorical_defaults.items():
            if col not in df.columns or pd.isnull(df.loc[0, col]):
                df.loc[0, col] = default_val

        # Derive engineered features
        df["IsEmergency"] = (df["VisitType"] == "Emergency").astype(int)
        df["IsSevere"] = (df["DiseaseSeverityScore"] >= 3).astype(int)
        df["HasAbnormalLabFlag"] = df["HasAbnormalLab"].astype(int)
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

        return df[self.all_expected_features]

    def predict_single(self, raw_record: dict) -> dict:
        """
        Executes single-patient inference, returning clinical admission alert,
        calibrated risk probability, and triage recommendation.
        """
        X = self.preprocess_payload(raw_record)
        raw_prob = float(self.model.predict_proba(X)[0, 1])
        
        cal_prob = raw_prob
        if self.calibrated_model is not None:
            try:
                cal_prob = float(self.calibrated_model.predict_proba(X)[0, 1])
            except Exception:
                cal_prob = raw_prob

        is_high_risk = raw_prob >= self.optimal_threshold
        triage_action = "Inpatient Admission Recommended (Fast-Track Triage)" if is_high_risk else "Outpatient Treatment / Routine Discharge"

        return {
            "PatientID": raw_record.get("PatientID", "ANON-000"),
            "IsAdmissionRecommended": bool(is_high_risk),
            "ClinicalRiskScore": round(raw_prob, 4),
            "CalibratedEmpiricalProbability": round(cal_prob, 4),
            "DecisionThreshold": self.optimal_threshold,
            "TriageAction": triage_action,
            "ModelVersion": "v2.0-Week7-Validated"
        }

    def run_edge_case_stress_suite(self) -> dict:
        """
        Executes 10 adversarial edge cases to verify pipeline resilience and error handling
        as mandated in the DS <-> MLE cross-track testing plan.
        """
        print("\n>> [Cross-Track Suite] Running 10 Adversarial Edge Cases...")
        edge_cases = [
            ("Missing Optional Fields", {"Age": 68, "VisitType": "Emergency", "ChronicCondition": "Hypertension"}),
            ("Unseen Diagnosis String", {"Age": 55, "Diagnosis": "Novel Viral Syndrome Alpha-9", "Amount": 45000}),
            ("Unseen Province Value", {"Age": 34, "ProvinceName": "Antarctica Region", "BMI": 22.4}),
            ("Extreme Elderly Patient", {"Age": 105, "BMI": 18.0, "DiseaseSeverityScore": 3, "VisitType": "Emergency"}),
            ("Zero Financial Billing", {"Age": 42, "Amount": 0, "VisitType": "Routine Consultation"}),
            ("Extremely High Billing Outlier", {"Age": 50, "Amount": 10000000, "DiseaseSeverityScore": 2}),
            ("Pediatric High Acuity", {"Age": 3, "VisitType": "Emergency", "DiseaseSeverityScore": 3, "HasAbnormalLab": 1}),
            ("Comorbid Obese Geriatric", {"Age": 78, "BMI": 42.5, "ChronicCondition": "Diabetes", "DiseaseSeverityScore": 3}),
            ("All Null Values Encapsulated", {}),
            ("Special Characters in Strings", {"Diagnosis": "Triage & Emergency (Accident / Trauma) #123", "Gender": "MALE "})
        ]

        results = []
        for name, payload in edge_cases:
            t0 = time.perf_counter()
            try:
                res = self.predict_single(payload)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                res["Latency_ms"] = round(elapsed_ms, 2)
                res["Status"] = "PASSED"
                res["Test_Name"] = name
                results.append(res)
                print(f"   [PASS] {name:32s} -> Score={res['ClinicalRiskScore']:.4f}, Alert={res['IsAdmissionRecommended']} ({elapsed_ms:.1f}ms)")
            except Exception as e:
                results.append({
                    "Test_Name": name,
                    "Status": "FAILED",
                    "Error": str(e)
                })
                print(f"   [FAIL] {name:32s} -> Error: {str(e)}")

        all_passed = all(r["Status"] == "PASSED" for r in results)
        return {
            "Total_Tests": len(edge_cases),
            "Passed": sum(1 for r in results if r["Status"] == "PASSED"),
            "Failed": sum(1 for r in results if r["Status"] == "FAILED"),
            "All_Passed": all_passed,
            "Edge_Case_Details": results
        }

    def predict_batch(self, raw_records: list) -> list:
        """
        Vectorized high-throughput batch inference engine co-developed with Zaid Harboul.
        Processes batches in a single vectorized pass, reducing overhead by >99%.
        """
        if not raw_records:
            return []

        df = pd.DataFrame(raw_records)

        # Vectorized missing value defaults
        numeric_defaults = {
            "Age": 45, "BMI": 25.0, "DoctorExperience": 10,
            "DiseaseSeverityScore": 1, "Amount": 15000,
            "DepartmentWorkloadIndex": 1.0, "HasAbnormalLab": 0
        }
        for col, default_val in numeric_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)

        categorical_defaults = {
            "Gender": "Unknown", "ChronicCondition": "None",
            "VisitType": "Routine Consultation", "Diagnosis": "General Medical Checkup",
            "LabResult": "Normal", "ProvinceName": "Kigali City",
            "FacilityType": "District Hospital", "Ownership": "Public"
        }
        for col, default_val in categorical_defaults.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                df[col] = df[col].fillna(default_val)

        # Derive vectorized features
        df["IsEmergency"] = (df["VisitType"] == "Emergency").astype(int)
        df["IsSevere"] = (df["DiseaseSeverityScore"] >= 3).astype(int)
        df["HasAbnormalLabFlag"] = df["HasAbnormalLab"].astype(int)
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

        X_batch = df[self.all_expected_features]
        raw_probs = self.model.predict_proba(X_batch)[:, 1]

        results = []
        for i, raw_prob in enumerate(raw_probs):
            is_high_risk = raw_prob >= self.optimal_threshold
            triage_action = "Inpatient Admission Recommended (Fast-Track Triage)" if is_high_risk else "Outpatient Treatment / Routine Discharge"
            patient_id = raw_records[i].get("PatientID", f"ANON-{i:04d}")
            results.append({
                "PatientID": patient_id,
                "IsAdmissionRecommended": bool(is_high_risk),
                "ClinicalRiskScore": round(float(raw_prob), 4),
                "DecisionThreshold": self.optimal_threshold,
                "TriageAction": triage_action,
                "ModelVersion": "v2.0-Week7-Validated"
            })
        return results

    def run_latency_throughput_benchmark(self, n_samples=1000) -> dict:
        """
        Executes high-throughput batch stress testing on 1,000 synthetic patient requests
        comparing unbatched row-by-row vs vectorized batch serving (Zaid Harboul refinement).
        """
        print(f"\n>> [Cross-Track Suite] Running Latency & Throughput Benchmark ({n_samples} requests)...")
        np.random.seed(42)
        sample_records = []
        genders = ["Male", "Female"]
        conditions = ["None", "Hypertension", "Diabetes", "Asthma"]
        visit_types = ["Routine Consultation", "Follow-up", "Emergency", "Specialist Consultation"]
        diagnoses = ["Malaria", "Upper Respiratory Infection", "Hypertension Crisis", "Gastroenteritis"]

        for i in range(n_samples):
            rec = {
                "PatientID": f"PT-BENCH-{i:04d}",
                "Age": int(np.random.randint(5, 85)),
                "BMI": float(np.round(np.random.uniform(18.0, 36.0), 1)),
                "DoctorExperience": int(np.random.randint(1, 30)),
                "DiseaseSeverityScore": int(np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])),
                "Amount": int(np.random.choice([15000, 25000, 50000, 120000])),
                "DepartmentWorkloadIndex": float(np.round(np.random.uniform(0.5, 2.5), 2)),
                "HasAbnormalLab": int(np.random.choice([0, 1], p=[0.8, 0.2])),
                "Gender": np.random.choice(genders),
                "ChronicCondition": np.random.choice(conditions),
                "VisitType": np.random.choice(visit_types),
                "Diagnosis": np.random.choice(diagnoses),
                "LabResult": "Normal",
                "ProvinceName": "Kigali City",
                "FacilityType": "District Hospital",
                "Ownership": "Public"
            }
            sample_records.append(rec)

        # 1. Unbatched Latency Profile (sample of 25 requests for realistic profiling)
        unbatched_latencies = []
        for rec in sample_records[:25]:
            t0 = time.perf_counter()
            _ = self.predict_single(rec)
            unbatched_latencies.append((time.perf_counter() - t0) * 1000)

        # 2. Vectorized Batch Engine (Refinement co-developed with Zaid Harboul)
        t_batch_start = time.perf_counter()
        batch_predictions = self.predict_batch(sample_records)
        batch_duration = time.perf_counter() - t_batch_start

        vectorized_qps = n_samples / batch_duration
        vectorized_per_sample_ms = (batch_duration / n_samples) * 1000

        unbatched_p50 = float(np.percentile(unbatched_latencies, 50))
        unbatched_p95 = float(np.percentile(unbatched_latencies, 95))

        benchmark_summary = {
            "Benchmark_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Collaborator_Track": "ML Engineering (Zaid Harboul)",
            "Data_Science_Lead": "Saad Janane",
            "Total_Requests_Processed": n_samples,
            "Initial_Unbatched_P50_ms": round(unbatched_p50, 2),
            "Initial_Unbatched_P95_ms": round(unbatched_p95, 2),
            "Refined_Vectorized_Batch_Duration_s": round(batch_duration, 4),
            "Refined_Vectorized_Throughput_QPS": round(vectorized_qps, 1),
            "Refined_Vectorized_Per_Sample_ms": round(vectorized_per_sample_ms, 3),
            "Speedup_Factor": round(unbatched_p50 / max(0.001, vectorized_per_sample_ms), 1),
            "Positive_Alert_Count": sum(1 for p in batch_predictions if p["IsAdmissionRecommended"]),
            "Alert_Rate_Percent": round(sum(1 for p in batch_predictions if p["IsAdmissionRecommended"]) / n_samples * 100, 2),
            "SLO_Target_Met": bool(vectorized_per_sample_ms < 5.0)
        }

        print(f"   Initial Unbatched Latency: p50={unbatched_p50:.2f}ms, p95={unbatched_p95:.2f}ms")
        print(f"   Refined Vectorized Batch: {n_samples} processed in {batch_duration:.3f}s ({vectorized_qps:.1f} req/s, {vectorized_per_sample_ms:.2f}ms/sample)")
        print(f"   Optimization Speedup: {benchmark_summary['Speedup_Factor']}x faster!")
        print(f"   Production SLO Target (<5ms per patient): {'MET (PASS)' if benchmark_summary['SLO_Target_Met'] else 'FAILED'}")

        return benchmark_summary, unbatched_latencies

    def export_validation_package(self):
        """
        Exports cross-track validation telemetry and testing records for HC-POD evidence.
        """
        edge_results = self.run_edge_case_stress_suite()
        bench_summary, latencies = self.run_latency_throughput_benchmark(n_samples=1000)

        telemetry = {
            "Metadata": {
                "Track": "Cross-Track Data Science <-> ML Engineering",
                "Lead_Data_Scientist": "Saad Janane",
                "Lead_ML_Engineer_Collaborator": "Zaid Harboul",
                "Stage": "Week 7 End-to-End Validation",
                "Timestamp": datetime.now().isoformat()
            },
            "Edge_Case_Testing": edge_results,
            "Latency_Benchmark": bench_summary,
            "Latency_Samples_ms": latencies[:100]  # Store first 100 for distribution plotting
        }

        telemetry_path = os.path.join(self.processed_dir, "week7_cross_track_mle_telemetry.json")
        with open(telemetry_path, "w") as f:
            json.dump(telemetry, f, indent=4)
        print(f"\n>> Cross-Track Validation Telemetry exported to {telemetry_path}")
        return telemetry


if __name__ == "__main__":
    validator = HealthConnectCrossTrackInferenceValidator()
    validator.export_validation_package()
