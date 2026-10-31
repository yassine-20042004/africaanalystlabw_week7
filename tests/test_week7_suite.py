"""
Automated Unit and Integration Test Suite for Week 7 HealthConnect Rwanda
Model Testing, Refinement, Calibration & Cross-Track Validation
Authors: Saad Janane (Lead Data Scientist) & Zaid Harboul (Lead ML Engineer - Collaborator)
File: tests/test_week7_suite.py
"""

import os
import json
import pytest
import joblib
import pandas as pd
import numpy as np


def test_week7_processed_artifacts_exist():
    """Verify that all Week 7 analytical results, sweeps, and models exist."""
    required_files = [
        "data/processed/week7_model_comparison.csv",
        "data/processed/week7_model_testing_results.json",
        "data/processed/week7_threshold_sweep.csv",
        "data/processed/week7_cross_track_mle_telemetry.json",
        "models/refined_admission_model.pkl"
    ]
    for path in required_files:
        assert os.path.exists(path), f"Missing required Week 7 artifact: {path}"


def test_refined_model_recall_threshold():
    """Verify that the refined model achieves clinical recall >= 80% at tuned threshold."""
    results_path = "data/processed/week7_model_testing_results.json"
    with open(results_path, "r") as f:
        data = json.load(f)

    w6_recall = data["Week6_Candidate_Evaluation"]["Recall"]
    w7_recall = data["Week7_Refined_Evaluation"]["Recall"]
    w7_optimal_tau = data["Week7_Refined_Evaluation"]["Optimal_Threshold"]

    assert w7_recall >= 0.80, f"Refined recall {w7_recall:.4f} is below 80% clinical safety target"
    assert w7_recall > w6_recall, f"Refined recall ({w7_recall:.4f}) did not improve over Week 6 ({w6_recall:.4f})"
    assert 0.20 <= w7_optimal_tau <= 0.50, f"Optimal threshold {w7_optimal_tau} outside reasonable bounds"


def test_probability_calibration_brier_score():
    """Verify that probability calibration improves calibration and maintains Brier score <= 0.15."""
    results_path = "data/processed/week7_model_testing_results.json"
    with open(results_path, "r") as f:
        data = json.load(f)

    brier = data["Week7_Refined_Evaluation"]["Brier_Score"]
    assert brier <= 0.15, f"Brier score {brier:.4f} exceeds target maximum of 0.15"


def test_subgroup_demographic_fairness():
    """Verify that every demographic age group achieves clinical recall > 50% in Week 7."""
    results_path = "data/processed/week7_model_testing_results.json"
    with open(results_path, "r") as f:
        data = json.load(f)

    w7_age_slices = data["Subgroup_Disparity"]["Week7_Slices_tau_0.38"]["By_Age_Group"]
    for grp, metrics in w7_age_slices.items():
        rec = metrics["Recall"]
        assert rec >= 0.50, f"Age group '{grp}' recall {rec:.4f} is unacceptably low (under 50%)"


def test_cross_track_mle_edge_cases():
    """Verify that the cross-track inference pipeline with Zaid Harboul passed all edge cases."""
    telemetry_path = "data/processed/week7_cross_track_mle_telemetry.json"
    with open(telemetry_path, "r") as f:
        data = json.load(f)

    edge_results = data["Edge_Case_Testing"]
    assert edge_results["All_Passed"] is True, "One or more edge cases failed in MLE integration"
    assert edge_results["Passed"] == edge_results["Total_Tests"], "Edge case pass count mismatch"


def test_cross_track_mle_latency_slo():
    """Verify that vectorized batch inference satisfies the SLO of < 5ms per patient."""
    telemetry_path = "data/processed/week7_cross_track_mle_telemetry.json"
    with open(telemetry_path, "r") as f:
        data = json.load(f)

    bench = data["Latency_Benchmark"]
    assert bench["SLO_Target_Met"] is True, "Vectorized latency did not satisfy SLO target (<5ms)"
    assert bench["Refined_Vectorized_Per_Sample_ms"] < 5.0, (
        f"Per sample latency {bench['Refined_Vectorized_Per_Sample_ms']}ms exceeds 5.0ms threshold"
    )
    assert bench["Speedup_Factor"] > 10.0, f"Speedup factor {bench['Speedup_Factor']}x is insufficient"


def test_publication_figures_exist():
    """Verify that all 6 Week 7 publication figures exist and are non-empty."""
    figures_dir = "reports/figures"
    fig_names = [f"fig{i:02d}" for i in range(16, 22)]
    for prefix in fig_names:
        matches = [f for f in os.listdir(figures_dir) if f.startswith(prefix) and f.endswith(".png")]
        assert len(matches) > 0, f"Missing figure {prefix} in {figures_dir}"
        file_path = os.path.join(figures_dir, matches[0])
        assert os.path.getsize(file_path) > 50000, f"Figure {file_path} is unexpectedly small (<50KB)"
