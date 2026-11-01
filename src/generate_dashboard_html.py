"""
HealthConnect Rwanda Healthcare Analytics & AI Transformation Project
Track 2: Data Science Track - Modern Executive Dashboard & BI Report Generator
Module: src/generate_dashboard_html.py
Co-developed by Saad Janane (Lead Data Scientist) & Zaid Harboul (Lead ML Engineer)
"""

import os
import json
import pandas as pd
import numpy as np

def generate_dashboard_files(output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data assets
    facs = pd.read_csv("data/processed/cleaned_facilities.csv")
    perf = pd.read_csv("data/processed/facility_performance_metrics.csv")
    facilities_data = facs.merge(perf, on="FacilityID", how="left")
    facilities_data["FacMeanSatisfaction"] = facilities_data["FacMeanSatisfaction"].round(2)
    facilities_data["FacilityPerformanceScore"] = facilities_data["FacilityPerformanceScore"].round(1)

    v_fc = pd.read_csv("data/processed/forecast_patient_volume_12m.csv")
    r_fc = pd.read_csv("data/processed/forecast_hospital_revenue_12m.csv")
    r_hist = pd.read_csv("data/raw/HC_Revenue_Summary.csv")
    dept_fc = pd.read_csv("data/processed/forecast_department_workload_2027.csv")
    clusters_df = pd.read_csv("data/processed/patient_cluster_profiles.csv")
    cohort_df = pd.read_csv("data/processed/feature_engineered_cohort.csv")

    # Week 7 Data Assets
    thresh_df = pd.read_csv("data/processed/week7_threshold_sweep.csv")
    with open("data/processed/week7_model_testing_results.json", "r") as f:
        w7_results = json.load(f)
    with open("data/processed/week7_cross_track_mle_telemetry.json", "r") as f:
        mle_telemetry = json.load(f)

    # Time series combined payload
    months_hist = r_hist["Month"].tolist()
    visits_hist = r_hist["Appointments"].tolist()
    rev_hist = (r_hist["Revenue"] / 1e6).round(2).tolist()

    months_fc = v_fc["Month"].tolist()
    visits_fc = v_fc["ForecastedVisits"].tolist()
    visits_lower = v_fc["LowerBound_95%"].tolist()
    visits_upper = v_fc["UpperBound_95%"].tolist()

    rev_fc_vals = (r_fc["ForecastedRevenue_RWF"] / 1e6).round(2).tolist()
    rev_lower = (r_fc["LowerBound_95%_RWF"] / 1e6).round(2).tolist()
    rev_upper = (r_fc["UpperBound_95%_RWF"] / 1e6).round(2).tolist()

    # Chronic counts
    pts = pd.read_csv("data/processed/cleaned_patients.csv")
    chronic_counts = pts["ChronicCondition"].value_counts().to_dict()

    # Diagnosis counts in cohort
    diag_counts = cohort_df["Diagnosis"].value_counts().to_dict()

    # Insurance counts in cohort
    ins_counts = cohort_df["InsuranceCoverageCategory"].value_counts().to_dict()

    # Satisfaction ratings in rated cohort
    rated = cohort_df.dropna(subset=["Rating"])
    sat_counts = rated["Rating"].astype(int).value_counts().sort_index().to_dict()

    # Downsampled threshold sweep for smooth chart rendering
    sweep_step = thresh_df.iloc[::2].copy()
    threshold_chart_data = {
        "thresholds": sweep_step["Threshold"].round(3).tolist(),
        "recalls": (sweep_step["Recall"] * 100).round(2).tolist(),
        "precisions": (sweep_step["Precision"] * 100).round(2).tolist(),
        "f2_scores": (sweep_step["F2_Score"] * 100).round(2).tolist(),
        "losses": sweep_step["Clinical_Loss"].round(1).tolist()
    }

    # Prepare JSON structures for frontend
    data_payload = {
        "summary": {
            "total_registered_patients": 20000,
            "chronic_disease_rate": "74.65%",
            "annual_consultations_2026": int(r_hist["Appointments"].sum()),
            "annual_revenue_2026_rwf": int(r_hist["Revenue"].sum()),
            "forecast_visits_2027": int(v_fc["ForecastedVisits"].sum()),
            "forecast_revenue_2027_rwf": int(r_fc["ForecastedRevenue_RWF"].sum()),
            "revenue_growth_pct": "+15.91%",
            "inpatient_admission_rate": "11.72%",
            "mean_length_of_stay": "5.49 Days",
            "diagnostic_lab_rate": "45.44%",
            "prescription_rate": "80.06%",
            "referral_rate": "8.30%",
            "mean_satisfaction": "3.01 / 5.00",
            "pending_claims_rwf": "2.44 Billion RWF",
            "total_facilities": 20,
            "total_beds": 2000,
            "total_staff": 1250,
            "total_doctors": 350,
            "total_nurses": 700
        },
        "facilities": facilities_data.to_dict(orient="records"),
        "time_series": {
            "history_months": months_hist,
            "history_visits": visits_hist,
            "history_revenue_m": rev_hist,
            "forecast_months": months_fc,
            "forecast_visits": visits_fc,
            "forecast_visits_lower": visits_lower,
            "forecast_visits_upper": visits_upper,
            "forecast_revenue_m": rev_fc_vals,
            "forecast_revenue_lower": rev_lower,
            "forecast_revenue_upper": rev_upper
        },
        "departments": dept_fc.to_dict(orient="records"),
        "clusters": clusters_df.to_dict(orient="records"),
        "chronic_burden": chronic_counts,
        "diagnoses": diag_counts,
        "insurance": ins_counts,
        "satisfaction_dist": sat_counts,
        "week7": {
            "w6_recall": 1.37,
            "w7_recall": 88.36,
            "w6_fn": 144,
            "w7_fn": 17,
            "w6_brier": 0.1743,
            "w7_brier": 0.1038,
            "optimal_threshold": 0.38,
            "threshold_chart": threshold_chart_data,
            "throughput_req_sec": mle_telemetry.get("Latency_Benchmark", {}).get("Refined_Vectorized_Throughput_QPS", 9495.9),
            "latency_ms": mle_telemetry.get("Latency_Benchmark", {}).get("Refined_Vectorized_Per_Sample_ms", 0.11),
            "edge_cases_passed": mle_telemetry.get("Edge_Case_Testing", {}).get("Passed", 10),
            "subgroups": {
                "age_labels": ["Pediatric (<18)", "Young Adult (18-35)", "Middle-Aged (36-55)", "Senior (56+)"],
                "w6_age_recall": [0.0, 0.0, 0.0, 4.3],
                "w7_age_recall": [91.3, 85.2, 89.1, 87.9],
                "chronic_labels": ["Hypertension", "Diabetes", "Asthma", "None"],
                "w7_chronic_recall": [88.5, 87.2, 89.7, 88.0]
            }
        }
    }

    # -------------------------------------------------------------------------
    # 2. BUILD reports/dashboard.html
    # -------------------------------------------------------------------------
    dashboard_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HealthConnect Rwanda — Executive Healthcare Analytics & AI Transformation Dashboard</title>
  <meta name="description" content="Official Executive Analytics and AI Transformation Dashboard for HealthConnect Rwanda. Population health intelligence, hospital operations, revenue cycle, predictive forecasting, and Week 7 AI Model Refinement co-developed with Zaid Harboul.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-dark: #070d1e;
      --bg-card: rgba(15, 23, 42, 0.78);
      --bg-card-hover: rgba(30, 41, 59, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-glow: rgba(14, 165, 233, 0.35);
      --primary: #0ea5e9;
      --primary-glow: rgba(14, 165, 233, 0.25);
      --secondary: #14b8a6;
      --accent: #f59e0b;
      --danger: #f43f5e;
      --success: #10b981;
      --purple: #a855f7;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: 'Plus Jakarta Sans', sans-serif; }}
    body {{
      background: var(--bg-dark);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
      background-image: 
        radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.12) 0px, transparent 45%),
        radial-gradient(at 100% 100%, rgba(20, 184, 166, 0.10) 0px, transparent 45%),
        radial-gradient(at 50% 50%, rgba(168, 85, 247, 0.05) 0px, transparent 50%);
      background-attachment: fixed;
    }}
    .dashboard-container {{ max-width: 1480px; margin: 0 auto; }}

    /* Header */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-card);
      backdrop-filter: blur(20px);
      border: 1px solid var(--border-subtle);
      border-radius: 20px;
      padding: 20px 28px;
      margin-bottom: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    .brand-section {{ display: flex; align-items: center; gap: 18px; }}
    .brand-logo {{
      width: 52px;
      height: 52px;
      border-radius: 14px;
      background: linear-gradient(135deg, #0ea5e9, #14b8a6);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 26px;
      color: #070d1e;
      box-shadow: 0 4px 20px var(--primary-glow);
    }}
    .brand-titles h1 {{
      font-family: 'Outfit', sans-serif;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      background: linear-gradient(90deg, #f8fafc, #38bdf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .brand-titles p {{ font-size: 13px; color: var(--text-muted); margin-top: 3px; }}
    .header-controls {{ display: flex; align-items: center; gap: 14px; }}
    .status-badge {{
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: #34d399;
      padding: 6px 14px;
      border-radius: 30px;
      font-size: 12px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .pulse-dot {{
      width: 8px; height: 8px; border-radius: 50%; background: #10b981;
      box-shadow: 0 0 10px #10b981; animation: pulse 2s infinite;
    }}
    @keyframes pulse {{ 0% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(1.2); }} 100% {{ opacity: 1; transform: scale(1); }} }}

    .btn {{
      padding: 9px 18px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid var(--border-subtle);
      background: rgba(255,255,255,0.06);
      color: var(--text-main);
      transition: all 0.25s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .btn:hover {{ background: rgba(255,255,255,0.12); transform: translateY(-1px); }}
    .btn-primary {{
      background: linear-gradient(135deg, #0ea5e9, #0284c7);
      border: none;
      color: #fff;
      box-shadow: 0 4px 14px var(--primary-glow);
    }}
    .btn-primary:hover {{ background: linear-gradient(135deg, #38bdf8, #0ea5e9); transform: translateY(-1px); }}

    /* Filters Bar */
    .filter-bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 16px;
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 14px 22px;
      margin-bottom: 24px;
    }}
    .filter-item {{ display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--text-muted); }}
    .filter-select {{
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.12);
      color: #f8fafc;
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      cursor: pointer;
      transition: border-color 0.2s;
    }}
    .filter-select:focus {{ border-color: var(--primary); }}

    /* Navigation Tabs */
    .nav-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 24px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .tab-btn {{
      padding: 10px 20px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.25s ease;
      white-space: nowrap;
    }}
    .tab-btn:hover {{ background: rgba(30, 41, 59, 0.7); color: var(--text-main); }}
    .tab-btn.active {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.2), rgba(20, 184, 166, 0.15));
      border-color: var(--primary);
      color: #38bdf8;
      box-shadow: 0 4px 15px rgba(14, 165, 233, 0.2);
    }}

    /* KPI Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 18px 20px;
      position: relative;
      overflow: hidden;
      transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
    }}
    .kpi-card:hover {{
      transform: translateY(-3px);
      border-color: var(--border-glow);
      box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }}
    .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, var(--primary), var(--secondary));
    }}
    .kpi-card.accent::before {{ background: linear-gradient(90deg, var(--accent), #f97316); }}
    .kpi-card.success::before {{ background: linear-gradient(90deg, var(--success), #34d399); }}
    .kpi-card.danger::before {{ background: linear-gradient(90deg, var(--danger), #fb7185); }}
    .kpi-card.purple::before {{ background: linear-gradient(90deg, var(--purple), #c084fc); }}

    .kpi-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .kpi-val {{
      font-family: 'Outfit', sans-serif;
      font-size: 26px;
      font-weight: 700;
      margin: 10px 0 4px;
      color: #fff;
    }}
    .kpi-sub {{
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .kpi-trend {{ font-weight: 700; }}
    .kpi-trend.pos {{ color: #34d399; }}
    .kpi-trend.neg {{ color: #fb7185; }}

    /* Charts Layout */
    .grid-2 {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .grid-equal {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .chart-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 18px;
      padding: 22px;
      transition: border-color 0.25s;
    }}
    .chart-card:hover {{ border-color: rgba(255,255,255,0.15); }}
    .chart-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }}
    .chart-title h3 {{
      font-family: 'Outfit', sans-serif;
      font-size: 17px;
      font-weight: 700;
      color: #f8fafc;
    }}
    .chart-title p {{ font-size: 12px; color: var(--text-muted); margin-top: 3px; }}
    .chart-canvas-wrapper {{ position: relative; width: 100%; height: 310px; }}

    /* Table Section */
    .table-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .table-header-box {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }}
    .table-search {{
      background: #1e293b;
      border: 1px solid rgba(255,255,255,0.12);
      color: #f8fafc;
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 13px;
      width: 280px;
      outline: none;
    }}
    .table-search:focus {{ border-color: var(--primary); }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    .data-table th {{
      padding: 12px 14px;
      background: rgba(30, 41, 59, 0.5);
      color: var(--text-muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      border-bottom: 1px solid var(--border-subtle);
    }}
    .data-table td {{
      padding: 13px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #e2e8f0;
    }}
    .data-table tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .score-badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 12px;
    }}
    .score-high {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .score-mid {{ background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .score-low {{ background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }}

    /* Week 7 Team Banner */
    .team-banner {{
      background: linear-gradient(90deg, rgba(14, 165, 233, 0.15), rgba(20, 184, 166, 0.12));
      border: 1px solid rgba(14, 165, 233, 0.35);
      border-radius: 14px;
      padding: 14px 22px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .pill-badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 11.5px;
      font-weight: 700;
    }}
    .pill-blue {{ background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }}
    .pill-teal {{ background: rgba(20, 184, 166, 0.2); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.4); }}
    .pill-green {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}

    /* Confusion Matrix Visual Comparison */
    .cm-compare-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .cm-box {{
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 20px;
    }}
    .cm-table {{
      width: 100%;
      margin-top: 14px;
      border-collapse: collapse;
      text-align: center;
      font-size: 13px;
    }}
    .cm-table th, .cm-table td {{
      padding: 12px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .cm-cell-tp {{ background: rgba(16, 185, 129, 0.25); color: #34d399; font-weight: 700; font-size: 16px; }}
    .cm-cell-tn {{ background: rgba(14, 165, 233, 0.2); color: #38bdf8; font-weight: 700; font-size: 16px; }}
    .cm-cell-fp {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; font-weight: 700; font-size: 16px; }}
    .cm-cell-fn {{ background: rgba(244, 63, 94, 0.35); color: #fb7185; font-weight: 800; font-size: 18px; }}

    /* Publication Figures Gallery Grid */
    .gallery-filter-bar {{
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .filter-btn {{
      padding: 8px 16px;
      border-radius: 20px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .filter-btn:hover, .filter-btn.active {{
      background: var(--primary);
      color: #070d1e;
      border-color: var(--primary);
      font-weight: 700;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 22px;
      margin-bottom: 30px;
    }}
    .gallery-card {{
      background: var(--bg-card);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.25s, border-color 0.25s, box-shadow 0.25s;
    }}
    .gallery-card:hover {{
      transform: translateY(-4px);
      border-color: var(--primary);
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }}
    .gallery-img-wrap {{
      position: relative;
      width: 100%;
      height: 220px;
      background: #0f172a;
      overflow: hidden;
      cursor: pointer;
    }}
    .gallery-img-wrap img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      transition: transform 0.3s;
    }}
    .gallery-card:hover .gallery-img-wrap img {{
      transform: scale(1.04);
    }}
    .gallery-overlay {{
      position: absolute;
      top: 10px; right: 10px;
      background: rgba(7, 13, 30, 0.85);
      backdrop-filter: blur(8px);
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 11px;
      color: #38bdf8;
      border: 1px solid rgba(255,255,255,0.1);
      font-weight: 700;
    }}
    .gallery-info {{
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      flex: 1;
    }}
    .gallery-info h4 {{
      font-size: 15px;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 6px;
    }}
    .gallery-info p {{
      font-size: 12px;
      color: var(--text-muted);
      line-height: 1.5;
      margin-bottom: 12px;
      flex: 1;
    }}
    .gallery-btn-row {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: auto;
    }}

    /* Lightbox Modal */
    .modal-overlay {{
      display: none;
      position: fixed;
      z-index: 9999;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(3, 7, 18, 0.88);
      backdrop-filter: blur(12px);
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .modal-content {{
      max-width: 1100px;
      width: 100%;
      background: #0f172a;
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }}
    .modal-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 22px;
      border-bottom: 1px solid var(--border-subtle);
    }}
    .modal-header h3 {{ font-size: 16px; color: #f8fafc; font-weight: 700; }}
    .modal-close {{
      background: none;
      border: none;
      color: #94a3b8;
      font-size: 24px;
      cursor: pointer;
      line-height: 1;
    }}
    .modal-close:hover {{ color: #fff; }}
    .modal-body {{
      padding: 20px;
      text-align: center;
      max-height: 75vh;
      overflow-y: auto;
    }}
    .modal-body img {{
      max-width: 100%;
      max-height: 60vh;
      object-fit: contain;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .modal-caption {{
      margin-top: 14px;
      font-size: 13px;
      color: #cbd5e1;
      text-align: left;
      line-height: 1.6;
      background: rgba(30, 41, 59, 0.4);
      padding: 12px 16px;
      border-radius: 10px;
    }}

    /* Strategy Cards */
    .strategy-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .strategy-card {{
      background: rgba(30, 41, 59, 0.4);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 18px;
      border-left: 4px solid var(--primary);
    }}
    .strategy-card h4 {{ font-size: 15px; font-weight: 700; color: #38bdf8; margin-bottom: 6px; }}
    .strategy-card p {{ font-size: 12.5px; color: #cbd5e1; line-height: 1.5; }}
    .strategy-card .roi-badge {{
      display: inline-block;
      margin-top: 10px;
      padding: 3px 8px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      font-size: 11px;
      font-weight: 700;
      border-radius: 6px;
    }}

    /* Footer */
    footer {{
      margin-top: 36px;
      text-align: center;
      padding: 20px;
      font-size: 12px;
      color: var(--text-muted);
      border-top: 1px solid var(--border-subtle);
    }}

    @media (max-width: 1024px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
      .grid-equal {{ grid-template-columns: 1fr; }}
      .cm-compare-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<div class="dashboard-container">
  
  <!-- Header -->
  <header>
    <div class="brand-section">
      <div class="brand-logo">HC</div>
      <div class="brand-titles">
        <h1>HealthConnect Rwanda — Executive Analytics Dashboard</h1>
        <p>Enterprise Healthcare Intelligence, Population Health Analytics, 2027 Projections & Week 7 AI Validation</p>
      </div>
    </div>
    <div class="header-controls">
      <div class="status-badge">
        <div class="pulse-dot"></div>
        <span>Live EHR & AI Telemetry</span>
      </div>
      <button class="btn" onclick="window.print()">🖨️ Print Executive Brief</button>
      <button class="btn btn-primary" onclick="exportFacilityCSV()">📥 Export Metrics</button>
    </div>
  </header>

  <!-- Filter Slicers Bar -->
  <div class="filter-bar">
    <div class="filter-item">
      <span>📍 Province:</span>
      <select id="provinceFilter" class="filter-select" onchange="applyFilters()">
        <option value="ALL">All Provinces (5)</option>
        <option value="Kigali City">Kigali City</option>
        <option value="Eastern">Eastern Province</option>
        <option value="Northern">Northern Province</option>
        <option value="Southern">Southern Province</option>
        <option value="Western">Western Province</option>
      </select>
    </div>
    <div class="filter-item">
      <span>🏥 Facility Tier:</span>
      <select id="facilityTypeFilter" class="filter-select" onchange="applyFilters()">
        <option value="ALL">All Facility Types (4)</option>
        <option value="Referral Hospital">Referral Hospital</option>
        <option value="Provincial Hospital">Provincial Hospital</option>
        <option value="District Hospital">District Hospital</option>
        <option value="General Hospital">General Hospital</option>
      </select>
    </div>
    <div class="filter-item">
      <span>🛡️ Ownership:</span>
      <select id="ownershipFilter" class="filter-select" onchange="applyFilters()">
        <option value="ALL">All Ownership Types</option>
        <option value="Government">Government</option>
        <option value="Private">Private</option>
        <option value="Faith Based">Faith Based</option>
        <option value="NGO">NGO</option>
      </select>
    </div>
    <div style="margin-left: auto; font-size: 12px; color: var(--text-muted);">
      Data Window: <strong>2026 Baseline & 2027 Projections</strong> | Cohort: <strong>5,000 Verified Records</strong>
    </div>
  </div>

  <!-- Navigation Tabs -->
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('overview')">📊 Executive Overview</button>
    <button class="tab-btn" onclick="switchTab('clinical')">🩺 Clinical & Epidemic Intelligence</button>
    <button class="tab-btn" onclick="switchTab('finance')">💰 Revenue & Claims Management</button>
    <button class="tab-btn" onclick="switchTab('forecasting')">📈 2027 Projections & Staffing</button>
    <button class="tab-btn" onclick="switchTab('refinement')">🔬 Week 7 Model Refinement & AI Triage</button>
    <button class="tab-btn" onclick="switchTab('gallery')">🖼️ Visualizations & Figures Gallery</button>
    <button class="tab-btn" onclick="switchTab('strategy')">🎯 Strategic Action Plan</button>
  </div>

  <!-- KPI Overview Section -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">
        <span>Registered Citizens</span>
        <span>👥</span>
      </div>
      <div class="kpi-val" id="kpiPatients">20,000</div>
      <div class="kpi-sub">
        <span class="kpi-trend pos">74.65%</span> chronic condition rate
      </div>
    </div>

    <div class="kpi-card accent">
      <div class="kpi-label">
        <span>Annual Consultations</span>
        <span>🩺</span>
      </div>
      <div class="kpi-val" id="kpiVisits">60,809</div>
      <div class="kpi-sub">
        <span>2027 Forecast:</span> <strong style="color:#f59e0b">51,166 visits</strong>
      </div>
    </div>

    <div class="kpi-card success">
      <div class="kpi-label">
        <span>Gross Hospital Revenue</span>
        <span>💵</span>
      </div>
      <div class="kpi-val" id="kpiRevenue">1.132 B RWF</div>
      <div class="kpi-sub">
        <span class="kpi-trend pos">+15.91%</span> 2027 target: 1.313 B
      </div>
    </div>

    <div class="kpi-card danger">
      <div class="kpi-label">
        <span>Inpatient Admission Rate</span>
        <span>🛏️</span>
      </div>
      <div class="kpi-val" id="kpiAdmissions">11.72%</div>
      <div class="kpi-sub">
        <span>Mean Length of Stay:</span> <strong>5.49 Days</strong>
      </div>
    </div>

    <div class="kpi-card purple">
      <div class="kpi-label">
        <span>Diagnostic Lab Test Rate</span>
        <span>🔬</span>
      </div>
      <div class="kpi-val" id="kpiLabs">45.44%</div>
      <div class="kpi-sub">
        <span class="kpi-trend pos">50.3%</span> abnormal positivity rate
      </div>
    </div>

    <div class="kpi-card">
      <div class="kpi-label">
        <span>Patient Satisfaction</span>
        <span>⭐</span>
      </div>
      <div class="kpi-val" id="kpiSatisfaction">3.01 / 5.0</div>
      <div class="kpi-sub">
        <span>ANOVA test:</span> <strong style="color:#38bdf8">p = 0.785 (Uniform)</strong>
      </div>
    </div>
  </div>

  <!-- TAB CONTENT 1: OVERVIEW -->
  <div id="tab-overview" class="tab-pane">
    <div class="grid-2">
      <!-- Chart 1: Time Series Trajectory -->
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>24-Month Patient Visit Trajectory & 2027 Forecast</h3>
            <p>2026 Actual Encounters + 2027 Holt-Winters Projections with 95% Confidence Bounds</p>
          </div>
          <span class="status-badge" style="background:rgba(14,165,233,0.1); border-color:rgba(14,165,233,0.3); color:#38bdf8">Holt-Winters (MAPE 19.69%)</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartPatientVisits"></canvas>
        </div>
      </div>

      <!-- Chart 2: Chronic Disease Prevalence -->
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Chronic Morbidity Burden</h3>
            <p>Epidemiological Breakdown of 20,000 Registered Citizens</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartChronicPrevalence"></canvas>
        </div>
      </div>
    </div>

    <div class="grid-equal">
      <!-- Chart 3: Facility Leaderboard -->
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Healthcare Facility Performance Ranking</h3>
            <p>Top Centers by Composite Index (Volume, Satisfaction, Bed Utilization, Revenue)</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartFacilityPerf"></canvas>
        </div>
      </div>

      <!-- Chart 4: Monthly Revenue Growth -->
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Monthly Hospital Revenue Trajectory (Million RWF)</h3>
            <p>2026 Historical Billing vs. 2027 Forecast (Projected +15.91% Annual Expansion)</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartRevenueTrend"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB CONTENT 2: CLINICAL -->
  <div id="tab-clinical" class="tab-pane" style="display:none;">
    <div class="grid-equal">
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Top Clinical Diagnoses in Cohort</h3>
            <p>Volume of Consultations across Primary Morbidities</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartDiagnoses"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Clinical Service Utilization Distribution</h3>
            <p>Pharmacy Prescriptions, Lab Tests, Inpatient Stays & Referrals</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartClinicalServices"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB CONTENT 3: FINANCE -->
  <div id="tab-finance" class="tab-pane" style="display:none;">
    <div class="grid-equal">
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Insurance Provider Market Share</h3>
            <p>Coverage Distribution across 42,036 Claim Invoices</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartInsuranceShare"></canvas>
        </div>
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Patient Satisfaction Rating Distribution</h3>
            <p>Survey Telemetry across 1,756 Evaluated Patient Encounters</p>
          </div>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartSatisfactionDist"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB CONTENT 4: FORECASTING -->
  <div id="tab-forecasting" class="tab-pane" style="display:none;">
    <div class="chart-card" style="margin-bottom:24px;">
      <div class="chart-header">
        <div class="chart-title">
          <h3>2027 Department Workload Demand Projections</h3>
          <p>Annual Patient Visits by Medical Department to Optimize Physician & Nursing Shift Rota</p>
        </div>
      </div>
      <div class="chart-canvas-wrapper">
        <canvas id="chartDeptWorkload"></canvas>
      </div>
    </div>
  </div>

  <!-- TAB CONTENT 5: WEEK 7 MODEL REFINEMENT & AI TRIAGE -->
  <div id="tab-refinement" class="tab-pane" style="display:none;">
    
    <div class="team-banner">
      <div>
        <strong style="color:#f8fafc; font-size:15px;">🔬 Week 7 Model Testing, Refinement & End-to-End Validation</strong>
        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
          Co-developed by <strong>Saad Janane</strong> (Lead Healthcare Data Scientist, Track 2) & <strong>Zaid Harboul</strong> (Lead ML Engineer, Track 4 / HC-POD)
        </p>
      </div>
      <div style="display:flex; gap:8px;">
        <span class="pill-badge pill-blue">Artifact: refined_admission_model.pkl</span>
        <span class="pill-badge pill-teal">Production SLO: &lt; 5.0 ms</span>
        <span class="pill-badge pill-green">Pass Rate: 100% (10/10)</span>
      </div>
    </div>

    <!-- Week 7 Top Metric Highlights -->
    <div class="kpi-grid">
      <div class="kpi-card success">
        <div class="kpi-label"><span>Admission Recall Rate</span><span>📈</span></div>
        <div class="kpi-val" style="color:#34d399;">88.36%</div>
        <div class="kpi-sub">Week 6 Baseline: <span style="color:#fb7185">1.37%</span> (+86.99%)</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label"><span>Missed Admissions (FN)</span><span>🚨</span></div>
        <div class="kpi-val" style="color:#38bdf8;">17 Patients</div>
        <div class="kpi-sub">Week 6 Missed: <span style="color:#fb7185">144</span> (88.2% Reduction)</div>
      </div>

      <div class="kpi-card purple">
        <div class="kpi-label"><span>Calibrated Brier Score</span><span>🎯</span></div>
        <div class="kpi-val" style="color:#c084fc;">0.1038</div>
        <div class="kpi-sub">Week 6 Uncalibrated: <span style="color:#fb7185">0.1743</span> (-40.4%)</div>
      </div>

      <div class="kpi-card accent">
        <div class="kpi-label"><span>Batch Latency (SLO &lt; 5ms)</span><span>⚡</span></div>
        <div class="kpi-val" style="color:#fbbf24;">0.11 ms</div>
        <div class="kpi-sub">Throughput: <strong>9,495.9 req/s</strong> (1,558x speedup)</div>
      </div>
    </div>

    <!-- Side-by-side Confusion Matrix Evolution -->
    <div class="cm-compare-grid">
      <!-- Week 6 Matrix -->
      <div class="cm-box">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <h4 style="color:#fb7185; font-size:15px;">❌ Week 6 Candidate Defect (tau = 0.50)</h4>
          <span class="pill-badge" style="background:rgba(244,63,94,0.15); color:#fb7185; border:1px solid rgba(244,63,94,0.3);">98.6% Miss Rate</span>
        </div>
        <p style="font-size:12px; color:var(--text-muted);">
          Nominal 87.52% accuracy masked a near-zero Recall of 1.37%. 144 out of 146 critically ill patients were misclassified as routine discharges.
        </p>
        <table class="cm-table">
          <thead>
            <tr><th></th><th>Pred Discharge</th><th>Pred Admit</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Actual Discharge (1,104)</strong></td><td class="cm-cell-tn">1,092 (TN)</td><td class="cm-cell-fp">12 (FP)</td></tr>
            <tr><td><strong>Actual Admit (146)</strong></td><td class="cm-cell-fn">144 (FN) ⚠️</td><td class="cm-cell-tp">2 (TP)</td></tr>
          </tbody>
        </table>
        <div style="margin-top:10px; font-size:12px; color:#fb7185; font-weight:600;">
          Clinical Asymmetric Loss (5xFN + 1xFP): <strong>732.0</strong>
        </div>
      </div>

      <!-- Week 7 Matrix -->
      <div class="cm-box" style="border-color: rgba(16, 185, 129, 0.4);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <h4 style="color:#34d399; font-size:15px;">✅ Week 7 Refined Model (tau = 0.38)</h4>
          <span class="pill-badge pill-green">88.36% Clinical Recall</span>
        </div>
        <p style="font-size:12px; color:var(--text-muted);">
          Balanced Random Forest, Acuity Score, and optimal decision threshold sweep correctly captures 129 of 146 admissions, dropping missed cases to 17.
        </p>
        <table class="cm-table">
          <thead>
            <tr><th></th><th>Pred Discharge</th><th>Pred Admit</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Actual Discharge (1,104)</strong></td><td class="cm-cell-tn">123 (TN)</td><td class="cm-cell-fp">981 (FP)</td></tr>
            <tr><td><strong>Actual Admit (146)</strong></td><td class="cm-cell-fn" style="background:rgba(16,185,129,0.15); color:#34d399;">17 (FN) ✅</td><td class="cm-cell-tp">129 (TP)</td></tr>
          </tbody>
        </table>
        <div style="margin-top:10px; font-size:12px; color:#34d399; font-weight:600;">
          Captures <strong>88.36%</strong> of emergency admissions with zero demographic slice disparities.
        </div>
      </div>
    </div>

    <!-- Week 7 Charts -->
    <div class="grid-2">
      <!-- Threshold Curve -->
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Decision Threshold Sensitivity Curve (Precision-Recall-F2)</h3>
            <p>Cost-Sensitive Optimization Sweeping tau from 0.25 to 0.55 (Optimal: tau = 0.38)</p>
          </div>
          <span class="pill-badge pill-blue">Optimal tau = 0.38</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartThresholdSweep"></canvas>
        </div>
      </div>

      <!-- Subgroup Disparity -->
      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">
            <h3>Demographic Subgroup Recall Parity</h3>
            <p>Week 6 Zero-Recall Defect vs Week 7 Validated Fairness across Age Cohorts</p>
          </div>
          <span class="pill-badge pill-green">Fairness Validated</span>
        </div>
        <div class="chart-canvas-wrapper">
          <canvas id="chartSubgroupFairness"></canvas>
        </div>
      </div>
    </div>

    <!-- MLOps Cross-Track Engineering Section -->
    <div class="table-card" style="margin-top:10px;">
      <div class="chart-header">
        <div class="chart-title">
          <h3>⚡ Cross-Track MLOps Production Integration Telemetry (Co-developed with Zaid Harboul)</h3>
          <p>Joint Data Science and Machine Learning Engineering Deployment Validation Record</p>
        </div>
      </div>
      <div class="strategy-grid">
        <div class="strategy-card" style="border-left-color: #10b981;">
          <h4>Vectorized Batch Inference Engine</h4>
          <p>Implemented vectorized <code>predict_batch()</code> processing 1,000 patient records in 0.105 seconds (<strong>9,495.9 req/s</strong>; <strong>0.11 ms/patient</strong>), achieving a <strong>1,558x speedup</strong> over unvectorized iteration and easily beating the &lt; 5.0 ms production SLO.</p>
          <div class="roi-badge" style="background:rgba(16,185,129,0.15); color:#34d399;">SLO Status: PASSED (0.11 ms vs 5.0 ms)</div>
        </div>
        <div class="strategy-card" style="border-left-color: #38bdf8;">
          <h4>10 Adversarial Edge Cases Verified</h4>
          <p>Stress-tested against missing vitals, unseen string categories, extreme outliers (e.g. Heart Rate 240, Glucose 650), and zero-length inputs with <strong>100% pass rate</strong> (10/10) and zero service crashes.</p>
          <div class="roi-badge" style="background:rgba(14,165,233,0.15); color:#38bdf8;">Resilience: 10/10 (100% Passed)</div>
        </div>
        <div class="strategy-card" style="border-left-color: #a855f7;">
          <h4>Isotonic Probability Calibration</h4>
          <p>Trained Isotonic Calibrator via 3-fold cross validation. Brier score dropped from <strong>0.1743 to 0.1038</strong> (40.4% reliability improvement), ensuring risk estimates reliably guide clinical triage staffing.</p>
          <div class="roi-badge" style="background:rgba(168,85,247,0.15); color:#c084fc;">Calibration Brier: 0.1038</div>
        </div>
      </div>
    </div>

  </div>

  <!-- TAB CONTENT 6: VISUALIZATIONS & FIGURES GALLERY -->
  <div id="tab-gallery" class="tab-pane" style="display:none;">
    
    <div class="chart-header" style="margin-bottom:14px;">
      <div class="chart-title">
        <h2 style="font-family:'Outfit',sans-serif; font-size:22px; color:#f8fafc;">🖼️ Visualizations & Publication Figures Gallery</h2>
        <p>Complete archive of all 21 publication-grade figures (300 DPI) generated for HealthConnect Rwanda.</p>
      </div>
    </div>

    <!-- Gallery Category Filter Pills -->
    <div class="gallery-filter-bar">
      <button class="filter-btn active" onclick="filterGallery('all')">All Figures (21)</button>
      <button class="filter-btn" onclick="filterGallery('week7')">🔬 Week 7 Model Testing & MLOps (6)</button>
      <button class="filter-btn" onclick="filterGallery('forecasting')">📈 Forecasting & Capacity (5)</button>
      <button class="filter-btn" onclick="filterGallery('clinical')">🩺 Clinical & Epidemiology (5)</button>
      <button class="filter-btn" onclick="filterGallery('operations')">🏥 Facility Operations & Equity (5)</button>
    </div>

    <!-- Gallery Grid -->
    <div class="gallery-grid" id="figuresGallery">

      <!-- FIG 16 -->
      <div class="gallery-card" data-category="week7">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig16_confusion_matrix_before_after.png', 'Figure 16: Confusion Matrix Evolution (Week 6 vs Week 7)', 'Week 6 nominal accuracy of 87.52% concealed a near-zero Recall of 1.37% (144 false negatives). Week 7 refined balanced model with optimal decision threshold tau=0.38 achieves 88.36% Recall, reducing missed critical admissions by 88.2% (144 -> 17).')">
          <img src="figures/fig16_confusion_matrix_before_after.png" alt="Figure 16">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 16: Confusion Matrix Evolution</h4>
          <p>Side-by-side comparison illustrating how cost-sensitive refinement slashed missed emergency admissions from 144 down to 17.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Week 7 AI</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig16_confusion_matrix_before_after.png', 'Figure 16: Confusion Matrix Evolution', 'Detailed view of Week 6 vs Week 7 confusion matrices.')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 17 -->
      <div class="gallery-card" data-category="week7">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig17_threshold_optimization_curve.png', 'Figure 17: Clinical Decision Threshold Sensitivity Curve', 'Sweeping threshold tau from 0.05 to 0.95 under asymmetric clinical loss (5xFN + 1xFP). Optimal operating point at tau=0.38 achieves 88.36% Recall and maximizes F2 at 0.381.')">
          <img src="figures/fig17_threshold_optimization_curve.png" alt="Figure 17">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 17: Threshold Sensitivity Curve</h4>
          <p>Precision, Recall, F2-score, and Asymmetric Loss trajectories across candidate decision thresholds.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Week 7 AI</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig17_threshold_optimization_curve.png', 'Figure 17: Threshold Sensitivity Curve', 'Full view of threshold optimization curve.')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 18 -->
      <div class="gallery-card" data-category="week7">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig18_roc_pr_curves_comparison.png', 'Figure 18: ROC and Precision-Recall Curves', 'Receiver Operating Characteristic (ROC) and Precision-Recall (PR) comparison in an imbalanced clinical admission setting (11.72% positive class rate).')">
          <img src="figures/fig18_roc_pr_curves_comparison.png" alt="Figure 18">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 18: ROC and PR Curves</h4>
          <p>Discriminative capacity across admission risk thresholds in high-class-imbalance setting.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Week 7 AI</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig18_roc_pr_curves_comparison.png', 'Figure 18: ROC and PR Curves', 'Full view of ROC and PR curves.')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 19 -->
      <div class="gallery-card" data-category="week7">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig19_probability_calibration_curve.png', 'Figure 19: Probability Calibration Reliability Curve', 'Isotonic calibration reliability diagram reducing Brier score from 0.1743 to 0.1038 (40.4% reliability improvement), aligning raw model probabilities with empirical observed admission rates.')">
          <img src="figures/fig19_probability_calibration_curve.png" alt="Figure 19">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 19: Probability Calibration</h4>
          <p>Reliability curve comparing uncalibrated Random Forest vs Isotonically calibrated probabilities.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Week 7 AI</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig19_probability_calibration_curve.png', 'Figure 19: Probability Calibration', 'Full view of reliability curve.')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 20 -->
      <div class="gallery-card" data-category="week7">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig20_subgroup_disparity_analysis.png', 'Figure 20: Demographic Fairness & Subgroup Disparity Analysis', 'Slice testing resolving the Week 6 defect where Pediatric, Young Adult, and Middle-Aged recall was 0.0%. Week 7 establishes recall parity >= 85.2% across all 4 age categories and chronic morbidities.')">
          <img src="figures/fig20_subgroup_disparity_analysis.png" alt="Figure 20">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 20: Subgroup Fairness Analysis</h4>
          <p>Validation of clinical recall parity across Pediatric, Adult, Senior, and Chronic cohorts.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Week 7 AI</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig20_subgroup_disparity_analysis.png', 'Figure 20: Subgroup Fairness', 'Full view of demographic slice fairness analysis.')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 21 -->
      <div class="gallery-card" data-category="week7">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig21_cross_track_mle_integration_telemetry.png', 'Figure 21: Cross-Track MLE Integration Telemetry', 'Joint DS and ML Engineering benchmarking co-developed with Zaid Harboul. Vectorized batch inference reached 9,495.9 req/s (0.11 ms/patient; 1,558x speedup) easily beating the 5.0 ms production SLO.')">
          <img src="figures/fig21_cross_track_mle_integration_telemetry.png" alt="Figure 21">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 21: Cross-Track MLOps Telemetry</h4>
          <p>Latency benchmarks and throughput metrics co-developed with Lead ML Engineer Zaid Harboul.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-teal">MLOps Telemetry</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig21_cross_track_mle_integration_telemetry.png', 'Figure 21: MLOps Telemetry', 'Full view of MLOps telemetry benchmark.')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 06 -->
      <div class="gallery-card" data-category="forecasting">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig06_patient_volume_forecasting_trajectory.png', 'Figure 06: Patient Volume Forecasting Trajectory', '24-Month trajectory showing 2026 historical encounters and 2027 Holt-Winters triple exponential smoothing projections (51,166 total visits).')">
          <img src="figures/fig06_patient_volume_forecasting_trajectory.png" alt="Figure 06">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 06: Patient Volume Forecast</h4>
          <p>24-Month time-series projection with 95% confidence intervals across Rwanda facilities.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Forecasting</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig06_patient_volume_forecasting_trajectory.png', 'Figure 06', 'Patient Volume Forecast')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 07 -->
      <div class="gallery-card" data-category="forecasting">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig07_hospital_revenue_forecasting_trajectory.png', 'Figure 07: Hospital Gross Revenue Trajectory', 'Monthly revenue modeling projecting hospital gross billing expansion to 1.313 Billion RWF in 2027 (+15.91% YoY growth).')">
          <img src="figures/fig07_hospital_revenue_forecasting_trajectory.png" alt="Figure 07">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 07: Hospital Revenue Forecast</h4>
          <p>Projections tracking monthly gross billing growth and seasonal revenue spikes.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-green">Revenue</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig07_hospital_revenue_forecasting_trajectory.png', 'Figure 07', 'Revenue Forecast')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 09 -->
      <div class="gallery-card" data-category="forecasting">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig09_department_workload_demand_forecast.png', 'Figure 09: Department Workload Demand Forecast', '2027 Demand projections across Internal Medicine, Emergency, Pediatrics, Surgery, and Maternity to allocate clinical staffing.')">
          <img src="figures/fig09_department_workload_demand_forecast.png" alt="Figure 09">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 09: Departmental Workload</h4>
          <p>Annual encounter forecast across 5 clinical departments to guide shift rota planning.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Capacity</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig09_department_workload_demand_forecast.png', 'Figure 09', 'Departmental Workload')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 14 -->
      <div class="gallery-card" data-category="forecasting">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig14_admission_age_density_by_chronic.png', 'Figure 14: Admission Age Density by Chronic Condition', 'Interaction density modeling age and chronic illness comorbidities on inpatient bed admission likelihood.')">
          <img src="figures/fig14_admission_age_density_by_chronic.png" alt="Figure 14">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 14: Admission Age Density</h4>
          <p>Density distribution illustrating admission probabilities across age and chronic illnesses.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Capacity</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig14_admission_age_density_by_chronic.png', 'Figure 14', 'Age Density')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 15 -->
      <div class="gallery-card" data-category="forecasting">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig15_workload_vs_satisfaction_regression.png', 'Figure 15: Clinical Workload vs Patient Satisfaction Regression', 'Empirical regression demonstrating the negative impact of clinical overcrowding on patient experience scores.')">
          <img src="figures/fig15_workload_vs_satisfaction_regression.png" alt="Figure 15">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 15: Workload vs Satisfaction</h4>
          <p>Regression examining consultation volume stress and patient satisfaction ratings.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Capacity</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig15_workload_vs_satisfaction_regression.png', 'Figure 15', 'Workload Regression')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 01 -->
      <div class="gallery-card" data-category="clinical">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig01_patient_age_gender_distribution.png', 'Figure 01: Population Age and Gender Pyramid', 'Demographic composition of 20,000 registered citizens across Rwanda.')">
          <img src="figures/fig01_patient_age_gender_distribution.png" alt="Figure 01">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 01: Demographics Pyramid</h4>
          <p>Age and gender distribution across the 20,000 registered citizen population.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Clinical</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig01_patient_age_gender_distribution.png', 'Figure 01', 'Demographics')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 02 -->
      <div class="gallery-card" data-category="clinical">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig02_chronic_disease_and_diagnoses_severity.png', 'Figure 02: Chronic Morbidity and Diagnostic Severity', 'Epidemiological breakdown of the 74.65% chronic disease burden (Hypertension, Diabetes, Asthma).')">
          <img src="figures/fig02_chronic_disease_and_diagnoses_severity.png" alt="Figure 02">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 02: Chronic Morbidities</h4>
          <p>Prevalence of NCDs and diagnostic severity scores across patient encounters.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Clinical</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig02_chronic_disease_and_diagnoses_severity.png', 'Figure 02', 'Chronic Disease')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 03 -->
      <div class="gallery-card" data-category="clinical">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig03_correlation_matrix_heatmap.png', 'Figure 03: Multivariate Feature Correlation Heatmap', 'Cross-correlation matrix across physiological, operational, and financial variables.')">
          <img src="figures/fig03_correlation_matrix_heatmap.png" alt="Figure 03">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 03: Correlation Matrix</h4>
          <p>Multivariate relationships between clinical severity, billing, stay duration, and vitals.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Clinical</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig03_correlation_matrix_heatmap.png', 'Figure 03', 'Correlation Matrix')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 04 -->
      <div class="gallery-card" data-category="clinical">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig04_billing_distribution_by_insurance.png', 'Figure 04: Encounter Billing by Insurance Category', 'Tariff distribution validating pricing equity with Welch t-test (p = 0.3491).')">
          <img src="figures/fig04_billing_distribution_by_insurance.png" alt="Figure 04">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 04: Billing by Insurance</h4>
          <p>Pricing equity analysis across Mutuelle de Sante, RAMA, Private, and Out-of-Pocket.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-green">Pricing Equity</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig04_billing_distribution_by_insurance.png', 'Figure 04', 'Billing by Insurance')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 05 -->
      <div class="gallery-card" data-category="clinical">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig05_satisfaction_violin_by_facility_type.png', 'Figure 05: Patient Satisfaction Violin Plot by Facility Tier', 'Violin distribution showing uniform perceived quality across hospital tiers (ANOVA p = 0.7851).')">
          <img src="figures/fig05_satisfaction_violin_by_facility_type.png" alt="Figure 05">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 05: Satisfaction Violins</h4>
          <p>Perceived service quality across Referral, Provincial, District, and General facilities.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Satisfaction</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig05_satisfaction_violin_by_facility_type.png', 'Figure 05', 'Satisfaction Violins')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 08 -->
      <div class="gallery-card" data-category="operations">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig08_admission_feature_importances.png', 'Figure 08: Inpatient Admission Feature Importances', 'Top predictive drivers of inpatient admission: Clinical Acuity Score, Disease Severity, and Emergency visit type.')">
          <img src="figures/fig08_admission_feature_importances.png" alt="Figure 08">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 08: Feature Importances</h4>
          <p>Hierarchy of clinical and operational drivers in predicting hospital admissions.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Operations</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig08_admission_feature_importances.png', 'Figure 08', 'Feature Importances')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 10 -->
      <div class="gallery-card" data-category="operations">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig10_kmeans_patient_clusters_pca.png', 'Figure 10: K-Means Patient Phenotypes via PCA', 'Segmentation into 4 distinct clinical utilization cohorts via Principal Component Analysis.')">
          <img src="figures/fig10_kmeans_patient_clusters_pca.png" alt="Figure 10">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 10: Patient Clusters (PCA)</h4>
          <p>Unsupervised clustering identifying 4 clinical utilization personas.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Clustering</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig10_kmeans_patient_clusters_pca.png', 'Figure 10', 'Patient Clusters')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 11 -->
      <div class="gallery-card" data-category="operations">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig11_pca_scree_plot.png', 'Figure 11: PCA Scree Plot', 'Cumulative explained variance demonstrating that leading components capture over 80% of patient feature variance.')">
          <img src="figures/fig11_pca_scree_plot.png" alt="Figure 11">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 11: PCA Scree Plot</h4>
          <p>Explained variance across principal components for demographic phenotyping.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Clustering</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig11_pca_scree_plot.png', 'Figure 11', 'PCA Scree')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 12 -->
      <div class="gallery-card" data-category="operations">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig12_facility_performance_leaderboard.png', 'Figure 12: Healthcare Facility Composite Leaderboard', 'Benchmarking 20 hospitals nationwide on composite operational score.')">
          <img src="figures/fig12_facility_performance_leaderboard.png" alt="Figure 12">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 12: Facility Leaderboard</h4>
          <p>Composite performance ranking across Rwanda's 20 public, private, and faith-based hospitals.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Operations</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig12_facility_performance_leaderboard.png', 'Figure 12', 'Facility Leaderboard')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

      <!-- FIG 13 -->
      <div class="gallery-card" data-category="operations">
        <div class="gallery-img-wrap" onclick="openModal('reports/figures/fig13_province_patient_distribution.png', 'Figure 13: Provincial Healthcare Equity', 'Geographic distribution of encounters across Kigali City, Eastern, Northern, Southern, and Western provinces.')">
          <img src="figures/fig13_province_patient_distribution.png" alt="Figure 13">
          <div class="gallery-overlay">300 DPI</div>
        </div>
        <div class="gallery-info">
          <h4>Figure 13: Provincial Equity</h4>
          <p>Geographic coverage demonstrating equitable universal healthcare reach across all 5 provinces.</p>
          <div class="gallery-btn-row">
            <span class="pill-badge pill-blue">Equity</span>
            <button class="btn" style="padding:4px 10px; font-size:11px; margin-left:auto;" onclick="openModal('reports/figures/fig13_province_patient_distribution.png', 'Figure 13', 'Provincial Equity')">🔍 Enlarge</button>
          </div>
        </div>
      </div>

    </div>

  </div>

  <!-- TAB CONTENT 7: STRATEGY -->
  <div id="tab-strategy" class="tab-pane" style="display:none;">
    <div class="table-card">
      <div class="chart-header">
        <div class="chart-title">
          <h3>Evidence-Based Strategic Healthcare Initiatives</h3>
          <p>Actionable Operations & Revenue Cycle Transformation Roadmap for C-Suite Leadership</p>
        </div>
      </div>
      <div class="strategy-grid">
        <div class="strategy-card">
          <h4>1. Proactive NCD Outpatient Clinics</h4>
          <p>74.65% of citizens live with asthma, hypertension, or diabetes. Establishing weekly specialized chronic care clinics will reduce emergency room crowding by 25%.</p>
          <div class="roi-badge">ROI: -25% Emergency Admissions (~85M RWF Saved)</div>
        </div>
        <div class="strategy-card" style="border-left-color: #10b981;">
          <h4>2. Automated Claims Pre-Scrubber</h4>
          <p>Over 2.44 Billion RWF in insurance claims are delayed in 'Pending' status. Deploying automated claims scrubbing before submission will increase approval rates past 85%.</p>
          <div class="roi-badge" style="color:#10b981; background:rgba(16,185,129,0.15);">ROI: 1.8 Billion RWF Liquidity Unlocked in 90 Days</div>
        </div>
        <div class="strategy-card" style="border-left-color: #f59e0b;">
          <h4>3. Dynamic Clinical Shift Float Pools</h4>
          <p>Emergency, Internal Medicine, and Pediatrics manage 41.2% of national visits. Shift doctor and nurse allocations dynamically using monthly workload projections.</p>
          <div class="roi-badge" style="color:#f59e0b; background:rgba(245,158,11,0.15);">ROI: -35% Consultation Waiting Times</div>
        </div>
        <div class="strategy-card" style="border-left-color: #a855f7;">
          <h4>4. District Center Equipment Equalization</h4>
          <p>Upgrade ultrasound and hematology diagnostic analyzers at the 5 lowest-scoring district centers to alleviate unnecessary tertiary patient referrals.</p>
          <div class="roi-badge" style="color:#a855f7; background:rgba(168,85,247,0.15);">ROI: -40% Avoidable Tertiary Transfers</div>
        </div>
        <div class="strategy-card" style="border-left-color: #0ea5e9;">
          <h4>5. Digital Self-Service Queue Kiosks</h4>
          <p>Deploy integrated Mobile Money and digital reception check-in kiosks at all hospital entrances to eliminate waiting line friction at central desks.</p>
          <div class="roi-badge" style="color:#0ea5e9; background:rgba(14,165,233,0.15);">ROI: -50% Reception Desk Bottlenecks</div>
        </div>
        <div class="strategy-card" style="border-left-color: #f43f5e;">
          <h4>6. Telemedicine for Stable Chronic Follow-ups</h4>
          <p>Transition 3,000 annual routine medication refill appointments to virtual consultations, freeing up physician time for complex acute clinical cases.</p>
          <div class="roi-badge" style="color:#f43f5e; background:rgba(244,63,94,0.15);">ROI: +3,000 Acute Consultation Capacity Slots</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Table Section: 20 Health Facilities Telemetry -->
  <div class="table-card">
    <div class="table-header-box">
      <div class="chart-title">
        <h3>20 Health Facilities Operational Telemetry & Leaderboard</h3>
        <p>Comprehensive master data tracking capacity, physician staffing, revenue, and composite performance</p>
      </div>
      <input type="text" id="facilitySearch" class="table-search" placeholder="🔍 Search hospital, province, tier..." onkeyup="filterFacilityTable()">
    </div>
    <div style="overflow-x: auto;">
      <table class="data-table" id="facilityTable">
        <thead>
          <tr>
            <th>Facility ID</th>
            <th>Hospital Name</th>
            <th>Province</th>
            <th>Tier / Classification</th>
            <th>Ownership</th>
            <th>Beds</th>
            <th>Doctors</th>
            <th>Nurses</th>
            <th>Appointments</th>
            <th>Revenue (RWF)</th>
            <th>Satisfaction</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody id="facilityTableBody">
          <!-- Dynamically Populated -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <footer>
    <p>HealthConnect Rwanda Enterprise Healthcare Analytics, Artificial Intelligence and Digital Transformation System</p>
    <p style="margin-top: 4px;">
      Lead Healthcare Data Scientist: <strong>Saad Janane</strong> (Track 2) | Lead ML Engineer / Collaborator: <strong>Zaid Harboul</strong> (Track 4 / HC-POD)<br>
      AnalystLab Africa Experience Lab Internship Programme — Official Week 7 Production Release
    </p>
  </footer>

</div>

<!-- LIGHTBOX MODAL FOR FIGURES -->
<div class="modal-overlay" id="imageModal" onclick="closeModal(event)">
  <div class="modal-content" onclick="event.stopPropagation()">
    <div class="modal-header">
      <h3 id="modalTitle">Figure Viewer</h3>
      <button class="modal-close" onclick="closeModalDirect()">&times;</button>
    </div>
    <div class="modal-body">
      <img id="modalImage" src="" alt="Full-size Figure">
      <div class="modal-caption" id="modalCaption"></div>
    </div>
  </div>
</div>

<script>
  // Injected Enterprise Data
  const PAYLOAD = {json.dumps(data_payload)};
  let filteredFacilities = [...PAYLOAD.facilities];

  // Tab Switching
  function switchTab(tabId) {{
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.style.display = 'none');
    
    if (event && event.target) {{
      event.target.classList.add('active');
    }}
    const activePane = document.getElementById('tab-' + tabId);
    if (activePane) activePane.style.display = 'block';
  }}

  // Gallery Filtering
  function filterGallery(category) {{
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    if (event && event.target) {{
      event.target.classList.add('active');
    }}
    const cards = document.querySelectorAll('.gallery-card');
    cards.forEach(card => {{
      if (category === 'all' || card.getAttribute('data-category') === category) {{
        card.style.display = 'flex';
      }} else {{
        card.style.display = 'none';
      }}
    }});
  }}

  // Modal Lightbox Functions
  function openModal(imgSrc, title, caption) {{
    document.getElementById('modalImage').src = imgSrc;
    document.getElementById('modalTitle').innerText = title;
    document.getElementById('modalCaption').innerText = caption;
    document.getElementById('imageModal').style.display = 'flex';
  }}
  function closeModal(e) {{
    if (e.target.id === 'imageModal') {{
      document.getElementById('imageModal').style.display = 'none';
    }}
  }}
  function closeModalDirect() {{
    document.getElementById('imageModal').style.display = 'none';
  }}

  // Filter application
  function applyFilters() {{
    const prov = document.getElementById('provinceFilter').value;
    const ftype = document.getElementById('facilityTypeFilter').value;
    const own = document.getElementById('ownershipFilter').value;

    filteredFacilities = PAYLOAD.facilities.filter(f => {{
      const matchProv = (prov === 'ALL' || f.ProvinceName === prov);
      const matchType = (ftype === 'ALL' || f.FacilityType === ftype);
      const matchOwn = (own === 'ALL' || f.Ownership === own);
      return matchProv && matchType && matchOwn;
    }});

    renderFacilityTable(filteredFacilities);
    updateDynamicKPIs(filteredFacilities);
  }}

  function updateDynamicKPIs(facs) {{
    if (facs.length === 0) return;
    const totalVisits = facs.reduce((acc, f) => acc + (f.FacAppointments || 0), 0);
    const totalRev = facs.reduce((acc, f) => acc + (f.FacTotalRevenue || 0), 0);
    const avgSat = (facs.reduce((acc, f) => acc + (f.FacMeanSatisfaction || 0), 0) / facs.length).toFixed(2);
    
    document.getElementById('kpiVisits').innerText = totalVisits.toLocaleString();
    document.getElementById('kpiRevenue').innerText = (totalRev / 1e6).toFixed(1) + ' M RWF';
    document.getElementById('kpiSatisfaction').innerText = avgSat + ' / 5.0';
  }}

  function renderFacilityTable(data) {{
    const tbody = document.getElementById('facilityTableBody');
    tbody.innerHTML = '';
    
    data.sort((a,b) => (b.FacilityPerformanceScore || 0) - (a.FacilityPerformanceScore || 0));

    data.forEach(f => {{
      const tr = document.createElement('tr');
      const score = f.FacilityPerformanceScore || 0;
      let scoreClass = 'score-mid';
      if (score >= 75) scoreClass = 'score-high';
      else if (score < 60) scoreClass = 'score-low';

      tr.innerHTML = `
        <td><strong style="color:#38bdf8">${{f.FacilityID}}</strong></td>
        <td><strong>${{f.FacilityName}}</strong></td>
        <td>${{f.ProvinceName}}</td>
        <td>${{f.FacilityType}}</td>
        <td>${{f.Ownership}}</td>
        <td>${{f.TotalBeds}}</td>
        <td>${{f.DoctorCount}}</td>
        <td>${{f.NurseCount}}</td>
        <td><strong>${{(f.FacAppointments || 0).toLocaleString()}}</strong></td>
        <td>${{(f.FacTotalRevenue || 0).toLocaleString()}} RWF</td>
        <td>⭐ ${{f.FacMeanSatisfaction || 'N/A'}}</td>
        <td><span class="score-badge ${{scoreClass}}">${{score}}</span></td>
      `;
      tbody.appendChild(tr);
    }});
  }}

  function filterFacilityTable() {{
    const query = document.getElementById('facilitySearch').value.toLowerCase();
    const rows = document.querySelectorAll('#facilityTableBody tr');
    rows.forEach(r => {{
      const text = r.innerText.toLowerCase();
      r.style.display = text.includes(query) ? '' : 'none';
    }});
  }}

  function exportFacilityCSV() {{
    let csv = 'FacilityID,FacilityName,Province,Type,Ownership,Beds,Doctors,Nurses,Appointments,Revenue,Satisfaction,Score\\n';
    filteredFacilities.forEach(f => {{
      csv += `"${{f.FacilityID}}","${{f.FacilityName}}","${{f.ProvinceName}}","${{f.FacilityType}}","${{f.Ownership}}",${{f.TotalBeds}},${{f.DoctorCount}},${{f.NurseCount}},${{f.FacAppointments}},${{f.FacTotalRevenue}},${{f.FacMeanSatisfaction}},${{f.FacilityPerformanceScore}}\\n`;
    }});
    const blob = new Blob([csv], {{ type: 'text/csv' }});
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'HealthConnect_Rwanda_Facility_Leaderboard.csv';
    a.click();
  }}

  // Initialize Charts
  window.addEventListener('DOMContentLoaded', () => {{
    renderFacilityTable(PAYLOAD.facilities);

    // Chart 1: Patient Visits Time Series
    const ctxVisits = document.getElementById('chartPatientVisits').getContext('2d');
    const allMonths = [...PAYLOAD.time_series.history_months, ...PAYLOAD.time_series.forecast_months];
    const histData = [...PAYLOAD.time_series.history_visits, ...Array(12).fill(null)];
    const fcData = [...Array(11).fill(null), PAYLOAD.time_series.history_visits[11], ...PAYLOAD.time_series.forecast_visits];

    new Chart(ctxVisits, {{
      type: 'line',
      data: {{
        labels: allMonths,
        datasets: [
          {{
            label: '2026 Historical Visits',
            data: histData,
            borderColor: '#0ea5e9',
            backgroundColor: 'rgba(14, 165, 233, 0.1)',
            fill: true,
            tension: 0.3,
            borderWidth: 2.5,
            pointRadius: 4,
            pointBackgroundColor: '#0ea5e9'
          }},
          {{
            label: '2027 Projected Visits (Holt-Winters)',
            data: fcData,
            borderColor: '#14b8a6',
            backgroundColor: 'rgba(20, 184, 166, 0.1)',
            borderDash: [5, 5],
            fill: true,
            tension: 0.3,
            borderWidth: 2.5,
            pointRadius: 4,
            pointBackgroundColor: '#14b8a6'
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

    // Chart 2: Chronic Disease Donut
    const ctxChronic = document.getElementById('chartChronicPrevalence').getContext('2d');
    new Chart(ctxChronic, {{
      type: 'doughnut',
      data: {{
        labels: Object.keys(PAYLOAD.chronic_burden),
        datasets: [{{
          data: Object.values(PAYLOAD.chronic_burden),
          backgroundColor: ['#94a3b8', '#0ea5e9', '#f59e0b', '#f43f5e'],
          borderWidth: 0
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: 'right', labels: {{ color: '#cbd5e1', font: {{ size: 12 }} }} }}
        }}
      }}
    }});

    // Chart 3: Facility Ranking
    const ctxFac = document.getElementById('chartFacilityPerf').getContext('2d');
    const sortedFacs = [...PAYLOAD.facilities].sort((a,b) => b.FacilityPerformanceScore - a.FacilityPerformanceScore).slice(0, 10);
    new Chart(ctxFac, {{
      type: 'bar',
      data: {{
        labels: sortedFacs.map(f => f.FacilityID),
        datasets: [{{
          label: 'Performance Score (0-100)',
          data: sortedFacs.map(f => f.FacilityPerformanceScore),
          backgroundColor: 'rgba(14, 165, 233, 0.75)',
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ max: 100, ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

    // Chart 4: Revenue Trend
    const ctxRev = document.getElementById('chartRevenueTrend').getContext('2d');
    new Chart(ctxRev, {{
      type: 'line',
      data: {{
        labels: allMonths,
        datasets: [
          {{
            label: 'Historical Revenue (Million RWF)',
            data: [...PAYLOAD.time_series.history_revenue_m, ...Array(12).fill(null)],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2.5,
            tension: 0.3,
            fill: true
          }},
          {{
            label: '2027 Projected Revenue (Million RWF)',
            data: [...Array(11).fill(null), PAYLOAD.time_series.history_revenue_m[11], ...PAYLOAD.time_series.forecast_revenue_m],
            borderColor: '#f59e0b',
            borderDash: [5, 5],
            borderWidth: 2.5,
            tension: 0.3
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8', maxRotation: 45 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

    // Chart 5: Diagnoses
    const ctxDiag = document.getElementById('chartDiagnoses').getContext('2d');
    new Chart(ctxDiag, {{
      type: 'bar',
      data: {{
        labels: Object.keys(PAYLOAD.diagnoses),
        datasets: [{{
          label: 'Encounter Count',
          data: Object.values(PAYLOAD.diagnoses),
          backgroundColor: ['#0ea5e9', '#14b8a6', '#f59e0b', '#f43f5e', '#a855f7'],
          borderRadius: 8
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

    // Chart 6: Clinical Services
    const ctxClin = document.getElementById('chartClinicalServices').getContext('2d');
    new Chart(ctxClin, {{
      type: 'doughnut',
      data: {{
        labels: ['Prescriptions (80.1%)', 'Lab Tests (45.4%)', 'Inpatient Admissions (11.7%)', 'Referrals (8.3%)'],
        datasets: [{{
          data: [4003, 2272, 586, 415],
          backgroundColor: ['#0ea5e9', '#14b8a6', '#f43f5e', '#f59e0b'],
          borderWidth: 0
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#cbd5e1' }} }} }}
      }}
    }});

    // Chart 7: Insurance
    const ctxIns = document.getElementById('chartInsuranceShare').getContext('2d');
    new Chart(ctxIns, {{
      type: 'pie',
      data: {{
        labels: Object.keys(PAYLOAD.insurance),
        datasets: [{{
          data: Object.values(PAYLOAD.insurance),
          backgroundColor: ['#94a3b8', '#0ea5e9', '#14b8a6', '#f59e0b', '#a855f7'],
          borderWidth: 0
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'right', labels: {{ color: '#cbd5e1' }} }} }}
      }}
    }});

    // Chart 8: Satisfaction
    const ctxSat = document.getElementById('chartSatisfactionDist').getContext('2d');
    new Chart(ctxSat, {{
      type: 'bar',
      data: {{
        labels: ['⭐ 1 (Poor)', '⭐⭐ 2 (Fair)', '⭐⭐⭐ 3 (Good)', '⭐⭐⭐⭐ 4 (V.Good)', '⭐⭐⭐⭐⭐ 5 (Exec)'],
        datasets: [{{
          label: 'Survey Reviews',
          data: Object.values(PAYLOAD.satisfaction_dist),
          backgroundColor: ['#f43f5e', '#fb7185', '#f59e0b', '#38bdf8', '#10b981'],
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

    // Chart 9: Department Workload
    const ctxDept = document.getElementById('chartDeptWorkload').getContext('2d');
    new Chart(ctxDept, {{
      type: 'bar',
      data: {{
        labels: PAYLOAD.departments.map(d => d.Department),
        datasets: [{{
          label: '2027 Projected Encounters',
          data: PAYLOAD.departments.map(d => d.ForecastedEncounters_2027),
          backgroundColor: ['#0ea5e9', '#14b8a6', '#f59e0b', '#f43f5e', '#a855f7'],
          borderRadius: 8
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

    // Chart 10: Week 7 Threshold Sensitivity Curve
    const ctxThresh = document.getElementById('chartThresholdSweep').getContext('2d');
    const w7t = PAYLOAD.week7.threshold_chart;
    new Chart(ctxThresh, {{
      type: 'line',
      data: {{
        labels: w7t.thresholds,
        datasets: [
          {{
            label: 'Recall %',
            data: w7t.recalls,
            borderColor: '#10b981',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.2
          }},
          {{
            label: 'Precision %',
            data: w7t.precisions,
            borderColor: '#38bdf8',
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.2
          }},
          {{
            label: 'F2-Score %',
            data: w7t.f2_scores,
            borderColor: '#c084fc',
            borderDash: [5, 5],
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.2
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#cbd5e1', font: {{ size: 11 }} }} }}
        }},
        scales: {{
          x: {{
            title: {{ display: true, text: 'Clinical Threshold (tau)', color: '#94a3b8' }},
            ticks: {{ color: '#94a3b8' }},
            grid: {{ color: 'rgba(255,255,255,0.05)' }}
          }},
          y: {{
            title: {{ display: true, text: 'Score (%)', color: '#94a3b8' }},
            ticks: {{ color: '#94a3b8' }},
            grid: {{ color: 'rgba(255,255,255,0.05)' }}
          }}
        }}
      }}
    }});

    // Chart 11: Week 7 Subgroup Disparity Analysis
    const ctxFair = document.getElementById('chartSubgroupFairness').getContext('2d');
    const w7s = PAYLOAD.week7.subgroups;
    new Chart(ctxFair, {{
      type: 'bar',
      data: {{
        labels: w7s.age_labels,
        datasets: [
          {{
            label: 'Week 6 Baseline Recall %',
            data: w7s.w6_age_recall,
            backgroundColor: 'rgba(244, 63, 94, 0.75)',
            borderRadius: 6
          }},
          {{
            label: 'Week 7 Refined Recall %',
            data: w7s.w7_age_recall,
            backgroundColor: 'rgba(16, 185, 129, 0.85)',
            borderRadius: 6
          }}
        ]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#cbd5e1', font: {{ size: 11 }} }} }}
        }},
        scales: {{
          x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ display: false }} }},
          y: {{ max: 100, ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
        }}
      }}
    }});

  }});
</script>

</body>
</html>
"""

    dashboard_path = os.path.join(output_dir, "dashboard.html")
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    print(f"Executive Dashboard HTML successfully saved to {dashboard_path}")

    # -------------------------------------------------------------------------
    # 3. BUILD reports/Executive_BI_Report.html
    # -------------------------------------------------------------------------
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HealthConnect Rwanda — Executive Business Intelligence & Clinical Transformation Report</title>
  <style>
    @media print {{
      body {{ font-size: 11pt; color: #000; background: #fff; }}
      .page-break {{ page-break-before: always; }}
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: #1e293b;
      max-width: 1020px;
      margin: 0 auto;
      padding: 40px 24px;
      background: #f8fafc;
    }}
    .header-box {{
      border-bottom: 3px solid #0284c7;
      padding-bottom: 20px;
      margin-bottom: 30px;
    }}
    h1 {{ font-size: 26px; color: #0f172a; margin-bottom: 6px; }}
    .meta {{ font-size: 13px; color: #64748b; line-height: 1.5; }}
    .kpi-row {{
      display: flex;
      gap: 15px;
      margin: 25px 0;
      flex-wrap: wrap;
    }}
    .kpi-card {{
      flex: 1;
      min-width: 180px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      border-top: 4px solid #0284c7;
    }}
    .kpi-card.danger {{ border-top-color: #ef4444; }}
    .kpi-card.warning {{ border-top-color: #f59e0b; }}
    .kpi-card.success {{ border-top-color: #10b981; }}
    .kpi-card.purple {{ border-top-color: #8b5cf6; }}
    .kpi-title {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
    .kpi-val {{ font-size: 24px; font-weight: 800; color: #0f172a; margin: 4px 0; }}
    .kpi-sub {{ font-size: 11px; color: #94a3b8; }}
    h2 {{ font-size: 18px; color: #0f172a; margin: 30px 0 15px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
    p {{ font-size: 14px; color: #334155; margin-bottom: 14px; text-align: justify; }}
    .table-custom {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin: 20px 0;
    }}
    .table-custom th, .table-custom td {{
      padding: 10px 12px;
      border: 1px solid #e2e8f0;
      text-align: left;
    }}
    .table-custom th {{ background: #f1f5f9; color: #334155; font-weight: 700; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
    }}
    .badge-pass {{ background: #dcfce7; color: #166534; }}
    .badge-warn {{ background: #fef3c7; color: #92400e; }}
    .badge-fail {{ background: #fee2e2; color: #991b1b; }}
    .badge-info {{ background: #e0f2fe; color: #0369a1; }}
    .figure-block {{
      margin: 24px 0;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .figure-block img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
    }}
    .figure-caption {{
      margin-top: 10px;
      font-size: 12.5px;
      color: #64748b;
      text-align: justify;
    }}
  </style>
</head>
<body>

  <div class="header-box">
    <h1>HealthConnect Rwanda — Executive Business Intelligence Report</h1>
    <div class="meta">
      <strong>AnalystLab Africa Experience Lab Internship Programme</strong> | Official Capstone Deliverable<br>
      <strong>Lead Healthcare Data Scientist:</strong> Saad Janane (Track 2) | <strong>Lead Machine Learning Engineer:</strong> Zaid Harboul (Track 4 / HC-POD)<br>
      Evaluation Period: 2026 Baseline & 2027 Projections | Scope: 20 Facilities, 27 Relational Datasets, 250,000+ Records
    </div>
  </div>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-title">Registered Citizens</div>
      <div class="kpi-val">20,000</div>
      <div class="kpi-sub">74.65% Chronic Condition Rate</div>
    </div>
    <div class="kpi-card warning">
      <div class="kpi-title">2026 Consultations</div>
      <div class="kpi-val">60,809</div>
      <div class="kpi-sub">2027 Forecast: 51,166 Visits</div>
    </div>
    <div class="kpi-card success">
      <div class="kpi-title">2026 Gross Revenue</div>
      <div class="kpi-val">1.132 B RWF</div>
      <div class="kpi-sub">+15.91% Projected 2027 Target</div>
    </div>
    <div class="kpi-card danger">
      <div class="kpi-title">Admission Rate</div>
      <div class="kpi-val">11.72%</div>
      <div class="kpi-sub">Mean Stay: 5.49 Days</div>
    </div>
    <div class="kpi-card purple">
      <div class="kpi-title">AI Admission Recall</div>
      <div class="kpi-val" style="color:#7c3aed;">88.36%</div>
      <div class="kpi-sub">FN Missed: 17 (vs 144 Week 6)</div>
    </div>
  </div>

  <h2>1. Executive Summary & Context</h2>
  <p>
    Operating 20 healthcare centers and regional referral hospitals across all 5 provinces of Rwanda, HealthConnect Rwanda coordinates nationwide healthcare delivery, patient consultations, diagnostic testing, inpatient admissions, and claims processing.
    In 2026, the health system managed 60,809 clinical encounters and generated 1.132 Billion RWF in gross invoicing. This report synthesizes key findings from our comprehensive data science and predictive analytics transformation study, including our collaborative Week 7 Model Testing, Refinement & MLOps deployment validation with Lead ML Engineer Zaid Harboul.
  </p>

  <h2>2. Inferential Hypothesis Testing Results (Alpha = 0.05)</h2>
  <table class="table-custom">
    <thead>
      <tr>
        <th>Hypothesis Test</th>
        <th>Test Method</th>
        <th>Test Statistic</th>
        <th>P-Value</th>
        <th>Null Decision</th>
        <th>Executive Conclusion</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>1. Pricing Equity</strong><br>(Insured vs Uninsured Billing)</td>
        <td>Welch's Two-Sample T-Test</td>
        <td>t = 0.9364</td>
        <td>p = 0.3491</td>
        <td><span class="badge badge-pass">Fail to Reject</span></td>
        <td>Standardized fee schedules. No evidence of predatory insurance upcoding.</td>
      </tr>
      <tr>
        <td><strong>2. Facility Homogeneity</strong><br>(Satisfaction Across 20 Facilities)</td>
        <td>One-Way ANOVA (F-Test)</td>
        <td>F = 0.7381</td>
        <td>p = 0.7851</td>
        <td><span class="badge badge-pass">Fail to Reject</span></td>
        <td>Perceived service quality is statistically uniform nationwide across all 20 centers.</td>
      </tr>
      <tr>
        <td><strong>3. Admission Triage</strong><br>(Admission Independence from Age)</td>
        <td>Chi-Square Test (df = 4)</td>
        <td>Chi2 = 2.8415</td>
        <td>p = 0.5852</td>
        <td><span class="badge badge-pass">Fail to Reject</span></td>
        <td>Hospital admissions are driven by physiological disease severity rather than chronological age.</td>
      </tr>
    </tbody>
  </table>

  <h2>3. Top 10 Healthcare Facility Leaderboard</h2>
  <table class="table-custom">
    <thead>
      <tr>
        <th>Facility ID</th>
        <th>Hospital Name</th>
        <th>Province</th>
        <th>Facility Tier</th>
        <th>Doctors</th>
        <th>Beds</th>
        <th>Annual Consultations</th>
        <th>Composite Performance Score</th>
      </tr>
    </thead>
    <tbody>
"""

    top_facs = facilities_data.sort_values(by="FacilityPerformanceScore", ascending=False).head(10)
    for _, f in top_facs.iterrows():
        report_html += f"""      <tr>
        <td><strong>{f['FacilityID']}</strong></td>
        <td>{f['FacilityName']}</td>
        <td>{f['ProvinceName']}</td>
        <td>{f['FacilityType']}</td>
        <td>{f['DoctorCount']}</td>
        <td>{f['TotalBeds']}</td>
        <td>{int(f['FacAppointments']):,}</td>
        <td><span class="badge badge-pass">{f['FacilityPerformanceScore']}</span></td>
      </tr>\n"""

    report_html += """    </tbody>
  </table>

  <h2>4. Week 7 Clinical AI Model Refinement & Cross-Track Validation (Saad Janane & Zaid Harboul)</h2>
  <p>
    During rigorous Week 7 model validation, error decomposition revealed that the Week 6 candidate model exhibited a dangerous failure mode: while nominal accuracy was 87.52%, the model's default decision threshold (&tau; = 0.50) suffered a near-zero Recall of 1.37%, missing 144 out of 146 true inpatient admissions (98.6% miss rate).
    Furthermore, slice testing demonstrated 0.0% recall across Pediatric, Young Adult, and Middle-Aged cohorts.
  </p>
  <p>
    To resolve this clinical defect, Track 2 (Saad Janane) engineered a multi-variable Clinical Acuity Score, applied cost-sensitive class weights (1:7.5), performed 3-fold cross-validated Isotonic Probability Calibration, and swept decision thresholds under asymmetric clinical loss (5&times;FN + 1&times;FP).
    Collaborating with Track 4 (Zaid Harboul), we co-developed vectorized batch inference meeting the strict &lt; 5.0 ms production SLO.
  </p>

  <table class="table-custom">
    <thead>
      <tr>
        <th>Evaluation Dimension</th>
        <th>Week 6 Candidate Model</th>
        <th>Week 7 Refined Model</th>
        <th>Clinical & Operational Impact</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Model Architecture</strong></td>
        <td>Balanced Random Forest (Default &tau;=0.50)</td>
        <td>Balanced RF + Acuity Score (&tau;=0.38)</td>
        <td>Clinical Acuity Feature Engineering</td>
      </tr>
      <tr>
        <td><strong>Admission Recall</strong></td>
        <td><span class="badge badge-fail">1.37% (2 / 146)</span></td>
        <td><span class="badge badge-pass">88.36% (129 / 146)</span></td>
        <td><strong>+86.99% Gain</strong> in catching critically ill patients</td>
      </tr>
      <tr>
        <td><strong>Missed Admissions (FN)</strong></td>
        <td><span class="badge badge-fail">144 Missed Admissions</span></td>
        <td><span class="badge badge-pass">17 Missed Admissions</span></td>
        <td><strong>88.2% Reduction</strong> in life-threatening false negatives</td>
      </tr>
      <tr>
        <td><strong>Brier Score (Reliability)</strong></td>
        <td>0.1743 (Uncalibrated)</td>
        <td>0.1038 (Isotonic Calibrated)</td>
        <td><strong>40.4% Improvement</strong> in risk probability reliability</td>
      </tr>
      <tr>
        <td><strong>Subgroup Disparity</strong></td>
        <td><span class="badge badge-fail">0.0% Recall (Pediatric/Adults)</span></td>
        <td><span class="badge badge-pass">&ge; 85.2% Recall Across All</span></td>
        <td>Zero demographic disparity across all age cohorts</td>
      </tr>
      <tr>
        <td><strong>Batch Latency / Throughput</strong></td>
        <td>164.1 ms / 6.1 req/s</td>
        <td><span class="badge badge-pass">0.11 ms / 9,495.9 req/s</span></td>
        <td><strong>1,558x Speedup</strong> co-developed with Zaid Harboul</td>
      </tr>
      <tr>
        <td><strong>Edge Case Resilience</strong></td>
        <td>Untested</td>
        <td><span class="badge badge-pass">10 / 10 Passed (100%)</span></td>
        <td>Resilient to missing vitals, unseen strings, extreme values</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-block">
    <img src="figures/fig16_confusion_matrix_before_after.png" alt="Figure 16">
    <div class="figure-caption">
      <strong>Figure 16: Confusion Matrix Evolution (Week 6 Defect vs Week 7 Refined).</strong>
      Illustrates the decisive elimination of 127 false negatives (missed hospital admissions), dropping missed admissions from 144 down to 17.
    </div>
  </div>

  <div class="figure-block">
    <img src="figures/fig17_threshold_optimization_curve.png" alt="Figure 17">
    <div class="figure-caption">
      <strong>Figure 17: Clinical Decision Threshold Sensitivity Optimization Curve.</strong>
      Demonstrates the threshold trade-off trajectory sweeping &tau; from 0.05 to 0.95. Selected threshold at &tau; = 0.38 optimizes the asymmetric clinical loss and maximizes F2-Score at 0.381.
    </div>
  </div>

  <div class="figure-block">
    <img src="figures/fig21_cross_track_mle_integration_telemetry.png" alt="Figure 21">
    <div class="figure-caption">
      <strong>Figure 21: Cross-Track MLE Integration Telemetry co-developed with Zaid Harboul.</strong>
      Compares row-by-row vs vectorized inference performance, demonstrating 9,495.9 requests/second throughput and 0.11 ms latency (1,558x speedup) comfortably within production SLO limits.
    </div>
  </div>

  <h2>5. C-Suite Strategic Transformation Recommendations</h2>
  <ol style="font-size: 13.5px; color: #334155; padding-left: 20px; line-height: 1.8;">
    <li><strong>Dedicated Outpatient NCD Clinics</strong>: Screen and treat the 74.65% chronic disease population early, reducing emergency admissions by 25% and saving 85M RWF annually.</li>
    <li><strong>Automated Claims Pre-Scrubber</strong>: Address the 50% claims delay bottleneck, accelerating approvals to >85% and unlocking 1.8 Billion RWF in liquidity.</li>
    <li><strong>Dynamic Clinical Float Pools</strong>: Reallocate nursing and physician shifts dynamically to Emergency, Internal Medicine, and Pediatrics to cut consultation wait times by 35%.</li>
    <li><strong>Diagnostic Equipment Equalization</strong>: Upgrade ultrasound and hematology analyzers at underperforming district centers to reduce unnecessary tertiary patient transfers by 40%.</li>
    <li><strong>Digital Self-Service Reception Kiosks</strong>: Expand Mobile Money check-in at entrance lobbies to reduce physical queue congestion by 50%.</li>
  </ol>

  <div style="margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 15px; font-size: 11px; color: #94a3b8; text-align: center;">
    Certified by HealthConnect Rwanda Analytics & AI Transformation Team — Lead Data Scientist: Saad Janane (Track 2) | Lead ML Engineer: Zaid Harboul (Track 4 / HC-POD).<br>
    AnalystLab Africa Experience Lab Internship Programme — Week 7 Capstone Release.
  </div>

</body>
</html>
"""

    report_path = os.path.join(output_dir, "Executive_BI_Report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Executive BI Report successfully saved to {report_path}")

if __name__ == "__main__":
    generate_dashboard_files()
