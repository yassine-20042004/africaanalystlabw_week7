"""
HealthConnect Rwanda — Executive Healthcare Analytics & AI Transformation Dashboard
Official Streamlit Application (Week 7 Model Testing, Refinement & End-to-End Validation)
Lead Healthcare Data Scientist: Saad Janane (Track 2)
Lead Machine Learning Engineer / Collaborator: Zaid Harboul (Track 4 / HC-POD)
Institution: AnalystLab Africa Experience Lab Internship Programme
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is importable
sys.path.append(os.path.abspath("."))
try:
    from src.week7_cross_track_validation import HealthConnectInferenceEngine
except Exception:
    HealthConnectInferenceEngine = None

# Set page configuration
st.set_page_config(
    page_title="HealthConnect Rwanda — Executive AI & BI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics
st.markdown("""
<style>
    :root {
        --primary: #0ea5e9;
        --secondary: #14b8a6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #f43f5e;
        --purple: #a855f7;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
        padding: 18px 22px;
        border-radius: 14px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(14, 165, 233, 0.4);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 800;
        font-family: 'Outfit', sans-serif, system-ui;
        color: #38bdf8;
        margin: 6px 0 2px;
    }
    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-teal { background: rgba(20, 184, 166, 0.15); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.3); }
    .badge-sky { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

    .team-banner {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.12), rgba(20, 184, 166, 0.12));
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .figure-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA AND MODEL CACHING
# ==============================================================================
@st.cache_data
def load_healthcare_data():
    """Loads all processed datasets for HealthConnect Rwanda."""
    facs = pd.read_csv("data/processed/cleaned_facilities.csv")
    perf = pd.read_csv("data/processed/facility_performance_metrics.csv")
    facilities_df = facs.merge(perf, on="FacilityID", how="left")
    
    v_fc = pd.read_csv("data/processed/forecast_patient_volume_12m.csv")
    r_fc = pd.read_csv("data/processed/forecast_hospital_revenue_12m.csv")
    r_hist = pd.read_csv("data/raw/HC_Revenue_Summary.csv")
    dept_fc = pd.read_csv("data/processed/forecast_department_workload_2027.csv")
    clusters_df = pd.read_csv("data/processed/patient_cluster_profiles.csv")
    cohort_df = pd.read_csv("data/processed/feature_engineered_cohort.csv")
    
    # Week 7 specific assets
    thresh_df = pd.read_csv("data/processed/week7_threshold_sweep.csv")
    with open("data/processed/week7_model_testing_results.json", "r") as f:
        w7_results = json.load(f)
    with open("data/processed/week7_cross_track_mle_telemetry.json", "r") as f:
        mle_telemetry = json.load(f)

    return facilities_df, v_fc, r_fc, r_hist, dept_fc, clusters_df, cohort_df, thresh_df, w7_results, mle_telemetry


@st.cache_resource
def load_inference_engine():
    """Loads serialized Refined Admission Model engine co-developed with Zaid Harboul."""
    if HealthConnectInferenceEngine is not None and os.path.exists("models/refined_admission_model.pkl"):
        return HealthConnectInferenceEngine(model_path="models/refined_admission_model.pkl")
    return None


# Load Assets
(
    facilities_df,
    v_fc,
    r_fc,
    r_hist,
    dept_fc,
    clusters_df,
    cohort_df,
    thresh_df,
    w7_results,
    mle_telemetry,
) = load_healthcare_data()

engine = load_inference_engine()


# ==============================================================================
# SIDEBAR NAVIGATION & GLOBAL FILTERS
# ==============================================================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 12px;">
    <div style="font-size: 38px;">🏥</div>
    <h2 style="margin: 0; font-size: 20px; font-weight: 700; background: linear-gradient(90deg, #38bdf8, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        HealthConnect Rwanda
    </h2>
    <p style="font-size: 11px; color: #94a3b8; margin: 4px 0 0;">Executive Analytics & AI Transformation</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="font-size: 12px; background: rgba(255,255,255,0.04); padding: 10px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 14px;">
    <strong>Lead Data Scientist:</strong> Saad Janane (Track 2)<br>
    <strong>Lead ML Engineer:</strong> Zaid Harboul (Track 4)<br>
    <strong>Experience Lab:</strong> Week 7 Final Release
</div>
""", unsafe_allow_html=True)

nav_choice = st.sidebar.radio(
    "Navigation Track",
    [
        "📊 Executive KPI Command Center",
        "🩺 Clinical & Population Health",
        "📈 2027 Projections & Capacity",
        "🔬 Week 7 AI Model Refinement & Triage Sandbox",
        "🖼️ Visualizations & Publication Gallery",
        "📑 Strategic Roadmap & Governance",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Operational Slicers")

# Filter: Province
provinces = ["All Provinces"] + sorted(facilities_df["ProvinceName"].dropna().unique().tolist())
selected_province = st.sidebar.selectbox("Filter Province", provinces)

# Filter: Facility Tier
fac_tiers = ["All Facility Tiers"] + sorted(facilities_df["FacilityType"].dropna().unique().tolist())
selected_tier = st.sidebar.selectbox("Filter Facility Tier", fac_tiers)

# Filter: Ownership
ownerships = ["All Ownerships"] + sorted(facilities_df["Ownership"].dropna().unique().tolist())
selected_ownership = st.sidebar.selectbox("Filter Ownership", ownerships)

# Apply filters to facilities dataset
filtered_facs = facilities_df.copy()
if selected_province != "All Provinces":
    filtered_facs = filtered_facs[filtered_facs["ProvinceName"] == selected_province]
if selected_tier != "All Facility Tiers":
    filtered_facs = filtered_facs[filtered_facs["FacilityType"] == selected_tier]
if selected_ownership != "All Ownerships":
    filtered_facs = filtered_facs[filtered_facs["Ownership"] == selected_ownership]

st.sidebar.markdown("---")
st.sidebar.caption(f"Active Facilities: **{len(filtered_facs)} of {len(facilities_df)}**")
st.sidebar.caption("AnalystLab Africa Experience Lab © 2026")


# ==============================================================================
# TRACK 1: EXECUTIVE KPI COMMAND CENTER
# ==============================================================================
if nav_choice == "📊 Executive KPI Command Center":
    st.title("📊 Executive Healthcare Command Center")
    st.markdown("Macro operational indicators, facility performance metrics, and 2026-2027 systemic throughput.")

    st.markdown("""
    <div class="team-banner">
        <div>
            <strong>🏥 HealthConnect Rwanda Nationwide Healthcare System</strong> — 20 Facilities across 5 Provinces
        </div>
        <div>
            <span class="badge-pill badge-sky">Track 2: Data Science</span>
            <span class="badge-pill badge-teal">Track 4: ML Engineering (Zaid Harboul)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">REGISTERED CITIZENS</div>
            <div class="metric-val">20,000</div>
            <div class="metric-sub"><span class="badge-pill badge-teal">74.65%</span> Chronic Condition Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">ANNUAL CONSULTATIONS</div>
            <div class="metric-val">{int(r_hist['Appointments'].sum()):,}</div>
            <div class="metric-sub">2027 Forecast: <strong style="color:#f59e0b">51,166 Visits</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">GROSS REVENUE (2026)</div>
            <div class="metric-val">1.132 B RWF</div>
            <div class="metric-sub"><span class="badge-pill badge-sky">+15.91%</span> 2027 Target: 1.313 B RWF</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">ADMISSION RATE / STAY</div>
            <div class="metric-val">11.72%</div>
            <div class="metric-sub">Mean Stay: <strong>5.49 Days</strong> (Bed Cap: 2,000)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts
    col_ts, col_fac = st.columns([1.6, 1.2])

    with col_ts:
        st.subheader("📈 24-Month Patient Visit Trajectory & 2027 Projections")
        hist_months = r_hist["Month"].tolist()
        hist_visits = r_hist["Appointments"].tolist()
        fc_months = v_fc["Month"].tolist()
        fc_visits = v_fc["ForecastedVisits"].tolist()

        ts_df = pd.DataFrame({
            "Month": hist_months + fc_months,
            "Historical Visits (2026)": hist_visits + [None] * len(fc_visits),
            "Forecasted Visits (2027)": [None] * (len(hist_visits) - 1) + [hist_visits[-1]] + fc_visits,
        }).set_index("Month")

        st.line_chart(ts_df, color=["#0ea5e9", "#f59e0b"], height=340)
        st.caption("Holt-Winters Triple Exponential Smoothing Model (MAPE: 19.69%, Normalized RMSE: 0.161).")

    with col_fac:
        st.subheader("🏥 Facility Composite Performance Score")
        sorted_facs = filtered_facs.sort_values(by="FacilityPerformanceScore", ascending=True).tail(8)
        
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4.2), facecolor="#0f172a")
        ax_bar.set_facecolor("#0f172a")
        colors = ["#38bdf8" if score >= 85 else "#14b8a6" for score in sorted_facs["FacilityPerformanceScore"]]
        bars = ax_bar.barh(sorted_facs["FacilityName"], sorted_facs["FacilityPerformanceScore"], color=colors, height=0.65)
        ax_bar.set_xlim(0, 100)
        ax_bar.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax_bar.spines.values():
            spine.set_color("#334155")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.2, color="#94a3b8")
        for bar in bars:
            w = bar.get_width()
            ax_bar.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}", va="center", color="#f8fafc", fontsize=8.5, fontweight="bold")
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # Row 3: Facilities Data Table
    st.subheader("📋 Sliced Facility Operations Table")
    display_cols = [
        "FacilityID", "FacilityName", "ProvinceName", "FacilityType", "Ownership",
        "DoctorCount", "TotalBeds", "FacAppointments", "FacAdmissions", "FacilityPerformanceScore"
    ]
    st.dataframe(
        filtered_facs[display_cols].rename(columns={
            "FacilityID": "ID",
            "FacilityName": "Facility Name",
            "ProvinceName": "Province",
            "FacilityType": "Tier",
            "DoctorCount": "Doctors",
            "TotalBeds": "Beds",
            "FacAppointments": "Consultations",
            "FacAdmissions": "Admissions",
            "FacilityPerformanceScore": "Composite Score (0-100)"
        }),
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# TRACK 2: CLINICAL & POPULATION HEALTH
# ==============================================================================
elif nav_choice == "🩺 Clinical & Population Health":
    st.title("🩺 Clinical & Epidemiological Intelligence")
    st.markdown("Chronic disease morbidity patterns, laboratory positivity rates, and diagnostic distributions.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">CHRONIC DISEASE PREVALENCE</div>
            <div class="metric-val">74.65%</div>
            <div class="metric-sub">14,930 of 20,000 Registered Citizens</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">LAB POSITIVITY RATE</div>
            <div class="metric-val">50.30%</div>
            <div class="metric-sub">2,272 Lab Tests Ordered Across 5,000 Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">PRESCRIPTION RATE</div>
            <div class="metric-val">80.06%</div>
            <div class="metric-sub">4,003 Encounters Dispensed Medication</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_pie, col_lab = st.columns(2)

    with col_pie:
        st.subheader("🫀 Chronic Morbidity Distribution")
        chronic_counts = cohort_df["ChronicCondition"].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4.2), facecolor="#0f172a")
        ax_pie.set_facecolor("#0f172a")
        wedges, texts, autotexts = ax_pie.pie(
            chronic_counts.values,
            labels=chronic_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#0ea5e9", "#14b8a6", "#f59e0b", "#a855f7"],
            textprops={"color": "#f8fafc", "fontsize": 10},
            wedgeprops={"edgecolor": "#0f172a", "linewidth": 2}
        )
        for at in autotexts:
            at.set_color("#070d1e")
            at.set_fontweight("bold")
        st.pyplot(fig_pie, use_container_width=True)
        plt.close(fig_pie)
        st.caption("Hypertension, Diabetes, and Asthma account for nearly 75% of clinical visit volume.")

    with col_lab:
        st.subheader("🔬 Diagnostic Breakdown (5,000 Cohort)")
        diag_counts = cohort_df["Diagnosis"].value_counts().head(6)
        fig_diag, ax_diag = plt.subplots(figsize=(6, 4.2), facecolor="#0f172a")
        ax_diag.set_facecolor("#0f172a")
        bars_d = ax_diag.bar(diag_counts.index, diag_counts.values, color="#14b8a6", width=0.6)
        ax_diag.tick_params(colors="#cbd5e1", labelsize=8.5, rotation=25)
        for spine in ax_diag.spines.values():
            spine.set_color("#334155")
        ax_diag.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
        for bar in bars_d:
            h = bar.get_height()
            ax_diag.text(bar.get_x() + bar.get_width() / 2, h + 15, f"{h}", ha="center", color="#f8fafc", fontsize=8.5, fontweight="bold")
        st.pyplot(fig_diag, use_container_width=True)
        plt.close(fig_diag)
        st.caption("Hypertension, Type 2 Diabetes, and Asthma represent primary outpatient encounter reasons.")

    st.subheader("⭐ Patient Satisfaction Uniformity (Inferential Testing)")
    st.info(
        "**One-Way ANOVA Hypothesis Test**: $F = 0.7381$, $p = 0.7851$ (Alpha = 0.05). "
        "Fail to Reject Null Hypothesis ($H_0$). Patient satisfaction is statistically uniform nationwide across all 20 hospital centers (mean: 3.01 / 5.00), demonstrating that operational bottlenecks are systemic rather than localized."
    )


# ==============================================================================
# TRACK 3: 2027 PROJECTIONS & CAPACITY
# ==============================================================================
elif nav_choice == "📈 2027 Projections & Capacity":
    st.title("📈 2027 Projections & Capacity Planning")
    st.markdown("Holt-Winters time-series forecasting for patient visits, gross revenue, and departmental workload.")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">2027 PROJECTED ENCOUNTERS</div>
            <div class="metric-val">51,166 Visits</div>
            <div class="metric-sub">Monthly Average: <strong>4,264 Visits</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">2027 REVENUE PROJECTION</div>
            <div class="metric-val">1.313 B RWF</div>
            <div class="metric-sub"><span class="badge-pill badge-sky">+15.91%</span> Growth vs 2026 Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">BED UTILIZATION STRESS</div>
            <div class="metric-val">1,215 Active Beds</div>
            <div class="metric-sub">Baseline Capacity: <strong>2,000 Total Beds</strong> (60.8% Occupancy)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_rev, col_dept = st.columns([1.5, 1.2])

    with col_rev:
        st.subheader("💵 24-Month Revenue Trajectory (Million RWF)")
        hist_rev = (r_hist["Revenue"] / 1e6).round(2).tolist()
        fc_rev = (r_fc["ForecastedRevenue_RWF"] / 1e6).round(2).tolist()

        rev_chart_df = pd.DataFrame({
            "Month": r_hist["Month"].tolist() + r_fc["Month"].tolist(),
            "2026 Actual Revenue": hist_rev + [None] * len(fc_rev),
            "2027 Forecasted Revenue": [None] * (len(hist_rev) - 1) + [hist_rev[-1]] + fc_rev
        }).set_index("Month")

        st.line_chart(rev_chart_df, color=["#10b981", "#f59e0b"], height=340)
        st.caption("Holt-Winters Revenue Forecast projecting sustained growth reaching 122.9 Million RWF in December 2027.")

    with col_dept:
        st.subheader("🏥 2027 Departmental Workload Demand")
        fig_dept, ax_dept = plt.subplots(figsize=(6, 4.2), facecolor="#0f172a")
        ax_dept.set_facecolor("#0f172a")
        bars_dept = ax_dept.bar(
            dept_fc["Department"],
            dept_fc["ForecastedEncounters_2027"],
            color=["#0ea5e9", "#14b8a6", "#f59e0b", "#f43f5e", "#a855f7"],
            width=0.6
        )
        ax_dept.tick_params(colors="#cbd5e1", labelsize=8.5, rotation=20)
        for spine in ax_dept.spines.values():
            spine.set_color("#334155")
        ax_dept.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
        for bar in bars_dept:
            h = bar.get_height()
            ax_dept.text(bar.get_x() + bar.get_width() / 2, h + 150, f"{h:,}", ha="center", color="#f8fafc", fontsize=8.5, fontweight="bold")
        st.pyplot(fig_dept, use_container_width=True)
        plt.close(fig_dept)
        st.caption("Emergency and Internal Medicine handle 44% of total forecasted inpatient bed stress.")


# ==============================================================================
# TRACK 4: WEEK 7 AI MODEL REFINEMENT & TRIAGE SANDBOX
# ==============================================================================
elif nav_choice == "🔬 Week 7 AI Model Refinement & Triage Sandbox":
    st.title("🔬 Week 7 AI Model Refinement & Clinical Triage Sandbox")
    st.markdown(
        "Interactive testing, asymmetric clinical cost optimization, demographic slice fairness, "
        "and real-time inference co-developed with **Zaid Harboul** (Lead ML Engineer)."
    )

    st.markdown("""
    <div class="team-banner">
        <div>
            <strong>Cross-Track Engineering Team:</strong> Saad Janane (Lead Healthcare Data Scientist) & Zaid Harboul (Lead ML Engineer)
        </div>
        <div>
            <span class="badge-pill badge-sky">Model v2.0-Week7-Validated</span>
            <span class="badge-pill badge-teal">Production SLO: &lt; 5.0 ms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Comparison Cards (Week 6 Defect vs Week 7 Refined)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">ADMISSION RECALL</div>
            <div class="metric-val" style="color: #10b981;">88.36%</div>
            <div class="metric-sub">Week 6 Baseline: <span style="color:#fb7185">1.37%</span> (+86.99%)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">MISSED ADMISSIONS (FN)</div>
            <div class="metric-val" style="color: #38bdf8;">17 Patients</div>
            <div class="metric-sub">Week 6 Missed: <span style="color:#fb7185">144</span> (88.2% Reduction)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">BRIER SCORE (CALIBRATION)</div>
            <div class="metric-val" style="color: #a855f7;">0.1038</div>
            <div class="metric-sub">Week 6 Uncalibrated: <span style="color:#fb7185">0.1743</span> (-40.4%)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-sub">VECTORIZED LATENCY</div>
            <div class="metric-val" style="color: #f59e0b;">0.11 ms</div>
            <div class="metric-sub">Throughput: <strong>9,495.9 req/s</strong> (1,558x speedup)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # INTERACTIVE THRESHOLD SLIDER SIMULATOR
    st.subheader("🎚️ Interactive Decision Threshold Sensitivity Simulator")
    st.markdown(
        "Adjust the clinical decision threshold $\\tau$ to simulate operational trade-offs between "
        "missed critical admissions (False Negatives, cost weight = 5) and triage over-allocation (False Positives, cost weight = 1)."
    )

    t_col1, t_col2 = st.columns([1.1, 1.9])

    with t_col1:
        selected_thresh = st.slider(
            "Select Clinical Decision Threshold (tau)",
            min_value=0.25,
            max_value=0.55,
            value=0.38,
            step=0.005,
            help="Optimal threshold determined by Cost-Sensitive Asymmetric Loss is tau = 0.38."
        )

        # Match closest threshold in sweep dataset
        closest_idx = (thresh_df["Threshold"] - selected_thresh).abs().idxmin()
        row = thresh_df.loc[closest_idx]

        cur_rec = row["Recall"] * 100
        cur_prec = row["Precision"] * 100
        cur_fn = int(row["False_Negatives"])
        cur_fp = int(row["False_Positives"])
        cur_loss = row["Clinical_Loss"]
        cur_f2 = row["F2_Score"]

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 14px; margin-top: 10px;">
            <div style="font-size: 13px; color: #94a3b8;">Simulated Operating Point (tau = {selected_thresh:.3f}):</div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                <span>Admission Recall:</span> <strong style="color: {'#10b981' if cur_rec >= 80 else '#fb7185'}">{cur_rec:.1f}%</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Precision:</span> <strong>{cur_prec:.1f}%</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>False Negatives (Missed):</span> <strong style="color: {'#10b981' if cur_fn <= 20 else '#fb7185'}">{cur_fn} / 146</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>False Positives (Over-triage):</span> <strong>{cur_fp} / 1104</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Clinical Loss (5xFN + FP):</span> <strong style="color: #38bdf8;">{cur_loss:.1f}</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>F2-Score (Recall Weighted):</span> <strong style="color: #a855f7;">{cur_f2:.3f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_thresh == 0.38:
            st.success("✅ **Optimal Production Threshold Selected**: Captures 88.36% of admissions while keeping false negatives strictly at 17.")
        elif selected_thresh > 0.45:
            st.warning("⚠️ High threshold creates dangerous clinical false negatives (missed critically ill patients)!")
        elif selected_thresh < 0.30:
            st.info("ℹ️ Low threshold catches nearly 100% admissions but over-allocates hospital beds.")

    with t_col2:
        # Dynamic Confusion Matrix visualization at selected threshold
        fig_cm, (ax_cm, ax_curve) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#0f172a")
        
        # Subplot 1: Dynamic Confusion Matrix Heatmap
        ax_cm.set_facecolor("#0f172a")
        tp = 146 - cur_fn
        tn = 1104 - cur_fp
        cm_data = np.array([[tn, cur_fp], [cur_fn, tp]])
        
        sns.heatmap(
            cm_data,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            annot_kws={"size": 13, "weight": "bold", "color": "#f8fafc"}
        )
        ax_cm.set_title(f"Dynamic Confusion Matrix (tau={selected_thresh:.2f})", color="#f8fafc", fontsize=10, pad=8)
        ax_cm.set_xlabel("Predicted Admission", color="#cbd5e1", fontsize=9)
        ax_cm.set_ylabel("True Admission", color="#cbd5e1", fontsize=9)
        ax_cm.set_xticklabels(["Discharge (0)", "Admit (1)"], color="#cbd5e1", fontsize=8.5)
        ax_cm.set_yticklabels(["Discharge (0)", "Admit (1)"], color="#cbd5e1", fontsize=8.5)

        # Subplot 2: Threshold Optimization Curve
        ax_curve.set_facecolor("#0f172a")
        ax_curve.plot(thresh_df["Threshold"], thresh_df["Recall"] * 100, label="Recall %", color="#10b981", lw=2)
        ax_curve.plot(thresh_df["Threshold"], thresh_df["Precision"] * 100, label="Precision %", color="#38bdf8", lw=2)
        ax_curve.plot(thresh_df["Threshold"], thresh_df["F2_Score"] * 100, label="F2 Score %", color="#a855f7", lw=2, linestyle="--")
        ax_curve.axvline(selected_thresh, color="#f43f5e", lw=2, linestyle=":", label=f"Selected (tau={selected_thresh:.2f})")
        ax_curve.set_title("Precision-Recall-F2 vs Threshold", color="#f8fafc", fontsize=10, pad=8)
        ax_curve.set_xlabel("Decision Threshold (tau)", color="#cbd5e1", fontsize=9)
        ax_curve.set_ylabel("Metric Value (%)", color="#cbd5e1", fontsize=9)
        ax_curve.tick_params(colors="#cbd5e1", labelsize=8)
        ax_curve.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=7.5, loc="center right")
        for spine in ax_curve.spines.values():
            spine.set_color("#334155")
        ax_curve.grid(alpha=0.15, color="#94a3b8")

        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    st.markdown("---")

    # LIVE PATIENT TRIAGE PREDICTOR SANDBOX
    st.subheader("🩺 Live Clinical Triage Inference Sandbox")
    st.markdown("Simulate an incoming patient encounter to test real-time risk scoring and bed pre-allocation.")

    with st.form("patient_triage_form"):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            patient_id = st.text_input("Patient ID", value="RW-PT-2027-0891")
            age = st.slider("Patient Age", 1, 95, 58)
            gender = st.selectbox("Gender", ["Female", "Male"])

        with f_col2:
            chronic = st.selectbox("Chronic Morbidity", ["Hypertension", "Diabetes", "Asthma", "None"])
            severity = st.slider("Disease Severity Score", 1, 5, 4)
            visit_type = st.selectbox("Visit Type", ["Emergency", "Routine Consultation", "Specialist Follow-up"])

        with f_col3:
            bmi = st.number_input("Body Mass Index (BMI)", value=28.4, min_value=12.0, max_value=60.0, step=0.1)
            lab_result = st.selectbox("Diagnostic Lab Status", ["Abnormal", "Normal", "Critical"])
            doc_exp = st.slider("Doctor Experience (Years)", 1, 35, 12)

        with f_col4:
            facility_type = st.selectbox("Facility Tier", ["Referral Hospital", "Provincial Hospital", "District Hospital", "General Hospital"])
            province = st.selectbox("Province", ["Kigali City", "Eastern", "Northern", "Southern", "Western"])
            amount = st.number_input("Estimated Encounter Billing (RWF)", value=32500, step=2500)

        submit_triage = st.form_submit_button("⚡ Run Real-Time Clinical Triage Inference", use_container_width=True)

    if submit_triage and engine is not None:
        raw_record = {
            "PatientID": patient_id,
            "Age": age,
            "Gender": gender,
            "ChronicCondition": chronic,
            "DiseaseSeverityScore": severity,
            "VisitType": visit_type,
            "BMI": bmi,
            "DoctorExperience": doc_exp,
            "Amount": amount,
            "HasAbnormalLab": 1 if lab_result in ["Abnormal", "Critical"] else 0,
            "LabResult": lab_result,
            "FacilityType": facility_type,
            "ProvinceName": province,
            "Ownership": "Public",
            "DepartmentWorkloadIndex": 1.25,
            "Diagnosis": f"{chronic} Complication" if chronic != "None" else "Acute Febrile Illness"
        }

        # Execute single prediction
        result = engine.predict_single(raw_record)
        
        # Calculate Acuity Score for explanation
        acuity = (
            (1 if visit_type == "Emergency" else 0) * 3 +
            (1 if severity >= 3 else 0) * 2 +
            (1 if lab_result in ["Abnormal", "Critical"] else 0) * 2 +
            (1 if chronic != "None" else 0)
        )

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            st.markdown("#### 🎯 Triage Decision Output")
            if result["IsAdmissionRecommended"]:
                st.error(f"🚨 **PREDICTED ACTION: {result['TriageAction']}**")
                st.markdown(f"""
                - **Admission Risk Score:** `{result['ClinicalRiskScore'] * 100:.1f}%`
                - **Calibrated Empirical Probability:** `{result['CalibratedEmpiricalProbability'] * 100:.1f}%`
                - **Decision Threshold (tau):** `{result['DecisionThreshold']:.2f}`
                - **Clinical Acuity Score:** `{acuity} / 8.0` (High Acuity)
                - **Bed Management Alert:** Bed Pre-allocation Protocol Activated!
                """)
            else:
                st.success(f"🟢 **PREDICTED ACTION: {result['TriageAction']}**")
                st.markdown(f"""
                - **Admission Risk Score:** `{result['ClinicalRiskScore'] * 100:.1f}%`
                - **Calibrated Empirical Probability:** `{result['CalibratedEmpiricalProbability'] * 100:.1f}%`
                - **Decision Threshold (tau):** `{result['DecisionThreshold']:.2f}`
                - **Clinical Acuity Score:** `{acuity} / 8.0` (Low Acuity)
                - **Bed Management Alert:** Routine Outpatient Queue Assigned.
                """)

        with res_col2:
            st.markdown("#### 🔍 Explainability Breakdown")
            st.markdown(f"""
            - **Emergency Acuity Weight:** `{3 if visit_type == 'Emergency' else 0} pts`
            - **Physiological Severity (Level {severity}):** `{2 if severity >= 3 else 0} pts`
            - **Diagnostic Lab Status ({lab_result}):** `{2 if lab_result in ['Abnormal', 'Critical'] else 0} pts`
            - **Chronic Condition Comorbidity ({chronic}):** `{1 if chronic != 'None' else 0} pt`
            
            *Model incorporates Isotonic Calibration trained via 3-fold cross validation to eliminate probability distortion.*
            """)

    st.markdown("---")

    # MLOPS TELEMETRY WITH ZAID HARBOUL
    st.subheader("⚡ Joint DS <-> MLE Production Telemetry (Co-developed with Zaid Harboul)")
    
    t_m1, t_m2, t_m3, t_m4 = st.columns(4)
    with t_m1:
        throughput = mle_telemetry.get("Latency_Benchmark", {}).get("Refined_Vectorized_Throughput_QPS", 9495.9)
        st.metric("Vectorized Throughput", f"{throughput:.1f} req/s", "+1,558x Speedup")
    with t_m2:
        latency = mle_telemetry.get("Latency_Benchmark", {}).get("Refined_Vectorized_Per_Sample_ms", 0.11)
        st.metric("Batch Inference Latency", f"{latency:.2f} ms", "SLO: < 5.0 ms")
    with t_m3:
        passed_tests = mle_telemetry.get("Edge_Case_Testing", {}).get("Passed", 10)
        total_tests = mle_telemetry.get("Edge_Case_Testing", {}).get("Total_Tests", 10)
        st.metric("Edge Cases Resilience", f"{passed_tests}/{total_tests} Passed", "100% Reliability")
    with t_m4:
        st.metric("Demographic Recall Parity", ">= 85.2%", "Zero Disparity Drop")


# ==============================================================================
# TRACK 5: VISUALIZATIONS & PUBLICATION GALLERY
# ==============================================================================
elif nav_choice == "🖼️ Visualizations & Publication Gallery":
    st.title("🖼️ Visualizations & Publication Figures Gallery")
    st.markdown("Comprehensive 300 DPI high-resolution figures generated across Week 6 and Week 7 milestones.")

    figures_meta = [
        # WEEK 7 FIGURES
        {
            "id": "fig16",
            "filename": "reports/figures/fig16_confusion_matrix_before_after.png",
            "category": "Week 7 Model Testing & Refinement",
            "title": "Figure 16: Confusion Matrix Evolution (Week 6 vs Week 7)",
            "subtitle": "Asymmetric Clinical Loss Minimization & False Negative Reduction",
            "takeaway": "Week 6 baseline model suffered from a 98.6% miss rate (144 false negatives). Refined model with balanced class weighting and optimal threshold (tau=0.38) drops missed admissions to 17 (88.2% reduction)."
        },
        {
            "id": "fig17",
            "filename": "reports/figures/fig17_threshold_optimization_curve.png",
            "category": "Week 7 Model Testing & Refinement",
            "title": "Figure 17: Clinical Decision Threshold Sensitivity Curve",
            "subtitle": "Precision, Recall, F2-Score, and Asymmetric Loss (5xFN + 1xFP) Trajectory",
            "takeaway": "Optimal clinical threshold at tau = 0.38 achieves 88.36% Recall and maximizes F2 at 0.381, balancing bed over-allocation against life-threatening false negatives."
        },
        {
            "id": "fig18",
            "filename": "reports/figures/fig18_roc_pr_curves_comparison.png",
            "category": "Week 7 Model Testing & Refinement",
            "title": "Figure 18: ROC and Precision-Recall Curves",
            "subtitle": "Discriminative Ability in Highly Imbalanced Clinical Setting (11.72% Admissions)",
            "takeaway": "Demonstrates precision-recall trade-offs in highly skewed healthcare populations, guiding clinical triage routing."
        },
        {
            "id": "fig19",
            "filename": "reports/figures/fig19_probability_calibration_curve.png",
            "category": "Week 7 Model Testing & Refinement",
            "title": "Figure 19: Probability Calibration Reliability Curve",
            "subtitle": "Raw Random Forest Probabilities vs Isotonically Calibrated Risk Scores",
            "takeaway": "Isotonic calibration improves Brier reliability score from 0.1743 down to 0.1038 (40.4% error reduction), ensuring predicted probabilities match true empirical admission risks."
        },
        {
            "id": "fig20",
            "filename": "reports/figures/fig20_subgroup_disparity_analysis.png",
            "category": "Week 7 Model Testing & Refinement",
            "title": "Figure 20: Demographic Fairness & Subgroup Slice Testing",
            "subtitle": "Admission Recall Parity Across Age Slices and Chronic Morbidities",
            "takeaway": "Solves the Week 6 disparity defect where Pediatric, Young Adult, and Middle-Aged recall was 0.0%. Week 7 achieves consistent >= 85.2% recall across all age groups and chronic conditions."
        },
        {
            "id": "fig21",
            "filename": "reports/figures/fig21_cross_track_mle_integration_telemetry.png",
            "category": "Week 7 Model Testing & Refinement",
            "title": "Figure 21: Cross-Track MLE Integration Telemetry (Saad Janane & Zaid Harboul)",
            "subtitle": "Row-by-Row vs Vectorized Batch Inference Latency and Throughput Benchmark",
            "takeaway": "Joint engineering with Zaid Harboul achieved 9,495.9 req/s throughput and 0.11 ms/patient latency (1,558x speedup), effortlessly exceeding the 5.0 ms production SLO."
        },
        # FORECASTING & CAPACITY FIGURES
        {
            "id": "fig06",
            "filename": "reports/figures/fig06_patient_volume_forecasting_trajectory.png",
            "category": "Forecasting & Capacity Projections",
            "title": "Figure 06: 24-Month Patient Visit Trajectory & 2027 Projections",
            "subtitle": "2026 Baseline Actuals vs 2027 Holt-Winters Triple Exponential Smoothing",
            "takeaway": "Projects 51,166 outpatient visits in 2027 with 95% confidence intervals, providing staffing targets for 20 facilities."
        },
        {
            "id": "fig07",
            "filename": "reports/figures/fig07_hospital_revenue_forecasting_trajectory.png",
            "category": "Forecasting & Capacity Projections",
            "title": "Figure 07: Hospital Gross Revenue Trajectory & 2027 Forecast",
            "subtitle": "Monthly Revenue Forecasting with Seasonal Decomposition",
            "takeaway": "Projects hospital gross revenue reaching 1.313 Billion RWF in 2027 (+15.91% year-over-year expansion)."
        },
        {
            "id": "fig09",
            "filename": "reports/figures/fig09_department_workload_demand_forecast.png",
            "category": "Forecasting & Capacity Projections",
            "title": "Figure 09: Departmental Workload & Clinical Encounter Forecast",
            "subtitle": "2027 Demand Projections Across Internal Med, Emergency, Pediatrics, and Surgery",
            "takeaway": "Internal Medicine and Emergency face the greatest clinical demand, requiring dynamic clinical float pool staffing."
        },
        {
            "id": "fig14",
            "filename": "reports/figures/fig14_admission_age_density_by_chronic.png",
            "category": "Forecasting & Capacity Projections",
            "title": "Figure 14: Patient Age Density Distribution by Admission Status",
            "subtitle": "Age Cohort Slicing vs Chronic Condition Comorbidity",
            "takeaway": "Admission probability increases significantly after age 50 among patients with multi-morbidities."
        },
        {
            "id": "fig15",
            "filename": "reports/figures/fig15_workload_vs_satisfaction_regression.png",
            "category": "Forecasting & Capacity Projections",
            "title": "Figure 15: Clinical Workload vs Patient Satisfaction Regression",
            "subtitle": "Analysis of Departmental Overcrowding and Patient Experience",
            "takeaway": "Demonstrates inverse relationship between peak physician workload and patient experience rating."
        },
        # CLINICAL & EPIDEMIOLOGY FIGURES
        {
            "id": "fig01",
            "filename": "reports/figures/fig01_patient_age_gender_distribution.png",
            "category": "Clinical & Epidemiology",
            "title": "Figure 01: Population Age and Gender Demographic Pyramid",
            "subtitle": "Demographic Profile of 20,000 Registered Citizens",
            "takeaway": "Even demographic distribution across gender, with median citizen age of 45 years."
        },
        {
            "id": "fig02",
            "filename": "reports/figures/fig02_chronic_disease_and_diagnoses_severity.png",
            "category": "Clinical & Epidemiology",
            "title": "Figure 02: Chronic Disease Prevalence and Diagnostic Severity",
            "subtitle": "Morbidity Burden Breakdown (Hypertension, Diabetes, Asthma)",
            "takeaway": "74.65% chronic disease prevalence underscores urgency of specialized primary care NCD clinics."
        },
        {
            "id": "fig03",
            "filename": "reports/figures/fig03_correlation_matrix_heatmap.png",
            "category": "Clinical & Epidemiology",
            "title": "Figure 03: Multivariate Clinical Feature Correlation Heatmap",
            "subtitle": "Inter-feature Dependencies Across Physiological and Operational Variables",
            "takeaway": "Identifies strong linkages between disease severity, emergency triage, and inpatient admission length of stay."
        },
        {
            "id": "fig04",
            "filename": "reports/figures/fig04_billing_distribution_by_insurance.png",
            "category": "Clinical & Epidemiology",
            "title": "Figure 04: Encounter Billing Distribution by Insurance Category",
            "subtitle": "Tariff Pricing Uniformity Across Mutuelle de Santé, RAMA, and Out-of-Pocket",
            "takeaway": "Validates standardized pricing equity with Welch's t-test (p = 0.3491), showing no predatory billing."
        },
        {
            "id": "fig05",
            "filename": "reports/figures/fig05_satisfaction_violin_by_facility_type.png",
            "category": "Clinical & Epidemiology",
            "title": "Figure 05: Patient Satisfaction Violin Plot by Facility Tier",
            "subtitle": "Perceived Quality Across Referral, Provincial, District, and General Centers",
            "takeaway": "Uniform satisfaction score distributions across all facility tiers (ANOVA p = 0.7851)."
        },
        # OPERATIONS & CLUSTERS
        {
            "id": "fig08",
            "filename": "reports/figures/fig08_admission_feature_importances.png",
            "category": "Operations & Clusters",
            "title": "Figure 08: Inpatient Admission Feature Importance Hierarchy",
            "subtitle": "Gini Impurity & Permutation Importance for Admission Prediction",
            "takeaway": "Clinical Acuity Score, Disease Severity, and Emergency status emerge as top drivers of inpatient bed admission."
        },
        {
            "id": "fig10",
            "filename": "reports/figures/fig10_kmeans_patient_clusters_pca.png",
            "category": "Operations & Clusters",
            "title": "Figure 10: K-Means Patient Phenotype Clustering via PCA",
            "subtitle": "Unsupervised Segmentation into 4 Distinct Healthcare Utilization Cohorts",
            "takeaway": "Segments patients into Routine Low-Risk (35%), Chronic Managed (38%), Acute High-Utilization (18%), and Complex Surgical (9%)."
        },
        {
            "id": "fig11",
            "filename": "reports/figures/fig11_pca_scree_plot.png",
            "category": "Operations & Clusters",
            "title": "Figure 11: PCA Scree Plot and Cumulative Explained Variance",
            "subtitle": "Dimensionality Reduction Validation for Patient Phenotyping",
            "takeaway": "Top principal components explain over 80% of aggregate demographic and clinical variance."
        },
        {
            "id": "fig12",
            "filename": "reports/figures/fig12_facility_performance_leaderboard.png",
            "category": "Operations & Clusters",
            "title": "Figure 12: Healthcare Facility Composite Performance Leaderboard",
            "subtitle": "Benchmarking 20 Hospitals on Throughput, Bed Utilization, and Quality",
            "takeaway": "Identifies top performers (Kigali Referral, Butare Provincial) and facilities needing equipment upgrades."
        },
        {
            "id": "fig13",
            "filename": "reports/figures/fig13_province_patient_distribution.png",
            "category": "Operations & Clusters",
            "title": "Figure 13: Provincial Healthcare Equity and Patient Geographic Distribution",
            "subtitle": "Population Health Reach Across Kigali City, Eastern, Northern, Southern, and Western",
            "takeaway": "Equitable geographic access across Rwanda's 5 provinces, supporting universal healthcare coverage objectives."
        }
    ]

    # Category Selector Pills
    cat_options = [
        "All Figures (21)",
        "Week 7 Model Testing & Refinement (6)",
        "Forecasting & Capacity Projections (5)",
        "Clinical & Epidemiology (5)",
        "Operations & Clusters (5)"
    ]
    selected_cat = st.radio("Filter Visualizations by Milestone", cat_options, horizontal=True)

    # Filter figures list
    filtered_figs = figures_meta
    if "Week 7" in selected_cat:
        filtered_figs = [f for f in figures_meta if f["category"] == "Week 7 Model Testing & Refinement"]
    elif "Forecasting" in selected_cat:
        filtered_figs = [f for f in figures_meta if f["category"] == "Forecasting & Capacity Projections"]
    elif "Clinical" in selected_cat:
        filtered_figs = [f for f in figures_meta if f["category"] == "Clinical & Epidemiology"]
    elif "Operations" in selected_cat:
        filtered_figs = [f for f in figures_meta if f["category"] == "Operations & Clusters"]

    st.markdown(f"Displaying **{len(filtered_figs)} publication figures** (300 DPI Publication Grade).")

    # Render Figures in Expandable Cards
    for fig in filtered_figs:
        with st.container():
            st.markdown(f"""
            <div class="figure-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <h3 style="margin: 0; font-size: 18px; color: #f8fafc;">{fig['title']}</h3>
                    <span class="badge-pill badge-sky">{fig['category']}</span>
                </div>
                <p style="color: #94a3b8; font-size: 13px; margin: 0 0 12px;">{fig['subtitle']}</p>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists(fig["filename"]):
                st.image(fig["filename"], use_container_width=True)
                
                exp_col1, exp_col2 = st.columns([2.5, 1])
                with exp_col1:
                    st.info(f"💡 **Analytical Interpretation & Clinical Takeaway:** {fig['takeaway']}")
                with exp_col2:
                    with open(fig["filename"], "rb") as file:
                        btn = st.download_button(
                            label=f"⬇️ Download {fig['id'].upper()} (PNG)",
                            data=file,
                            file_name=os.path.basename(fig["filename"]),
                            mime="image/png",
                            use_container_width=True
                        )
            else:
                st.warning(f"Figure file not found: `{fig['filename']}`")

            st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# TRACK 6: STRATEGIC ROADMAP & GOVERNANCE
# ==============================================================================
elif nav_choice == "📑 Strategic Roadmap & Governance":
    st.title("📑 Strategic Transformation Roadmap & Governance")
    st.markdown("C-Suite implementation priorities, MLOps production governance, and Week 8 handoff plan.")

    st.markdown("""
    <div class="team-banner">
        <div>
            <strong>AnalystLab Africa Experience Lab</strong> — Official Track 2 & Track 4 Capstone Milestone
        </div>
        <div>
            <span class="badge-pill badge-teal">Status: Production Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 5 C-Suite Transformation Action Items")
    st.markdown("""
    1. **Dedicated Outpatient NCD Clinics**:
       - *Finding*: 74.65% chronic disease burden (Hypertension, Diabetes, Asthma).
       - *Action*: Establish specialized nurse-led chronic disease management clinics in all 20 centers to screen early and cut emergency inpatient admissions by 25%.
       - *Financial Impact*: Estimated 85 Million RWF annual savings in acute care avoidance.

    2. **Automated Claims Pre-Scrubber**:
       - *Finding*: 50% claims delay bottleneck resulting in 2.44 Billion RWF pending claims liquidity freeze.
       - *Action*: Implement automated diagnostic pre-scrubber rules before submission to Mutuelle and RAMA.
       - *Financial Impact*: Accelerate reimbursement approvals to >85% within 14 days, unlocking 1.8 Billion RWF in liquid capital.

    3. **Dynamic Clinical Float Pools**:
       - *Finding*: Emergency and Internal Medicine handle 44% of total inpatient demand.
       - *Action*: Deploy dynamic nurse and doctor shift allocation across high-stress departments based on Holt-Winters 2027 workload forecasts.
       - *Impact*: Cut outpatient wait times by 35% and alleviate clinical burnout.

    4. **Diagnostic Equipment Equalization**:
       - *Finding*: Significant disparity in lab abnormal positivity tracking between referral and district hospitals.
       - *Action*: Upgrade hematology analyzers and ultrasound at 8 lower-performing district hospitals.
       - *Impact*: Reduce unnecessary patient transfers to Kigali referral centers by 40%.

    5. **Digital Self-Service Check-In Kiosks**:
       - *Finding*: Morning registration bottlenecks create peak queue congestion.
       - *Action*: Expand mobile money and national ID check-in kiosks at hospital entrance lobbies.
       - *Impact*: Decrease check-in queue times by 50%.
    """)

    st.markdown("---")

    st.subheader("🛡️ Cross-Track Production Governance & Sign-Off")
    st.markdown(f"""
    | Engineering Domain | Owner | Responsibility | Validation Status |
    | :--- | :--- | :--- | :--- |
    | **Data Science (Track 2)** | **Saad Janane** | Model testing, error decomposition, calibration, threshold tuning, and fairness | <span class="badge-pill badge-teal">APPROVED (v2.0)</span> |
    | **ML Engineering (Track 4)** | **Zaid Harboul** | Vectorized inference engine, batch optimization, adversarial edge-case resilience | <span class="badge-pill badge-teal">APPROVED (v2.0)</span> |
    | **Production Latency SLO** | Joint Team | Target: < 5.0 ms / patient | **0.11 ms p50 (PASSED)** |
    | **High Throughput SLO** | Joint Team | Target: > 500 req / s | **9,495.9 req/s (PASSED)** |
    | **Adversarial Resilience** | Joint Team | 10 Edge Cases (missing, unseen, extreme) | **10 / 10 (100% PASSED)** |
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🚀 Week 8 Transition Plan")
    st.markdown("""
    - **Step 1: Containerization**: Package `refined_admission_model.pkl` and `HealthConnectInferenceEngine` into a lightweight Docker microservice with FastAPI endpoints (`/predict`, `/health`, `/metrics`).
    - **Step 2: Real-Time Monitoring**: Deploy Prometheus metrics exporter to track prediction latency, input drift (Kolmogorov-Smirnov test), and calibration stability.
    - **Step 3: Hospital EHR Integration**: Pilot integration with open-source OpenMRS / Rwanda Health Information System (RHIS) for automated admission risk flagging at triage.
    """)

    st.caption("HealthConnect Rwanda Healthcare Analytics & AI Transformation | Certified by Lead Data Scientist Saad Janane and Lead ML Engineer Zaid Harboul.")
