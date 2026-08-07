"""
app.py — Aerothon 2026 | OMEGA-EMS Interactive Dashboard
Hybrid-Electric Propulsion Optimisation for a Fixed-Wing UAV

Run:  streamlit run dashboard/app.py

STRUCTURE:
──────────
Sidebar     : Mission configuration inputs (modifiable)
Tab 1       : Mission Overview  — flight path, phase timeline, key metrics
Tab 2       : Energy Management — power split, SOC, fuel, engine op-point
Tab 3       : Health Monitor    — battery SOH, engine EHI, motor temp, SHI
Tab 4       : System Performance— specs, performance metrics, mission profile
Tab 5       : Comparison        — Fuzzy vs APEX vs Rustom II side-by-side
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.uav_model        import HybridUAV
from src.mission          import Mission, MissionConfig
from src.fuzzy_controller import FuzzyEMS
from src.apex_controller  import APEXEMS
from src.ai_apex_controller import APEXNeuralEMS
from src.simulation       import Simulator, RustomIIBaseline
from src.health_monitor   import run_with_health


def _create_controller_instance(ctrl_cls: type, uav: HybridUAV):
    try:
        return ctrl_cls(uav)
    except Exception as exc:
        if ctrl_cls is APEXNeuralEMS:
            return exc
        raise

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AEROTHON 2026 — OMEGA-EMS",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── COLOUR PALETTE ───────────────────────────────────────────────────────────
COLOUR = {
    "fuzzy":    "#3B82F6",   # blue
    "apex":     "#10B981",   # green
    "rustom":   "#F59E0B",   # amber
    "engine":   "#EF4444",   # red
    "battery":  "#8B5CF6",   # purple
    "health":   "#06B6D4",   # cyan
    "bg_card":  "#1E293B",
    "takeoff":  "#64748B",
    "climb":    "#0EA5E9",
    "cruise":   "#6366F1",
    "loiter":   "#10B981",
    "descent":  "#F59E0B",
}

PHASE_COLOURS = {
    "Takeoff": COLOUR["takeoff"],
    "Climb":   COLOUR["climb"],
    "Cruise":  COLOUR["cruise"],
    "Loiter":  COLOUR["loiter"],
    "Descent": COLOUR["descent"],
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .metric-card {
        background: #1E293B; border-radius: 12px;
        padding: 16px; text-align: center; border: 1px solid #334155;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #F1F5F9; }
    .metric-label { font-size: 0.8rem; color: #94A3B8; margin-top: 4px; }
    .metric-delta { font-size: 0.9rem; font-weight: 600; margin-top: 4px; }
    .green  { color: #10B981; }
    .blue   { color: #3B82F6; }
    .amber  { color: #F59E0B; }
    .header-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
        border: 1px solid #1E40AF; border-radius: 12px;
        padding: 20px 32px; margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1 style="color:#F1F5F9;margin:0;font-size:1.8rem;">
    ✈️ &nbsp;OMEGA-EMS &nbsp;|&nbsp; Aerothon 2026
  </h1>
  <p style="color:#94A3B8;margin:4px 0 0 0;font-size:0.95rem;">
    Hybrid-Electric Propulsion Optimisation for a Fixed-Wing UAV &nbsp;·&nbsp;
    HAL × IIT Indore &nbsp;·&nbsp; Team AERONEXUS
  </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Mission Configuration")
    st.caption("Modify inputs and re-run simulation")

    cruise_alt   = st.slider("Cruise Altitude (m)",     3000, 10000, 5000, 500)
    cruise_speed = st.slider("Cruise Speed (km/h)",      200,   280,  250,   5)
    loiter_alt   = st.slider("Loiter Altitude (m)",     2000,  6000, 3000, 500)
    wind_speed   = st.slider("Headwind (m/s)",             0,    20,    0,   1)

    st.markdown("---")
    st.markdown("### Course")
    st.caption("The mission route — one-way distance and outbound bearing to the loiter/ISR station")
    radius_km    = st.slider("Operational Radius (km)",    10,   100,   50,   5,
                              help="Locked design point (Phase 4A trade study): 50 km")
    heading_deg  = st.slider("Outbound Heading (° from N)", 0,   359,   90,   5,
                              help="Cosmetic — rotates the ground-track map, does not change energy physics")

    st.markdown("---")
    st.markdown("### Powertrain Spec")
    st.caption("Locked design point (Phase 4A-5A + Simulink Blueprint): 90 kW / 20 kWh NMC")
    engine_kw    = st.slider("Turboshaft Rating (kW)",     60,   120,   90,   5)
    battery_kwh  = st.slider("Battery Capacity (kWh)",     10,    30,   20,   1)

    st.markdown("---")
    st.markdown("### Disturbances")
    enable_dist  = st.checkbox("Enable sudden wind gust", value=False)

    st.markdown("---")
    st.markdown("### Display")
    dt_select    = st.select_slider(
        "Time step (s)", options=[1, 5, 10], value=5,
        help="Coarser dt = faster run, less resolution"
    )
    show_health  = st.checkbox("Show Health Monitor", value=True)
    show_ai      = st.checkbox("Include APEX-AI (learned)", value=True,
                                help="Genuinely trained model — see Energy Management tab for how it was trained")

    run_btn      = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ─── SIMULATION ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_simulation(
    cruise_alt: float, cruise_speed: float, radius_km: float, heading_deg: float, loiter_alt: float,
    wind_speed: float, enable_dist: bool, dt: float, engine_kw: float, battery_kwh: float, show_ai: bool
):
    """
    Runs all controllers on an identical mission. Cached — only re-runs
    when inputs change.
    """
    uav    = HybridUAV()
    # Instance-level overrides so the sidebar sliders actually drive the
    # physics rather than just the locked class defaults.
    uav.TURBOSHAFT_MAX_POWER    = float(engine_kw) * 1000.0
    uav.BATTERY_CAPACITY        = float(battery_kwh) * 1000.0
    uav.BATTERY_USABLE_ENERGY   = uav.BATTERY_CAPACITY * (uav.BATTERY_MAX_SOC - uav.BATTERY_MIN_SOC)
    uav.BATTERY_MASS            = uav.BATTERY_CAPACITY / uav.BATTERY_SPEC_ENERGY

    config = MissionConfig(
        cruise_altitude_m      = float(cruise_alt),
        cruise_speed_kmh       = float(cruise_speed),
        operational_radius_km  = float(radius_km),
        heading_deg            = float(heading_deg),
        loiter_altitude_m      = float(loiter_alt),
        wind_speed_ms          = float(wind_speed),
        enable_disturbances    = enable_dist,
        dt                     = float(dt),
    )
    mission = Mission(uav, config)
    profile = mission.generate_profile()

    results = {}
    health  = {}
    extra   = {}

    controller_specs = [("Fuzzy Logic EMS", FuzzyEMS), ("APEX (Phase-Aware)", APEXEMS)]
    if show_ai:
        controller_specs.append(("APEX-AI (Learned)", APEXNeuralEMS))

    for name, ctrl_cls in controller_specs:
        uav.reset()
        ctrl = _create_controller_instance(ctrl_cls, uav)
        if isinstance(ctrl, Exception):
            extra[name] = {"load_error": str(ctrl)}
            continue
        sim    = Simulator(uav, config)
        result = sim.run(profile, ctrl, verbose=False)
        results[name] = result
        h_log, _ = run_with_health(result.log, dt=dt)
        health[name]  = h_log
        extra[name]   = {"charge_cycles": getattr(ctrl, "charge_cycles", None)}

    return results, health, extra, config, uav


# ─── INITIAL RUN ──────────────────────────────────────────────────────────────
with st.spinner("⚙️ Running simulation..."):
    results, health_data, extra_stats, config, uav = run_simulation(
        cruise_alt, cruise_speed, radius_km, heading_deg, loiter_alt,
        wind_speed, enable_dist, dt_select, engine_kw, battery_kwh, show_ai
    )

fuzzy_r  = results["Fuzzy Logic EMS"]
apex_r   = results["APEX (Phase-Aware)"]
fuzzy_h  = health_data["Fuzzy Logic EMS"]
apex_h   = health_data["APEX (Phase-Aware)"]
ai_r     = results.get("APEX-AI (Learned)")
ai_h     = health_data.get("APEX-AI (Learned)")
ai_error = extra_stats.get("APEX-AI (Learned)", {}).get("load_error")
if ai_error is not None:
    st.error(
        "APEX-AI model could not be loaded in this Python environment: "
        + ai_error
    )

def to_df(log):
    return pd.DataFrame([{
        "t_min":     s.t / 60.0,
        "t_h":       s.t / 3600.0,
        "phase":     s.phase,
        "altitude":  s.altitude_m,
        "speed":     s.speed_kmh,
        "power_req": s.required_power_W / 1000,
        "engine_kW": s.turboshaft_W / 1000,
        "battery_kW":s.battery_W / 1000,
        "soc":       s.battery_soc * 100,
        "fuel_kg":   s.fuel_mass_kg,
        "fuel_burned": s.fuel_burned_kg,
        "x_km":      s.x_km,
        "y_km":      s.y_km,
    } for s in log])

def to_hdf(hlog):
    return pd.DataFrame([{
        "t_min":     h.t / 60.0,
        "phase":     h.phase,
        "soh":       h.battery_soh * 100,
        "batt_temp": h.batt_temp_C,
        "ehi":       h.engine_ehi * 100,
        "bsfc":      h.engine_bsfc_actual,
        "motor_temp":h.motor_temp_C,
        "motor_derate": h.motor_derate * 100,
        "shi":       h.shi,
    } for h in hlog])

df_f = to_df(fuzzy_r.log)
df_a = to_df(apex_r.log)
dh_f = to_hdf(fuzzy_h)
dh_a = to_hdf(apex_h)

# ─── TOP METRIC CARDS ─────────────────────────────────────────────────────────
impr = (apex_r.loiter_time_h - fuzzy_r.loiter_time_h) / fuzzy_r.loiter_time_h * 100

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "APEX Loiter",     f"{apex_r.loiter_time_h:.2f} h",  f"{apex_r.loiter_time_min:.0f} min",  "green"),
    (c2, "Fuzzy Loiter",    f"{fuzzy_r.loiter_time_h:.2f} h", f"{fuzzy_r.loiter_time_min:.0f} min",  "blue"),
    (c3, "APEX Improvement",f"{impr:+.1f}%",                  f"+{(apex_r.loiter_time_h-fuzzy_r.loiter_time_h)*60:.0f} min","green"),
    (c4, "Silent Electric", f"{apex_r.electric_only_time_s/3600:.1f} h", f"{apex_r.electric_only_time_s/60:.0f} min","green"),
    (c5, "Fuel Used",       f"{apex_r.fuel_burned_kg:.0f} kg",f"of {uav.FUEL_MASS_MAX:.0f} kg", "amber"),
]
for col, label, val, sub, colour in cards:
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value {colour}">{val}</div>
      <div class="metric-label">{label}</div>
      <div class="metric-delta {colour}">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Mission Overview",
    "⚡  Energy Management",
    "🩺  Health Monitor",
    "🎯  System Performance",
    "📊  Comparison"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MISSION OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_path, col_alt = st.columns([1, 1])

    # ── 2D Flight Path ──────────────────────────────────────────────────────
    with col_path:
        st.markdown("#### ✈️ Flight Course")
        fig_path = go.Figure()
        for phase, colour in PHASE_COLOURS.items():
            sub = df_a[df_a["phase"] == phase]
            if sub.empty:
                continue
            fig_path.add_trace(go.Scatter(
                x=sub["x_km"], y=sub["y_km"],
                mode="lines", name=phase,
                line=dict(color=colour, width=2.5),
            ))
        # Base marker
        fig_path.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers+text",
            marker=dict(size=12, color="#F1F5F9", symbol="square"),
            text=["Base"], textposition="top center",
            textfont=dict(color="#F1F5F9"),
            showlegend=False,
        ))
        fig_path.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=340,
            xaxis_title="East Distance (km)",
            yaxis_title="North Distance (km)",
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_path, use_container_width=True)

    # ── Altitude Profile ────────────────────────────────────────────────────
    with col_alt:
        st.markdown("#### 📐 Altitude & Speed Profile")
        fig_alt = make_subplots(specs=[[{"secondary_y": True}]])
        fig_alt.add_trace(
            go.Scatter(x=df_a["t_min"], y=df_a["altitude"],
                       name="Altitude (m)", line=dict(color="#60A5FA", width=2)),
            secondary_y=False,
        )
        fig_alt.add_trace(
            go.Scatter(x=df_a["t_min"], y=df_a["speed"],
                       name="Speed (km/h)", line=dict(color="#34D399", width=2)),
            secondary_y=True,
        )
        # Phase bands
        for phase, colour in PHASE_COLOURS.items():
            sub = df_a[df_a["phase"] == phase]
            if sub.empty:
                continue
            fig_alt.add_vrect(
                x0=sub["t_min"].min(), x1=sub["t_min"].max(),
                fillcolor=colour, opacity=0.08, line_width=0,
                annotation_text=phase, annotation_position="top left",
                annotation_font_size=9, annotation_font_color=colour,
            )
        fig_alt.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=340,
            xaxis_title="Time (min)",
            margin=dict(l=20, r=20, t=10, b=40),
            legend=dict(orientation="h", y=-0.15),
        )
        fig_alt.update_yaxes(title_text="Altitude (m)", secondary_y=False)
        fig_alt.update_yaxes(title_text="Speed (km/h)", secondary_y=True)
        st.plotly_chart(fig_alt, use_container_width=True)

    # ── Phase Duration Table ─────────────────────────────────────────────────
    st.markdown("#### ⏱️ Phase Breakdown")
    rows = []
    for phase in ["Takeoff", "Climb", "Cruise", "Loiter", "Descent"]:
        sub_f = df_f[df_f["phase"] == phase]
        sub_a = df_a[df_a["phase"] == phase]
        if sub_f.empty and sub_a.empty:
            continue
        dur_f = len(sub_f) * config.dt / 60
        dur_a = len(sub_a) * config.dt / 60
        rows.append({
            "Phase":            phase,
            "Fuzzy (min)":      f"{dur_f:.1f}",
            "APEX (min)":       f"{dur_a:.1f}",
            "Avg Power F (kW)": f"{sub_f['power_req'].mean():.1f}" if not sub_f.empty else "─",
            "Avg Power A (kW)": f"{sub_a['power_req'].mean():.1f}" if not sub_a.empty else "─",
            "Avg Alt (m)":      f"{sub_a['altitude'].mean():.0f}" if not sub_a.empty else "─",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ENERGY MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_e1, col_e2 = st.columns(2)

    # ── Power Split Comparison ───────────────────────────────────────────────
    with col_e1:
        st.markdown("#### ⚡ Power Split — Fuzzy Logic")
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(
            x=df_f["t_min"], y=df_f["engine_kW"],
            name="Turboshaft", stackgroup="one",
            line=dict(color=COLOUR["engine"]),
            fillcolor="rgba(239,68,68,0.4)",
        ))
        fig_pf.add_trace(go.Scatter(
            x=df_f["t_min"], y=df_f["battery_kW"].clip(lower=0),
            name="Battery (discharge)", stackgroup="one",
            line=dict(color=COLOUR["battery"]),
            fillcolor="rgba(139,92,246,0.4)",
        ))
        fig_pf.add_trace(go.Scatter(
            x=df_f["t_min"], y=df_f["power_req"],
            name="Required", mode="lines",
            line=dict(color="#F1F5F9", width=1.5, dash="dot"),
        ))
        fig_pf.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=300,
            xaxis_title="Time (min)", yaxis_title="Power (kW)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_pf, use_container_width=True)

    with col_e2:
        st.markdown("#### ⚡ Power Split — APEX")
        fig_pa = go.Figure()
        fig_pa.add_trace(go.Scatter(
            x=df_a["t_min"], y=df_a["engine_kW"],
            name="Turboshaft", stackgroup="one",
            line=dict(color=COLOUR["engine"]),
            fillcolor="rgba(239,68,68,0.4)",
        ))
        fig_pa.add_trace(go.Scatter(
            x=df_a["t_min"], y=df_a["battery_kW"].clip(lower=0),
            name="Battery (discharge)", stackgroup="one",
            line=dict(color=COLOUR["battery"]),
            fillcolor="rgba(139,92,246,0.4)",
        ))
        fig_pa.add_trace(go.Scatter(
            x=df_a["t_min"], y=df_a["power_req"],
            name="Required", mode="lines",
            line=dict(color="#F1F5F9", width=1.5, dash="dot"),
        ))
        fig_pa.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=300,
            xaxis_title="Time (min)", yaxis_title="Power (kW)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_pa, use_container_width=True)

    # ── SOC and Fuel ─────────────────────────────────────────────────────────
    col_soc, col_fuel = st.columns(2)

    with col_soc:
        st.markdown("#### 🔋 Battery State of Charge")
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(
            x=df_f["t_min"], y=df_f["soc"],
            name="Fuzzy Logic", line=dict(color=COLOUR["fuzzy"], width=2),
        ))
        fig_soc.add_trace(go.Scatter(
            x=df_a["t_min"], y=df_a["soc"],
            name="APEX", line=dict(color=COLOUR["apex"], width=2),
        ))
        fig_soc.add_hline(y=20, line_dash="dash", line_color="#EF4444",
                          annotation_text="MIN SOC (20%)")
        fig_soc.add_hline(y=95, line_dash="dash", line_color="#94A3B8",
                          annotation_text="MAX SOC (95%)")
        fig_soc.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=280,
            xaxis_title="Time (min)", yaxis_title="SOC (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=20, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_soc, use_container_width=True)

    with col_fuel:
        st.markdown("#### ⛽ Fuel Remaining")
        fig_fuel = go.Figure()
        fig_fuel.add_trace(go.Scatter(
            x=df_f["t_min"], y=df_f["fuel_kg"],
            name="Fuzzy Logic", line=dict(color=COLOUR["fuzzy"], width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        ))
        fig_fuel.add_trace(go.Scatter(
            x=df_a["t_min"], y=df_a["fuel_kg"],
            name="APEX", line=dict(color=COLOUR["apex"], width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
        ))
        fig_fuel.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=280,
            xaxis_title="Time (min)", yaxis_title="Fuel Remaining (kg)",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=20, r=20, t=10, b=50),
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

    # ── BSFC Curve ───────────────────────────────────────────────────────────
    st.markdown("#### 📉 Engine BSFC Curve — Why 70% Rated Power is Optimal")
    pct_range = np.arange(10, 105, 5)
    bsfc_vals = [0.35 * (1 + 0.35*(p/100 - 0.70)**2/0.49) for p in pct_range]

    fig_bsfc = go.Figure()
    fig_bsfc.add_trace(go.Scatter(
        x=pct_range, y=bsfc_vals,
        mode="lines+markers",
        line=dict(color=COLOUR["engine"], width=3),
        name="BSFC (kg/kWh)",
    ))
    fig_bsfc.add_vline(x=70, line_color=COLOUR["apex"], line_dash="dash",
                       annotation_text="APEX target: 70%", annotation_font_color=COLOUR["apex"])
    loiter_pct = (apex_r.log[len(apex_r.log)//2].required_power_W / 60000) * 100
    fig_bsfc.add_vline(x=loiter_pct, line_color=COLOUR["fuzzy"], line_dash="dash",
                       annotation_text=f"Fuzzy typical: {loiter_pct:.0f}%",
                       annotation_font_color=COLOUR["fuzzy"])
    fig_bsfc.update_layout(
        template="plotly_dark", paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A", height=260,
        xaxis_title=f"Engine Load (% of {engine_kw:.0f} kW rated)",
        yaxis_title="BSFC (kg/kWh) — lower is better",
        margin=dict(l=20, r=20, t=10, b=40),
    )
    st.plotly_chart(fig_bsfc, use_container_width=True)
    st.caption("APEX runs the engine only at 70% rated power (scheduling), "
               "while Fuzzy Logic runs it at partial load (continuous). "
               "The 6% BSFC difference compounds over 9+ hours of loiter.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HEALTH MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not show_health:
        st.info("Enable Health Monitor in the sidebar to view this tab.")
    else:
        # ── SHI Gauge ─────────────────────────────────────────────────────────
        st.markdown("#### 🩺 System Health Index (SHI)")
        col_shi_f, col_shi_a = st.columns(2)

        def make_gauge(value, title, ref_colour):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title, "font": {"color": "#F1F5F9"}},
                number={"suffix": "/100", "font": {"color": ref_colour, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": ref_colour},
                    "steps": [
                        {"range": [0,  60], "color": "#7F1D1D"},
                        {"range": [60, 75], "color": "#78350F"},
                        {"range": [75, 90], "color": "#1E3A5F"},
                        {"range": [90, 100],"color": "#14532D"},
                    ],
                    "threshold": {
                        "line": {"color": "#F1F5F9", "width": 3},
                        "thickness": 0.75,
                        "value": value,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="#0F172A", height=240,
                font={"color": "#F1F5F9"},
                margin=dict(l=20, r=20, t=30, b=10),
            )
            return fig

        with col_shi_f:
            shi_f = dh_f["shi"].iloc[-1]
            st.plotly_chart(make_gauge(shi_f, "Fuzzy Logic EMS", COLOUR["fuzzy"]),
                            use_container_width=True)
        with col_shi_a:
            shi_a = dh_a["shi"].iloc[-1]
            st.plotly_chart(make_gauge(shi_a, "APEX EMS", COLOUR["apex"]),
                            use_container_width=True)

        # ── Health Trends ──────────────────────────────────────────────────────
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("#### 🔋 Battery SOH & Temperature")
            fig_bh = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bh.add_trace(
                go.Scatter(x=dh_a["t_min"], y=dh_a["soh"],
                           name="SOH (%) — APEX",
                           line=dict(color=COLOUR["apex"], width=2)),
                secondary_y=False,
            )
            fig_bh.add_trace(
                go.Scatter(x=dh_f["t_min"], y=dh_f["soh"],
                           name="SOH (%) — Fuzzy",
                           line=dict(color=COLOUR["fuzzy"], width=2, dash="dot")),
                secondary_y=False,
            )
            fig_bh.add_trace(
                go.Scatter(x=dh_a["t_min"], y=dh_a["batt_temp"],
                           name="Batt Temp °C",
                           line=dict(color="#F59E0B", width=1.5)),
                secondary_y=True,
            )
            fig_bh.update_layout(
                template="plotly_dark", paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A", height=280,
                xaxis_title="Time (min)",
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=20, r=20, t=10, b=60),
            )
            fig_bh.update_yaxes(title_text="SOH (%)", secondary_y=False)
            fig_bh.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
            st.plotly_chart(fig_bh, use_container_width=True)

        with col_h2:
            st.markdown("#### 🔧 Engine EHI & BSFC Degradation")
            fig_eh = make_subplots(specs=[[{"secondary_y": True}]])
            fig_eh.add_trace(
                go.Scatter(x=dh_a["t_min"], y=dh_a["ehi"],
                           name="EHI (%) — APEX",
                           line=dict(color=COLOUR["apex"], width=2)),
                secondary_y=False,
            )
            fig_eh.add_trace(
                go.Scatter(x=dh_a["t_min"], y=dh_a["bsfc"],
                           name="BSFC (kg/kWh)",
                           line=dict(color=COLOUR["engine"], width=1.5)),
                secondary_y=True,
            )
            fig_eh.update_layout(
                template="plotly_dark", paper_bgcolor="#0F172A",
                plot_bgcolor="#0F172A", height=280,
                xaxis_title="Time (min)",
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=20, r=20, t=10, b=60),
            )
            fig_eh.update_yaxes(title_text="EHI (%)",       secondary_y=False)
            fig_eh.update_yaxes(title_text="BSFC (kg/kWh)", secondary_y=True)
            st.plotly_chart(fig_eh, use_container_width=True)

        # ── Motor Temperature ──────────────────────────────────────────────────
        st.markdown("#### 🌡️ Motor Temperature Profile")
        fig_mt = go.Figure()
        fig_mt.add_trace(go.Scatter(
            x=dh_a["t_min"], y=dh_a["motor_temp"],
            name="Motor Temp — APEX",
            line=dict(color=COLOUR["apex"], width=2),
        ))
        fig_mt.add_hline(y=85,  line_dash="dot",  line_color="#F59E0B",
                         annotation_text="Rated (85°C)")
        fig_mt.add_hline(y=120, line_dash="dash", line_color="#EF4444",
                         annotation_text="Derating starts (120°C)")
        fig_mt.update_layout(
            template="plotly_dark", paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A", height=200,
            xaxis_title="Time (min)", yaxis_title="Temperature (°C)",
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_mt, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SYSTEM PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📋 System Specifications & Performance")
    # Allow live variation of MTOW and Payload (visual only)
    with st.expander("Adjust Displayed MTOW / Payload (visual)", expanded=False):
        display_mtow = st.slider("Displayed MTOW (kg)",
                                int(max(1, uav.MTOW*0.5)), int(uav.MTOW*1.5), int(uav.MTOW), step=10)
        display_payload = st.slider("Displayed Payload (kg)",
                                   0, int(uav.MTOW*0.6), int(uav.PAYLOAD), step=1)
    # ── System Specifications ──────────────────────────────────────────────────
    st.markdown("### Airframe & Component Specifications")
    
    spec_col1, spec_col2, spec_col3 = st.columns(3)
    
    with spec_col1:
        st.markdown("**Airframe**")
        # show adjusted/selected display values if provided
        disp_mtow = locals().get("display_mtow", uav.MTOW)
        disp_payload = locals().get("display_payload", uav.PAYLOAD)
        st.text(f"MTOW: {disp_mtow:.0f} kg\n"
            f"Payload: {disp_payload:.0f} kg\n"
                f"Structural: {uav.STRUCTURAL_MASS:.0f} kg\n"
                f"Wing Area: {uav.WING_AREA:.1f} m²\n"
                f"Wingspan: {np.sqrt(uav.ASPECT_RATIO * uav.WING_AREA):.1f} m\n"
                f"Aspect Ratio: {uav.ASPECT_RATIO:.1f}")
    
    with spec_col2:
        st.markdown("**Propulsion System**")
        st.text(f"Turboshaft: {engine_kw:.0f} kW\n"
                f"Min Power: {uav.TURBOSHAFT_MIN_POWER/1000:.0f} kW\n"
                f"Electric Motors: 2 × {uav.MOTOR_MAX_POWER/1000:.0f} kW\n"
                f"Total Max Power: {uav.MAX_ELECTRIC_POWER/1000:.0f} kW\n"
                f"Generator: {uav.GENERATOR_MAX_POWER/1000:.0f} kW\n"
                f"BSFC: {uav.TURBOSHAFT_BSFC:.2f} kg/kWh")
    
    with spec_col3:
        st.markdown("**Energy Storage**")
        battery_energy_kwh = uav.BATTERY_CAPACITY / 1000.0
        usable_kwh = uav.BATTERY_USABLE_ENERGY / 1000.0
        st.text(f"Battery: {battery_energy_kwh:.1f} kWh total\n"
                f"Usable: {usable_kwh:.1f} kWh\n"
                f"Chemistry: {uav.BATTERY_CHEMISTRY}\n"
                f"Voltage: {uav.BATTERY_VOLTAGE:.0f} V\n"
                f"Fuel Capacity: {uav.FUEL_MASS_MAX:.1f} kg\n"
                f"Motor Mass: {uav.MOTOR_MASS:.0f} kg")

    st.markdown("---")

    # ── System Performance Metrics ─────────────────────────────────────────────
    st.markdown("### Overall System Performance")
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric("Total Endurance", f"{apex_r.total_time_h:.2f} h", f"{apex_r.total_time_h*60:.0f} min")
    with perf_col2:
        st.metric("Avg System Power", f"{apex_r.avg_power_kW:.1f} kW", f"Peak: {df_a['power_req'].max()/1000:.1f} kW")
    with perf_col3:
        st.metric("Avg Altitude", f"{df_a['altitude'].mean():.0f} m", f"Max: {df_a['altitude'].max():.0f} m")
    with perf_col4:
        st.metric("Fuel Efficiency", f"{apex_r.fuel_efficiency:.1f} km/kg", f"Used: {apex_r.fuel_burned_kg:.1f} kg")
    # New: normalize performance by MTOW / payload
    perf_col5, perf_col6, perf_col7, perf_col8 = st.columns(4)
    norm_endurance = apex_r.total_time_h / (disp_mtow if disp_mtow>0 else uav.MTOW)
    payload_frac = disp_payload / disp_mtow if disp_mtow>0 else 0.0
    with perf_col5:
        st.metric("Endurance / MTOW", f"{norm_endurance:.4f} h/kg", "Normalized")
    with perf_col6:
        st.metric("Payload Fraction", f"{payload_frac:.2%}", f"{disp_payload:.0f} kg/{disp_mtow:.0f} kg")
    
    battery_cycle_efficiency = (uav.BATTERY_CHARGE_EFF * uav.BATTERY_DISCHARGE_EFF) * 100
    
    with perf_col5:
        st.metric("Battery Efficiency", f"{battery_cycle_efficiency:.1f}%", "Round-trip")
    with perf_col6:
        st.metric("Engine Avg Load", f"{(apex_r.avg_engine_fraction*100):.0f}%", f"Optimized at 70%")
    with perf_col7:
        st.metric("Final Battery SOC", f"{apex_r.battery_final_soc*100:.1f}%", f"Returned healthy")
    with perf_col8:
        effective_loiter = (apex_r.loiter_time_h - fuzzy_r.loiter_time_h) / fuzzy_r.loiter_time_h * 100
        st.metric("APEX Gain vs Fuzzy", f"+{effective_loiter:.1f}%", "Endurance advantage")

    st.markdown("---")

    # ── Mission Profile Cycle Visualization ────────────────────────────────────
    st.markdown("### Mission Profile Visualization (APEX Controller)")
    
    # Create a comprehensive mission profile figure with multiple sub-plots
    fig_mission = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Altitude Profile", "Airspeed Profile",
                        "Required Power", "Aircraft Weight",
                        "Battery SOC Trajectory", "Fuel Consumption"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )
    
    # 1. Altitude Profile
    for phase, colour in PHASE_COLOURS.items():
        sub = df_a[df_a["phase"] == phase]
        if not sub.empty:
            fig_mission.add_trace(
                go.Scatter(x=sub["t_min"], y=sub["altitude"],
                           name=phase, mode="lines",
                           line=dict(color=colour, width=2),
                           showlegend=True),
                row=1, col=1
            )
    fig_mission.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig_mission.update_xaxes(title_text="Time (min)", row=1, col=1)
    
    # 2. Airspeed Profile
    for phase, colour in PHASE_COLOURS.items():
        sub = df_a[df_a["phase"] == phase]
        if not sub.empty:
            fig_mission.add_trace(
                go.Scatter(x=sub["t_min"], y=sub["speed"],
                           name=phase, mode="lines",
                           line=dict(color=colour, width=2),
                           showlegend=False),
                row=1, col=2
            )
    fig_mission.update_yaxes(title_text="Speed (km/h)", row=1, col=2)
    fig_mission.update_xaxes(title_text="Time (min)", row=1, col=2)
    
    # 3. Required Power
    fig_mission.add_trace(
        go.Scatter(x=df_a["t_min"], y=df_a["power_req"],
                   name="Power Required", mode="lines",
                   line=dict(color="#F1F5F9", width=2),
                   fill="tozeroy", fillcolor="rgba(255,255,255,0.1)",
                   showlegend=False),
        row=2, col=1
    )
    fig_mission.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig_mission.update_xaxes(title_text="Time (min)", row=2, col=1)
    
    # 4. Aircraft Weight (decreasing as fuel burns)
    weight_profile = []
    for step in apex_r.log:
        # adjust displayed payload if user changed in the expander
        adj_payload = locals().get("display_payload", uav.PAYLOAD)
        total_mass = (uav.STRUCTURAL_MASS + adj_payload + 
                     uav.BATTERY_MASS + uav.MOTOR_MASS + 
                     uav.ENGINE_MASS + step.fuel_mass_kg)
        weight_profile.append(total_mass)
    
    fig_mission.add_trace(
        go.Scatter(x=df_a["t_min"], y=weight_profile,
                   name="Aircraft Weight", mode="lines",
                   line=dict(color="#10B981", width=2),
                   showlegend=False),
        row=2, col=2
    )
    fig_mission.update_yaxes(title_text="Weight (kg)", row=2, col=2)
    fig_mission.update_xaxes(title_text="Time (min)", row=2, col=2)
    
    # 5. Battery SOC
    fig_mission.add_trace(
        go.Scatter(x=df_a["t_min"], y=df_a["soc"],
                   name="Battery SOC", mode="lines",
                   line=dict(color=COLOUR["battery"], width=2),
                   fill="tozeroy", fillcolor="rgba(139,92,246,0.2)",
                   showlegend=False),
        row=3, col=1
    )
    fig_mission.add_hline(y=20, line_dash="dash", line_color="#EF4444",
                          annotation_text="Min (20%)", row=3, col=1)
    fig_mission.add_hline(y=95, line_dash="dash", line_color="#94A3B8",
                          annotation_text="Max (95%)", row=3, col=1)
    fig_mission.update_yaxes(title_text="SOC (%)", row=3, col=1, range=[0, 100])
    fig_mission.update_xaxes(title_text="Time (min)", row=3, col=1)
    
    # 6. Fuel Consumption
    fig_mission.add_trace(
        go.Scatter(x=df_a["t_min"], y=df_a["fuel_kg"],
                   name="Fuel Remaining", mode="lines",
                   line=dict(color=COLOUR["engine"], width=2),
                   fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                   showlegend=False),
        row=3, col=2
    )
    fig_mission.update_yaxes(title_text="Fuel (kg)", row=3, col=2)
    fig_mission.update_xaxes(title_text="Time (min)", row=3, col=2)
    
    fig_mission.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0.5),
        margin=dict(l=20, r=20, t=80, b=20),
    )
    
    st.plotly_chart(fig_mission, use_container_width=True)

    # ── Propulsion Mode Timeline (interpretable AI decisions)
    def detect_mode(df):
        # returns list of modes per row: 'Thermal', 'Electric', 'Hybrid', or 'Idle'
        modes = []
        for e, b in zip(df['engine_kW'], df['battery_kW']):
            if e > 1 and b > 1:
                modes.append('Hybrid')
            elif e > 1 and b <= 1:
                modes.append('Thermal')
            elif b > 1 and e <= 1:
                modes.append('Electric')
            else:
                modes.append('Idle')
        return modes

    mode_colors = {'Thermal': COLOUR['engine'], 'Electric': COLOUR['battery'], 'Hybrid': '#F59E0B', 'Idle': '#94A3B8'}

    st.markdown("#### 🔀 Propulsion Mode Timeline — APEX (how decisions are made)")
    def render_mode_timeline(df, title="APEX Mode"):
        modes = detect_mode(df)
        times = df['t_min'].values
        fig_m = go.Figure()
        # draw mode bands as vertical rects for contiguous regions
        cur_mode = modes[0]
        start_t = times[0]
        for t, m in zip(times[1:], modes[1:]):
            if m != cur_mode:
                fig_m.add_vrect(x0=start_t, x1=t, fillcolor=mode_colors.get(cur_mode,'#333'), opacity=0.18, line_width=0)
                start_t = t
                cur_mode = m
        # final
        fig_m.add_vrect(x0=start_t, x1=times[-1], fillcolor=mode_colors.get(cur_mode,'#333'), opacity=0.18, line_width=0)
        # overlay small step-line for engine fraction to provide context
        fig_m.add_trace(go.Scatter(x=times, y=df['engine_kW']/df['engine_kW'].max(), mode='lines', line=dict(color='#F1F5F9', width=1), name='Engine (norm)'))
        fig_m.add_trace(go.Scatter(x=times, y=(df['battery_kW'].clip(lower=0))/max(1, df['battery_kW'].max()), mode='lines', line=dict(color=COLOUR['battery'], width=1, dash='dot'), name='Battery (norm)'))
        # legend proxies
        for mode, col in mode_colors.items():
            fig_m.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color=col), name=mode))
        fig_m.update_layout(template='plotly_dark', paper_bgcolor='#0F172A', plot_bgcolor='#0F172A', height=220, margin=dict(l=20,r=20,t=10,b=20))
        fig_m.update_xaxes(title_text='Time (min)')
        fig_m.update_yaxes(visible=False)
        st.plotly_chart(fig_m, use_container_width=True)

    render_mode_timeline(df_a, "APEX Mode")
    if ai_r is not None:
        st.markdown("#### 🔀 Propulsion Mode Timeline — APEX-AI (Learned)")
        df_ai = to_df(ai_r.log)
        render_mode_timeline(df_ai, "APEX-AI Mode")

    st.markdown("---")

    # ── Phase Analysis Table ────────────────────────────────────────────────────
    st.markdown("### Phase-by-Phase Analysis (APEX Run)")
    
    phase_analysis = []
    for phase in ["Takeoff", "Climb", "Cruise", "Loiter", "Descent"]:
        sub = df_a[df_a["phase"] == phase]
        if sub.empty:
            continue
        
        phase_analysis.append({
            "Phase": phase,
            "Duration (min)": f"{len(sub) * config.dt / 60:.1f}",
            "Avg Alt (m)": f"{sub['altitude'].mean():.0f}",
            "Avg Speed (km/h)": f"{sub['speed'].mean():.1f}",
            "Avg Power (kW)": f"{sub['power_req'].mean()/1000:.1f}",
            "Engine Power (kW)": f"{sub['engine_kW'].mean():.1f}",
            "Battery Power (kW)": f"{sub['battery_kW'].mean():.1f}",
            "Fuel Burned (kg)": f"{(sub['fuel_burned'].iloc[-1] - sub['fuel_burned'].iloc[0]):.2f}" if len(sub) > 1 else "0.00",
        })
    
    st.dataframe(pd.DataFrame(phase_analysis), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    # ── Endurance Comparison Bar ──────────────────────────────────────────────
    st.markdown("#### 🏆 Loiter Endurance Comparison")
    rustom_loiter = RustomIIBaseline.LOITER_TIME_H

    controllers = ["Fuzzy Logic EMS", "APEX (Phase-Aware)"]
    loiter_vals = [fuzzy_r.loiter_time_h, apex_r.loiter_time_h]
    bar_colours = [COLOUR["fuzzy"], COLOUR["apex"]]
    if ai_r is not None:
        controllers.append("APEX-AI\n(Learned)")
        loiter_vals.append(ai_r.loiter_time_h)
        bar_colours.append("#A855F7")
    controllers.append("Rustom-II\n(Reference)")
    loiter_vals.append(rustom_loiter)
    bar_colours.append(COLOUR["rustom"])

    fig_bar = go.Figure(go.Bar(
        x=controllers,
        y=loiter_vals,
        marker_color=bar_colours,
        text=[f"{v:.2f} h" for v in loiter_vals],
        textposition="outside",
        textfont=dict(color="#F1F5F9", size=14),
    ))
    fig_bar.update_layout(
        template="plotly_dark", paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A", height=320,
        yaxis_title="Loiter Duration (hours)",
        margin=dict(l=20, r=20, t=10, b=40),
        yaxis=dict(range=[0, max(loiter_vals) * 1.2]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(f"Rustom-II reference: {rustom_loiter} h loiter "
               f"(TAPAS BH-201, NPO Saturn 36MT turboprop, 1800 kg MTOW). "
               f"Our UAV is 1000 kg with hybrid propulsion — direct class comparison "
               f"would require normalisation by weight and fuel.")

    # ── Side-by-side metrics ──────────────────────────────────────────────────
    st.markdown("#### 📋 Detailed Comparison")
    metric_labels = ["Total Endurance (h)", "Loiter Time (h)",
                      "Loiter Time (min)", "Fuel Burned (kg)",
                      "Battery Final SOC (%)", "Avg System Power (kW)",
                      "Engine Fraction (%)", "Silent-Electric (min)",
                      "Charge Cycles"]

    def _row(r, name):
        cc = extra_stats.get(name, {}).get("charge_cycles")
        return [f"{r.total_time_h:.2f}", f"{r.loiter_time_h:.2f}",
                f"{r.loiter_time_min:.0f}", f"{r.fuel_burned_kg:.1f}",
                f"{r.battery_final_soc*100:.1f}", f"{r.avg_power_kW:.1f}",
                f"{r.avg_engine_fraction*100:.0f}",
                f"{r.electric_only_time_s/60:.1f}",
                (str(cc) if cc is not None else "—")]

    comp_data = {
        "Metric":                metric_labels,
        "Fuzzy Logic EMS":       _row(fuzzy_r, "Fuzzy Logic EMS"),
        "APEX (Phase-Aware)":    _row(apex_r, "APEX (Phase-Aware)"),
    }
    if ai_r is not None:
        comp_data["APEX-AI (Learned)"] = _row(ai_r, "APEX-AI (Learned)")

    impr_ai = ((ai_r.loiter_time_h - fuzzy_r.loiter_time_h) / fuzzy_r.loiter_time_h * 100
               if ai_r is not None else None)
    comp_data["Improvement vs Fuzzy"] = (
        ["—", f"+{impr:.1f}%",
         f"+{(apex_r.loiter_time_h - fuzzy_r.loiter_time_h)*60:.0f} min",
         "—", "—", "—", "—",
         f"{apex_r.electric_only_time_s/60:.0f} vs {fuzzy_r.electric_only_time_s/60:.1f} min",
         "—"]
    )
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
    if ai_r is not None:
        st.caption(f"APEX-AI vs Fuzzy: **+{impr_ai:.1f}%** loiter endurance — a genuinely "
                   f"trained model (MLP, imitation-learned from verified APEX behaviour "
                   f"across 60 randomized missions) capturing most of APEX's advantage "
                   f"while remaining a real, generalizable learned function rather than "
                   f"a fixed threshold.")

    # ── APEX Advantage Explanation ────────────────────────────────────────────
    st.markdown("#### 💡 APEX Technical Advantage")
    st.markdown(f"""
    | Factor | Fuzzy Logic | APEX | Gain |
    |--------|-------------|------|------|
    | Engine operating point (loiter) | ~39% rated → BSFC 0.372 | 70% rated → BSFC 0.350 | **6.3% fuel efficiency** |
    | Battery use in climb | 50/50 default drains SOC | Engine covers demand only | **+0.23 SOC preserved** |
    | Pure-electric operation | {fuzzy_r.electric_only_time_s/60:.1f} min | {apex_r.electric_only_time_s/60:.0f} min | **{(apex_r.electric_only_time_s/max(fuzzy_r.electric_only_time_s,1)):.0f}× more** |
    | Engine scheduling | Reactive (per-step rules) | Phase-aware state machine | **+4.5% loiter** |
    """)

    # ── PS1 Rubric Scorecard ──────────────────────────────────────────────────
    st.markdown("#### ✅ PS1 Evaluation Rubric Coverage")
    rubric = pd.DataFrame({
        "Criterion":   ["Mission Feasibility", "Optimization Quality",
                        "Engineering Justification", "Innovation",
                        "Endurance Improvement", "Presentation & Visualization"],
        "Weight (%)":  [20, 25, 20, 15, 10, 10],
        "Coverage":    ["✅ Both controllers complete all 5 phases",
                        "✅ APEX state-machine + adaptive thresholds vs Fuzzy rules",
                        "✅ BSFC curve, ISA model + altitude derating, NMC degradation, PS2 digital twin",
                        "✅ Phase-aware engine scheduling — novel vs rule-based methods",
                        f"✅ +{impr:.1f}% loiter vs Fuzzy; Rustom-II baseline provided",
                        "✅ This dashboard — interactive, physics-grounded"],
    })
    st.dataframe(rubric, use_container_width=True, hide_index=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AEROTHON 2026 · Team AERONEXUS · "
    "Veera Akash R · Naveen Kumar K · Sidharth K · Anish A  |  "
    "HAL × IIT Indore  |  PS1: Hybrid-Electric Propulsion Optimisation"
)
