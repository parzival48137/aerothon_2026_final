"""
mission.py — Aerothon 2026 | Problem Statement 1
Mission Profile Definition and Environment Model

WHAT THIS FILE DOES:
────────────────────
Generates a time-stepped record of every second of flight.
At each second, it tells us:
  - Where the UAV is        (altitude, position)
  - How fast it is going    (airspeed)
  - What the air is like    (density, temperature — from ISA model)
  - How much power physics  demands right now (Watts)

The controller (Fuzzy Logic or APEX) reads each second and answers:
  "How much of this power comes from the turboshaft vs the battery?"

KEY INSIGHT — WHY SEPARATE MISSION FROM CONTROLLER:
────────────────────────────────────────────────────
The environment (physics, phases, weather) is the same for BOTH
controllers. Only the energy decisions differ. Keeping them separate
means we can run the same mission through both controllers and get a
fair, controlled comparison.

MISSION PHASES (from Problem Statement):
  1. Takeoff    — high power, short duration
  2. Climb      — elevated power, altitude gain
  3. Cruise     — steady power, 1 hour transit
  4. Loiter     — minimum power, THIS IS WHAT WE OPTIMISE
  5. Descent    — low power, gravity does the work

MODIFIABLE INPUTS (tunable via MissionConfig):
  - Cruise altitude and speed
  - Cruise duration
  - Loiter altitude
  - Wind conditions
  - Enable sudden disturbances (for APEX adaptive testing)
"""

import numpy as np
import sys
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.uav_model import HybridUAV, get_air_properties


# ─── PHASE NAMES ──────────────────────────────────────────────────────────────
class Phase(str, Enum):
    """
    The 5 mission phases from the Problem Statement.
    Using string enum so we can use them directly as labels in plots.
    """
    TAKEOFF = "Takeoff"
    CLIMB   = "Climb"
    CRUISE  = "Cruise"
    LOITER  = "Loiter"
    DESCENT = "Descent"


# ─── MODIFIABLE INPUTS ────────────────────────────────────────────────────────
@dataclass
class MissionConfig:
    """
    All user-tunable parameters for the mission.

    These will become sliders and inputs in the Streamlit dashboard.
    Change these to explore different mission scenarios.

    DESIGN CHOICES EXPLAINED:
    ─────────────────────────
    cruise_altitude_m = 5000  → midpoint of PS range (3–10 km)
                                Good balance: thin air (less drag) but
                                ISA temp manageable for turboshaft
    
    loiter_altitude_m = 3000  → lower than cruise for two reasons:
                                1. Denser air → wings more efficient at
                                   low speed → less power needed
                                2. Better target visibility for sensors

    cruise_duration_s = 3600  → 1 hour transit is realistic for MALE UAV
                                We fix this so the loiter comparison is fair
    """
    # ── Mission Profile ───────────────────────────────────────────────────
    cruise_altitude_m:   float = 5000.0    # m   — target cruise altitude
    cruise_speed_kmh:    float = 250.0     # km/h — from problem statement
    operational_radius_km: float = 50.0   # km  — "course": one-way transit
                                           # leg to station. Phase 4A locked
                                           # design point (was 100 km provisional).
    heading_deg:          float = 90.0    # deg — outbound course, 0=North,
                                           # 90=East, clockwise. Cosmetic
                                           # (rotates the ground-track plot);
                                           # does not affect energy physics,
                                           # which is heading-independent.
    cruise_duration_s:   Optional[float] = None  # s — derived from radius
                                           # below unless explicitly overridden
    loiter_altitude_m:   float = 3000.0   # m   — loiter/surveillance orbit

    # ── Environment ───────────────────────────────────────────────────────
    wind_speed_ms:       float = 0.0      # m/s  — steady headwind (cruise)
    # NOTE: Positive = headwind (increases drag)
    #       Negative = tailwind (reduces drag, but UAV is on return leg)

    # ── Disturbance Scenarios (for APEX testing) ──────────────────────────
    # These simulate real-world unexpected events:
    # APEX should detect and adapt — Fuzzy Logic handles it but less gracefully
    enable_disturbances: bool  = False    # Toggle on/off
    disturbance_time_s:  float = 7200.0  # When it hits (default: during loiter)

    # ── Simulation Resolution ─────────────────────────────────────────────
    dt:                  float = 1.0      # seconds per time step
    # dt=1.0 → one data point per second → good accuracy, acceptable speed
    # dt=10.0 → faster but coarser (use for quick tests)

    def __post_init__(self):
        # Derive one-way cruise-leg duration from the operational radius
        # ("course") unless the caller explicitly fixed a duration.
        if self.cruise_duration_s is None:
            self.cruise_duration_s = (
                self.operational_radius_km / max(self.cruise_speed_kmh, 1.0)
            ) * 3600.0


# ─── ONE SECOND OF FLIGHT ─────────────────────────────────────────────────────
@dataclass
class TimeStep:
    """
    Everything we know about one instant in the mission.

    The controller receives this and returns a power split decision.
    The simulation then updates fuel, battery, and UAV state.

    FIELDS EXPLAINED:
    ─────────────────
    required_power_W : Power the PHYSICS demand at this instant.
                       This is non-negotiable — the UAV must produce this
                       or it falls out of the sky.

    CL, LD           : Aerodynamic coefficients — logged for dashboard
                       and engineering justification in the report.

    x_km, y_km       : 2D position on the ground grid (east/north).
                       Used for the course visualisation on the dashboard.

    disturbance       : Description of any active sudden event.
                       APEX reads this to trigger adaptive behaviour.
    """
    t:               float   # elapsed mission time (s)
    phase:           str     # current phase name
    altitude_m:      float   # altitude above sea level (m)
    speed_kmh:       float   # true airspeed (km/h)
    wind_ms:         float   # effective headwind (m/s)
    rho:             float   # air density (kg/m³)
    temp_K:          float   # ambient temperature (K)
    required_power_W:float   # power physics demand (W)
    CL:              float   # lift coefficient (dimensionless)
    LD:              float   # lift-to-drag ratio (dimensionless)
    x_km:            float   # position east of base (km)
    y_km:            float   # position north of base (km)
    disturbance:     str     # active disturbance description ("" if none)


# ─── MAIN MISSION CLASS ───────────────────────────────────────────────────────
class Mission:
    """
    Generates the complete time-stepped mission environment profile.

    USAGE:
        uav     = HybridUAV()
        config  = MissionConfig()
        mission = Mission(uav, config)
        profile = mission.generate_profile()
        mission.print_summary(profile)

    The returned profile is a list of TimeStep objects.
    Pass it to the Simulation class along with a controller.
    """

    # ─── PHASE PERFORMANCE PARAMETERS ─────────────────────────────────────
    # These come from engineering estimates for a 1000 kg MALE-class UAV

    # Rate of Climb — derived from excess power analysis:
    # Below 2000m: engine produces more thrust in denser air → faster climb
    # Above 2000m: diminishing returns as air thins
    ROC_LOW_MS  = 5.0    # m/s  — rate of climb below 2000m
    ROC_HIGH_MS = 3.0    # m/s  — rate of climb above 2000m
    # Average ≈ 4 m/s → climbing 5000m takes ≈ 21 minutes (realistic for class)

    # Rate of Descent — engine at idle, mild nose-down pitch
    ROD_MS      = 4.5    # m/s  — rate of descent

    # Takeoff
    TO_SPEED_KMH = 150.0  # km/h — liftoff speed (CL ≈ 0.92 at this speed)
    TO_ALT_M     = 100.0  # m    — end of takeoff phase (obstacles cleared)
    TO_DURATION  = 120    # s    — includes ground roll + rotation + initial climb

    # Loiter budget cap — simulation stops early when energy exhausted
    MAX_LOITER_H = 20.0   # hours maximum (practical limit, not energy limit).
    # Raised from 10.0: at the corrected 90kW/20kWh/119.5kg design point
    # both controllers had enough energy to fly past the old 10h cap,
    # which was silently hiding the Fuzzy-vs-APEX energy-management
    # difference this comparison exists to measure. 20h is still just a
    # safety ceiling — actual endurance is energy-limited, not this cap.

    def __init__(self, uav: HybridUAV, config: Optional[MissionConfig] = None):
        self.uav    = uav
        self.config = config or MissionConfig()

    # ─── WIND MODEL ───────────────────────────────────────────────────────────
    def _get_wind(self, t: float, phase: str) -> float:
        """
        Returns effective headwind at time t during a given phase.

        WHY WIND MATTERS:
        Headwind increases the drag force (and thus required power) because
        the aircraft must push through more relative air mass.
        Power ∝ drag, and drag ∝ (airspeed + headwind)²

        Steady wind: only applied during cruise (UAV is transiting).
        Gust disturbance: sudden step change — tests APEX adaptation.
        """
        cfg = self.config
        wind = 0.0

        if phase == Phase.CRUISE:
            wind = cfg.wind_speed_ms

        if cfg.enable_disturbances and t >= cfg.disturbance_time_s:
            # Sudden +15 m/s gust scenario.
            # Real effect: required power jumps ~20-30%
            # APEX: detects power spike, shifts more load to battery immediately
            # Fuzzy Logic: reacts but with membership function lag
            wind += 15.0

        return wind

    # ─── POWER WITH WIND CORRECTION ───────────────────────────────────────────
    def _get_wind_corrected_power(
            self, altitude_m: float, speed_kmh: float,
            wind_ms: float, weight_N: float
    ) -> tuple:
        """
        Compute required power accounting for headwind.

        PHYSICS:
        The aircraft's airspeed through the air mass determines lift and drag.
        We fly at cruise speed (ground speed target), but if there's a headwind,
        we need more thrust to maintain that speed over the ground.

        Simplified model: headwind adds an equivalent drag increment.
        Wind drag penalty ≈ proportional to (wind / airspeed)

        Returns: (power_W, CL, LD)
        """
        P_base, CL, _, LD = self.uav.get_required_power(
            altitude_m, speed_kmh, weight_N=weight_N
        )
        # Wind drag correction factor
        V_kmh = max(speed_kmh, 1.0)
        wind_frac    = (wind_ms * 3.6) / V_kmh          # wind as fraction of airspeed
        wind_penalty = 1.0 + 0.5 * max(wind_frac, 0.0)  # headwind increases power
        return P_base * wind_penalty, CL, LD

    # ─── PROFILE GENERATOR ────────────────────────────────────────────────────
    def generate_profile(self) -> List[TimeStep]:
        """
        Builds the complete time-stepped mission profile.

        Walks through all 5 phases second by second.
        Returns a list of TimeStep records ready for the controller.

        IMPORTANT: Uses MTOW weight throughout profile generation.
        The simulation will recalculate power with actual (decreasing)
        fuel weight at each step — making results more accurate.
        The profile is the 'worst case' power demand template.
        """
        cfg    = self.config
        dt     = cfg.dt
        uav    = self.uav
        W_full = uav.MTOW * 9.81   # full-weight power demand (conservative)

        profile: List[TimeStep] = []
        t    = 0.0
        x_km = 0.0
        y_km = 0.0

        # ── PHASE 1: TAKEOFF ─────────────────────────────────────────────────
        # Duration: TO_DURATION seconds (120 s)
        # Altitude: 0 → TO_ALT_M (0 → 100 m)
        # Speed:    0 → TO_SPEED_KMH (0 → 150 km/h)
        # Power:    Maximum — fighting gravity + accelerating mass
        #
        # Power multiplier of 1.8× accounts for:
        # - Inertia (F=ma, we need to accelerate 1000 kg)
        # - Ground effect transitioning out
        # - Flap drag during takeoff roll
        print("  [1/5] Generating TAKEOFF phase...")
        n_to = int(self.TO_DURATION / dt)
        for i in range(n_to):
            frac     = i / max(n_to - 1, 1)
            alt      = self.TO_ALT_M * frac
            speed    = max(self.TO_SPEED_KMH * frac, 40.0)
            wind     = self._get_wind(t, Phase.TAKEOFF)
            rho, _, T = get_air_properties(alt)
            P, CL, LD = self._get_wind_corrected_power(alt, speed, wind, W_full)
            P_eff    = P * 1.8   # takeoff power multiplier

            profile.append(TimeStep(
                t=t, phase=Phase.TAKEOFF, altitude_m=alt,
                speed_kmh=speed, wind_ms=wind, rho=rho, temp_K=T,
                required_power_W=P_eff, CL=CL, LD=LD,
                x_km=x_km, y_km=y_km, disturbance=""
            ))
            x_km += (speed / 3.6) * dt / 1000.0
            t += dt

        # ── PHASE 2: CLIMB ───────────────────────────────────────────────────
        # Altitude: TO_ALT_M → cruise_altitude_m
        # Speed:    TO_SPEED → cruise_speed (linear ramp)
        # Power:    High — P_level + power to gain altitude
        #
        # Climbing power formula:
        #   P_climb = P_level + (W × ROC)
        #   The second term is literally lifting the aircraft weight at
        #   rate ROC. At 3 m/s and 9810 N weight: extra 29.4 kW just to climb.
        print("  [2/5] Generating CLIMB phase...")
        alt_c = self.TO_ALT_M
        spd_c = self.TO_SPEED_KMH
        while alt_c < cfg.cruise_altitude_m:
            roc        = self.ROC_LOW_MS if alt_c < 2000 else self.ROC_HIGH_MS
            alt_c      = min(alt_c + roc * dt, cfg.cruise_altitude_m)
            climb_frac = ((alt_c - self.TO_ALT_M) /
                          max(cfg.cruise_altitude_m - self.TO_ALT_M, 1.0))
            spd_c      = (self.TO_SPEED_KMH +
                          (cfg.cruise_speed_kmh - self.TO_SPEED_KMH) * climb_frac)
            wind       = self._get_wind(t, Phase.CLIMB)
            rho, _, T  = get_air_properties(alt_c)
            P_lv, CL, LD = self._get_wind_corrected_power(alt_c, spd_c, wind, W_full)
            # Add climbing component
            P_climb    = P_lv + W_full * roc

            profile.append(TimeStep(
                t=t, phase=Phase.CLIMB, altitude_m=alt_c,
                speed_kmh=spd_c, wind_ms=wind, rho=rho, temp_K=T,
                required_power_W=P_climb, CL=CL, LD=LD,
                x_km=x_km, y_km=y_km, disturbance=""
            ))
            x_km += (spd_c / 3.6) * dt / 1000.0
            t += dt

        # ── PHASE 3: CRUISE ──────────────────────────────────────────────────
        # Constant altitude and speed for cruise_duration_s
        # This is the transit phase — fixed 1 hour in our baseline config.
        # Headwind (if configured) increases power demand here.
        print("  [3/5] Generating CRUISE phase...")
        alt_cr   = cfg.cruise_altitude_m
        spd_cr   = cfg.cruise_speed_kmh
        n_cruise = int(cfg.cruise_duration_s / dt)
        for i in range(n_cruise):
            wind      = self._get_wind(t, Phase.CRUISE)
            rho, _, T = get_air_properties(alt_cr)
            P, CL, LD = self._get_wind_corrected_power(alt_cr, spd_cr, wind, W_full)

            disturb = ""
            if cfg.enable_disturbances and t >= cfg.disturbance_time_s:
                disturb = "GUST +15m/s headwind — APEX adaptive mode activated"

            profile.append(TimeStep(
                t=t, phase=Phase.CRUISE, altitude_m=alt_cr,
                speed_kmh=spd_cr, wind_ms=wind, rho=rho, temp_K=T,
                required_power_W=P, CL=CL, LD=LD,
                x_km=x_km, y_km=y_km, disturbance=disturb
            ))
            x_km += (spd_cr / 3.6) * dt / 1000.0
            t += dt

        # ── PHASE 4: LOITER ──────────────────────────────────────────────────
        # THIS IS THE CRITICAL PHASE — the one we optimise.
        #
        # Flying at MINIMUM POWER SPEED (not cruise speed).
        # Minimum power speed is where total power is lowest:
        #   V_mp = V_md / ³√3  ≈ 0.76 × V_md
        # where V_md is the minimum drag speed.
        #
        # At V_mp: propulsive power is minimised → battery + fuel last longest.
        # The controller's job is to stretch this phase as long as possible.
        #
        # ORBITAL PATTERN: UAV circles the target (surveillance orbit).
        # We model this as a circle with radius R ≈ 1 km.
        # Position oscillates — x and y loop around the orbit centre.
        print("  [4/5] Generating LOITER phase...")
        alt_l   = cfg.loiter_altitude_m
        spd_l   = uav.get_loiter_speed(alt_l)   # physics-optimal loiter speed
        rho_l, _, T_l = get_air_properties(alt_l)
        P_l, CL_l, LD_l = self._get_wind_corrected_power(alt_l, spd_l, 0.0, W_full)

        n_loiter  = int(self.MAX_LOITER_H * 3600.0 / dt)
        orbit_r   = 1.0    # km — surveillance orbit radius
        orbit_cx  = x_km   # orbit centre (where we arrived from cruise)
        orbit_cy  = y_km
        orbit_omega = (spd_l / 3.6) / (orbit_r * 1000.0)  # rad/s

        for i in range(n_loiter):
            theta = orbit_omega * (i * dt)
            x_km  = orbit_cx + orbit_r * np.cos(theta)
            y_km  = orbit_cy + orbit_r * np.sin(theta)
            wind  = self._get_wind(t, Phase.LOITER)

            disturb = ""
            if cfg.enable_disturbances and t >= cfg.disturbance_time_s:
                disturb = "GCS COMMAND: extend loiter — APEX re-optimising"

            profile.append(TimeStep(
                t=t, phase=Phase.LOITER, altitude_m=alt_l,
                speed_kmh=spd_l, wind_ms=wind, rho=rho_l, temp_K=T_l,
                required_power_W=P_l, CL=CL_l, LD=LD_l,
                x_km=x_km, y_km=y_km, disturbance=disturb
            ))
            t += dt

        # ── PHASE 5: DESCENT AND LANDING ─────────────────────────────────────
        # Altitude: loiter_alt → 0m
        # Engine near idle — gravity provides most of the energy.
        # Slight power needed to maintain control and approach speed.
        #
        # ENERGY RECOVERY OPPORTUNITY:
        # With a generator on the turboshaft, we can run it at idle and use
        # the airflow through the propeller as a windmill to trickle-charge
        # the battery during descent. Our controllers will model this.
        # Power factor 0.30 = 70% reduction from level-flight power.
        print("  [5/5] Generating DESCENT phase...")
        alt_d = cfg.loiter_altitude_m
        spd_d = spd_l
        x_km  = orbit_cx   # return toward base
        while alt_d > 0.1:
            alt_d = max(alt_d - self.ROD_MS * dt, 0.0)
            # Slowly accelerate back toward approach speed
            spd_d = min(spd_d + 0.02 * dt, cfg.cruise_speed_kmh * 0.8)
            wind  = self._get_wind(t, Phase.DESCENT)
            rho, _, T = get_air_properties(alt_d)
            P, CL, LD = self._get_wind_corrected_power(alt_d, spd_d, 0.0, W_full)
            P_desc = P * 0.30   # 70% reduction — gravity assist during glide

            profile.append(TimeStep(
                t=t, phase=Phase.DESCENT, altitude_m=alt_d,
                speed_kmh=spd_d, wind_ms=wind, rho=rho, temp_K=T,
                required_power_W=P_desc, CL=CL, LD=LD,
                x_km=x_km, y_km=y_km, disturbance=""
            ))
            # Moving back toward base
            x_km = max(x_km - (spd_d / 3.6) * dt / 1000.0, 0.0)
            t += dt

        # Rotate the whole ground track to the configured outbound heading.
        # Energy physics is heading-independent (flat-earth, no true wind
        # direction coupling beyond the scalar headwind already applied),
        # so this is a pure post-hoc rotation of the (x_km, y_km) trace —
        # it changes what the course looks like on the dashboard map, not
        # any power/fuel/battery number.
        theta = np.radians(90.0 - cfg.heading_deg)  # convert compass→math angle
        cos_h, sin_h = np.cos(theta), np.sin(theta)
        for step in profile:
            x0, y0 = step.x_km, step.y_km
            step.x_km = x0 * cos_h - y0 * sin_h
            step.y_km = x0 * sin_h + y0 * cos_h

        print(f"  Profile complete: {len(profile):,} time steps "
              f"({len(profile)*dt/3600:.1f} h theoretical max)\n")
        return profile

    # ─── ANALYSIS METHODS ─────────────────────────────────────────────────────

    def get_phase_stats(self, profile: List[TimeStep]) -> dict:
        """
        Aggregate statistics per phase.
        Used by the dashboard for the phase breakdown panel.
        """
        stats = {}
        for phase in Phase:
            steps = [s for s in profile if s.phase == phase]
            if not steps:
                continue
            powers = np.array([s.required_power_W for s in steps])
            alts   = np.array([s.altitude_m       for s in steps])
            speeds = np.array([s.speed_kmh        for s in steps])
            lds    = np.array([s.LD               for s in steps])
            stats[phase.value] = {
                "n_steps":        len(steps),
                "duration_s":     len(steps) * self.config.dt,
                "duration_min":   len(steps) * self.config.dt / 60.0,
                "duration_h":     len(steps) * self.config.dt / 3600.0,
                "avg_power_kW":   float(np.mean(powers)) / 1000.0,
                "max_power_kW":   float(np.max(powers))  / 1000.0,
                "avg_alt_m":      float(np.mean(alts)),
                "avg_speed_kmh":  float(np.mean(speeds)),
                "avg_LD":         float(np.mean(lds)),
                "energy_demand_Wh": float(np.sum(powers) * self.config.dt / 3600.0),
            }
        return stats

    def print_summary(self, profile: List[TimeStep]):
        """Print mission summary table — run after generate_profile() to verify."""
        stats    = self.get_phase_stats(profile)
        total_s  = len(profile) * self.config.dt
        cfg      = self.config

        print("\n" + "="*65)
        print("  AEROTHON 2026 — MISSION PROFILE SUMMARY")
        print("="*65)
        print(f"  Config: cruise={cfg.cruise_altitude_m:.0f}m, "
              f"{cfg.cruise_speed_kmh:.0f}km/h, "
              f"wind={cfg.wind_speed_ms:.0f}m/s, "
              f"disturbances={'ON' if cfg.enable_disturbances else 'OFF'}")
        print(f"  Total profile length: {total_s/3600:.1f} h  "
              f"({len(profile):,} steps at dt={cfg.dt}s)")
        print("─"*65)
        hdr = f"  {'Phase':<12} {'Duration':>10} {'Avg P (kW)':>11} " \
              f"{'Max P (kW)':>11} {'Avg Alt':>9} {'L/D':>6}"
        print(hdr)
        print("  " + "─"*61)
        for pname, s in stats.items():
            print(f"  {pname:<12} "
                  f"{s['duration_min']:>8.1f}min "
                  f"{s['avg_power_kW']:>10.1f}  "
                  f"{s['max_power_kW']:>10.1f}  "
                  f"{s['avg_alt_m']:>8.0f}m "
                  f"{s['avg_LD']:>6.1f}")
        print("─"*65)

        # Key numbers
        loiter = stats.get(Phase.LOITER.value, {})
        cruise = stats.get(Phase.CRUISE.value, {})
        if loiter:
            uav    = self.uav
            spd_l  = uav.get_loiter_speed(cfg.loiter_altitude_m)
            print(f"\n  LOITER SPEED (physics-optimal): {spd_l:.0f} km/h")
            print(f"  LOITER POWER (avg):             {loiter['avg_power_kW']:.1f} kW")
            print(f"  LOITER L/D:                     {loiter['avg_LD']:.1f}")
        if cruise:
            print(f"\n  CRUISE POWER (avg):             {cruise['avg_power_kW']:.1f} kW")
            print(f"  CRUISE L/D:                     {cruise['avg_LD']:.1f}")

        print(f"\n  ENDURANCE TARGET: maximise loiter duration.")
        print(f"  Controller benchmark: Fuzzy Logic vs APEX vs Rustom II baseline.")
        print("="*65 + "\n")


# ─── VERIFICATION RUN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.uav_model import HybridUAV

    print("Aerothon 2026 — Mission Profile Verification")
    print("─" * 45)

    uav     = HybridUAV()
    config  = MissionConfig()   # default settings
    mission = Mission(uav, config)

    print("Generating profile (this takes ~10 seconds for 10h loiter)...")
    profile = mission.generate_profile()
    mission.print_summary(profile)

    # Phase timeline — show transition points
    print("  PHASE TRANSITIONS:")
    current_phase = None
    for step in profile:
        if step.phase != current_phase:
            current_phase = step.phase
            print(f"    t = {step.t/60:7.1f} min │ {step.phase:<10} │ "
                  f"alt={step.altitude_m:6.0f}m │ "
                  f"speed={step.speed_kmh:5.0f}km/h │ "
                  f"P_req={step.required_power_W/1000:5.1f}kW")

    # Loiter speed verification
    spd = uav.get_loiter_speed(config.loiter_altitude_m)
    P_l, _, _, LD_l = uav.get_required_power(config.loiter_altitude_m, spd)
    P_c, _, _, LD_c = uav.get_required_power(config.cruise_altitude_m,
                                               config.cruise_speed_kmh)
    print(f"\n  LOITER vs CRUISE:")
    print(f"    Loiter speed : {spd:.0f} km/h  →  Power = {P_l/1000:.1f} kW  "
          f"(L/D = {LD_l:.1f})")
    print(f"    Cruise speed : {config.cruise_speed_kmh:.0f} km/h  →  "
          f"Power = {P_c/1000:.1f} kW  (L/D = {LD_c:.1f})")
    print(f"    Power saving in loiter: {(1 - P_l/P_c)*100:.0f}%")
    print(f"\n  ✓ Verification complete. Profile ready for simulation.\n")
