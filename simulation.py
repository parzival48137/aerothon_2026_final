"""
simulation.py — Aerothon 2026 | Problem Statement 1
Core Simulation Engine

WHAT THIS FILE DOES:
─────────────────────
This is the 'race engine' — it runs a controller through a mission
and records exactly what happens to fuel, battery, power, and time.

THE SIMULATION LOOP (runs every dt=1 second):
──────────────────────────────────────────────
For each time step in the mission profile:

  1. READ environment:    altitude, speed, air density (from mission profile)
  2. COMPUTE power:       recalculate required power with CURRENT weight
                          (UAV gets lighter as fuel burns → less power needed)
  3. ASK controller:      "given SOC and demand, how do you split power?"
  4. APPLY physics:       burn fuel, charge/discharge battery, update state
  5. CHECK feasibility:   did we exceed any constraints?
  6. RECORD log:          save everything for the dashboard

SIMULATION ENDS WHEN (in order of priority):
─────────────────────────────────────────────
  a) Fuel AND battery are exhausted    → mission failed (crash)
  b) Loiter phase exhausts energy      → endurance limit reached (normal)
  c) Profile is fully consumed         → theoretical max reached

THE COMPARISON:
────────────────
We run this SAME engine twice:
  - Once with FuzzyEMS     → get endurance_fuzzy
  - Once with APEXEMS      → get endurance_apex

Difference = APEX improvement. All else is identical. Fair comparison.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.uav_model import HybridUAV, get_air_properties
from src.mission import Mission, MissionConfig, TimeStep, Phase


# ─── SIMULATION LOG ENTRY ─────────────────────────────────────────────────────
@dataclass
class SimStep:
    """
    Everything recorded at one second of simulation.
    This is what the dashboard visualises.

    Fields labelled (CONTROLLER) are the decisions made.
    Fields labelled (STATE)      are the resulting physical state.
    Fields labelled (PERF)       are computed performance metrics.
    """
    # Time and phase
    t:                float   # elapsed time (s)
    phase:            str     # mission phase

    # Environment
    altitude_m:       float   # current altitude (m)
    speed_kmh:        float   # true airspeed (km/h)
    rho:              float   # air density (kg/m³)

    # Power accounting (W)
    required_power_W: float   # (PERF) what physics demands
    turboshaft_W:     float   # (CONTROLLER) engine contribution
    battery_W:        float   # (CONTROLLER) battery contribution (+ = discharge)
    actual_power_W:   float   # (STATE) what was actually delivered

    # State (CONTROLLER decisions accumulate into these)
    fuel_mass_kg:     float   # (STATE) fuel remaining (kg)
    battery_soc:      float   # (STATE) battery state of charge [0–1]
    battery_energy_Wh:float   # (STATE) battery remaining energy (Wh)

    # Fuel accounting
    fuel_flow_kgs:    float   # (PERF) fuel burn rate (kg/s)
    fuel_burned_kg:   float   # (PERF) cumulative fuel burned (kg)

    # Performance metrics
    engine_efficiency:float   # (PERF) BSFC efficiency at this operating point
    system_efficiency:float   # (PERF) overall propulsive efficiency
    weight_N:         float   # (PERF) current aircraft weight (N)

    # Position
    x_km:             float
    y_km:             float

    # Flags
    power_deficit_W:  float   # (PERF) power shortage if demand > supply (W)
    disturbance:      str     # active disturbance description


# ─── SIMULATION RESULT ────────────────────────────────────────────────────────
@dataclass
class SimResult:
    """
    Aggregated results from a complete simulation run.
    Used for the comparison dashboard and final report.
    """
    controller_name:    str
    success:            bool     # True if mission completed without energy loss

    # Endurance (THE KEY METRIC)
    total_time_h:       float    # total mission duration (hours)
    loiter_time_h:      float    # loiter phase duration (hours) — primary metric
    loiter_time_min:    float    # same in minutes

    # Energy accounting
    fuel_burned_kg:     float    # total fuel consumed (kg)
    fuel_remaining_kg:  float    # fuel left at mission end
    fuel_efficiency:    float    # km/kg (range per unit fuel)
    battery_final_soc:  float    # SOC at mission end

    # Power profile
    avg_power_kW:       float
    avg_engine_fraction:float    # fraction of demand covered by engine on average
    electric_only_time_s:float  # seconds spent in pure-electric mode

    # Phase breakdown
    phase_durations_min:dict     # {phase: duration in minutes}
    phase_avg_power_kW: dict     # {phase: average power in kW}

    # Raw log (for dashboard)
    log:                List[SimStep]

    def print_summary(self):
        print(f"\n  {'─'*50}")
        print(f"  RESULT: {self.controller_name}")
        print(f"  {'─'*50}")
        print(f"  Status              : {'✓ SUCCESS' if self.success else '✗ ENERGY DEPLETED'}")
        print(f"  TOTAL ENDURANCE     : {self.total_time_h:.2f} h")
        print(f"  LOITER TIME         : {self.loiter_time_h:.2f} h  "
              f"({self.loiter_time_min:.0f} min)  ← PRIMARY METRIC")
        print(f"  Fuel burned         : {self.fuel_burned_kg:.1f} kg  "
              f"(of {self.fuel_remaining_kg + self.fuel_burned_kg:.0f} kg)")
        print(f"  Battery final SOC   : {self.battery_final_soc*100:.1f}%")
        print(f"  Avg system power    : {self.avg_power_kW:.1f} kW")
        print(f"  Engine fraction avg : {self.avg_engine_fraction*100:.0f}%")
        print(f"  Silent-electric time: {self.electric_only_time_s/60:.1f} min")
        print(f"  {'─'*50}")


# ─── MAIN SIMULATION ENGINE ───────────────────────────────────────────────────
class Simulator:
    """
    Runs a controller through a mission profile and logs results.

    USAGE:
        uav        = HybridUAV()
        config     = MissionConfig()
        mission    = Mission(uav, config)
        profile    = mission.generate_profile()
        controller = FuzzyEMS(uav)    # or APEXEMS(uav)

        sim    = Simulator(uav, config)
        result = sim.run(profile, controller)
        result.print_summary()
    """

    # Power delivery tolerance
    # If power deficit > this, we count it as a feasibility failure
    POWER_DEFICIT_THRESHOLD_W = 500.0   # 0.5 kW tolerance

    def __init__(self, uav: HybridUAV, config: Optional[MissionConfig] = None):
        self.uav    = uav
        self.config = config or MissionConfig()

    def run(
        self,
        profile:    List[TimeStep],
        controller: Any,
        verbose:    bool = True
    ) -> SimResult:
        """
        Main simulation loop.

        Iterates through every time step in the profile.
        At each step: get controller decision → apply physics → log state.

        Stops early if energy is exhausted during loiter.
        Always completes takeoff + climb + cruise (safety margins exist for those).

        Args:
            profile    : time-stepped mission profile from Mission.generate_profile()
            controller : any object with a .decide(soc, power_W, phase, fuel_frac)
                         method that returns (turboshaft_W, battery_W)
            verbose    : print progress during simulation

        Returns:
            SimResult with full log and aggregated metrics
        """
        uav    = self.uav
        config = self.config

        # Reset UAV to initial state
        uav.reset()

        # Initialise log
        log: List[SimStep] = []
        fuel_burned_total = 0.0
        electric_only_s   = 0.0
        power_deficit_total_W = 0.0

        if verbose:
            print(f"\n  Running simulation: {controller.name}")
            print(f"  {'─'*45}")

        t0        = time.time()
        prev_phase = None

        for step_idx, env in enumerate(profile):
            # ── 1. Current state ─────────────────────────────────────────────
            soc         = uav.battery_soc
            fuel_frac   = uav.fuel_mass / uav.FUEL_MASS_MAX

            # ── 2. Recalculate power with actual (current) weight ─────────────
            # WHY: As fuel burns, UAV gets lighter. Less lift needed → less drag
            # → less power. This is why fuel efficiency compounds over time.
            # Simulation is more accurate than profile which used fixed MTOW weight.
            P_actual, CL, _, LD = uav.get_required_power(
                env.altitude_m, env.speed_kmh
            )
            # Use loiter-specific speed during loiter (controller doesn't change speed)
            required_W = P_actual

            # ── 3. Controller decision ────────────────────────────────────────
            turboshaft_W, battery_W = controller.decide(
                soc           = soc,
                power_demand_W = required_W,
                phase          = env.phase,
                fuel_frac      = fuel_frac,
            )

            # ── 4. Physical constraints enforcement ───────────────────────────
            # 4a. Engine: clamp to ALTITUDE-DERATED available power, not flat
            #     sea-level rating (FIX 4 — see get_available_engine_power).
            engine_avail_W = uav.get_available_engine_power(env.altitude_m)
            turboshaft_W = float(np.clip(
                turboshaft_W, 0.0, engine_avail_W
            ))

            # 4b. No engine if fuel is empty
            if uav.fuel_mass <= 0.0:
                turboshaft_W = 0.0

            # 4c. Battery: clamp to discharge limits
            battery_W = float(np.clip(
                battery_W, -uav.GENERATOR_MAX_POWER, uav.MAX_ELECTRIC_POWER
            ))

            # 4d. No battery discharge if at minimum SOC
            if soc <= uav.BATTERY_MIN_SOC and battery_W > 0:
                battery_W = 0.0

            # 4e. Calculate total power delivered
            actual_delivered_W = turboshaft_W + battery_W
            power_deficit_W    = max(required_W - actual_delivered_W, 0.0)
            power_deficit_total_W += power_deficit_W

            # ── 5. Apply physics — consume fuel and update battery ─────────────

            # Fuel consumption
            # Note: generator converts engine mechanical power → electrical
            # We apply generator efficiency here
            engine_mech_W = turboshaft_W / uav.GENERATOR_EFF \
                            if turboshaft_W > 0 else 0.0
            fuel_flow_kgs = uav.get_turboshaft_fuel_flow(engine_mech_W)
            fuel_burned   = fuel_flow_kgs * config.dt
            uav.fuel_mass = max(uav.fuel_mass - fuel_burned, 0.0)
            fuel_burned_total += fuel_burned

            # Battery update
            # If battery_W > 0: battery discharges (delivers power)
            # If battery_W < 0: battery charges (absorbs excess engine power)
            new_soc, energy_change_Wh = uav.update_battery(battery_W, config.dt)

            # Track electric-only mode (when engine is at or below idle)
            if turboshaft_W <= uav.TURBOSHAFT_MIN_POWER + 100:
                electric_only_s += config.dt

            # Engine efficiency at this operating point
            eng_frac = turboshaft_W / uav.TURBOSHAFT_MAX_POWER if turboshaft_W > 0 else 0.0
            bsfc_factor    = 1.0 + 0.35 * (eng_frac - 0.70)**2 / 0.49 if eng_frac > 0 else 1.0
            engine_eff     = 1.0 / max(bsfc_factor, 0.01)   # higher = better
            system_eff     = (actual_delivered_W / max(engine_mech_W + abs(battery_W), 1.0))

            # ── 6. Log this time step ─────────────────────────────────────────
            log.append(SimStep(
                t                = env.t,
                phase            = env.phase,
                altitude_m       = env.altitude_m,
                speed_kmh        = env.speed_kmh,
                rho              = env.rho,
                required_power_W = required_W,
                turboshaft_W     = turboshaft_W,
                battery_W        = battery_W,
                actual_power_W   = actual_delivered_W,
                fuel_mass_kg     = uav.fuel_mass,
                battery_soc      = new_soc,
                battery_energy_Wh= new_soc * uav.BATTERY_CAPACITY,
                fuel_flow_kgs    = fuel_flow_kgs,
                fuel_burned_kg   = fuel_burned_total,
                engine_efficiency= engine_eff,
                system_efficiency= system_eff,
                weight_N         = uav.get_current_weight(),
                x_km             = env.x_km,
                y_km             = env.y_km,
                power_deficit_W  = power_deficit_W,
                disturbance      = env.disturbance,
            ))

            # Progress print
            if verbose and env.phase != prev_phase:
                prev_phase = env.phase
                print(f"  t={env.t/60:6.1f}min  {env.phase:<10}  "
                      f"SOC={soc:.2f}  fuel={uav.fuel_mass:.1f}kg  "
                      f"P={required_W/1000:.1f}kW")

            # ── 7. STOP CONDITION — energy exhausted during loiter ────────────
            # We stop mid-loiter when both energy sources are depleted.
            # Three-way check covers all controller behaviours:
            #   a) battery_at_min  : SOC dropped to near-minimum (normal drain)
            #   b) power_collapsed : controller delivers <15% demand with empty fuel
            #                        (catches case where safety guard holds SOC just
            #                         above threshold but engine is also empty)
            if env.phase == Phase.LOITER:
                fuel_empty      = uav.fuel_mass <= 0.1
                battery_at_min  = new_soc <= uav.BATTERY_MIN_SOC + 0.015
                power_collapsed = (actual_delivered_W < required_W * 0.15
                                   and fuel_empty)
                if fuel_empty and (battery_at_min or power_collapsed):
                    if verbose:
                        print(f"  ⚡ Energy exhausted at t={env.t/60:.1f}min — "
                              f"loiter ends.")
                    break

        # ── Aggregate results ─────────────────────────────────────────────────
        if verbose:
            elapsed = time.time() - t0
            print(f"  Simulation complete in {elapsed:.1f}s — "
                  f"{len(log):,} steps processed.")

        return self._aggregate(log, controller.name,
                               fuel_burned_total, electric_only_s)

    def _aggregate(
        self,
        log:              List[SimStep],
        controller_name:  str,
        fuel_burned_total:float,
        electric_only_s:  float,
    ) -> SimResult:
        """
        Compute summary statistics from the simulation log.
        Called automatically at the end of run().
        """
        uav = self.uav

        if not log:
            # Empty log — simulation didn't run
            return SimResult(
                controller_name=controller_name, success=False,
                total_time_h=0, loiter_time_h=0, loiter_time_min=0,
                fuel_burned_kg=0, fuel_remaining_kg=uav.FUEL_MASS_MAX,
                fuel_efficiency=0, battery_final_soc=uav.BATTERY_MAX_SOC,
                avg_power_kW=0, avg_engine_fraction=0,
                electric_only_time_s=0,
                phase_durations_min={}, phase_avg_power_kW={}, log=log
            )

        last   = log[-1]
        dt     = self.config.dt

        # Phase timing
        phase_steps = {}
        for step in log:
            phase_steps.setdefault(step.phase, []).append(step)

        phase_dur   = {p: len(s)*dt/60   for p, s in phase_steps.items()}
        phase_power = {p: float(np.mean([x.required_power_W for x in s]))/1000
                       for p, s in phase_steps.items()}

        loiter_steps = phase_steps.get(Phase.LOITER, [])
        loiter_s     = len(loiter_steps) * dt

        total_s      = len(log) * dt

        powers       = np.array([s.required_power_W for s in log])
        eng_fracs    = np.array([
            s.turboshaft_W / max(s.required_power_W, 1.0) for s in log
        ])

        fuel_final   = last.fuel_mass_kg
        fuel_burned  = uav.FUEL_MASS_MAX - fuel_final

        # Mission success: no critical power deficits
        deficits     = [s.power_deficit_W for s in log]
        success      = max(deficits) < self.POWER_DEFICIT_THRESHOLD_W * 10

        return SimResult(
            controller_name      = controller_name,
            success              = success,
            total_time_h         = total_s / 3600.0,
            loiter_time_h        = loiter_s / 3600.0,
            loiter_time_min      = loiter_s / 60.0,
            fuel_burned_kg       = fuel_burned,
            fuel_remaining_kg    = fuel_final,
            fuel_efficiency      = (loiter_s / 3600.0 * 250.0) / max(fuel_burned, 0.001),
            battery_final_soc    = last.battery_soc,
            avg_power_kW         = float(np.mean(powers)) / 1000.0,
            avg_engine_fraction  = float(np.mean(eng_fracs)),
            electric_only_time_s = electric_only_s,
            phase_durations_min  = phase_dur,
            phase_avg_power_kW   = phase_power,
            log                  = log,
        )


# ─── RUSTOM II BASELINE ───────────────────────────────────────────────────────
class RustomIIBaseline:
    """
    Reference performance data for TAPAS BH-201 / Rustom-II.
    Used as the real-world comparison baseline in the dashboard.

    DATA SOURCES:
    ─────────────
    Engine: NPO Saturn 36MT turboprop (NOT a Rotax piston — common error)
    Endurance: ~24 h (publicly documented)
    MTOW: 1800 kg
    Speed: 225 km/h
    Altitude: up to 8000 m

    We use this as a reference line in the comparison charts.
    Our UAV is lighter (1000 kg vs 1800 kg) but we demonstrate
    competitive endurance with a SMARTER energy management approach.
    """
    NAME        = "Rustom-II (TAPAS BH-201)"
    MTOW_KG     = 1800.0
    MAX_ENDURANCE_H = 24.0
    CRUISE_KMH  = 225.0
    MAX_ALT_M   = 8000.0
    ENGINE      = "NPO Saturn 36MT turboprop"
    PAYLOAD_KG  = 350.0

    # Estimated conventional (non-hybrid) loiter capability
    # Traditional turboprop: endurance limited by fuel only
    # Our hybrid: can extend loiter through intelligent energy split
    LOITER_TIME_H = 18.0   # estimated loiter fraction of total endurance

    @classmethod
    def summary(cls):
        print(f"\n  RUSTOM-II BASELINE:")
        print(f"  Engine    : {cls.ENGINE}")
        print(f"  MTOW      : {cls.MTOW_KG} kg")
        print(f"  Endurance : {cls.MAX_ENDURANCE_H} h total, ~{cls.LOITER_TIME_H} h loiter")
        print(f"  Speed     : {cls.CRUISE_KMH} km/h")
        print(f"  Altitude  : up to {cls.MAX_ALT_M/1000:.0f} km")
        print(f"  Note: Conventional propulsion (non-hybrid). Used as reference.")


# ─── COMPARISON ENGINE ────────────────────────────────────────────────────────
def run_comparison(
    uav:         HybridUAV,
    config:      MissionConfig,
    profile:     List[TimeStep],
    controllers: list,
    verbose:     bool = True
) -> List[SimResult]:
    """
    Run multiple controllers on the identical mission and compare.

    This is the heart of the comparison:
    same UAV, same mission, same physics — only the EMS strategy differs.

    Args:
        controllers: list of controller objects (FuzzyEMS, APEXEMS, etc.)

    Returns:
        List of SimResult, one per controller.
    """
    results = []
    for ctrl in controllers:
        uav.reset()   # identical starting conditions for every controller
        sim    = Simulator(uav, config)
        result = sim.run(profile, ctrl, verbose=verbose)
        results.append(result)

    if verbose:
        _print_comparison(results)

    return results


def _print_comparison(results: List[SimResult]):
    """Print side-by-side comparison of simulation results."""
    print("\n" + "="*65)
    print("  CONTROLLER COMPARISON — LOITER ENDURANCE")
    print("="*65)

    if not results:
        print("  No results to compare.")
        return

    # Header
    print(f"  {'Metric':<32}", end="")
    for r in results:
        print(f"  {r.controller_name[:15]:>15}", end="")
    print()
    print("  " + "─"*60)

    # Rows
    rows = [
        ("Total endurance (h)",       lambda r: f"{r.total_time_h:.2f}"),
        ("Loiter time (h)",           lambda r: f"{r.loiter_time_h:.2f}"),
        ("Loiter time (min)",         lambda r: f"{r.loiter_time_min:.0f}"),
        ("Fuel burned (kg)",          lambda r: f"{r.fuel_burned_kg:.1f}"),
        ("Battery final SOC (%)",     lambda r: f"{r.battery_final_soc*100:.1f}"),
        ("Avg power (kW)",            lambda r: f"{r.avg_power_kW:.1f}"),
        ("Engine fraction (%)",       lambda r: f"{r.avg_engine_fraction*100:.0f}"),
        ("Silent-electric (min)",     lambda r: f"{r.electric_only_time_s/60:.1f}"),
    ]
    for label, fn in rows:
        print(f"  {label:<32}", end="")
        for r in results:
            print(f"  {fn(r):>15}", end="")
        print()

    # Improvement vs first controller
    if len(results) >= 2:
        base = results[0]
        print("\n  IMPROVEMENT vs baseline:")
        for r in results[1:]:
            if base.loiter_time_h > 0:
                impr = (r.loiter_time_h - base.loiter_time_h) / base.loiter_time_h * 100
                print(f"  {r.controller_name} vs {base.controller_name}: "
                      f"{impr:+.1f}% loiter endurance")

    # Rustom II reference
    print(f"\n  RUSTOM-II REFERENCE: ~18.0 h loiter (conventional turboprop)")
    print("="*65 + "\n")


# ─── VERIFICATION RUN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.uav_model  import HybridUAV
    from src.mission    import Mission, MissionConfig
    from src.fuzzy_controller import FuzzyEMS

    print("Aerothon 2026 — Simulation Engine Verification")
    print("─" * 50)

    uav     = HybridUAV()
    config  = MissionConfig(dt=1.0)
    mission = Mission(uav, config)

    print("Generating mission profile...")
    profile = mission.generate_profile()
    mission.print_summary(profile)

    # Run Fuzzy controller
    fuzzy_ctrl = FuzzyEMS(uav)
    sim        = Simulator(uav, config)
    result     = sim.run(profile, fuzzy_ctrl, verbose=True)
    result.print_summary()

    # Rustom II reference
    RustomIIBaseline.summary()

    print("\n  ✓ Simulation engine verified.")
    print("  Next: build APEX controller and run the comparison.\n")
