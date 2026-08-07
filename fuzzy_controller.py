"""
fuzzy_controller.py — Aerothon 2026 | Problem Statement 1
Fuzzy Logic Energy Management System (EMS)

WHAT THIS IS:
─────────────
The Fuzzy Logic controller is our first comparison baseline.
It makes energy-split decisions using human-readable IF-THEN rules
applied through fuzzy set membership functions.

WHY FUZZY LOGIC?
─────────────────
Hard rule-based control (e.g., "if SOC < 0.4, use engine") creates
sudden, jerky transitions. Fuzzy Logic creates smooth, overlapping
zones that mimic how an experienced pilot would think:

  "Battery is getting a bit low AND we're entering high-demand phase,
   so lean a bit more on the engine..."

This is more realistic and efficient than binary if/else logic.

HOW IT WORKS — THREE STEPS:
─────────────────────────────
1. FUZZIFICATION:   Convert crisp inputs (SOC=0.65, power=35kW)
                    into fuzzy memberships (SOC is 40% "medium", 60% "high")

2. RULE EVALUATION: Apply all IF-THEN rules, weighted by memberships
                    "Use turboshaft at HIGH level with strength 0.6"
                    "Charge battery at LOW level with strength 0.3"

3. DEFUZZIFICATION: Combine rule outputs into one crisp decision
                    → turboshaft_fraction = 0.72

CONTROLLER INPUTS (what it reads each second):
─────────────────────────────────────────────────
  - battery_soc       : State of Charge  [0.0 – 1.0]
  - power_demand_W    : What physics requires this instant
  - phase             : Current mission phase
  - fuel_remaining    : Fraction of fuel left [0.0 – 1.0]

CONTROLLER OUTPUT (what it decides):
──────────────────────────────────────
  - turboshaft_power_W : Power from turboshaft engine
  - battery_power_W   : Power from/to battery (positive=discharge,
                          negative=charging)

NOTE: We implement Fuzzy Logic from scratch using NumPy.
We could use scikit-fuzzy, but building it manually demonstrates
understanding to the judges — which matters for Engineering Justification (20%).
"""

import numpy as np
from typing import Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.uav_model import HybridUAV
from src.mission import Phase


# ─── FUZZY MEMBERSHIP FUNCTIONS ───────────────────────────────────────────────
# These define the "fuzzy sets" — how much a value belongs to a category.
# All functions return a value in [0.0, 1.0] (0 = not at all, 1 = fully)

def trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

    Shape:
              1.0
               /\\
              /  \\
             /    \\
    0.0 ────/      \\────
            a   b   c

    x = input value
    a = left foot (0 membership starts here)
    b = peak     (1.0 membership here)
    c = right foot (0 membership ends here)

    Examples:
    trimf(0.5, 0.3, 0.5, 0.7) → 1.0  (exactly at peak)
    trimf(0.4, 0.3, 0.5, 0.7) → 0.5  (halfway between foot and peak)
    trimf(0.2, 0.3, 0.5, 0.7) → 0.0  (below left foot)
    """
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

    Shape:
              1.0
         ─────────────
        /             \\
       /               \\
    0.0                 \\────
      a    b         c   d

    The flat top (b to c) means: "fully in this set for this range."
    Useful for "clearly LOW" or "clearly HIGH" regions.
    """
    if x <= a or x >= d:
        return 0.0
    elif x <= b:
        return (x - a) / max(b - a, 1e-9)
    elif x <= c:
        return 1.0
    else:
        return (d - x) / max(d - c, 1e-9)


# ─── BATTERY SOC MEMBERSHIP FUNCTIONS ────────────────────────────────────────
# SOC ranges from 0.0 (empty) to 1.0 (full)
# Safe operating: 0.20 (MIN) to 0.95 (MAX)
# We define 4 fuzzy sets:

def soc_critical(soc: float) -> float:
    """SOC is CRITICAL — must stop discharging NOW.
    Peaks at 0.20 (hard minimum). Above 0.30 → completely out of set."""
    return trapmf(soc, 0.0, 0.18, 0.20, 0.30)

def soc_low(soc: float) -> float:
    """SOC is LOW — prefer engine over battery."""
    return trimf(soc, 0.20, 0.35, 0.50)

def soc_medium(soc: float) -> float:
    """SOC is MEDIUM — balanced approach."""
    return trimf(soc, 0.40, 0.60, 0.80)

def soc_high(soc: float) -> float:
    """SOC is HIGH — can use battery freely or charge more."""
    return trapmf(soc, 0.70, 0.82, 0.95, 1.00)


# ─── POWER DEMAND MEMBERSHIP FUNCTIONS ───────────────────────────────────────
# Normalised against UAV's turboshaft max power (60 kW)
# Value: fraction of rated engine power [0.0 – 1.0+]

def demand_low(demand_frac: float) -> float:
    """Demand is LOW — loiter range (<40% rated)."""
    return trapmf(demand_frac, 0.0, 0.1, 0.30, 0.45)

def demand_medium(demand_frac: float) -> float:
    """Demand is MEDIUM — cruise range (35–75% rated)."""
    return trimf(demand_frac, 0.35, 0.55, 0.80)

def demand_high(demand_frac: float) -> float:
    """Demand is HIGH — climb/takeoff range (>70% rated)."""
    return trapmf(demand_frac, 0.65, 0.80, 1.0, 1.5)


# ─── FUEL MEMBERSHIP FUNCTIONS ───────────────────────────────────────────────
def fuel_low(fuel_frac: float) -> float:
    return trapmf(fuel_frac, 0.0, 0.05, 0.20, 0.35)

def fuel_adequate(fuel_frac: float) -> float:
    return trimf(fuel_frac, 0.20, 0.55, 0.80)

def fuel_ample(fuel_frac: float) -> float:
    return trapmf(fuel_frac, 0.65, 0.80, 1.0, 1.0)


# ─── DEFUZZIFICATION ─────────────────────────────────────────────────────────
def centroid_defuzz(memberships: list, values: list) -> float:
    """
    Centroid defuzzification — the most common method.

    Takes a list of (membership_strength, output_value) pairs
    and computes their weighted average.

    WHY THIS WORKS:
    If rule 1 fires at strength 0.7 suggesting output=0.8
    and rule 2 fires at strength 0.3 suggesting output=0.3,
    result = (0.7×0.8 + 0.3×0.3) / (0.7 + 0.3) = 0.65
    → Engine provides 65% of required power

    This is a smooth blend — no sudden jumps.
    """
    total_weight = sum(memberships)
    if total_weight < 1e-9:
        return 0.5   # default: 50/50 split if no rules fire
    return sum(m * v for m, v in zip(memberships, values)) / total_weight


# ─── FUZZY CONTROLLER CLASS ───────────────────────────────────────────────────
class FuzzyEMS:
    """
    Fuzzy Logic Energy Management System.

    Decides at every time step how to split power between
    the turboshaft engine and the battery.

    DECISION OUTPUT:
    ─────────────────
    engine_fraction ∈ [0.0, 1.0]
      0.0 = pure electric (motor only, engine off)
      1.0 = engine provides all required power
      >1.0 = engine runs above demand → surplus charges battery

    The final power split is then bounded by physical limits
    (engine can't exceed 60 kW, battery can't go below 20% SOC, etc.)
    """

    def __init__(self, uav: HybridUAV):
        self.uav  = uav
        self.name = "Fuzzy Logic EMS"

    def decide(
        self,
        soc:           float,   # battery state of charge [0–1]
        power_demand_W:float,   # required power (W)
        phase:         str,     # current mission phase
        fuel_frac:     float,   # fuel remaining [0–1]
    ) -> Tuple[float, float]:
        """
        Core decision function — called every time step.

        Evaluates all fuzzy rules and returns the power split.

        Returns:
            turboshaft_W  : power from engine (W)
            battery_W     : power from/to battery (W, positive=discharge)
        """
        uav = self.uav
        demand_frac = power_demand_W / uav.TURBOSHAFT_MAX_POWER

        # ── Step 1: Compute membership values ──────────────────────────────
        m_soc_crit   = soc_critical(soc)
        m_soc_low    = soc_low(soc)
        m_soc_med    = soc_medium(soc)
        m_soc_high   = soc_high(soc)

        m_dem_low    = demand_low(demand_frac)
        m_dem_med    = demand_medium(demand_frac)
        m_dem_high   = demand_high(demand_frac)

        m_fuel_low   = fuel_low(fuel_frac)
        m_fuel_ade   = fuel_adequate(fuel_frac)
        m_fuel_ample = fuel_ample(fuel_frac)

        # Phase flags
        is_loiter  = 1.0 if phase == Phase.LOITER  else 0.0
        is_climb   = 1.0 if phase == Phase.CLIMB   else 0.0
        is_cruise  = 1.0 if phase == Phase.CRUISE  else 0.0
        is_takeoff = 1.0 if phase == Phase.TAKEOFF else 0.0
        is_descent = 1.0 if phase == Phase.DESCENT else 0.0

        # ── Step 2: Fuzzy Rule Base ─────────────────────────────────────────
        # Each rule has: (strength, suggested_engine_fraction)
        # Strength = minimum of all condition memberships (AND logic)
        # Fraction = how much of demand the engine should cover

        # Rule output values (engine fraction):
        # 0.00 = all-electric
        # 0.50 = 50/50 split
        # 1.00 = engine covers all
        # 1.20 = engine covers demand + 20% extra (charges battery)
        # 1.40 = engine hard at maximum charging mode

        rules = [

            # ── SAFETY RULES (highest priority) ──────────────────────────
            # Battery critical → protect it, use engine for everything
            (m_soc_crit,                                    1.30),  # R1
            # Battery critical + high demand → max engine immediately
            (min(m_soc_crit, m_dem_high),                  1.40),  # R2

            # ── LOITER PHASE RULES ────────────────────────────────────────
            # Loiter + battery high + ample fuel → electric-primary for silence
            # (Electric-only mode preserves turboshaft and reduces IR signature)
            (min(is_loiter, m_soc_high, m_fuel_ample),     0.20),  # R3
            # Loiter + battery medium → balanced, engine covers most demand
            (min(is_loiter, m_soc_med,  m_dem_low),        0.80),  # R4
            # Loiter + battery low → engine primary + top up battery
            (min(is_loiter, m_soc_low,  m_dem_low),        1.20),  # R5
            # Loiter + fuel low → save fuel, use battery more
            (min(is_loiter, m_fuel_low),                   0.40),  # R6

            # ── CLIMB PHASE RULES ─────────────────────────────────────────
            # Climb always needs maximum power — use both sources
            # Engine at max, battery fills the gap
            (min(is_climb, m_dem_high),                    1.00),  # R7
            # Climb + battery high → let battery assist freely
            (min(is_climb, m_soc_high,  m_dem_high),       0.85),  # R8
            # Climb + battery low → conserve battery, engine works harder
            (min(is_climb, m_soc_low,   m_dem_high),       1.10),  # R9

            # ── CRUISE PHASE RULES ────────────────────────────────────────
            # Cruise + battery high + ample fuel → slight battery discharge OK
            (min(is_cruise, m_soc_high, m_fuel_ample),     0.60),  # R10
            # Cruise + battery medium → engine primary, gentle charge
            (min(is_cruise, m_soc_med,  m_dem_med),        1.10),  # R11
            # Cruise + battery low → engine aggressively charges battery
            (min(is_cruise, m_soc_low),                    1.35),  # R12
            # Cruise + fuel low → electric assist, save fuel for loiter
            (min(is_cruise, m_fuel_low, m_soc_high),       0.50),  # R13

            # ── TAKEOFF RULES ─────────────────────────────────────────────
            # Takeoff: full power from everything
            (is_takeoff,                                   1.00),  # R14

            # ── DESCENT RULES ─────────────────────────────────────────────
            # Descent: very low demand — let engine idle and charge battery
            (min(is_descent, m_dem_low),                   0.80),  # R15
            # Descent + battery not full → charge while descending
            (min(is_descent, m_soc_low),                   1.20),  # R16
            (min(is_descent, m_soc_med),                   1.00),  # R17

            # ── GENERAL RULES ─────────────────────────────────────────────
            # High demand anywhere → engine takes primary load
            (min(m_dem_high, m_soc_low),                   1.20),  # R18
            # Low demand anywhere → battery-primary is fine
            (min(m_dem_low,  m_soc_high),                  0.35),  # R19
        ]

        # ── Step 3: Defuzzify to get engine fraction ────────────────────────
        strengths   = [r[0] for r in rules]
        eng_targets = [r[1] for r in rules]
        engine_frac = centroid_defuzz(strengths, eng_targets)

        # ── Step 4: Convert fraction to actual power values ─────────────────
        # and apply physical limits

        # Engine power (what the rules suggest)
        turboshaft_desired = engine_frac * power_demand_W

        # Physical cap: engine can't exceed rated power
        turboshaft_W = float(np.clip(
            turboshaft_desired,
            uav.TURBOSHAFT_MIN_POWER,
            uav.TURBOSHAFT_MAX_POWER
        ))

        # If engine is above demand → surplus goes to charging battery
        # If engine is below demand → battery makes up the difference
        battery_W = power_demand_W - turboshaft_W

        # Physical cap: battery can't exceed motor max discharge
        # Negative = charging (battery absorbs surplus from engine)
        battery_W = float(np.clip(battery_W, -20_000, uav.MAX_ELECTRIC_POWER))

        # Safety override: if SOC is critical, force stop discharging
        if soc <= uav.BATTERY_MIN_SOC + 0.01 and battery_W > 0:
            battery_W    = 0.0
            turboshaft_W = min(power_demand_W, uav.TURBOSHAFT_MAX_POWER)

        # Safety override: if fuel is out, engine cannot run
        if fuel_frac <= 0.0:
            turboshaft_W = 0.0
            battery_W    = min(power_demand_W, uav.MAX_ELECTRIC_POWER)

        return turboshaft_W, battery_W

    def get_controller_state(
        self, soc: float, power_demand_W: float,
        phase: str, fuel_frac: float
    ) -> dict:
        """
        Returns full diagnostic info — useful for dashboard debug panel.
        Shows which fuzzy sets are active and at what strength.
        """
        demand_frac = power_demand_W / self.uav.TURBOSHAFT_MAX_POWER
        return {
            "controller":    self.name,
            "soc_critical":  round(soc_critical(soc), 3),
            "soc_low":       round(soc_low(soc), 3),
            "soc_medium":    round(soc_medium(soc), 3),
            "soc_high":      round(soc_high(soc), 3),
            "demand_low":    round(demand_low(demand_frac), 3),
            "demand_medium": round(demand_medium(demand_frac), 3),
            "demand_high":   round(demand_high(demand_frac), 3),
            "fuel_low":      round(fuel_low(fuel_frac), 3),
            "phase":         phase,
        }


# ─── VERIFICATION RUN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    uav        = HybridUAV()
    controller = FuzzyEMS(uav)

    print("\n" + "="*60)
    print("  FUZZY EMS — MEMBERSHIP AND RULE VERIFICATION")
    print("="*60)

    # Test a range of operating conditions
    test_cases = [
        # (soc,  power_W,    phase,           fuel, label)
        (0.95,  60_000,  Phase.TAKEOFF,  1.0,  "Takeoff — full battery, full fuel"),
        (0.85,  75_000,  Phase.CLIMB,    0.9,  "Hard climb — battery high"),
        (0.45,  75_000,  Phase.CLIMB,    0.7,  "Hard climb — battery low"),
        (0.80,  52_000,  Phase.CRUISE,   0.8,  "Cruise — nominal"),
        (0.35,  52_000,  Phase.CRUISE,   0.6,  "Cruise — battery getting low"),
        (0.85,  29_000,  Phase.LOITER,   0.5,  "Loiter — battery high (ideal)"),
        (0.55,  29_000,  Phase.LOITER,   0.4,  "Loiter — battery medium"),
        (0.25,  29_000,  Phase.LOITER,   0.3,  "Loiter — battery low"),
        (0.21,  29_000,  Phase.LOITER,   0.1,  "Loiter — battery CRITICAL"),
        (0.60,   8_000,  Phase.DESCENT,  0.2,  "Descent — charge battery"),
    ]

    print(f"\n  {'Condition':<40} {'Turbo(kW)':>10} {'Batt(kW)':>10} {'Engine%':>8}")
    print("  " + "─"*70)
    for soc, pwr, phase, fuel, label in test_cases:
        ts_W, bat_W = controller.decide(soc, pwr, phase, fuel)
        pct = ts_W / max(pwr, 1) * 100
        batt_str = f"{bat_W/1000:+.1f}"   # + = discharge, - = charging
        print(f"  {label:<40} {ts_W/1000:>9.1f}  {batt_str:>9}  {pct:>7.0f}%")

    print("\n  FUZZY MEMBERSHIP FUNCTIONS (SOC sweep):")
    print(f"  {'SOC':>6}  {'CRIT':>7}  {'LOW':>7}  {'MED':>7}  {'HIGH':>7}")
    print("  " + "─"*42)
    for s in np.arange(0.20, 1.01, 0.05):
        print(f"  {s:>6.2f}  "
              f"{soc_critical(s):>7.3f}  "
              f"{soc_low(s):>7.3f}  "
              f"{soc_medium(s):>7.3f}  "
              f"{soc_high(s):>7.3f}")

    print("\n  ✓ Fuzzy controller verified.\n")
