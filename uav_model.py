"""
uav_model.py — Aerothon 2026 | Problem Statement 1
Hybrid-Electric Propulsion Optimisation for a Fixed-Wing UAV
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

g = 9.81
R = 287.05

def get_air_properties(altitude_m: float) -> tuple[float, float, float]:
    T0, P0, L = 288.15, 101325.0, 0.0065
    T   = T0 - L * altitude_m
    P   = P0 * (T / T0) ** (g / (L * R))
    rho = P / (R * T)
    return rho, P, T

class HybridUAV:
    MTOW: float               = 1000.0
    PAYLOAD: float            = 200.0
    CRUISE_SPEED: float       = 250.0
    CRUISE_ALT: float         = 5000.0
    STRUCTURAL_MASS    = 400.0
    FUEL_MASS_MAX      = 119.5     # kg — Phase 4A locked (blueprint FUEL_KG)
    BATTERY_MASS: float       = 133.3    # kg — 20,000 Wh / 150 Wh/kg (NMC, Phase 4B)
    MOTOR_MASS: float         = 30.0
    ENGINE_MASS: float        = 60.0
    WING_AREA: float          = 10.0
    ASPECT_RATIO: float       = 12.0
    CD0: float                = 0.025
    OSWALD_EFF: float         = 0.85
    # Locked design point (Phase 4A-5A trade study + OMEGA-EMS Simulink
    # Blueprint, both confirm 90 kW / 20 kWh NMC / 50 km radius, 10.72 h
    # target endurance). Supersedes the earlier 60 kW / 17.6 kWh LiPo point.
    TURBOSHAFT_MAX_POWER: float = 90_000.0
    TURBOSHAFT_MIN_POWER: float = 12_000.0
    TURBOSHAFT_BSFC: float    = 0.35
    GENERATOR_EFF: float      = 0.92
    GENERATOR_MAX_POWER: float= 55_000.0
    BATTERY_CHEMISTRY  = "NMC"
    BATTERY_SPEC_ENERGY: float= 150.0     # Wh/kg — NMC baseline, Phase 4B cell table
    BATTERY_CAPACITY: float   = 20_000.0  # Wh — 20 kWh, Phase 4B locked
    BATTERY_VOLTAGE: float    = 400.0
    BATTERY_MAX_SOC: float    = 0.95
    BATTERY_MIN_SOC: float    = 0.20
    BATTERY_USABLE_ENERGY: float = 20_000.0 * (0.95 - 0.20)
    ALTITUDE_DERATE_EXP = 0.70      # (rho/rho0)^0.70 — matches matlab/turboshaft_model.m
    BATTERY_CHARGE_EFF = 0.95
    BATTERY_DISCHARGE_EFF = 0.97
    NUM_MOTORS: int         = 2
    MOTOR_MAX_POWER: float  = 20_000.0
    MOTOR_EFF: float        = 0.93
    PROPULSIVE_EFF: float   = 0.82
    MAX_ELECTRIC_POWER: float = 2 * 20_000.0

    def __init__(self) -> None:
        self.fuel_mass   = self.FUEL_MASS_MAX
        self.battery_soc = self.BATTERY_MAX_SOC

    def get_current_weight(self) -> float:
        mass = (self.STRUCTURAL_MASS + self.PAYLOAD + self.BATTERY_MASS +
                self.MOTOR_MASS + self.ENGINE_MASS + self.fuel_mass)
        return mass * g

    def get_required_power(
        self,
        altitude_m: float,
        speed_kmh: float,
        weight_N: Optional[float] = None,
    ) -> tuple[float, float, float, float]:
        if weight_N is None:
            weight_N = self.get_current_weight()
        V   = max(speed_kmh, 1.0) / 3.6
        rho, _, _ = get_air_properties(altitude_m)
        q   = 0.5 * rho * V**2
        CL  = weight_N / (q * self.WING_AREA)
        k   = 1.0 / (np.pi * self.ASPECT_RATIO * self.OSWALD_EFF)
        CD  = self.CD0 + k * CL**2
        LD  = CL / CD
        D   = q * self.WING_AREA * CD
        P   = (D * V) / self.PROPULSIVE_EFF
        return P, CL, CD, LD

    def get_loiter_speed(self, altitude_m: float) -> float:
        rho, _, _ = get_air_properties(altitude_m)
        W   = self.get_current_weight()
        k   = 1.0 / (np.pi * self.ASPECT_RATIO * self.OSWALD_EFF)
        CL_mp = np.sqrt(3.0 * self.CD0 / k)
        V_mp  = np.sqrt(2.0 * W / (rho * self.WING_AREA * CL_mp))
        return V_mp * 3.6

    def get_available_engine_power(self, altitude_m: float) -> float:
        """
        FIX 4 (project history: day2_v2_isr_mission_m.txt, matlab/turboshaft_model.m SS3):
        A sea-level-RATED turboshaft does not deliver its full rated power at
        altitude — thinner air reduces compressor mass flow. This was the
        physical reason the 60 kW design point was infeasible at 5000 m
        (only ~36 kW available vs ~52 kW cruise demand) and why the design
        was locked at 90 kW instead.

        P_avail = P_rated * (rho/rho0)^0.70   (AGARD CP-537 / Janes AAD
        correlation; matches matlab/turboshaft_model.m exactly)
        """
        rho, _, _ = get_air_properties(altitude_m)
        rho0 = 1.22500
        derate = (rho / rho0) ** self.ALTITUDE_DERATE_EXP
        return self.TURBOSHAFT_MAX_POWER * derate

    def get_turboshaft_fuel_flow(self, power_W: float) -> float:
        if power_W <= 0:
            return 0.0
        frac = power_W / self.TURBOSHAFT_MAX_POWER
        bsfc_factor  = 1.0 + 0.35 * (frac - 0.70)**2 / 0.49
        bsfc_current = self.TURBOSHAFT_BSFC * bsfc_factor
        return bsfc_current * (power_W / 1000.0) / 3600.0

    def update_battery(self, net_power_W: float, dt_s: float) -> tuple[float, float]:
        if net_power_W >= 0:
            energy_Wh = net_power_W * dt_s / 3600.0 / self.BATTERY_DISCHARGE_EFF
        else:
            energy_Wh = net_power_W * dt_s / 3600.0 * self.BATTERY_CHARGE_EFF
        soc_delta = -energy_Wh / self.BATTERY_CAPACITY
        self.battery_soc = float(np.clip(
            self.battery_soc + soc_delta,
            self.BATTERY_MIN_SOC,
            self.BATTERY_MAX_SOC
        ))
        return self.battery_soc, energy_Wh

    def reset(self) -> None:
        self.fuel_mass   = self.FUEL_MASS_MAX
        self.battery_soc = self.BATTERY_MAX_SOC

    def print_summary(self) -> None:
        print("\n" + "="*55)
        print("  AEROTHON 2026 — HYBRID UAV SPEC SHEET")
        print("="*55)
        print(f"  MTOW              :  {self.MTOW} kg")
        print(f"  Payload           :  {self.PAYLOAD} kg")
        print(f"  Fuel Capacity     :  {self.FUEL_MASS_MAX} kg")
        print(f"  Battery           :  {self.BATTERY_CAPACITY/1000:.1f} kWh  ({self.BATTERY_USABLE_ENERGY/1000:.1f} kWh usable)")
        print(f"  Turboshaft        :  {self.TURBOSHAFT_MAX_POWER/1000} kW")
        print(f"  Electric Motors   :  {self.NUM_MOTORS} × {self.MOTOR_MAX_POWER/1000} kW = {self.MAX_ELECTRIC_POWER/1000} kW total")
        print(f"  Wing Area         :  {self.WING_AREA} m²")
        print(f"  Wingspan          :  {np.sqrt(self.ASPECT_RATIO * self.WING_AREA):.1f} m")
        print("="*55)
        P, CL, CD, LD = self.get_required_power(self.CRUISE_ALT, self.CRUISE_SPEED)
        loiter_v = self.get_loiter_speed(self.CRUISE_ALT)
        P_loiter, _, _, LD_l = self.get_required_power(self.CRUISE_ALT, loiter_v)
        print(f"\n  CRUISE ({self.CRUISE_ALT}m, {self.CRUISE_SPEED}km/h):")
        print(f"  CL={CL:.3f}  CD={CD:.4f}  L/D={LD:.1f}  Power={P/1000:.1f}kW ({P/self.TURBOSHAFT_MAX_POWER*100:.0f}% engine)")
        print(f"\n  LOITER ({self.CRUISE_ALT}m, {loiter_v:.0f}km/h):")
        print(f"  Power={P_loiter/1000:.1f}kW  L/D={LD_l:.1f}  Saving vs cruise={(1-P_loiter/P)*100:.0f}%")
        print("="*55+"\n")

if __name__ == "__main__":
    uav = HybridUAV()
    uav.print_summary()
