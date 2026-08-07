"""
generate_training_data.py — Aerothon 2026 | APEX-AI training data generation

WHAT THIS DOES:
────────────────
Runs the VERIFIED, physically-grounded deterministic APEX controller across
many randomized missions (different altitudes, speeds, radii, winds), and
records every LOITER-phase timestep as a (features → state) example.

WHY THIS APPROACH (imitation learning, not RL-from-scratch):
──────────────────────────────────────────────────────────────
- The deterministic APEX logic is already verified against real BSFC
  physics and known to outperform Fuzzy Logic (+6.3% at the locked design
  point). It is a credible "expert" to imitate.
- Training a classifier on its behaviour across a WIDE distribution of
  missions (not just the one nominal mission) is what makes the result
  actually AI: the network has to learn the *shape* of the SOC-threshold
  decision boundary from data, and — unlike the hand-coded rule — can
  interpolate/generalize to conditions between the training grid.
- This is achievable and verifiable in hours, not days, unlike training an
  RL agent from scratch under deadline pressure.

FEATURES (per loiter timestep):
────────────────────────────────
  soc                 : battery state of charge [0-1]
  power_demand_frac   : instantaneous demand / engine rated power
  fuel_frac           : fuel remaining [0-1]
  prev_state_charging : was the controller charging last step? (0/1)
  time_in_state_min   : minutes spent in the current state so far

LABEL:
  next_state_charging : 1 if APEX is in CHARGING state this step, else 0
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.uav_model import HybridUAV
from src.mission import Mission, MissionConfig, Phase
from src.apex_controller import APEXEMS

RNG = np.random.default_rng(42)

def random_config():
    """Sample a realistic mission configuration."""
    return MissionConfig(
        cruise_altitude_m      = float(RNG.uniform(3500, 7000)),
        cruise_speed_kmh        = float(RNG.uniform(220, 270)),
        operational_radius_km   = float(RNG.uniform(25, 75)),
        loiter_altitude_m       = float(RNG.uniform(2500, 4500)),
        wind_speed_ms            = float(RNG.uniform(0, 12)),
        heading_deg              = float(RNG.uniform(0, 360)),
        dt                        = 2.0,   # coarser step = faster data generation
    )


def collect_episode(config: MissionConfig) -> list:
    """Run one mission with deterministic APEX, log loiter timesteps."""
    uav        = HybridUAV()
    mission    = Mission(uav, config)
    profile    = mission.generate_profile()
    controller = APEXEMS(uav)

    rows = []
    state_entry_t = 0.0
    prev_charging = 1  # loiter typically enters CHARGING unless SOC already high

    for step in profile:
        soc       = uav.battery_soc
        fuel_frac = uav.fuel_mass / uav.FUEL_MASS_MAX

        P_actual, _, _, _ = uav.get_required_power(step.altitude_m, step.speed_kmh)
        ts_W, bt_W = controller.decide(
            soc=soc, power_demand_W=P_actual, phase=step.phase, fuel_frac=fuel_frac
        )

        if step.phase == Phase.LOITER:
            cur_charging = 1 if controller.loiter_state == "CHARGING" else 0
            if cur_charging != prev_charging:
                state_entry_t = step.t
            rows.append({
                "soc":                  soc,
                "power_demand_frac":    P_actual / uav.TURBOSHAFT_MAX_POWER,
                "fuel_frac":            fuel_frac,
                "prev_state_charging":  prev_charging,
                "time_in_state_min":    (step.t - state_entry_t) / 60.0,
                "label_charging":       cur_charging,
            })
            prev_charging = cur_charging

        # Apply physics exactly like the real simulator does, so SOC/fuel
        # evolve realistically across the episode (engine clamped to
        # altitude-derated available power — same fix as simulation.py).
        avail_W = uav.get_available_engine_power(step.altitude_m)
        ts_W = float(np.clip(ts_W, 0.0, avail_W))
        engine_mech_W = ts_W / uav.GENERATOR_EFF if ts_W > 0 else 0.0
        fuel_flow = uav.get_turboshaft_fuel_flow(engine_mech_W)
        uav.fuel_mass = max(uav.fuel_mass - fuel_flow * config.dt, 0.0)
        uav.update_battery(bt_W, config.dt)

        if uav.fuel_mass <= 0.1 and soc <= uav.BATTERY_MIN_SOC + 0.02:
            break

    return rows


if __name__ == "__main__":
    N_EPISODES = 60
    all_rows = []
    print(f"Generating training data from {N_EPISODES} randomized missions...")
    for i in range(N_EPISODES):
        cfg = random_config()
        rows = collect_episode(cfg)
        all_rows.extend(rows)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_EPISODES}] episodes done, {len(all_rows):,} samples so far")

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "loiter_training_data.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} samples to {out_path}")
    print(f"Class balance: CHARGING={df['label_charging'].mean()*100:.1f}%  "
          f"ELECTRIC={(1-df['label_charging'].mean())*100:.1f}%")
