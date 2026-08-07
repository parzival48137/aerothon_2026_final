"""
main.py — Aerothon 2026 | OMEGA-EMS
Run this first to verify the full stack before opening the dashboard.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.uav_model        import HybridUAV
from src.mission          import Mission, MissionConfig
from src.fuzzy_controller import FuzzyEMS
from src.apex_controller  import APEXEMS
from src.simulation       import Simulator, run_comparison, RustomIIBaseline
from src.health_monitor   import run_with_health

def main():
    print("\n" + "="*65)
    print("  AEROTHON 2026 — OMEGA-EMS Full Stack Verification")
    print("  Team AERONEXUS | PS1: Hybrid-Electric Propulsion")
    print("="*65)

    uav    = HybridUAV()
    uav.print_summary()

    config = MissionConfig(dt=1.0)
    print("Generating mission profile...")
    mission = Mission(uav, config)
    profile = mission.generate_profile()
    mission.print_summary(profile)

    print("Running controller comparison...")
    results = run_comparison(
        uav         = uav,
        config      = config,
        profile     = profile,
        controllers = [FuzzyEMS(uav), APEXEMS(uav)],
        verbose     = True,
    )

    print("Attaching health monitor to APEX run...")
    apex_log  = results[1].log
    h_log, monitor = run_with_health(apex_log, dt=config.dt)
    monitor.print_summary(h_log)

    RustomIIBaseline.summary()

    print("\n  FULL STACK VERIFIED.")
    print("  To launch dashboard:  streamlit run dashboard/app.py\n")

if __name__ == "__main__":
    main()
