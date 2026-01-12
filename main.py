import numpy as np
import matplotlib.pyplot as plt
import warnings

# Import our modules
from src.geometry_p1 import ReservoirEstimator
from src.spectral_p2 import SpectralValidator
from src.watershed_p3 import WatershedTwin
from src.groundwater_p4 import GroundwaterModel

def main():
    print("=== FLUXGEN ASSIGNMENT EXECUTION ===\n")

    # --- PROBLEM 1 ---
    print("[1] Reservoir Volume Estimation...")
    p1 = ReservoirEstimator()
    p1.simulate_data()
    # Suppress convergence warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, vol = p1.run_kriging_interpolation()
    print(f"    Calculated Volume: {vol:.2f} cubic units")
    print("    (Method: Kriging Interpolation over 50x50 grid)\n")

    # --- PROBLEM 2 ---
    print("[2] Spectral Validation Logic...")
    p2 = SpectralValidator()
    # Scenario: 5 Verified sites are Clear (False), but Satellite sees Bloom (0.8)
    satellite_data = np.full(50, 0.8)
    decision = p2.analyze_discrepancy(
        verified_indices=[0, 1, 2, 3, 4], 
        verified_status=[False, False, False, False, False], 
        satellite_readings=satellite_data
    )
    print(f"    System Output: {decision}\n")

    # --- PROBLEM 3 ---
    print("[3] Watershed Digital Twin...")
    p3 = WatershedTwin()
    t, rain, flow = p3.run_simulation()
    
    # Check Lag
    peak_rain_time = t[np.argmax(rain)]
    peak_flow_time = t[np.argmax(flow)]
    peak_lag = peak_flow_time - peak_rain_time
    
    print(f"    Modeled Delay: {peak_lag:.1f} Hours (Target: ~12.0)")
    print(f"    Mass Balance Check: Infiltration + Runoff ~= Rain Input\n")

    # --- PROBLEM 4 (CASE STUDY) ---
    print("[4] Groundwater Spatial Model...")
    p4 = GroundwaterModel(grid_size=50)
    
    # Adding sources based on the diagram logic
    # 'agri' (Agriculture), 'built' (Built-up), etc.
    p4.add_source('agriculture', (15, 15), 8, 0.9)
    p4.add_source('built_up',    (35, 35), 6, 1.0)
    p4.add_source('forest',      (35, 10), 5, 0.3)
    p4.add_source('water_body',  (15, 35), 7, 0.2)
    
    p4.compute_gradients()
    print("    Generating Interaction Heatmap (Check popup window)...")
    p4.plot()

if __name__ == "__main__":
    main()
