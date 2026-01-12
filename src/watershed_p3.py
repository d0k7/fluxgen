import numpy as np

class WatershedTwin:
    """
    Solves Problem 3: The Balancing Act.
    Models the 'Delay' using Unit Hydrograph Convolution and tracks Mass Balance.
    """
    def run_simulation(self):
        # Time array (0 to 48 hours)
        t = np.linspace(0, 48, 200)
        dt = t[1] - t[0]
        
        # 1. INPUT: Rain Event (Gaussian Pulse centered at t=2)
        rain_inlet = 1000 * np.exp(-0.5 * ((t - 2.0) / 1.0)**2)
        total_rain = np.trapz(rain_inlet, t)
        
        # 2. MASS BALANCE: Account for the 'Missing' 400 units
        # Logic: It is stored in soil (Infiltration)
        infiltration_ratio = 0.4 # 400/1000
        effective_runoff = rain_inlet * (1 - infiltration_ratio)
        
        # 3. DYNAMICS: Model the 12-hour Delay
        # Use a Log-Normal Transfer Function (Unit Hydrograph)
        lag_target = 12.0
        sigma = 0.4
        
        # Math Adjustment: For LogNormal, Peak (Mode) = exp(mu - sigma^2)
        # We want Peak = lag_target. So mu = ln(lag_target) + sigma^2
        mu = np.log(lag_target) + sigma**2
        
        # Calculate Kernel h(t) safely ignoring t=0 error
        with np.errstate(divide='ignore', invalid='ignore'):
            h_t = (1/(t * sigma * np.sqrt(2*np.pi))) * np.exp(-((np.log(t)-mu)**2)/(2*sigma**2))
        
        h_t = np.nan_to_num(h_t) # Replace NaNs (at t=0) with 0
        h_t /= np.trapz(h_t, t)  # Normalize to conserve mass
        
        # Convolve Input with Kernel to get Output
        outlet_flow = np.convolve(effective_runoff, h_t, mode='full')[:len(t)] * dt
        
        return t, rain_inlet, outlet_flow
