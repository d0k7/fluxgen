import numpy as np

class SpectralValidator:
    """
    Solves Problem 2: The Spectral Discrepancy.
    Implements the Validation Framework and Spatial Weighting Logic.
    """
    def __init__(self, n_sites=50):
        self.n_sites = n_sites
        self.site_ids = np.arange(n_sites)
        
    def analyze_discrepancy(self, verified_indices, verified_status, satellite_readings):
        """
        Decides whether to cancel the alert based on spatial weighting.
        
        Args:
            verified_indices: List of IDs for the 5 manual sites.
            verified_status: List of booleans (True=Bloom, False=Clear).
            satellite_readings: Array of satellite 'Greenness' (0.0 to 1.0).
        """
        # 1. Calculate Error Rate on Verified Sites
        satellite_flags = satellite_readings[verified_indices] > 0.6 # Threshold
        ground_truth = np.array(verified_status)
        
        # False Positives: Satellite says Bloom, Ground says Clear
        false_positives = np.sum(satellite_flags & (~ground_truth))
        error_rate = false_positives / len(verified_indices)
        
        print(f"[P2 Logic] Verified Error Rate: {error_rate*100:.1f}%")

        # 2. Secondary Data Check (Validation Framework)
        # Check specific secondary data: Temperature & Wind
        is_safe_conditions = self.check_secondary_data(temp=18, wind_speed=15)
        
        # 3. Final Decision Logic
        if error_rate > 0.6 and is_safe_conditions:
            return "DECISION: CANCEL ALERT. High probability of Atmospheric Interference."
        elif error_rate > 0.6 and not is_safe_conditions:
             return "DECISION: DOWNGRADE ALERT. investigate sensors, conditions favorable for bloom."
        else:
            return "DECISION: MAINTAIN ALERT. Discrepancy is statistically insignificant."

    def check_secondary_data(self, temp, wind_speed):
        """
        Secondary Data Filter. 
        Algae needs high temp (>20C) and low wind (<10km/h).
        If temp is low or wind is high, it's likely NOT algae.
        """
        if temp < 20 or wind_speed > 10:
            return True # Conditions indicate 'Safe' (No biological bloom likely)
        return False
