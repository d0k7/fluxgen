import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

class GroundwaterModel:
    """
    Solves Problem 4: Case Study on Spatial-Mathematical Modeling.
    Simulates groundwater head dynamics with 4 interacting consumption sources.
    """
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        # Layers for different sources
        self.sources = {
            'agriculture': np.zeros((grid_size, grid_size)),
            'built_up': np.zeros((grid_size, grid_size)),
            'forest': np.zeros((grid_size, grid_size)),
            'water_body': np.zeros((grid_size, grid_size))
        }
        
    def add_source(self, name, center, radius, intensity):
        """Adds a consumption source (circular cluster) to the grid."""
        # Check if source name is valid
        if name not in self.sources:
            # map short names to keys if necessary, or just create new
            self.sources[name] = np.zeros((self.grid_size, self.grid_size))

        cx, cy = center
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        self.sources[name][mask] = intensity

    def compute_gradients(self):
        """
        Simulates 'Cell-to-Cell Interaction Mechanics'.
        """
        # 1. Aggregated Consumption (Local Stress)
        total_consumption = sum(self.sources.values())
        
        # 2. Propagate Effects (Gaussian Diffusion)
        # This simulates the physics of groundwater drawdown spreading
        self.head_gradient = gaussian_filter(total_consumption, sigma=3.0)
        
        return self.head_gradient

    def plot(self):
        """Generates the visualization requested in the Case Study."""
        if not hasattr(self, 'head_gradient'):
            self.compute_gradients()
            
        plt.figure(figsize=(8, 6))
        # cmap="RdYlGn_r" makes High Impact (Red) and Low Impact (Green)
        sns.heatmap(self.head_gradient, cmap="RdYlGn_r", annot=False)
        plt.title("Problem 4: Groundwater Interaction Gradients\n(Red = Critical Zone of Combined Stress)")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")
        plt.show()
