import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class ReservoirEstimator:
    def __init__(self, domain_size=50):
        self.domain_size = domain_size
        self.x = np.linspace(0, 10, domain_size)
        self.y = np.linspace(0, 10, domain_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
    def simulate_data(self):
        # Synthetic Truth (Irregular Bowl Shape)
        # We create a deep bowl (approx -15 units deep)
        self.Z_true = -1 * (np.sin(self.X/3.5) * np.sin(self.Y/3.5) * 15)
        # Ensure no water above ground (cutoff at 0)
        self.Z_true = np.where(self.Z_true > 0, 0, self.Z_true)
        
        # Sample 65% (approx 40 points for this scale demo)
        rng = np.random.RandomState(42)
        idx_mask = rng.choice(self.domain_size**2, 40, replace=False)
        self.X_train = np.column_stack((self.X.ravel()[idx_mask], self.Y.ravel()[idx_mask]))
        self.y_train = self.Z_true.ravel()[idx_mask]
        
    def run_kriging_interpolation(self):
        """
        Uses Gaussian Process Regression (Kriging).
        """
        # Kernel: Constant * RBF (Radial Basis Function)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=2.0, length_scale_bounds=(1e-1, 10.0))
        
        # FIX: normalize_y=True helps the optimizer handle depth values > 1.0
        # alpha=0.1 adds a little noise tolerance to prevent overfitting/convergence errors
        gp = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, 
            normalize_y=True,
            alpha=0.1 
        )
        
        gp.fit(self.X_train, self.y_train)
        
        # Predict
        X_grid = np.column_stack((self.X.ravel(), self.Y.ravel()))
        y_pred, sigma = gp.predict(X_grid, return_std=True)
        Z_pred = y_pred.reshape(self.X.shape)
        
        # Volume Integration (Sum of depths * cell area)
        dx = 10 / self.domain_size
        dy = 10 / self.domain_size
        cell_area = dx * dy
        
        # We take abs() because depth is negative, we want positive volume magnitude
        volume = np.sum(np.abs(Z_pred)) * cell_area
        
        return Z_pred, sigma, volume
