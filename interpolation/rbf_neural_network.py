import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def interpolating_rbf(s_over_k_range: np.ndarray, implied_volatility: np.ndarray, num_centers: int, reg: float = 1e-1):
    """
    Interpolates the implied volatility across spot/strikes using a custom RBF network with 
    Ridge regularization and standardization of the input data.
    
    Parameters:
      - s_over_k_range: 1D array of values (e.g., S/K ratios).
      - implied_volatility: 2D array (n_simulations x n_points) of implied volatility values.
      - num_centers: Number of RBF neurons.
      - reg: Regularization parameter for ridge regression (default=1e-1).
      
    Returns:
      - predictors: numpy array of predictor functions (one per simulation).
    """

    scaler = StandardScaler()
    X_train = np.array(s_over_k_range).reshape(-1, 1)
    X_train_scaled = scaler.fit_transform(X_train)
    
    n, _ = implied_volatility.shape
    predictors = []
    
    for i in tqdm(range(n), desc="Training RBF models"):
        model = RBFNetwork(num_centers=num_centers, reg=reg)
        y_train = implied_volatility[i]
        model.fit(X_train_scaled, y_train)
        predictor = lambda X: model.predict(scaler.transform(np.array(X).reshape(-1, 1)))
        predictors.append(predictor)
        
    return np.array(predictors)


class RBFNetwork:
    def __init__(self, num_centers, sigma=None, reg=1e-1):
        """
        Initialize the RBF network.
        
        Parameters:
          - num_centers: Number of RBF neurons.
          - sigma: Fixed width for all neurons (if None, widths are estimated).
          - reg: Regularization parameter for Ridge regression.
        """
        self.num_centers = num_centers
        self.sigma = sigma
        self.reg = reg
        self.centers = None
        self.widths = None
        self.weights = None
        self.bias = None

    def _rbf(self, x, center, width):
        return np.exp(- (x - center)**2 / (width**2))

    def _compute_activations(self, X):
        """
        Compute RBF activations vectorized.
        X has shape (n_samples, 1) and centers has shape (num_centers, 1).
        Returns an array of shape (n_samples, num_centers).
        """
        diff = X - self.centers.reshape(1, -1)
        activations = np.exp(- (diff**2) / (self.widths.reshape(1, -1) ** 2))
        return activations

    def fit(self, X, y):
        """
        Train the RBF network on X (inputs) and y (targets).

        Steps:
          1. Determine centers using KMeans.
          2. Estimate widths as the mean distance to other centers (or fixed sigma).
          3. Compute RBF activations.
          4. Solve the ridge regression problem: (A^T A + reg * R) w = A^T y.
        """
        kmeans = KMeans(n_clusters=self.num_centers, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_.reshape(-1, 1)
        
        self.widths = np.zeros(self.num_centers)
        for i in range(self.num_centers):
            distances = np.abs(self.centers[i] - self.centers.flatten())
            distances = distances[distances > 0]
            self.widths[i] = np.mean(distances) if distances.size > 0 else 1.0
        if self.sigma is not None:
            self.widths = np.full(self.num_centers, self.sigma)
        
        A = self._compute_activations(X)  # shape: (n_samples, num_centers)
        A_bias = np.hstack([A, np.ones((A.shape[0], 1))])
        y
        n_features = A_bias.shape[1]
        R = np.eye(n_features)
        R[-1, -1] = 0.0 
        lhs = A_bias.T @ A_bias + self.reg * R
        rhs = A_bias.T @ y
        w = np.linalg.solve(lhs, rhs)
        
        self.weights = w[:-1]
        self.bias = w[-1]

    def predict(self, X):
        """
        Compute the network's output for inputs X.
        """
        A = self._compute_activations(X)
        A_bias = np.hstack([A, np.ones((A.shape[0], 1))])
        return A_bias.dot(np.concatenate([self.weights, [self.bias]]))

