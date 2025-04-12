from sklearn.cluster import KMeans
import numpy as np


def interpolating_rbf(
    s_over_k_range: np.ndarray, implied_volatility: np.ndarray, num_centers: int
):
    """
    Interpolates the implied volatility accros spot/strikes using cubic splines

    Parameters :
    s_over_k_range : the range of spot/strikes accros which to interpolate
    implied_volatility : the implied volatilities to interpolate
    """
    X_train = np.array(s_over_k_range).reshape(-1, 1)
    n, _ = np.shape(implied_volatility)
    predictors = []
    for i in range(n):
        model = RBFNetwork(num_centers=num_centers)
        y_train = implied_volatility[i]
        model.fit(X_train, y_train)
        predictors.append(model.predict)
    return np.array(predictors)


class RBFNetwork:
    def __init__(self, num_centers, sigma=None):
        """
        Initialize the RBF network.

        Parameters:
        - num_centers: Number of RBF neurons
        - sigma: Fixed width for all neurons (if None, the width will be estimated)
        """
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.widths = None
        self.weights = None
        self.bias = None

    def _rbf(self, x, center, width):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (width**2))

    def _compute_activations(self, X):
        n_samples = X.shape[0]
        activations = np.zeros((n_samples, self.num_centers))
        for i in range(n_samples):
            for j in range(self.num_centers):
                activations[i, j] = self._rbf(X[i], self.centers[j], self.widths[j])
        return activations

    def fit(self, X, y):
        """
        Trains the RBF network on X (inputs) and y (targets).

        Steps:
        1. Determination of centers using KMeans.
        2. Estimation of widths using the average of the distances between centers.
        3. Calculation of RBF activations.
        4. Linear regression (least squares) for the output weights and bias.
        """

        kmeans = KMeans(n_clusters=self.num_centers, random_state=0).fit(X)
        self.centers = kmeans.cluster_centers_

        self.widths = np.zeros(self.num_centers)
        for i in range(self.num_centers):
            distances = np.linalg.norm(self.centers[i] - self.centers, axis=1)
            distances = distances[distances > 0]
            if len(distances) > 0:
                self.widths[i] = np.mean(distances)
            else:
                self.widths[i] = 1.0
        if self.sigma is not None:
            self.widths = np.full(self.num_centers, self.sigma)

        A = self._compute_activations(X)  # (n_samples, num_centers)
        A_bias = np.hstack([A, np.ones((A.shape[0], 1))])

        w, residuals, rank, s = np.linalg.lstsq(A_bias, y, rcond=None)
        self.weights = w[:-1]
        self.bias = w[-1]

    def predict(self, X):
        A = self._compute_activations(X)
        A_bias = np.hstack([A, np.ones((A.shape[0], 1))])
        return A_bias.dot(np.concatenate([self.weights, [self.bias]]))
