import numpy as np
import math


def interpolating_kernelreg(s_over_k_range: np.ndarray, implied_volatility: np.ndarray):
    """
    Interpolates the implied volatility accros spot/strikes using kernel
    regression

    Parameters :
    s_over_k_range : the range of spot/strikes accros which to interpolate
    implied_volatility : the implied volatilities to interpolate
    """
    n, _ = np.shape(implied_volatility)
    return np.array(
        [apply_NW_estimator(s_over_k_range, implied_volatility[i]) for i in range(n)]
    )


def gaussian_kernel(u: float, h: float):
    """
    Gaussian kernel which will be used for the kernel regression

    Parameters:
    - u : point at which it is evaluated
    - h : bandwidth
    """
    return 1 / np.sqrt(2 * math.pi) * np.exp(-((u / h) ** 2) / 2)


def NW_estimator(X: np.array, Y: np.array, h: float):
    """
    Computes the Nadaraya-Watson estimator

    Parameters:
    - X : explanatory variable
    - Y : target variable
    - h : bandwidth
    """
    return lambda x: sum(
        gaussian_kernel(x - X[i], h) * Y[i] for i in range(len(X))
    ) / sum(gaussian_kernel(x - X[i], h) for i in range(len(X)))


def NW_estimator_for_fast(X: np.array, Y: np.array, h: float):
    """
    Computes the Nadaraya-Watson estimator in a compatible format for the faster cross-validation

    Parameters:
    - X : explanatory variable
    - Y : target variable
    - h : bandwidth
    """
    weight_denom = lambda x: sum(gaussian_kernel(x - X[i], h) for i in range(len(X)))
    NW_estimator_fct = lambda x: sum(
        gaussian_kernel(x - X[i], h) * Y[i] for i in range(len(X))
    ) / weight_denom(x)

    return weight_denom, NW_estimator_fct


def find_optimal_h_CV(X: np.array, Y: np.array):
    """
    Finds the optimal bandwidth for given data using cross-validation

    Parameters:
    Parameters:
    - X : explanatory variable
    - Y : target variable
    """
    h_min = 1 / 20 * (X.max() - X.min())
    h_max = X.max() - X.min()
    ### Defining the domain on which we search for the optimal value
    h_candidates = np.linspace(h_min, h_max, 20)
    ### Saving the value of the objective function for every candidate h
    CV_err_h = np.empty_like(h_candidates)

    for i, h_iter in enumerate(h_candidates):
        CV_err_iter = np.empty_like(X)
        for j in range(len(X)):
            ### Validation set
            X_val = X[j]
            Y_val = Y[j]

            ### Training set
            X_tr = np.delete(X, j)
            Y_tr = np.delete(Y, j)

            Y_val_predict = NW_estimator(X_tr, Y_tr, h_iter)
            CV_err_iter[j] = (Y_val - Y_val_predict(X_val)) ** 2

        CV_err_h[i] = np.mean(CV_err_iter)

    return h_candidates[np.argmin(CV_err_h)]


def find_optimal_h_CV_fast(X: np.array, Y: np.array):
    """
    Finds the optimal bandwidth for given data using cross-validation but in a less computational way

    Parameters:
    Parameters:
    - X : explanatory variable
    - Y : target variable
    """

    h_min = 1 / 20 * (X.max() - X.min())
    h_max = X.max() - X.min()
    ### Defining the domain on which we search for the optimal value
    h_candidates = np.linspace(h_min, h_max, 20)
    ### Saving the value of the objective function for every candidate h
    CV_err_h = np.empty_like(h_candidates)

    for i, h_iter in enumerate(h_candidates):

        weight_denom, NW_estimator_use = NW_estimator_for_fast(X, Y, h_iter)

        CV_err_h[i] = np.mean(
            (
                (Y - NW_estimator_use(X))
                / (1 - 1 / np.sqrt(2 * math.pi) / weight_denom(X))
            )
            ** 2
        )

    return h_candidates[np.argmin(CV_err_h)]


def apply_NW_estimator(X: np.array, Y: np.array):
    """
    Finds optimal value of h with faster cross-validation for the given data, "calibrates" the estimator and returns regression on X

    Parameters:
    - X : explanatory variable
    - Y : target variable
    """
    h_opt_CV = find_optimal_h_CV_fast(X, Y)
    return NW_estimator(X, Y, h_opt_CV)
