import numpy as np
from scipy.interpolate import make_smoothing_spline

def interpolating_cs(s_over_k_range: np.ndarray,implied_volatility : np.ndarray,weights : np.ndarray = None,lam : float = None):
    """
    Interpolates the implied volatility accros spot/strikes using cubic splines
    
    Parameters :
    s_over_k_range : the range of spot/strikes accros which to interpolate
    implied_volatility : the implied volatilities to interpolate
    weights : the weight of each node of the interpolating spline
    lambd : smoothing parameter
    """
    n,_ = np.shape(implied_volatility)
    return np.array([make_smoothing_spline(s_over_k_range,implied_volatility[i],weights[i],lam) for i in range(n)])