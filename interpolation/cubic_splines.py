import numpy as np
from scipy.interpolate import make_smoothing_spline

def interpolating_cs(strikes: np.ndarray,implied_volatility : np.ndarray,weights : np.ndarray = None,lam : float = None):
    """
    Interpolates the implied volatility accros strikes using cubic splines
    
    Parameters :
    strikes : the range of strikes accros which to interpolate
    implied_volatility : the implied volatilities to interpolate
    weights : the weight of each node of the interpolating spline
    lambd : smoothing parameter
    """
    return make_smoothing_spline(strikes,implied_volatility,weights,lam)