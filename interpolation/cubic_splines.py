import numpy as np
from scipy.interpolate import CubicSpline

def interpolating_cs(strikes: np.ndarray,implied_volatility):
    """
    Interpolates the implied volatility accros strikes using cubic splines
    
    Parameters :
    strikes : the range of strikes accros which to interpolate
    implied_volatility : the implied volatilities to interpolate
    """
    return CubicSpline(strikes,implied_volatility)