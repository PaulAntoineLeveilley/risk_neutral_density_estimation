import numpy as np
from config import COARSE_STRIKE_RANGE

def derive_rnd(call_prices : np.ndarray,spot_prices: np.ndarray,maturity : float,r : float):
    """
    Derives the call prices twice with respect to the
    strike and multiplies by discount factor to get 
    risk neutral density.

    parameters : 
    - call_prices : an array containing the call prices
    - spot_prices : array containing the spot price corresponding
    to each call prices set.
    - maturity : maturity of the calls
    - r : interest rate
    """
    #need to divide by spot because call prices are
    #given as a function of K/S.
    first_derivative = np.gradient(call_prices,COARSE_STRIKE_RANGE,axis=1)/spot_prices[:, np.newaxis]
    second_derivative = np.gradient(first_derivative,COARSE_STRIKE_RANGE,axis=1)/spot_prices[:, np.newaxis]
    return np.exp(r*maturity)*second_derivative
    