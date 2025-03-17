import numpy as np
from scipy.stats import norm


def black_scholes_price(S: float, K: float, T: float, r: float, delta: float, sigma: float):
    """
    Computes the price of an european call option in the black scholes model.

    Parameters : 
    - S : spot price
    - K : strike
    - T : time to maturity
    - r : interest rate
    - delta : divident yield
    - sigma : volatility
    """
    d1 = (np.log(S/K)+(r-delta + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-delta*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
