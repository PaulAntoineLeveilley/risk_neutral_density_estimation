import numpy as np
import math
from scipy.stats import norm


def black_scholes_price(
    S: float, K: float, T: float, r: float, delta: float, sigma: float
):
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
    d1 = (np.log(S / K) + (r - delta + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def rnd_bs(S: float, T: float, r: float, sigma: float, delta: float, x: float):
    """
    Computes the risk neutral density of the Black and Scholes model

    Parameters :
    - S : initial spot
    - T : time to maturity
    - r : interest rate
    - sigma : volatility
    - delta : dividend rate
    - x : point at which the density is computed
    """
    return np.exp(
        -((np.log(x / S) - T * (r - delta - (sigma**2) / 2)) ** 2) / (2 * T * sigma**2)
    ) / (x * np.sqrt(2 * math.pi * T * sigma**2))


def vega_bs(S: float, K: float, T: float, r: float, delta: float, sigma: float):
    """
    Computes the vega of an european call option in the black scholes model.

    Parameters :
    - S : spot price
    - K : strike
    - T : time to maturity
    - r : interest rate
    - delta : divident yield
    - sigma : volatility
    """
    d1 = (np.log(S / K) + (r - delta + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * np.exp(-(d1**2) / 2) / np.sqrt(2 * math.pi)
