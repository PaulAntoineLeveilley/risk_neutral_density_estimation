import numpy as np
from scipy.stats import norm
from config import (
    RELATIVE_STRIKE_LOWER_BOUND,
    RELATIVE_STRIKE_UPPER_BOUND,
    COARSE_S_OVER_K_RANGE,
    NUMBER_OF_RND,
)


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float):
    """
    Computes the price of an european call option in the black scholes model.

    Parameters :
    - S : spot price
    - K : strike
    - T : time to maturity
    - r : interest rate
    - sigma : volatility
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def estimators_to_prediction(estimators: np.ndarray):
    """
    Takes as input an array of models and outputs the prediction
    of theses models.

    Parameters :
    - estimators : an array containing functions
    """
    return np.array(
        [estimators[i](COARSE_S_OVER_K_RANGE) for i in range(NUMBER_OF_RND)]
    )


def implied_volatility_to_call_prices(
    implied_volatility: np.ndarray,
    spot_prices: np.ndarray,
    maturity: float,
    r: float,
):
    """
    Transform implied volatility array into corresponding call prices
    by applying black scholes formula.

    Parameters :
    - implied_volatility : a 2 dimensional array containing implied volatilities.
    (take care that implied volatility here must be a given as a function
     of S/K at points K evenly spaced between S*RELATIVE_STRIKE_LOWER_BOUND
     and S*RELATIVE_STRIKE_UPPER_BOUND)
    - spot_prices : 1 dimensional array containing spot prices corresponding
    to the spot price associated to the i th line of implied_volatility
    - maturity : mafurity of the calls
    - r : interest rate
    """
    n, p = np.shape(implied_volatility)
    call_prices = np.empty_like(implied_volatility)
    for i in range(n):
        spot = spot_prices[i]
        strike_range = np.linspace(
            spot * RELATIVE_STRIKE_LOWER_BOUND, spot * RELATIVE_STRIKE_UPPER_BOUND, p
        )[::-1]
        # need to reverse because implied_volatility
        # is given as a function of S/K
        for j in range(p):
            strike = strike_range[j]
            sigma = implied_volatility[i, j]
            call_prices[i, j] = black_scholes_price(spot, strike, maturity, r, sigma)
    # outputs call prices as a function of K at points K evenly spaced between S*RELATIVE_STRIKE_LOWER_BOUND
    # and S*RELATIVE_STRIKE_UPPER_BOUND
    return call_prices[:, ::-1]
