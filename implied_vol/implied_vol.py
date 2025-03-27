import numpy as np
import math
from scipy.stats import norm


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
    d1 = (np.log(S/K)+(r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)


def f(S: float, K: float, T: float, r: float, c: float, sigma: float):
    """
    Function to equalize to zero

    Parameters :
    - S : spot price 
    - K : strike
    - T : maturity
    - r : interest rate
    - c : observed call price on the market
    - sigma : volatility
    """
    return black_scholes_price(S, K, T, r, sigma)-c


def derivative_of_f(S: float, K: float, T: float, r: float, sigma: float):
    """
    Computes the derivative of f (viewed as a function of sigma)

    Parameters :  
    - S : spot price 
    - K : strike
    - T : maturity
    - r : interest rate
    - sigma : volatility
    """
    d1 = (np.log(S/K)+(r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    return S*np.sqrt(T)*np.exp(-(d1**2)/2)/np.sqrt(2*math.pi)


def implied_vol(S: float, K: float, T, r: float, c: float, error: float, maxiter: int):
    """
    Computes the implied volatility of the call of strike K, maturity T, 
    interest rate r and market price c, up to an error of error.

    Parameters : 
    - S : spot price
    - K : strike
    - T : maturity
    - r : interest rate
    - c : market price
    - error : error 
    - maxiter : maximum number of iteration
    """
    sigma = 0.2
    iteration = 0
    while np.abs(f(S, K, T, r, c, sigma)) > error and iteration < maxiter:
        sigma -= f(S, K, T, r, c, sigma)/derivative_of_f(S, K, T, r, sigma)
        iteration += 1
    return sigma
