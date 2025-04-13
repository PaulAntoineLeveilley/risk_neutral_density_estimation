import numpy as np
import math
from py_vollib.black_scholes.implied_volatility import implied_volatility

from config import (
    RELATIVE_STRIKE_UPPER_BOUND,
    RELATIVE_STRIKE_LOWER_BOUND,
    NUMBER_OF_STRIKES,
    STD_ERROR,
)

from models.black_scholes import monte_carlo_simulations_bs
from pricing.black_scholes_pricing import black_scholes_price, vega_bs

from models.heston import monte_carlo_simulations_heston
from pricing.heston_pricing import heston_prices

from models.bakshi import monte_carlo_simulations_bakshi
from pricing.bakshi_pricing import bakshi_prices


def generate_call_prices(
    T: float,
    maturity: float,
    model: str,
    model_parameters: dict,
    n: int,
    compute_vega: bool,
    n_steps: int = 100,
    upper_bound: int = 1000,
):
    """
    Generate a sample of 50 call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model : model to use
    - model_parameters : parameters of the model
    - n : number of set of call prices to generate
    - compute_vega : True if one needs to compute the vega of the option (for spline interpolation)
    - n_steps : number of steps for monte carlo simulations
    - upper_bound : upper bound for the integral computation
    """
    if model == "black_scholes":
        return generate_call_prices_bs(T, maturity, model_parameters, n, compute_vega)
    elif model == "heston":
        return generate_call_prices_heston(
            T, maturity, model_parameters, n, n_steps, upper_bound, compute_vega
        )
    elif model == "bakshi":
        return generate_call_prices_bakshi(
            T, maturity, model_parameters, n, n_steps, upper_bound, compute_vega
        )
    else:
        print("Model must be one of 'black_scholes','heston', 'bakshi'.")
        return 0


def generate_call_prices_bs(
    T: float, maturity: float, model_parameters: dict, n: int, compute_vega: bool
):
    """
    Generates a sample of 50 call prices using black scholes model to simulate
    spot price and using black scholes formula to compute call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model_parameters : parameters of the model
    - n : number of set of call prices to generate
    - compute_vega : True if one needs to compute the vega of the option (for spline interpolation)
    """
    S0 = model_parameters["S0"]
    r = model_parameters["r"]
    delta = model_parameters["delta"]
    sigma = model_parameters["sigma"]
    spot_prices = monte_carlo_simulations_bs(S0, T, r, sigma, n)
    call_prices = np.zeros((n, NUMBER_OF_STRIKES))
    if compute_vega:
        vega = np.zeros((n, NUMBER_OF_STRIKES))
    for i, spot in enumerate(spot_prices):
        strike_range = np.linspace(
            RELATIVE_STRIKE_LOWER_BOUND * spot,
            RELATIVE_STRIKE_UPPER_BOUND * spot,
            NUMBER_OF_STRIKES,
        )
        for j, strike in enumerate(strike_range):
            call_price = black_scholes_price(spot, strike, maturity, r, delta, sigma)
            intrinsic_value = np.maximum(spot - strike * np.exp(-r * maturity), 0)
            noise = np.random.normal(0, STD_ERROR * call_price)
            while not call_price + noise > intrinsic_value:
                noise = np.random.normal(0, STD_ERROR * call_price)
            call_prices[i, j] = call_price + noise
            if compute_vega:
                vega[i, j] = vega_bs(spot, strike, maturity, r, delta, sigma)
    if compute_vega:
        return {
            "call_prices": call_prices,
            "spot_prices": spot_prices,
            "vols": None,
            "vega": vega,
        }
    else:
        return {"call_prices": call_prices, "spot_prices": spot_prices, "vols": None}


def generate_call_prices_heston(
    T: float,
    maturity: float,
    model_parameters: dict,
    n: int,
    n_steps: int,
    upper_bound: float,
    compute_vega: bool,
):
    """
    Generates a sample of 50 call prices using heston model to simulate
    spot price and using heston formula to compute call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model_parameters : parameters of the model
    - n : number of set of call prices to generate
    - n_steps : number of steps for monte carlo simulations
    - upper_bound : upper bound for the integral computation
    - compute_vega : True if one needs to compute the vega of the option (for spline interpolation)
    """
    S0 = model_parameters["S0"]
    V0 = model_parameters["V0"]
    r = model_parameters["r"]
    sigma = model_parameters["sigmav"]
    kappa = model_parameters["kappa"]
    theta = model_parameters["theta"]
    rho = model_parameters["rho"]
    lambd = model_parameters["lambd"]
    spot_prices, vols = monte_carlo_simulations_heston(
        S0, V0, T, r, sigma, kappa, theta, rho, n_steps, n
    )
    call_prices = np.zeros((n, NUMBER_OF_STRIKES))
    if compute_vega:
        vega = np.zeros((n, NUMBER_OF_STRIKES))
    for i in range(n):
        spot = spot_prices[i]
        vol = vols[i]
        strike_range = np.linspace(
            RELATIVE_STRIKE_LOWER_BOUND * spot,
            RELATIVE_STRIKE_UPPER_BOUND * spot,
            NUMBER_OF_STRIKES,
        )
        for j, strike in enumerate(strike_range):
            call_price = heston_prices(
                spot,
                strike,
                vol,
                maturity,
                r,
                kappa,
                theta,
                lambd,
                rho,
                sigma,
                upper_bound,
            )
            intrinsic_value = np.maximum(spot - strike * np.exp(-r * maturity), 0)
            noise = np.random.normal(0, STD_ERROR * call_price)
            while not call_price + noise > intrinsic_value:
                noise = np.random.normal(0, STD_ERROR * call_price)
            call_prices[i, j] = call_price + noise
            if compute_vega:
                vega[i, j] = vega_bs(spot, strike, maturity, r, 0, sigma)
    if compute_vega:
        return {
            "call_prices": call_prices,
            "spot_prices": spot_prices,
            "vols": vols,
            "vega": vega,
        }
    else:
        return {"call_prices": call_prices, "spot_prices": spot_prices, "vols": vols}


def generate_call_prices_bakshi(
    T: float,
    maturity: float,
    model_parameters: dict,
    n: int,
    n_steps: int,
    upper_bound: float,
    compute_vega: bool,
):
    """
    Generates a sample of 50 call prices using bakshi model to simulate
    spot price and using bakshi formula to compute call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model_parameters : parameters of the model
    - n : number of set of call prices to generate
    - n_steps : number of steps for monte carlo simulations
    - upper_bound : upper bound for the integral computation
    - compute_vega : True if one needs to compute the vega of the option (for spline interpolation)
    """
    S0 = model_parameters["S0"]
    V0 = model_parameters["V0"]
    r = model_parameters["r"]
    sigma = model_parameters["sigmav"]
    kappa = model_parameters["kappa"]
    theta = model_parameters["theta"]
    rho = model_parameters["rho"]
    lambda_jump = model_parameters["lambda_jump"]
    muj = model_parameters["muj"]
    sigmaj = model_parameters["sigmaj"]
    spot_prices, vols = monte_carlo_simulations_bakshi(
        S0, V0, T, r, sigma, kappa, theta, rho, lambda_jump, muj, sigmaj, n_steps, n
    )
    call_prices = np.zeros((n, NUMBER_OF_STRIKES))
    if compute_vega:
        vega = np.zeros((n, NUMBER_OF_STRIKES))
    for i in range(n):
        spot = spot_prices[i]
        vol = vols[i]
        strike_range = np.linspace(
            RELATIVE_STRIKE_LOWER_BOUND * spot,
            RELATIVE_STRIKE_UPPER_BOUND * spot,
            NUMBER_OF_STRIKES,
        )
        for j, strike in enumerate(strike_range):
            call_price = bakshi_prices(
                spot,
                strike,
                vol,
                maturity,
                r,
                kappa,
                theta,
                lambda_jump,
                rho,
                sigma,
                muj,
                sigmaj,
                upper_bound,
            )
            intrinsic_value = np.maximum(spot - strike * np.exp(-r * maturity), 0)
            noise = np.random.normal(0, STD_ERROR * call_price)
            while not call_price + noise > intrinsic_value:
                noise = np.random.normal(0, STD_ERROR * call_price)
            call_prices[i, j] = call_price + noise
            if compute_vega:
                vega[i, j] = vega_bs(spot, strike, maturity, r, 0, sigma)
    if compute_vega:
        return {
            "call_prices": call_prices,
            "spot_prices": spot_prices,
            "vols": vols,
            "vega": vega,
        }
    else:
        return {"call_prices": call_prices, "spot_prices": spot_prices, "vols": vols}


def compute_implied_volatility(call_prices, spot_prices, maturity: float, r: float):
    """
    Computes implied volatility

    Parameters :
    - call_prices : call prices from which to compute the implied volatility
    - spot_prices : initial spot associated to each call price
    - maturity : maturity of the option
    - r : interest rate
    """
    _, p = np.shape(call_prices)
    implied_volatility_array = np.empty_like(call_prices)
    for i, spot in enumerate(spot_prices):
        for j, call_price in enumerate(call_prices[i]):
            strike = RELATIVE_STRIKE_LOWER_BOUND * spot + (
                (RELATIVE_STRIKE_UPPER_BOUND - RELATIVE_STRIKE_LOWER_BOUND) * spot
            ) * j / (p - 1)
            implied_volatility_array[i, j] = implied_volatility(
                call_price, spot, strike, maturity, r, flag="c"
            )
    return implied_volatility_array
