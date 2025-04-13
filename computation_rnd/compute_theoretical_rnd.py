import numpy as np
from config import COARSE_STRIKE_RANGE, SIZE_COARSE_RANGE
from pricing.black_scholes_pricing import rnd_bs
from pricing.heston_pricing import rnd_heston
from pricing.bakshi_pricing import rnd_bakshi


def compute_theoretical_rnd(
    state_dict: dict,
    model: str,
    model_parameters: dict,
    maturity: float,
    upper_bound: float,
):
    """
    Computes theoretical risk neutral density associated to each spot price.

    Parameters :
    - state_dict : dict containing arrays containing spot values and
    volatity values
    - model : the model name
    - model_parameters : dict containing the parameters
    - maturity : maturity of the calls
    - upper_bound : upper bound for integal computations
    """
    spot_prices = state_dict["spot_prices"]
    vols = state_dict["vols"]
    match model:
        case "black_scholes":
            return compute_theoretical_rnd_bs(spot_prices, model_parameters, maturity)
        case "heston":
            return compute_theoretical_rnd_heston(
                spot_prices, vols, model_parameters, maturity, upper_bound
            )
        case "bakshi":
            return compute_theoretical_rnd_bakshi(
                spot_prices, vols, model_parameters, maturity, upper_bound
            )


def compute_theoretical_rnd_bs(
    spot_prices: np.ndarray, model_parameters: dict, maturity: float
):
    """
    Computes theoretical black scholes risk neutral
    density associated to the spot prices.

    Parameters :
    - spot_prices : array that contain the spot values
    - model_parameters : dict containing the model parameters
    - maturity : maturity of the calls
    """
    r = model_parameters["r"]
    delta = model_parameters["delta"]
    sigma = model_parameters["sigma"]
    n = np.shape(spot_prices)[0]
    theoretical_rnds = np.zeros((n, SIZE_COARSE_RANGE))
    for i in range(n):
        for j in range(SIZE_COARSE_RANGE):
            spot = spot_prices[i]
            theoretical_rnds[i, j] = rnd_bs(
                spot, maturity, r, sigma, delta, spot * COARSE_STRIKE_RANGE[j]
            )
    return theoretical_rnds


def compute_theoretical_rnd_heston(
    spot_prices: np.ndarray,
    vols: np.ndarray,
    model_parameters: dict,
    maturity: float,
    upper_bound: float,
):
    """
    Computes theoretical heston risk neutral
    density associated to the spot prices.

    Parameters :
    - spot_prices : array that contain the spot values
    - vols : array containing the vols values
    - model_parameters : dict containing the model parameters
    - maturity : maturity of the calls
    - upper_bound : upper bound for integal computations
    """
    r = model_parameters["r"]
    sigma = model_parameters["sigmav"]
    kappa = model_parameters["kappa"]
    theta = model_parameters["theta"]
    rho = model_parameters["rho"]
    lambd = model_parameters["lambd"]
    n = np.shape(spot_prices)[0]
    theoretical_rnds = np.zeros((n, SIZE_COARSE_RANGE))
    for i in range(n):
        for j in range(SIZE_COARSE_RANGE):
            spot = spot_prices[i]
            vol = vols[i]
            theoretical_rnds[i, j] = rnd_heston(
                spot,
                vol,
                maturity,
                r,
                kappa,
                theta,
                lambd,
                rho,
                sigma,
                upper_bound,
                spot * COARSE_STRIKE_RANGE[j],
            )
    return theoretical_rnds


def compute_theoretical_rnd_bakshi(
    spot_prices: np.ndarray,
    vols: np.ndarray,
    model_parameters: dict,
    maturity: float,
    upper_bound: float,
):
    """
    Computes theoretical bakshi risk neutral
    density associated to the spot prices.

    Parameters :
    - spot_prices : array that contain the spot values
    - vols : array containing the vols values
    - model_parameters : dict containing the model parameters
    - maturity : maturity of the calls
    - upper_bound : upper bound for integal computations
    """
    r = model_parameters["r"]
    sigma = model_parameters["sigmav"]
    kappa = model_parameters["kappa"]
    theta = model_parameters["theta"]
    rho = model_parameters["rho"]
    lambda_jump = model_parameters["lambda_jump"]
    muj = model_parameters["muj"]
    sigmaj = model_parameters["sigmaj"]
    n = np.shape(spot_prices)[0]
    theoretical_rnds = np.zeros((n, SIZE_COARSE_RANGE))
    for i in range(n):
        for j in range(SIZE_COARSE_RANGE):
            spot = spot_prices[i]
            vol = vols[i]
            theoretical_rnds[i, j] = rnd_bakshi(
                spot,
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
                spot * COARSE_STRIKE_RANGE[j],
            )
    return theoretical_rnds
    return None
