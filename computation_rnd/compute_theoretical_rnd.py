import numpy as np
from config import COARSE_STRIKE_RANGE,SIZE_COARSE_RANGE
from pricing.black_scholes_pricing import rnd_bs

def compute_theoretical_rnd(spot_prices :np.ndarray,model : str,model_parameters : dict,maturity : float):
    """
    Computes theoretical risk neutral density associated to each spot price.
    
    Parameters :
    - spot_prices : array that contains the spot values
    - model : the model name
    - model_parameters : dict containing the parameters
    - maturity : maturity of the calls
    """
    match model:
        case "black_scholes":
            return compute_theoretical_rnd_bs(spot_prices,model_parameters,maturity)
        case "heston":
            return compute_theoretical_rnd_heston(spot_prices, model_parameters,maturity)
        case "bakshi":
            return compute_theoretical_rnd_bakshi(spot_prices, model_parameters,maturity)
    
def compute_theoretical_rnd_bs(spot_prices : np.ndarray,model_parameters : dict, maturity : float):
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
    theoretical_rnds = np.zeros((n,SIZE_COARSE_RANGE))
    for i in range(n):
        for j in range(SIZE_COARSE_RANGE):
            spot = spot_prices[i]
            theoretical_rnds[i,j] = rnd_bs(spot,maturity,r,sigma,delta,spot * COARSE_STRIKE_RANGE[j])
    return theoretical_rnds

def compute_theoretical_rnd_heston(spot_prices : np.ndarray,model_parameters : dict, maturity : float):
    """
    Computes theoretical heston risk neutral
    density associated to the spot prices.
     
    Parameters :
    - spot_prices : array that contain the spot values
    - model_parameters : dict containing the model parameters
    - maturity : maturity of the calls
    """
    return None

def compute_theoretical_rnd_bakshi(spot_prices : np.ndarray,model_parameters : dict, maturity : float):
    """
    Computes theoretical bakshi risk neutral
    density associated to the spot prices.
     
    Parameters :
    - spot_prices : array that contain the spot values
    - model_parameters : dict containing the model parameters
    - maturity : maturity of the calls
    """
    return None