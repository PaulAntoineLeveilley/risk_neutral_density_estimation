import matplotlib.pyplot as plt
import numpy as np
from config import STRIKE_RANGE,COARSE_STRIKE_RANGE,COARSE_S_OVER_K_RANGE,S_OVER_K_RANGE

def plot_call_prices_and_estimation(call_prices : np.ndarray,estimated_call_price, args: dict):
    """
    Plots one sample of generated call prices and interpolation
    
    Parameters
    - call_prices : an 2d array containing generated call prices
    - estimated_call_price : a 2d array containing the interpolation of the call prices
    - args : a dict containing information for the plot
    """
    model = args["model"]
    maturity = args["maturity"]
    interpolation_method = args["interpolation_method"]
    plt.figure(figsize=(7,5))
    plt.scatter(STRIKE_RANGE,call_prices[0], marker = '+', color = 'g',label = 'Generated call prices')
    plt.plot(COARSE_STRIKE_RANGE,estimated_call_price[0],color = 'r',label = f'Interpolation method : {interpolation_method}')
    plt.xlabel('K/S')
    plt.title(f'Generated and estimated call prices, {model} model and maturity {int(252*maturity)} days, using {interpolation_method}.')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/call_prices/{model}_{int(252*maturity)}_days_call_prices_{interpolation_method}.png")
    plt.close()
    return None
    
def plot_implied_vol_and_estimation(implied_vol : np.ndarray,estimated_implied_vol : np.ndarray, args: dict):
    """
    Plots one sample of generated call prices and interpolation
    
    Parameters
    - impled_vol : an 2d array containing generated implied vol
    - estimated_impled_vol : a 2d arrafy containing the interpolation of the implied vol
    - args : a dict containing information for the plot
    """
    model = args["model"]
    maturity = args["maturity"]
    interpolation_method = args["interpolation_method"]
    plt.figure(figsize=(7,5))
    plt.scatter(S_OVER_K_RANGE,implied_vol[0], marker = '+', color = 'g',label = 'Generated implied volatility')
    plt.plot(COARSE_S_OVER_K_RANGE,estimated_implied_vol[0],color = 'r',label = f'Interpolation method : {interpolation_method}')
    plt.xlabel('S/K')
    plt.title(f'Generated and estimated implied volatility, {model} model and maturity {int(252*maturity)} days, using {interpolation_method}.')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results/implied_vol/{model}_{int(252*maturity)}_days_implied_vol_{interpolation_method}.png")
    plt.close()
    return None