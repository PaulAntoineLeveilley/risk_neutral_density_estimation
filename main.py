from data_generating_process import generate_call_prices, compute_implied_volatility
from interpolation.kernel_regression import interpolating_kernelreg
from interpolation.cubic_splines import interpolating_cs
from interpolation.rbf_neural_network import interpolating_rbf
from computation_rnd.computation_rnd import implied_volatility_to_call_prices,estimators_to_prediction
from config import (
    S_OVER_K_RANGE,
    NUMBER_OF_RND,COARSE_S_OVER_K_RANGE
)

import matplotlib.pyplot as plt
import numpy as np


def main():
    #model parameters
    S0 = 7616 #initial spot
    V0 = 0.01 #initial vol (for heston and bakshi)
    r = 0.045 #interest rate
    sigma = 0.13764 #vol
    sigmav = 0.2 #vol of vol (for heston and bakshi)
    delta = 0 #dividend rate (for black scholes only)
    kappa = 9 #speed of mean reversion (for heston and bakshi)
    theta = 0.0189 #longrun mean of vol  (for heston and bakshi)
    lambd = 0  # market price of vol (heston)
    lambdajump = 0.59  # jump rate (bakshi)
    muj = -0.05 #mean size of jumps (bakshi)
    sigmaj = 0.07 #standard deviation of jumps (bakshi)
    rho = 0.1 #correlation between brownian motions driving spot and vol (for heston and bakshi)

    #upper bound for integration
    upper_bound = 1000

    #other parameters
    T = 252 / 365 #time horizon for monter carlo simulations
    maturity = 63 / 365 #maturity of the calls
    model = "bakshi" #model to use

    #True if you need to compute the vega (for the weights of the cubic 
    # spline interpolation) 
    compute_vega = True

    #specifying model parameters
    model_parameters = {
        "S0": S0,
        "V0": V0,
        "r": r,
        "delta": delta,
        "sigma": sigma,
        "sigmav": sigmav,
        "kappa": kappa,
        "theta": theta,
        "rho": rho,
        "lambda_jump": lambdajump,
        "muj": muj,
        "sigmaj": sigmaj,
        "lambd": lambd,
    }

    #generate the spot prices and call prices attached to the grid  corresponding to each spot price
    #option to compute the vega of the calls, for cubic spline interpolation
    data = generate_call_prices(
        T, maturity, model, model_parameters, NUMBER_OF_RND, compute_vega, 100, upper_bound
    )

    spot_prices = data["spot_prices"]
    call_prices = data["call_prices"]
    vega = data["vega"]

    #transforming call prices into implied volatilities
    implied_volatility = compute_implied_volatility(
        call_prices, spot_prices, maturity, r
    )

    #need to reverse because the interpolation is done on the
    #implied vol vs S/K space
    implied_vol_reversed = implied_volatility[:,::-1]
    vega_reversed = vega[:,::-1]

    #fitting an estimation model to the implied volatilities

    # estimators  = interpolating_cs(S_OVER_K_RANGE,implied_vol_reversed,vega_reversed,lam = 0.9)
    # estimators = interpolating_kernelreg(S_OVER_K_RANGE,implied_vol_reversed)
    estimators = interpolating_rbf(S_OVER_K_RANGE,implied_vol_reversed,num_centers=5)

    #computing the predictions of the models on a grid
    predictions = estimators_to_prediction(estimators)

    #transforming the implied volatility prediction into call prices prediction
    estimated_call_prices = implied_volatility_to_call_prices(predictions,spot_prices,maturity,r)
    
    plt.scatter(S_OVER_K_RANGE,call_prices[0],color = 'b',label = 'True call prices', marker = 'o')
    plt.plot(COARSE_S_OVER_K_RANGE,estimated_call_prices[0],color = 'r',label = 'estimated call prices')
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
