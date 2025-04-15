from data_generating_process import generate_call_prices, compute_implied_volatility
from interpolation.kernel_regression import interpolating_kernelreg
from interpolation.cubic_splines import interpolating_cs
from interpolation.rbf_neural_network import interpolating_rbf
from computation_rnd.compute_call_price import (
    implied_volatility_to_call_prices,
    estimators_to_prediction,
)
from computation_rnd.derive_rnd import derive_rnd
from computation_rnd.compute_theoretical_rnd import compute_theoretical_rnd
from config import S_OVER_K_RANGE, NUMBER_OF_RND, STRIKE_RANGE, COARSE_STRIKE_RANGE, NUM_SAMPLES,P_VALUE_TRESHOLD
from kolmogorov_smirnov_test.array_to_pdf import  arrays_to_pdfs
from kolmogorov_smirnov_test.sample import sample_from_pdfs
from kolmogorov_smirnov_test.kolmogorov_test import perform_ks_test

from mean_estimated_rnd.average_rnd import compute_mean_and_confidence_interval_rnds 

import matplotlib.pyplot as plt
import numpy as np
import time 

def main():
    start = time.time()
    # model parameters
    S0 = 7616  # initial spot
    V0 = 0.01  # initial vol (for heston and bakshi)
    r = 0.045  # interest rate
    sigma = 0.13764  # vol
    sigmav = 0.2  # vol of vol (for heston and bakshi)
    delta = 0  # dividend rate (for black scholes only)
    kappa = 9  # speed of mean reversion (for heston and bakshi)
    theta = 0.0189  # longrun mean of vol  (for heston and bakshi)
    lambd = 0  # market price of vol (heston)
    lambdajump = 0.59  # jump rate (bakshi)
    muj = -0.05  # mean size of jumps (bakshi)
    sigmaj = 0.07  # standard deviation of jumps (bakshi)
    rho = 0.1  # correlation between brownian motions driving spot and vol (for heston and bakshi)

    # upper bound for integration
    upper_bound = 100

    # other parameters
    T = 252 / 365  # time horizon for monter carlo simulations
    maturity = 63 / 365  # maturity of the calls
    model = "bakshi"  # model to use

    # True if you need to compute the vega (for the weights of the cubic
    # spline interpolation)
    compute_vega = True

    # specifying model parameters
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

    # generate the spot prices and call prices attached to the grid  corresponding to each spot price
    # option to compute the vega of the calls, for cubic spline interpolation
    print("Generating the data : ")
    data = generate_call_prices(
        T,
        maturity,
        model,
        model_parameters,
        NUMBER_OF_RND,
        compute_vega,
        100,
        upper_bound,
    )

    spot_prices = data["spot_prices"]
    vols = data["vols"]
    call_prices = data["call_prices"]
    vega = data["vega"]

    # transforming call prices into implied volatilities
    print("computing implied volatilities")
    implied_volatility = compute_implied_volatility(
        call_prices, spot_prices, maturity, r
    )

    # need to reverse because the interpolation is done on the
    # implied vol vs S/K space
    implied_vol_reversed = implied_volatility[:, ::-1]
    vega_reversed = vega[:, ::-1]

    # fitting an estimation model to the implied volatilities
    print("fitting models")
    estimators = interpolating_cs(
        S_OVER_K_RANGE, implied_vol_reversed, vega_reversed, lam=0.5
    )
    # estimators = interpolating_kernelreg(S_OVER_K_RANGE,implied_vol_reversed)
    # estimators = interpolating_rbf(S_OVER_K_RANGE,implied_vol_reversed,num_centers=5)

    # computing the predictions of the models on a grid
    print("Compute predictions")
    predictions = estimators_to_prediction(estimators)

    # transforming the implied volatility prediction into call prices prediction
    print("Transforming predicted implied volatility into call prices")
    estimated_call_prices = implied_volatility_to_call_prices(
        predictions, spot_prices, maturity, r
    )

    # computing rnd from call prices
    print("Computing risk neutral densities from call prices")
    rnds = derive_rnd(estimated_call_prices, spot_prices, maturity, r)
    # compute statistic features of estimated rnds
    mean_rnd, confidence_upper, confidence_lower = compute_mean_and_confidence_interval_rnds(rnds)
    # compute associated theoretical rnd
    state_dict = {"spot_prices": spot_prices, "vols": vols}
    print("computing theoretical rnd")
    theoretical_rnds = compute_theoretical_rnd(
        state_dict, model, model_parameters, maturity, upper_bound
    )
    print("Performing Kolmogorov Smirnov test")
    # computing a valid rnd from estimated rnd
    estimated_pdfs = arrays_to_pdfs(rnds)    
    true_pdfs = arrays_to_pdfs(theoretical_rnds)
    print("Sampling")
    # sampling from estimated densities
    samples_estimated_rnd = sample_from_pdfs(estimated_pdfs,NUM_SAMPLES,spot_prices)
    samples_true_rnd = sample_from_pdfs(true_pdfs,NUM_SAMPLES,spot_prices)

    # performing the tests :
    print("Performing the tests")
    p_values = perform_ks_test(samples_estimated_rnd,samples_true_rnd)
    
    mean_pvalues = np.mean(p_values)
    std_pvalues = np.std(p_values)
    percentage_rejected_H0 = np.mean(p_values>P_VALUE_TRESHOLD)

    print("mean p-values : "+str(mean_pvalues))
    print("Standard deviation of p-values :"+str(std_pvalues))
    print("Percentage of rejection of H0 : "+ str(percentage_rejected_H0))
    

    end  = time.time()
    print("Total time for procedure :"+ str(end-start)+" seconds")

    for rnd in rnds:
        plt.plot(
            S0 * COARSE_STRIKE_RANGE, rnd,alpha = 0.1
        )
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        theoretical_rnds[0],
        color="b",linewidth = 1,
        label="theoretical rnd",
    )
    
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        mean_rnd,
        color="r",linewidth = 1,
        label="mean estimated rnd",
    )
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        confidence_upper,
        color="g",linewidth = 1,label = '95% confidence interval estimated rnd'
    )
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        confidence_lower,
        color="g",linewidth = 1
    )

    plt.grid(True)
    plt.xlabel("S")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
