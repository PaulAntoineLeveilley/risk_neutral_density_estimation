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
from config import (
    S_OVER_K_RANGE,
    NUMBER_OF_RND,
    NUM_SAMPLES,
    P_VALUE_TRESHOLD,
    MODEL_PARAMETERS,
    UPPER_BOUND,
    NUMBER_MC_STEPS,
    LAM,
)
from kolmogorov_smirnov_test.array_to_pdf import arrays_to_pdfs
from kolmogorov_smirnov_test.sample import sample_from_pdfs
from kolmogorov_smirnov_test.kolmogorov_test import perform_ks_test
from plots.plot import plots
from plots.boxplots import boxplot_pvalues
from plots.plot_call_prices import plot_call_prices_and_estimation, plot_implied_vol_and_estimation
from make_result_directory import make_result_directory

import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd


def full_estimating_procedure(
    T: float, maturities: list[float], model: str, interpolation_method: str
):
    """
    Generate data according to data generation procedure,
    performs risk neutral density estimation on the data
    and tests the goodness of fit of the estimated rnd.

    Parameters :
    - T : time horifzon for the monte carlo simulations
    - maturities : set of maturities
    - model : name of the model to use, one of 'black_scholes', 'heston', 'bakshi'.
    - interpolation_method : name of the interpolation method to use. One of
    'cubic_splines', 'kernel_regression', 'rbf_network'.
    """
    start = time.time()
    make_result_directory()
    match interpolation_method:
        case "cubic_splines":
            compute_vega = True
        case _:
            compute_vega = False
    list_p_values = []
    list_mean_p_values = []
    list_std_p_values = []
    list_percentage_rejection = []
    print("Generating the data : ")
    for maturity in maturities:
        print("Maturity : " + str(maturity))
        data = generate_call_prices(
            T,
            maturity,
            model,
            MODEL_PARAMETERS,
            NUMBER_OF_RND,
            compute_vega,
            NUMBER_MC_STEPS,
            UPPER_BOUND,
        )

        spot_prices = data["spot_prices"]
        vols = data["vols"]
        call_prices = data["call_prices"]

        if compute_vega:
            vega = data["vega"]

        print("computing implied volatilities")
        r = MODEL_PARAMETERS["r"]
        implied_volatility = compute_implied_volatility(
            call_prices, spot_prices, maturity, r
        )

        # need to reverse because the interpolation is done on the
        # implied vol vs S/K space
        implied_vol_reversed = implied_volatility[:, ::-1]
        if compute_vega:
            vega_reversed = vega[:, ::-1]

        print("fitting models")
        match interpolation_method:
            case "cubic_splines":
                estimators = interpolating_cs(
                    S_OVER_K_RANGE, implied_vol_reversed, vega_reversed, lam=LAM
                )
            case "kernel_regression":
                estimators = interpolating_kernelreg(
                    S_OVER_K_RANGE, implied_vol_reversed
                )
            case "rbf_network":
                estimators = interpolating_rbf(
                    S_OVER_K_RANGE, implied_vol_reversed, num_centers=5
                )

        print("Compute predictions")
        predictions = estimators_to_prediction(estimators)

        plot_implied_vol_and_estimation(implied_vol_reversed,predictions,{"model": model,"maturity" :maturity ,'interpolation_method': interpolation_method})

        print("Transforming predicted implied volatility into call prices")
        estimated_call_prices = implied_volatility_to_call_prices(
            predictions, spot_prices, maturity, r
        )

        plot_call_prices_and_estimation(call_prices,estimated_call_prices,{"model": model,"maturity" :maturity ,'interpolation_method': interpolation_method})

        print("Computing risk neutral densities from call prices")
        rnds = derive_rnd(estimated_call_prices, spot_prices, maturity, r)
        # compute associated theoretical rnd
        state_dict = {"spot_prices": spot_prices, "vols": vols}
        print("computing theoretical rnd")
        theoretical_rnds = compute_theoretical_rnd(
            state_dict, model, MODEL_PARAMETERS, maturity, UPPER_BOUND
        )
        print("Performing Kolmogorov Smirnov test")
        estimated_pdfs = arrays_to_pdfs(rnds)
        true_pdfs = arrays_to_pdfs(theoretical_rnds)
        print("Sampling")
        samples_estimated_rnd = sample_from_pdfs(
            estimated_pdfs, NUM_SAMPLES, spot_prices
        )
        samples_true_rnd = sample_from_pdfs(true_pdfs, NUM_SAMPLES, spot_prices)

        print("Performing the tests")
        p_values = perform_ks_test(samples_estimated_rnd, samples_true_rnd)

        list_p_values.append(p_values)

        mean_pvalues = np.mean(p_values)
        std_pvalues = np.std(p_values)
        percentage_rejected_H0 = np.mean(p_values < P_VALUE_TRESHOLD)

        list_mean_p_values.append(mean_pvalues)
        list_std_p_values.append(std_pvalues)
        list_percentage_rejection.append(percentage_rejected_H0)

        print("mean p-values : " + str(mean_pvalues))
        print("Standard deviation of p-values :" + str(std_pvalues))
        print("Percentage of rejection of H0 : " + str(percentage_rejected_H0))

        args = {"model": model, "maturity": maturity, "upper_bound": UPPER_BOUND, "interpolation_method": interpolation_method}
        plots(rnds, args)

    titles = ["T = " + f"{int(252*maturity)}" + " days" for maturity in maturities]
    args = {
        "titles": titles,
        "interpolation_method": interpolation_method,
        "model": model,
    }
    boxplot_pvalues(list_p_values, args)

    df = pd.DataFrame(
        {
            "Mean p-values": list_mean_p_values,
            "Standard deviation p-values": list_std_p_values,
        "Percentage of rejection of H0": list_percentage_rejection},
         index=maturities
    )
    df.to_csv(f"results/kolmogorov_test/kolmogorov_test_{model}_{interpolation_method}.csv")
    end = time.time()
    print("Total time for procedure :" + str(end - start) + " seconds")
    return None
