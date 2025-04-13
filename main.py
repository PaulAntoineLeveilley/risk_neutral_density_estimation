from data_generating_process import generate_call_prices, compute_implied_volatility
from interpolation.kernel_regression import interpolating_kernelreg
from interpolation.cubic_splines import interpolating_cs
from interpolation.rbf_neural_network import interpolating_rbf
from config import (
    S_OVER_K_RANGE,
    RELATIVE_STRIKE_LOWER_BOUND,
    RELATIVE_STRIKE_UPPER_BOUND,
)

import matplotlib.pyplot as plt
import numpy as np


def main():
    S0 = 7616
    V0 = 0.01
    r = 0.045
    sigma = 0.13764
    sigmav = 0.2
    delta = 0
    kappa = 9
    theta = 0.0189
    lambd = 0  # market price of vol
    lambdajump = 0.59  # jump rate
    muj = -0.05
    sigmaj = 0.07
    rho = 0.1

    upper_bound = 1000

    T = 252 / 365
    maturity = 63 / 365
    model = "bakshi"

    compute_vega = False

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
    data = generate_call_prices(
        T, maturity, model, model_parameters, 2, compute_vega, 100, upper_bound
    )

    spot_prices = data["spot_prices"]
    call_prices = data["call_prices"]

    implied_volatility = compute_implied_volatility(
        call_prices, spot_prices, maturity, r
    )
    implied_vol_reversed = implied_volatility[:,::-1]

    X_train = np.array(S_OVER_K_RANGE)
    y_train = implied_vol_reversed[0]

    estimator = interpolating_kernelreg(S_OVER_K_RANGE,implied_vol_reversed)[0]
    strike_range = np.linspace(
    RELATIVE_STRIKE_LOWER_BOUND, RELATIVE_STRIKE_UPPER_BOUND, 500
    )
    coarse_s_over_k_range = np.array([1 / K for K in strike_range[::-1]])
    interpolating_line = estimator(coarse_s_over_k_range)
    
    plt.figure(figsize=(10, 5))
    plt.plot(X_train, y_train, "o", label="Target implied volatility")
    plt.plot(coarse_s_over_k_range, interpolating_line, "-", label="kernel regression estimation")
    plt.xlabel("S/K")
    plt.ylabel("Implied volatility")
    plt.title("kernel regression Estimation of Implied Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
