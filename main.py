from data_generating_process import generate_call_prices, compute_implied_volatility
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
    model = "black_scholes"

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
        T, maturity, model, model_parameters, 1, compute_vega, 100, upper_bound
    )

    spot_prices = data["spot_prices"]
    call_prices = data["call_prices"]

    implied_volatility = compute_implied_volatility(
        call_prices, spot_prices, maturity, r
    )
    implied_vol_reversed = implied_volatility[:,::-1]


    interpolation_rbf = interpolating_rbf(S_OVER_K_RANGE,implied_vol_reversed,num_centers=5)

    
    
    X_train = np.array(S_OVER_K_RANGE).reshape(-1, 1)
    y_train = implied_vol_reversed[0]
    plt.figure(figsize=(10, 5))
    plt.plot(X_train, y_train, "o", label="Implied volatility (target)")
    plt.xlabel("S/K")
    plt.ylabel("Implied volatility")
    plt.title("Target data for training the RBF network")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    X_test = np.linspace(1/RELATIVE_STRIKE_UPPER_BOUND, 1/RELATIVE_STRIKE_LOWER_BOUND, 200).reshape(-1, 1)
    y_pred = interpolation_rbf[0](X_test)
    
    plt.figure(figsize=(10, 5))
    plt.plot(X_train, y_train, "o", label="Target implied volatility")
    plt.plot(X_test, y_pred, "-", label="RBF network estimation")
    plt.xlabel("S/K")
    plt.ylabel("Implied volatility")
    plt.title("RBF Network Estimation of Implied Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    


if __name__ == "__main__":
    main()
