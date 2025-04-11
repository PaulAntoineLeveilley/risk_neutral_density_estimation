from pricing.black_scholes_pricing import black_scholes_price, rnd_bs
from pricing.heston_pricing import heston_prices, rnd_heston
from pricing.bakshi_pricing import bakshi_prices, rnd_bakshi
from models.black_scholes import monte_carlo_simulations_bs
from models.heston import monte_carlo_simulations_heston
from models.bakshi import monte_carlo_simulations_bakshi
from data_generating_process import generate_call_prices, compute_implied_volatility
from interpolation.cubic_splines import interpolating_cs
from config import (
    S_OVER_K_RANGE,
    RELATIVE_STRIKE_LOWER_BOUND,
    RELATIVE_STRIKE_UPPER_BOUND,
)

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import math


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

    T = 1
    maturity = 21 / 365
    model = "black_scholes"

    compute_vega = False

    lam = 0.1

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
    implied_vol_reversed = implied_volatility[0][::-1]

    plt.plot(call_prices[0], "o")
    plt.show()

    plt.plot(S_OVER_K_RANGE, implied_vol_reversed, "o")
    plt.show()


if __name__ == "__main__":
    main()
