from pricing.black_scholes_pricing import black_scholes_price, rnd_bs
from pricing.heston_pricing import heston_prices, rnd_heston
from pricing.bakshi_pricing import bakshi_prices, rnd_bakshi
from models.black_scholes import monte_carlo_simulations_bs
from models.heston import monte_carlo_simulations_heston
from models.bakshi import monte_carlo_simulations_bakshi
from implied_vol.implied_vol import implied_vol

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def main():
    Smin = 30
    Smax = 180
    n = 100
    S = [Smin + (Smax-Smin)*i/n for i in range(n+1)]
    V = 0.2
    K = 100
    T = 1
    r = 0.05
    sigma = 0.20
    kappa = 1
    theta = 0.2
    lambd = 0  # market price of vol
    lambdajump = 0.5  # jump rate
    muj = -0.4
    sigmaj = 0.07
    rho = 0.1
    upper_bound = 1000

    error = 0.01
    maxiter = 50

    bs_iv = []
    h_iv = []
    ba_iv = []

    bs_prices = []
    h_prices = []
    ba_prices = []

    for i in range(n+1):
        bs_price = black_scholes_price(S[i], K, T, r, 0, sigma)
        bs_prices.append(bs_price)
        bs_iv.append(implied_vol(S[i], K, T, r, bs_price, error, maxiter))
        h_price = heston_prices(
            S[i], K, V, T, r, kappa, theta, lambd, rho, sigma, upper_bound)
        h_prices.append(h_price)
        h_iv.append(implied_vol(S[i], K, T, r, h_price, error, maxiter))
        ba_price = bakshi_prices(
            S[i], K, V, T, r, kappa, theta, lambdajump, rho, sigma, muj, sigmaj, upper_bound)
        ba_prices.append(ba_price)
        ba_iv.append(implied_vol(S[i], K, T, r, ba_price, error, maxiter))

    plt.plot(S, bs_iv, label="Black Scholes implied volatility")
    plt.plot(S, h_iv, label="Heston implied volatility")
    plt.plot(S, ba_iv, label="Bakshi implied volatility")
    plt.xlabel("S0")
    plt.ylabel("Implied volatility")
    plt.legend(loc = "upper left")
    plt.show()

    plt.plot(S, bs_prices, label="Black Scholes call prices")
    plt.plot(S, h_prices, label="Heston call prices")
    plt.plot(S, ba_prices, label="Bakshi call prices")
    plt.xlabel("S0")
    plt.ylabel("Call prices")
    plt.legend(loc = "upper left")
    plt.show()


if __name__ == "__main__":
    main()
