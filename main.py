from pricing.black_scholes_pricing import black_scholes_price, rnd_bs
from pricing.heston_pricing import heston_prices, rnd_heston
from pricing.bakshi_pricing import bakshi_prices, rnd_bakshi
from models.black_scholes import monte_carlo_simulations_bs
from models.heston import monte_carlo_simulations_heston
from models.bakshi import monte_carlo_simulations_bakshi

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def main():
    S = 100
    V = 0.2
    K = 100
    T = 1
    r = 0.05
    delta = 0
    sigma = 0.2
    kappa = 1
    theta = 0.2
    lambd = 0  # market price of vol
    lambdajump = 0.5  # jump rate
    muj = -0.4
    sigmaj = 0.07
    rho = 0.1
    upper_bound = 1000

    xmax = 500
    X = np.linspace(1, xmax, xmax)

    def rndh(x): return rnd_heston(S, V, T, r, kappa,
                                   theta, lambd, rho, sigma, upper_bound, x)

    def rndb(x): return rnd_bakshi(S, V, T, r, kappa, theta,
                                   lambdajump, rho, sigma, muj, sigmaj, upper_bound, x)

    def rndbs(x): return rnd_bs(S, T, r, sigma, delta, x)
    Y = np.zeros((xmax))
    Z = np.zeros((xmax))
    A = np.zeros((xmax))
    for i in range(1, xmax+1):
        Y[i-1] = rndh(i)
        Z[i-1] = rndbs(i)
        A[i-1] = rndb(i)
    plt.plot(X, Y, label='heston')
    plt.plot(X, Z, label="bs")
    plt.plot(X, A, label="bakshi")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
