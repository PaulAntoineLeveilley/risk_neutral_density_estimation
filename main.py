from pricing.black_scholes_pricing import black_scholes_price, rnd_bs
from pricing.heston_pricing import heston_prices, rnd_heston
from pricing.bakshi_pricing import bakshi_prices, rnd_bakshi
from models.black_scholes import monte_carlo_simulations_bs
from models.heston import monte_carlo_simulations_heston
from models.bakshi import monte_carlo_simulations_bakshi
from implied_vol.implied_vol import implied_vol
from data_generating_process import generate_call_prices, compute_implied_volatility
from interpolation.cubic_splines import interpolating_cs

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def main():
    S0= 100
    V = 0.04
    # K = 100
    # T = 1
    r = 0.05
    sigma = 0.20
    delta = 0
    kappa = 1
    theta = 0.2
    lambd = 0  # market price of vol
    lambdajump = 0.5  # jump rate
    muj = -0.1
    sigmaj = 0.07
    rho = 0.1
    upper_bound = 1000

    # error = 0.01
    # maxiter = 50
    T = 1
    std_error = 0.05
    maturity =  0.5
    model ="bakshi" 
    model_parameters = {"S0" : S0,"V0": V,"r" : r,"delta": delta, "sigma": sigma,"kappa": kappa,"theta": theta,"rho": rho, "lambda_jump": lambdajump,"muj": muj, "sigmaj":sigmaj,"lambd": lambd}
    call_prices,spot_prices = generate_call_prices(T,maturity, model, model_parameters,std_error,1,100,upper_bound)
    implied_volatility = compute_implied_volatility(call_prices, spot_prices,0.5,r)

    spot = spot_prices[0]
    implied_vol = implied_volatility[0]

    strike_range  = np.linspace(0.8*spot,1.2*spot,50)
    coarse_strike_range = np.linspace(0.8*spot,1.2*spot,200)

    cs = interpolating_cs(strike_range,implied_volatility[0])

    plt.plot(strike_range,call_prices[0],"o")
    plt.show()

    plt.plot(strike_range,implied_vol,"o")
    plt.plot(coarse_strike_range,cs(coarse_strike_range))
    plt.show()


if __name__ == "__main__":
    main()
