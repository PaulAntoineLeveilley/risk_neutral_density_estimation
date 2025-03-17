from pricing.black_scholes_pricing import black_scholes_price
from pricing.heston_pricing import heston_prices
import matplotlib.pyplot as plt
import numpy as np


def main():
    S = 100
    V = 0.2
    # K = 100
    T = 1
    r = 0.05
    delta =0
    sigma = 0.2
    kappa = 1
    theta = 0.2
    lambd =  0
    rho = 0.1
    upper_bound = 1000
    Kmin = 20
    Kmax = 200
    nk = 50
    K= np.linspace(Kmin,Kmax,nk)
    prices_h= np.zeros((nk))
    prices_bs= np.zeros((nk))
    for i in range(nk):
        prices_h[i] = heston_prices(S,K[i],V,T,r,kappa, theta, lambd, rho,sigma,upper_bound)
        prices_bs[i] = black_scholes_price(S,K[i],T,r,delta,sigma)
    plt.plot(K,prices_h, label=  'Heston')
    plt.plot(K,prices_bs,label = 'Black Scholes')
    plt.legend()
    plt.xlabel('Strike')
    plt.ylabel('Price')
    plt.title('Call price vs strike')
    plt.show()
    
if __name__ == "__main__":
    main()
 