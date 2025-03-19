from pricing.black_scholes_pricing import black_scholes_price
from pricing.heston_pricing import heston_prices
from pricing.bakshi_pricing import bakshi_prices
from models.black_scholes import monte_carlo_simulations_bs
from models.heston import monte_carlo_simulations_heston
from models.bakshi import monte_carlo_simulations_bakshi
import matplotlib.pyplot as plt
import numpy as np


def main():
    S = 100
    V = 0.2
    # K = 100
    T = 1
    r = 0.05
#     delta =0
    sigma = 0.2
    kappa = 1
    theta = 0.2
#     lambd =  0 #market price of vol
    lambdajump = 0.5 #jump rate
    muj = -0.4
    sigmaj = 0.07
    rho = 0.1
#     upper_bound = 1000
#     Kmin = 20
#     Kmax = 200
#     nk = 50
#     K= np.linspace(Kmin,Kmax,nk)
#     prices_h= np.zeros((nk))
#     prices_bs= np.zeros((nk))
#     prices_b = np.zeros((nk))
#     for i in range(nk):
#         prices_h[i] = heston_prices(S,K[i],V,T,r,kappa, theta, lambd, rho,sigma,upper_bound)
#         prices_bs[i] = black_scholes_price(S,K[i],T,r,delta,sigma)
#         prices_b[i] = bakshi_prices(S,K[i],V,T,r,kappa, theta,lambdajump,rho,sigma,muj,sigmaj,upper_bound)
#     plt.plot(K,prices_h, label=  'Heston')
#     plt.plot(K,prices_bs,label = 'Black Scholes')
#     plt.plot(K,prices_b,label = 'Bakshi')
#     plt.legend()
#     plt.xlabel('Strike')
#     plt.ylabel('Price')
#     plt.title('Call price vs strike')
#     plt.show()  
    spot_prices = monte_carlo_simulations_bakshi(S,V,T,r,sigma,kappa,theta,rho,lambdajump,muj,sigmaj,100,100)
    plt.hist(spot_prices)
    plt.show()
if __name__ == "__main__":
    main()
 