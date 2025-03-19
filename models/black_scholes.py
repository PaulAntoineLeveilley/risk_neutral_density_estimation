import numpy as np

def monte_carlo_simulations_bs(S,T,r, sigma,n_paths):
    """
    Computes monte carlo simulations for black scholes model

    Parameters :
    - S : spot price
    - T : time horizon
    - r : interest rate 
    - sigma : volatility
    - n_paths : number of paths to simulate 
    """
    N = np.random.normal(0,1,n_paths)
    return S*np.exp((r-(sigma**2)/2)*T+sigma*np.sqrt(T)*N)