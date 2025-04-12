import numpy as np


def monte_carlo_simulations_bakshi(S0, V0, T, r, sigma, kappa, theta, rho, lambd, muj, sigmaj, n_steps, n_paths):
    """
    Computes monte carlo simulations for bakshi model

    Parameters : 
    - S0 : spot price
    - V0 : initial volatility
    - T : time horizon
    - r : interest rate
    - sigma : volatility of volatility
    - kappa : speed of mean reversion
    - theta : long run mean of volatility
    - rho : correlation
    - lambd : jump rate
    - muj : jump mean
    - sigmaj : jump standard deviation
    - n_steps : number of time steps for euler scheme
    - n_paths : number of paths to simulate 
    """
    dt = T/n_steps
    dW1 = np.sqrt(dt)*np.random.normal(0, 1, (n_steps, n_paths))
    dZ = np.sqrt(dt)*np.random.normal(0, 1, (n_steps, n_paths))
    dW2 = rho*dW1+np.sqrt(1-rho**2)*dZ
    S = S0*np.ones((n_paths))
    V = V0*np.ones((n_paths))
    for i in range(n_steps):
        J = 0
        p = np.random.uniform(0, 1)
        if p < lambd*dt:
            J = np.exp(np.random.normal(np.log(1+muj)-(sigmaj**2)/2, sigmaj))-1
        S += (r-lambd*muj)*S*dt + S*np.sqrt(np.maximum(V, 0))*dW1[i, :] + J
        V += kappa*(theta - V)*dt + sigma*np.sqrt(np.maximum(V, 0))*dW2[i, :]
    return S,V
