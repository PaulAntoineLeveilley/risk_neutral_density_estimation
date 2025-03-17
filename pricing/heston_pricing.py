import numpy as np
import math
from scipy.integrate import quad

def heston_prices(S: float, K: float, V: float, T: float, r: float, kappa: float, theta: float, lambd: float, rho: float, sigma: float, upper_bound : float):
    """
    Computes the price of an european call option in the Heston model.

    Parameters : 
    - S : spot price
    - K : strike 
    - V : initial value for volatility
    - T : time to maturity
    - r : interest rate
    - kappa : mean reversion parameter
    - theta : long run mean of volatity
    - lambd : market price of volatility
    - rho : correlation
    - sigma : volatility of volatility
    - upper_bounb : upper bound for integral evaluation
    """
    return S*P(1,S, T, V, K,r, kappa, theta, lambd, rho, sigma, upper_bound) - K*np.exp(-r*T)*P(2,S, T, V, K,r, kappa, theta, lambd, rho, sigma,upper_bound)


def P(i : int,S: float, T: float, V: float, K: float,r : float, kappa: float, theta: float, lambd: float, rho: float, sigma: float, upper_bound : float):
    """
    Computes the i^th term in the Heston formula for a call option

    Parameters :
    - i : index of the term
    - S : spot price 
    - K : strike 
    - T : time to maturity
    - r : interest rate
    - kappa : mean reversion parameter
    - theta : long run mean of volatity
    - lambd : market price of volatility
    - rho : correlation
    - sigma : volatility of volatility     
    - upper_bound : upper bound for integral evaluation
    """
    integrand = lambda phi : (np.exp(-1j*phi*np.log(K))*f(i,S,V,T,r, kappa, theta, lambd,rho, sigma,phi)/(1j*phi)).real   
    integral= quad(integrand,0,upper_bound)[0]
    return 0.5+integral/math.pi

def f(i : int, S : float, V : float, T : float,r : float, kappa : float, theta : float , lambd : float, rho : float, sigma : float ,phi : complex):
    """
    Computes term in the integrand for the complex integral.

    Parameters : 
    - i : index of the term
    - S : Spot price
    - V : volatility
    - T : time to maturity
    - r : interest rate
    - kappa : mean reversion parameter
    - theta : long run mean of the volatility
    - lambd : market price of volatility
    - rho : correlation
    - sigma : volatility of volatility
    - phi : integration variable
    """
    C,D = compute_C_D(i,T,r, kappa, theta, lambd, rho,sigma,phi)
    return np.exp(C +D*V+1j*phi*np.log(S))

def compute_C_D(i : int, T : float,r : float, kappa : float, theta: float, lambd : float, rho : float, sigma : float, phi : complex) : 
    """
    Computes the 'C' and the 'D' term in the expression of the function f

    Parameters :  
    - i : index of the term
    - T : time to maturity
    - r : interest rate
    - kappa : mean reversion parameter
    - theta : long run mean of the volatility
    - lambd : market price of volatility
    - rho : correlation
    - sigma : volatility of volatility
    - phi : integration variable
    """ 
    a = kappa*theta
    b = kappa+lambd
    if i == 1 :   
        b -=rho*sigma
        u= 0.5
    elif i ==2 :
        u = -0.5
    d =np.sqrt((1j*rho*sigma*phi-b)**2-(sigma**2)*(2j*u*phi-phi**2))
    h = b-1j*rho*sigma*phi
    g = (h+d)/(h-d)
    C = 1j*r*phi*T+((h+d)*T-2*np.log((1-g*np.exp(d*T))/(1-g)))*a/sigma**2
    D = (h+d)*(1-np.exp(d*T))/((1-g*np.exp(d*T))*(sigma**2))
    return C,D 


















