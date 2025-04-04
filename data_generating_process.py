import numpy as np

from models.black_scholes import monte_carlo_simulations_bs
from pricing.black_scholes_pricing import black_scholes_price

from models.heston import monte_carlo_simulations_heston
from pricing.heston_pricing import heston_prices

from models.bakshi import monte_carlo_simulations_bakshi
from pricing.bakshi_pricing import bakshi_prices

from implied_vol.implied_vol import implied_vol

def generate_call_prices(T : float, maturity : float, model : str, model_parameters : dict,std_error,n :int,n_steps: int = 100, upper_bound: int = 1000):
    """
    Generate a sample of 50 call prices.
    
    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model : model to use
    - model_parameters : parameters of the model 
    - std_error : standard deviation of the error term
    - n : number of set of call prices to generate
    - n_steps : number of steps for monte carlo simulations
    - upper_bound : upper bound for the integral computation
    """
    if model == "black_scholes" :  
        return generate_call_prices_bs(T, maturity,model_parameters,std_error,n)
    elif model == "heston" :  
        return generate_call_prices_heston(T, maturity, model_parameters,std_error,n,n_steps,upper_bound)
    elif model == "bakshi": 
        return generate_call_prices_bakshi(T, maturity, model_parameters,std_error,n,n_steps,upper_bound)
    else :
        print("Model must be one of 'black_scholes','heston', 'bakshi'.")
        return 0
    
def generate_call_prices_bs(T : float, maturity : float, model_parameters : dict,std_error : float,n : int):
    """
    Generates a sample of 50 call prices using black scholes model to simulate 
    spot price and using black scholes formula to compute call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model_parameters : parameters of the model 
    - std_error : standard deviation of the error term
    - n : number of set of call prices to generate
    """
    S0 = model_parameters["S0"]
    r = model_parameters["r"]
    delta = model_parameters["delta"]
    sigma = model_parameters["sigma"]
    spot_prices = monte_carlo_simulations_bs(S0, T,r,sigma, n)
    call_prices = np.zeros((n,50))
    for i,spot in enumerate(spot_prices):
        strike_range = np.linspace(0.8*spot,1.2*spot, 50)
        for j, strike in enumerate(strike_range) :
            epsilon = np.random.normal(0,std_error)
            call_prices[i,j]= black_scholes_price(spot,strike,maturity,r,delta, sigma)*(1+epsilon)
    return call_prices,spot_prices


def generate_call_prices_heston(T : float, maturity : float, model_parameters : dict,std_error : float,n : int,n_steps:int,upper_bound : float):
    """
    Generates a sample of 50 call prices using heston model to simulate 
    spot price and using heston formula to compute call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model_parameters : parameters of the model 
    - std_error : standard deviation of the error term
    - n : number of set of call prices to generate
    - n_steps : number of steps for monte carlo simulations
    - upper_bound : upper bound for the integral computation
    """
    S0 = model_parameters["S0"]
    V0 = model_parameters["V0"]
    r = model_parameters["r"]
    sigma = model_parameters["sigma"] 
    kappa = model_parameters["kappa"]
    theta = model_parameters["theta"]
    rho = model_parameters["rho"]
    lambd = model_parameters["lambd"]
    spot_prices,vols = monte_carlo_simulations_heston(S0,V0,T,r,sigma,kappa, theta,rho,n_steps,n)
    call_prices = np.zeros((n,50))
    for i in range(n):
        spot = spot_prices[i]
        vol = vols[i] 
        epsilon = np.random.normal(0,std_error)
        strike_range = np.linspace(0.8*spot,1.2*spot, 50)
        for j, strike in enumerate(strike_range) :
            epsilon = np.random.normal(0,std_error)
            call_prices[i,j] = heston_prices(spot,strike,vol,maturity,r,kappa,theta,lambd,rho,sigma,upper_bound)*(1+epsilon)
    return call_prices,spot_prices

def generate_call_prices_bakshi(T : float, maturity : float, model_parameters : dict,std_error : float,n : int,n_steps:int,upper_bound : float):
    """  
    Generates a sample of 50 call prices using bakshi model to simulate 
    spot price and using bakshi formula to compute call prices.

    Parameters :
    - T : Time horizon for the monte carlo simulations
    - maturity : maturity of the calls
    - model_parameters : parameters of the model 
    - std_error : standard deviation of the error term
    - n : number of set of call prices to generate
    - n_steps : number of steps for monte carlo simulations
    - upper_bound : upper bound for the integral computation
    """
    S0 = model_parameters["S0"]
    V0 = model_parameters["V0"]
    r = model_parameters["r"]
    sigma = model_parameters["sigma"] 
    kappa = model_parameters["kappa"]
    theta = model_parameters["theta"]
    rho = model_parameters["rho"]
    lambda_jump = model_parameters["lambda_jump"]
    muj= model_parameters["muj"]
    sigmaj = model_parameters["sigmaj"]
    spot_prices,vols = monte_carlo_simulations_bakshi(S0,V0,T,r,sigma,kappa,theta,rho,lambda_jump,muj,sigmaj,n_steps,n)
    call_prices = np.zeros((n,50))
    for i in range(n):
        spot = spot_prices[i]
        vol = vols[i] 
        epsilon = np.random.normal(0,std_error)
        strike_range = np.linspace(0.8*spot,1.2*spot, 50)
        for j, strike in enumerate(strike_range) :
            epsilon = np.random.normal(0,std_error)
            call_prices[i,j] = bakshi_prices(spot,strike,vol,maturity,r,kappa,theta,lambda_jump,rho,sigma,muj,sigmaj,upper_bound)*(1+epsilon)
    return call_prices,spot_prices


def compute_implied_volatility(call_prices, spot_prices, maturity: float,r : float):
    """
    Computes implied volatility
    
    Parameters : 
    - call_prices : call prices from which to compute the implied volatility
    - spot_prices : initial spot associated to each call price
    - maturity : maturity of the option
    - r : interest rate
    """
    _,p = np.shape(call_prices)
    implied_volatility = np.empty_like(call_prices)
    for i,spot in enumerate(spot_prices):
        for j,call_price in enumerate(call_prices[i]):
            strike = 0.8*spot+(0.4*spot)*j/(p-1) 
            implied_volatility[i,j] = implied_vol(spot,strike,maturity,r,call_price,0.01,30)
    return implied_volatility