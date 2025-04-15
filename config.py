import numpy as np

RELATIVE_STRIKE_UPPER_BOUND = 1.2
RELATIVE_STRIKE_LOWER_BOUND = 0.8
NUMBER_OF_STRIKES = 50
NUMBER_OF_RND = 10
STRIKE_RANGE = np.linspace(
    RELATIVE_STRIKE_LOWER_BOUND, RELATIVE_STRIKE_UPPER_BOUND, NUMBER_OF_STRIKES
)
S_OVER_K_RANGE = np.array([1 / K for K in STRIKE_RANGE[::-1]])

SIZE_COARSE_RANGE = 500
COARSE_STRIKE_RANGE = np.linspace(
    RELATIVE_STRIKE_LOWER_BOUND, RELATIVE_STRIKE_UPPER_BOUND, SIZE_COARSE_RANGE
    )

COARSE_S_OVER_K_RANGE = np.array([1 / K for K in COARSE_STRIKE_RANGE[::-1]])

STD_ERROR = 0.05
QUANTILE = 1.96
NUM_SAMPLES = 100
P_VALUE_TRESHOLD = 0.05
MAX_RND_TO_DISPLAY = 50

# model parameters
S0 = 7616  # initial spot
V0 = 0.01  # initial vol (for heston and bakshi)
r = 0.045  # interest rate
sigma = 0.13764  # vol
sigmav = 0.2  # vol of vol (for heston and bakshi)
delta = 0  # dividend rate (for black scholes only)
kappa = 9  # speed of mean reversion (for heston and bakshi)
theta = 0.0189  # longrun mean of vol  (for heston and bakshi)
lambd = 0  # market price of vol (heston)
lambdajump = 0.59  # jump rate (bakshi)
muj = -0.05  # mean size of jumps (bakshi)
sigmaj = 0.07  # standard deviation of jumps (bakshi)
rho = 0.1  # correlation between brownian motions driving spot and vol (for heston and bakshi)


MODEL_PARAMETERS = {
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
