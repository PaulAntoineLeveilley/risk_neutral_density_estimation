import matplotlib.pyplot as plt
import numpy as np
from config import COARSE_STRIKE_RANGE ,MODEL_PARAMETERS,MAX_RND_TO_DISPLAY,Y_INF,Y_SUP
from mean_estimated_rnd.average_rnd import compute_mean_and_confidence_interval_rnds 
from computation_rnd.compute_theoretical_rnd import compute_theoretical_rnd

def plots(rnds : np.array, args : dict):
    """
    Plots true rnd and average rnd along with confidence interval.

    Parameters :
    - rnds : 2D array containing the estimated rnds
    - args : a dict with additionals arguments
    """
    S0 = MODEL_PARAMETERS["S0"]
    V0 = MODEL_PARAMETERS["V0"]
    state_dict ={"spot_prices": np.array([S0]),"vols": np.array([V0])}
    model = args["model"]
    maturity = args["maturity"]
    upper_bound = args["upper_bound"]
    theoretical_rnd = compute_theoretical_rnd(state_dict=state_dict,model=model,model_parameters=MODEL_PARAMETERS,maturity=maturity,upper_bound = upper_bound)[0]
    mean_rnd, confidence_upper, confidence_lower = compute_mean_and_confidence_interval_rnds(rnds)
    S0 = MODEL_PARAMETERS["S0"]
    for rnd in rnds[:MAX_RND_TO_DISPLAY]:
        plt.plot(
            S0 * COARSE_STRIKE_RANGE, rnd,alpha = 0.1
        )
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        theoretical_rnd,
        color="b",linewidth = 1,
        label="theoretical rnd",
    )
    
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        mean_rnd,
        color="r",linewidth = 1,
        label="mean estimated rnd",
    )
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        confidence_upper,
        color="g",linewidth = 1,label = '95% confidence interval estimated rnd'
    )
    plt.plot(
        S0 * COARSE_STRIKE_RANGE,
        confidence_lower,
        color="g",linewidth = 1
    )

    plt.grid(True)
    plt.xlabel("S")
    plt.legend()
    plt.ylim(Y_INF,Y_SUP)
    plt.show()
    return None