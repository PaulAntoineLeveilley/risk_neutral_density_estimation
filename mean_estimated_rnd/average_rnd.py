import numpy as np
from config import QUANTILE

def compute_average_rnds(rnds :np.ndarray):
    """
    Computes the average of the estimated rnds.

    Parameters :
    - rnds : 2D array whoses lines are the estimated
    risk neutral densities.
    """
    return np.mean(rnds, axis=0)

def compute_std_rnds(rnds : np.ndarray):
    """
    Computes standard deviation of estimated rnds.

    Parameters :
    - rnds : 2D array whoses lines are the estimated
    risk neutral densities.
    """
    return np.std(rnds,axis=0)

def compute_mean_and_confidence_interval_rnds(rnds : np.ndarray):
    """
    Computes mean and 95% confidence interval for the estimated rnd.
    
    Parameters :
    - rnds : 2D array whoses lines are the estimated
    risk neutral densities.
    """
    sample_size = np.shape(rnds)[0]
    mean_rnd = compute_average_rnds(rnds)
    std_rnd = compute_std_rnds(rnds)
    upper_confidence_bound = mean_rnd+ QUANTILE*std_rnd/np.sqrt(sample_size)
    lower_confidence_bound = mean_rnd- QUANTILE*std_rnd/np.sqrt(sample_size)
    return mean_rnd,upper_confidence_bound, lower_confidence_bound