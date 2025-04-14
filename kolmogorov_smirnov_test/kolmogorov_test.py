import numpy as np
from scipy.stats import ks_2samp
from tqdm import tqdm

def perform_ks_test(sample_estimated_rnd : np.ndarray, sample_true_rnd : np.ndarray):
    """
    Performs Kolmogorov Smirnov tests

    Parameters :
    - sample_estimated_rnd : 2D array where each line is
    a sample from the estimated rnd
    - sample_true_rnd : 2D array where each line is
    a sample from the true rnd
    """
    n,p = np.shape(sample_true_rnd)
    return np.array([ks_2samp(sample_true_rnd[i],sample_estimated_rnd[i]).pvalue for i in tqdm(range(n))])