import numpy as np
from tqdm import tqdm

from config import COARSE_STRIKE_RANGE

def sample_from_pdf(pdf : np.ndarray,num_sample: int, spot : float):
    """
    Samples num_samples realizations of a 
    random variable with probability distribution
    function given by pdf

    Parameters : 
    - pdf : a 1D array representing a probability density function
    - num_sample : the number of realizations to sample
    - spot : the spot associated to the rnd
    """
    return np.random.choice(spot*COARSE_STRIKE_RANGE,size=num_sample,p = pdf)

def sample_from_pdfs(pdfs : np.ndarray,num_sample : int, spot_prices : np.ndarray):
    """
    Loop over the lines of pdfs to sample.

    Parameters :  
    - pdfs : an 2D array which lines represents probability
    density functions
    - num_sample : the number of realizations to sampe from each pdf
    - spot_prices : spot price associated to each rnd
    """
    return np.array([sample_from_pdf(pdf,num_sample,spot_prices[i]) for i, pdf in tqdm(enumerate(pdfs))])