import numpy as np

def arrays_to_pdfs(arrays : np.ndarray):
    """
    Computes discretes probability distribution functions
    from an array of arrays representing densities.

    Parameters : 
    - arrays : an array of shape n,p where 
    each line represents a probability density
    function
    """
    positive_part = np.maximum(arrays,0)
    return positive_part/np.sum(positive_part, axis = 1)[:,np.newaxis]