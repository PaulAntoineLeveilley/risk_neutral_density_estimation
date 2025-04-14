import numpy as np

RELATIVE_STRIKE_UPPER_BOUND = 1.2
RELATIVE_STRIKE_LOWER_BOUND = 0.8
NUMBER_OF_STRIKES = 50
NUMBER_OF_RND = 2
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

NUM_SAMPLES = 10