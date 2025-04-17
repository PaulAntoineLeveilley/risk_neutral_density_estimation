import os

def make_result_directory():
    """
    Creates the directory to save the files
    """
    if not os.path.isdir('results'):
        os.makedirs("results")
    if not os.path.isdir('results/boxplots'):
        os.makedirs('results/boxplots')
    if not os.path.isdir('results/rnd_plots'):
        os.makedirs('results/rnd_plots')
    if not os.path.isdir('results/kolmogorov_test'):
        os.makedirs('results/kolmogorov_test')
    if not os.path.isdir('results/call_prices'):
        os.makedirs('results/call_prices')
    if not os.path.isdir('results/implied_vol'):
        os.makedirs('results/implied_vol')
    return None