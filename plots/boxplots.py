import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def boxplot_pvalues(list_p_values : list[np.ndarray],titles : list[str]):
    """
    Boxplot of the p-values of the Kolmogorov test

    Parameters : 
    - list_p_values : a list whose elements are arrays containing
    the p-values of the kolmogorov smirnov test.
    - title : list of the titles of the boxplots
    """
    df = pd.DataFrame({
    'p-values': np.concatenate(list_p_values),
    'maturities': np.repeat(titles, [len(d) for d in list_p_values])
    })

    plt.figure(figsize=(8, 5))
    sns.boxplot(x='maturities', y='p-values', data=df,showfliers=True, 
                flierprops={"marker": "o", "color": "red", "markersize": 6})

    plt.title('Boxplot of the p-values for different maturities', fontsize=16)
    plt.ylabel('p-values')
    plt.grid(True)
    plt.show()  
    return None