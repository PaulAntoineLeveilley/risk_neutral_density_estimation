# Comparison of methods to estimate risk neutral densities

## Introduction

## To use the project

### Installation of the dependencies
To use the project you need to install the dependencies listed in the `pyproject.toml` file. You can do so manually (not recomended) or use poetry.  
Poetry (https://python-poetry.org/) is a python dependency manager. If you have not installed it yet you can install it by runing `pipx install poetry` (you might need to install pipx first, see the documentation : https://pipx.pypa.io/stable/installation/).
Once you have poetry installed, to install the dependencies of the project, cd into the project folder and simply run `poetry install`. Now the project has a dedicated virtual environement attached to it (that you can check by runing `poetry env info`) with all the defined dependencies installed. To run the project with this virtual environement, do `poetry run python main.py`.

### Usage
If you want to produce the results you have to edit the main file to input a model (variable `model`, one of
`'black_scholes'`, `'heston'`,`'bakshi'`) and and interpolation method (variable `interpolation_method`, one of `'cubic_splines'`, `'kernel_regression'`, `'rbf_network'`) that you want to test. You also have to specity a list of maturities (variable `maturities`, in years) for which you want the results. Finally you must set the time horizon `T` (years) for the monte carlo simulations.
The code will produce :
- A plot of the estimated risk neutral density with a 95% confidence interval (with also the true density)
- A boxplot of the p-values of the Kolmogorov-Smirnov 2 sample tests.
- A csv file containing for each maturity the mean p-value, the standard deviation of the p-values and the percentage of rejection of the null hypothesis.

Note that the code will save theses files in a directory on your machine (and it will create a directory if
it does not find one).
