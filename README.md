# Comparison of methods to estimate risk neutral densities

## To use the project

### Installation of the dependencies
To use the project you need to install the dependencies listed in the `pyproject.toml` file. You can do so manually (not recomended) or use poetry.  
Poetry (https://python-poetry.org/) is a python dependency manager. If you have not installed it yet you can install it by runing `pipx install poetry` (you might need to install pipx first, see the documentation : https://pipx.pypa.io/stable/installation/).
Once you have poetry installed, to install the dependencies of the project, cd into the project folder and simply run `poetry install`. Now the project has a dedicated virtual environement attached to it (that you can check by runing `poetry env info`) with all the defined dependencies installed. To run the project with this virtual environement, do `poetry run python main.py`. If you have any problem, please contact me (paul-antoine.leveilley@dauphine.eu).

### Usage
The code will produce :
- Plots of the estimated risk neutral density with a 95% confidence interval (with also the true density)
- Boxplots of the p-values of the Kolmogorov-Smirnov 2 sample tests.
- Csv files containing for each maturity the mean p-value, the standard deviation of the p-values and the percentage of rejection of the null hypothesis.
- Plots of the generated call prices and interpolating line.
- Plots of the implied volatilities and interpolating lines

Note that the code will save theses files in a directory on your machine (and it will create a directory if
it does not find one).

By default, the code will do 5000 rounds of simulation. It may take a long time to run. You can edit the `NUMBER_OF_RND` variable in the `config.py` file if you want to test the code with fewer simulations.
