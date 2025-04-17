from full_estimating_procedure import full_estimating_procedure
import time
def main():
    start = time.time()
    T = 1
    maturities = [21/252,42/252,63/252,84/252,105/252,126/252]
    # model = 'black_scholes'
    # interpolation_method = 'cubic_splines'

    models = ['black_scholes','heston',"bakshi"]
    interpolation_methods = ["cubic_splines", 'kernel_regression','rbf_network']
    for model in models:
        for interpolation_method in interpolation_methods:
            try : 
                full_estimating_procedure(T,maturities,model,interpolation_method)
            except ValueError as e: 
                print(e)
if __name__ == "__main__":
    main()
