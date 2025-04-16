from full_estimating_procedure import full_estimating_procedure
import time
def main():
    start = time.time()
    T = 1
    models = ['black_scholes','heston','bakshi']
    interpolation_methods = ['cubic_splines','kernel_regression','rbf_network']
    for model in models:
        if model =='bakshi':
            #somehow, maturity 21/252 causes bakshi call prices to be negative
            #which generate an error at the 
            maturities = [42/252,63/252,84/252,105/252]
        else : 
            maturities = [21/252,42/252,63/252,84/252,105/252]
        for interpolation_method in interpolation_methods:
            try :  
                full_estimating_procedure(T,maturities,model,interpolation_method)
            except ValueError as e:
                print(e)
    end = time.time()
    print(f'Produced all the results in {end-start} seconds')
if __name__ == "__main__":
    main()
