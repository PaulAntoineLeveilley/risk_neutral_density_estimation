from full_estimating_procedure import full_estimating_procedure

def main():
    T = 1
    maturities = [21/252,42/252,63/252]
    model = 'black_scholes'
    interpolation_method = 'cubic_splines'
    full_estimating_procedure(T,maturities,model,interpolation_method)

if __name__ == "__main__":
    main()
