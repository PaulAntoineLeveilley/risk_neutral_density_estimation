from full_estimating_procedure import full_estimating_procedure

def main():
    T = 1
    maturities = [63/252]
    model = 'bakshi'
    interpolation_method = 'rbf_network'
    full_estimating_procedure(T,maturities,model,interpolation_method)

if __name__ == "__main__":
    main()
