from full_estimating_procedure import full_estimating_procedure

def main():
    T = 1
    maturities = [42/252,63/252,84/252,105/252]
    model = 'bakshi'
    interpolation_method = 'kernel_regression'
    full_estimating_procedure(T,maturities,model,interpolation_method)

if __name__ == "__main__":
    main()
