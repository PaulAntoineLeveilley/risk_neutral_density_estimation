from pricing.black_scholes_pricing import black_scholes_price

def main() : 
    S = 100    
    K = 120
    T = 1
    r = 0.05
    delta = 0.02
    sigma = 0.2
    price = black_scholes_price(S,K,T,r,delta,sigma)
    print('price = '+ str(price))
    
if __name__ == "__main__" : 
    main()