import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, Poly, sympify
import ast
import re

def plot_graph(func_name, x_min, x_max, func_params=None):
    x = np.linspace(x_min, x_max, 400)
    if func_name == "y = x":
        y = x
    elif func_name == "y = x^2":
        y = x**2
    elif func_name == "y = sin(x)":
        y = np.sin(x)
    elif func_name == "y = cos(x)":
        y = np.cos(x)
    elif func_name == "polynomial" and func_params:
        y = np.polyval(func_params, x)
    else:
        print("Unknown function")
        return

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Plot of {func_name}")
    plt.grid(True)
    plt.show()

def parse_polynomial(poly_str):
    # Clean input
    poly_str = poly_str.replace(" ", "").replace("–", "-").replace("^", "**")
    
    # Extract terms
    terms = re.findall(r'([+-]?[^+-]+)', poly_str)
    
    # Initialize coefficients for up to 4th-degree polynomial
    coeffs = [0, 0, 0, 0, 0]
    
    for term in terms:
        # Constant term
        if 'x' not in term:
            coeffs[0] = float(term)
        
        # x term
        elif term == 'x':
            coeffs[4] = 1
        elif term == '-x':
            coeffs[4] = -1
        
        # x^2 term
        elif 'x**2' in term or 'x^2' in term:
            coeff = term.split('x')[0]
            coeffs[3] = float(coeff) if coeff and coeff not in ['+', '-'] else (1 if '+x**2' in term else -1)
        
        # x^3 term
        elif 'x**3' in term or 'x^3' in term:
            coeff = term.split('x')[0]
            coeffs[2] = float(coeff) if coeff and coeff not in ['+', '-'] else (1 if '+x**3' in term else -1)
        
        # x term with coefficient
        else:
            coeff = term.split('x')[0]
            coeffs[4] = float(coeff)
    
    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs.pop(0)
    
    return coeffs


def main():
    while True:
        user_input = input("What graph do you want to plot? (Type 'exit' to quit): ").strip().lower()
        if "exit" in user_input:
            print("Thank you – that’s it for today, bye!")
            break
        
        if "sin" in user_input:
            llm_response = "y = sin(x),-5,5" 
        elif "cos" in user_input:
            llm_response = "y = cos(x),-5,5" 
        elif user_input.strip() == "x^2" or user_input.strip() == "y = x^2":
            llm_response = "y = x^2,-5,5"
        elif user_input.strip() == "x" or user_input.strip() == "y = x":
            llm_response = "y = x,-5,5"
        else:
            coefficients = parse_polynomial(user_input)
            llm_response = f"polynomial,-5,5,{coefficients}" 
        
        parts = llm_response.split(",")
        func_name = parts[0].strip()
        x_min = float(parts[1])
        x_max = float(parts[2])
        
        func_params = None
        if func_name == "polynomial" and len(parts) > 3:
            func_params = [float(c) for c in parts[3].strip('[]').split(',')]

        plot_graph(func_name, x_min, x_max, func_params)

if __name__ == "__main__":
    main()