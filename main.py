import matplotlib.pyplot as plt
import numpy as np
import re


def plot_graph(func_name, x_min, x_max, func_params=None, k=None):
    x = np.linspace(x_min, x_max, 400)

    if func_name == "x":
        y = x
    elif func_name == "x^2":
        y = x**2
    elif func_name == "sin":
        k = k if k is not None else 1
        y = np.sin(k * x)
    elif func_name == "cos":
        k = k if k is not None else 1
        y = np.cos(k * x)
    elif "polynomial" in func_name and func_params:
        # Polynomial evaluation
        y = np.polyval(func_params, x)
    else:
        print("Unknown function")
        return

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"{func_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Graph of {func_name}")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()


def parse_polynomial(poly_str):
    # Clean up the polynomial string
    poly_str = poly_str.replace("y =", "").replace("–", "-").replace("^", "**").replace(" ", "")

    # Regex to find terms
    terms = re.findall(r"([+-]?[^+-]+)", poly_str)

    # Determine the highest degree of the polynomial
    max_degree = 0
    for term in terms:
        if "x" in term:
            if "**" in term:
                degree = int(term.split("**")[1])
            elif "x" in term:
                degree = 1
            else:
                degree = 0
            max_degree = max(max_degree, degree)

    # Initialize coefficients
    coeffs = [0] * (max_degree + 1)

    for term in terms:
        if "x" in term:
            if "**" in term:
                degree = int(term.split("**")[1])
                coeff = term.split("x")[0]
            elif "x" in term:
                degree = 1
                coeff = term.split("x")[0]
            else:
                degree = 0
                coeff = term
            coeffs[max_degree - degree] = float(coeff if coeff not in ["", "+", "-"] else coeff + "1")
        else:
            coeffs[-1] = float(term)

    return coeffs


def main():
    while True:
        user_input = input("Enter the equation to plot (or 'exit' to quit): ").strip().lower()

        if user_input == "exit":
            print("Goodbye!")
            break

        try:
            # Handle `sin(kx)` or `cos(kx)`
            trig_match = re.match(r"(sin|cos)\(([\d\.]*)x\)", user_input)
            if trig_match:
                func_name = trig_match.group(1)
                k = float(trig_match.group(2)) if trig_match.group(2) else 1
                plot_graph(func_name, -5, 5, k=k)
            elif user_input in ["x", "y = x"]:
                plot_graph("x", -5, 5)
            elif user_input in ["x^2", "y = x^2"]:
                plot_graph("x^2", -5, 5)
            else:
                # Polynomial case
                coefficients = parse_polynomial(user_input)
                plot_graph("polynomial", -10, 10, coefficients)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
