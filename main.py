import os
from typing import Any

from mistralai import Mistral
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import re


API_KEY = os.environ["MISTRAL_API_KEY"]
MODEL_NAME = "mistral-large-latest"

client = Mistral(api_key=API_KEY)

def get_llm_response(user_message: str) -> str:
    """
    Communicates with the LLM to extract function details and interval.
    Handles potential errors and provides informative messages.
    """
    prompt = f"""
    You are a helpful assistant that extracts function and interval for graph plotting.

    **Supported Functions:**
    - Polynomials (e.g., "x^3 - 2x + 1")
    - Trigonometric functions (e.g., "sin(3x)", "cos(0.5x)")

    **Instructions:**
    1. **Extract function type:** 
        - 'polynomial' for polynomials
        - 'sin' or 'cos' for trigonometric functions
    2. **Extract function parameters:**
        - For polynomials: provide a comma-separated list of coefficients in square brackets (e.g., [1, -2, 1])
        - For trigonometric functions: provide the scale factor 'k' as a single number.
    3. **Extract interval:** 
        - Specify the interval as "from x_min to x_max" or "between x_min and x_max".

    **Return format:**
    - "function_type,function_parameters,x_min,x_max" 
      (e.g., "polynomial,[1,-2,1],-5,5")
    - Return "exit" to end the session.
    - Return "unknown" if the request cannot be understood.

    User message: {user_message}
    Assistant:
    """

    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a graph plotting assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        llm_response = response.choices[0].message.content.strip()
        if not llm_response:
            return "unknown"  # Handle empty responses
        return llm_response
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        return "error"

def parse_llm_output(llm_response: str) -> tuple:
    """Parses the LLM response and extracts function details and interval.
    Handles potential errors and provides informative messages.
    """
    if llm_response in ["exit", "unknown", "error"]:
        return llm_response, None, None, None, None, None  # Return 6 values

    try:
        parts = llm_response.split(',')

        if len(parts) < 4:  # Check if at least 4 parts are present
            return "unknown", None, None, None, None, "Invalid LLM response format."

        function_type = parts[0]

        if function_type == "polynomial":
            try:
                coefficients = eval(parts[1])  # Evaluate the coefficient string
            except (SyntaxError, ValueError):
                return "unknown", None, None, None, None, "Invalid coefficient format."

            try:
                x_min = float(parts[2])
                x_max = float(parts[3])
            except ValueError:
                return "unknown", None, None, None, None, "Invalid interval format."

            return "plot", "polynomial", coefficients, x_min, x_max

        elif function_type in ["sin", "cos"]:
            try:
                k = float(parts[1])
            except ValueError:
                return "unknown", None, None, None, None, "Invalid scale factor."

            try:
                x_min = float(parts[2])
                x_max = float(parts[3])
            except ValueError:
                return "unknown", None, None, None, None, "Invalid interval format."

            return "plot", function_type, k, x_min, x_max

        elif function_type == "linear":  # Handle linear equations (y = x, y = 2x + 1)
            try:
                # Extract slope (if present, otherwise assume slope = 1)
                slope = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0 
                x_min = float(parts[2])
                x_max = float(parts[3])
            except ValueError:
                return "unknown", None, None, None, None, "Invalid interval format."
            return "plot", "linear", slope, x_min, x_max

        else:
            return "unknown", None, None, None, None, "Unsupported function type."

    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return "unknown", None, None, None, None, "General parsing error."

def plot_graph(function_type: str, params: Any, x_min: float, x_max: float):
    """
    Plots the graph based on the extracted function and interval.
    """
    x = np.linspace(x_min, x_max, 400)
    y = np.zeros_like(x)

    if function_type == "polynomial":
        for i, coeff in enumerate(params):
            y += coeff * x ** (len(params) - 1 - i)
        title = "y = " + " + ".join(f"{coeff}x^{len(params) - 1 - i}" for i, coeff in enumerate(params))
    elif function_type == "sin":
        y = np.sin(params * x)
        title = f"y = sin({params}x)"
    elif function_type == "cos":
        y = np.cos(params * x)
        title = f"y = cos({params}x)"
    else:
        print("Unsupported function type.")
        return False

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()
    return True

if __name__ == "__main__":
    print("Welcome to the Graph Plotting Assistant!")
    while True:
        user_input = input("What graph do you want to plot? (type 'bye' to exit): ")
        if user_input.lower() == "bye":
            print("Goodbye!")
            break

        try:
            llm_response = get_llm_response(user_input)
            action, func_type, func_params, x_min, x_max, error_message = parse_llm_output(llm_response)
        except Exception as e:
            print(f"Error communicating with LLM: {e}")
            print("Consider using Bard workspace extension for LLM interactions.")
            continue

        if action == "plot":
            if plot_graph(func_type, func_params, x_min, x_max):
                print("Graph plotted successfully!")
            else:
                print("Error plotting graph.")
        elif action == "unknown":
            print(f"Invalid request: {error_message if error_message else ''}")
        else:
            print("An unexpected error occurred.")