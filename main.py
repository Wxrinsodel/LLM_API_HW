import os
from typing import Any

from mistralai import Mistral
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import re


API_KEY = "8PMXajgoFMgiCQ3ILhya9YTNcInnYWZU"
MODEL_NAME = "mistral-large-latest"

client = Mistral(api_key=API_KEY)

def get_llm_response(user_message: str) -> str:
    """Communicates with the LLM to extract function details and interval."""
    prompt = f"""
    You are a helpful assistant that extracts function and interval for graph plotting.
    You can plot:
    - Polynomials of power 1 to 4 (e.g., "x^3 - 2x + 1").
    - Scaled sine and cosine functions (e.g., "sin(3x)", "cos(0.5x)").

    For polynomials, extract the coefficients as a list, starting from the highest power.
    For scaled trig functions, extract the function name (sin or cos) and the scale factor 'k'.

    Allowed interval format is "from x_min to x_max" or "between x_min and x_max".

    If the user asks to plot a graph, extract:
    - function_type (polynomial, sin, cos)
    - function_parameters (list of coefficients for polynomial, scale factor 'k' for trig)
    - x_min (numeric value from the interval)
    - x_max (numeric value from the interval)

    **IMPORTANT INSTRUCTIONS: Return ONLY the result in the format: "function_type,function_parameters,x_min,x_max".**
    **For polynomials, function_parameters should be a comma-separated list of coefficients INSIDE SQUARE BRACKETS (e.g., [1,-2,1]).**
    **For sin(kx) and cos(kx), function_parameters should be just the value of k (a single number).**
    **Do not include any extra text, explanations, or preambles. Just the comma-separated output.**

    If the user wants to end the session, return "exit".
    If you cannot understand the request, return "unknown".

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
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with the LLM: {e}")
        return "error"

def parse_llm_output(llm_response: str) -> tuple:
    """Parses the LLM response and extracts function details and interval."""
    if llm_response in ["exit", "unknown", "error"]:
        return llm_response, None, None, None, None

    try:
        parts = llm_response.split(',')
        if parts[0] == "polynomial":
            coefficients = eval(parts[1])
            x_min, x_max = float(parts[2]), float(parts[3])
            return "plot", "polynomial", coefficients, x_min, x_max
        elif parts[0] in ["sin", "cos"]:
            k = float(parts[1])
            x_min, x_max = float(parts[2]), float(parts[3])
            return "plot", parts[0], k, x_min, x_max
        else:
            return "unknown", None, None, None, None
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return "unknown", None, None, None, None

def plot_graph(function_type: str, params: Any, x_min: float, x_max: float):
    """Plots the graph based on the extracted function and interval."""
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
        try:
            llm_response = get_llm_response(user_input)
            action, func_type, func_params, x_min, x_max = parse_llm_output(llm_response)
        except Exception as e:
            print(f"Error communicating with LLM: {e}")
            # Explore alternative LLM interaction methods here (e.g., Bard workspace extension)
            print("Consider using Bard workspace extension for LLM interactions.")
            continue

        if action == "exit":
            print("Goodbye!")
            break
        elif action == "plot":
            plot_graph(func_type, func_params, x_min, x_max)
        else:
            print("Invalid request. Please try again.")