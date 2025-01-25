import matplotlib.pyplot as plt
import numpy as np

def plot_graph(func_name, x_min, x_max):
    x = np.linspace(x_min, x_max, 400)
    if func_name == "y=x":
        y = x
    elif func_name == "y=x^2":
        y = x**2
    elif func_name == "y=sin(x)":
        y = np.sin(x)
    elif func_name == "y=cos(x)":
        y = np.cos(x)
    else:
        print("Unknown function")
        return

    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Plot of {func_name}")
    plt.grid(True)
    plt.show()

def main():
    while True:
        user_input = input("What graph do you want to plot? (Type 'exit' to quit): ")
        if "exit" in user_input.lower():
            print("Thank you – that’s it for today, bye!")
            break
        
        if "sin" in user_input.lower():
            llm_response = "y=sin(x),-5,5"
        elif "cos" in user_input.lower():
            llm_response = "y=cos(x),-5,5"
        elif "x^2" in user_input.lower():
            llm_response = "y=x^2,-5,5"
        elif "x" in user_input.lower():
            llm_response = "y=x,-5,5"
        else:
            print("Could not understand. Please be more specific.")
            continue

        parts = llm_response.split(",") 
        func_name = parts[0].strip()
        x_min = float(parts[1])
        x_max = float(parts[2])

    
        plot_graph(func_name, x_min, x_max)

if __name__ == "__main__":
    main()
