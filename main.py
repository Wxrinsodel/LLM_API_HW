def main():
    while True:
        user_input = input("What graph do you want to plot? (Type 'exit' to quit): ")
        if "exit" in user_input.lower():
            print("Thank you – that’s it for today, bye!")
            break
        print(f"Your answer: {user_input}")

if __name__ == "__main__":
    main()