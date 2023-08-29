def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero!")
    return x / y

def exponent(base, exponent):
    if base < 0 and exponent % 1 != 0:
        raise ValueError("Cannot compute the exponent of a negative base with a fractional exponent!")
    return base ** exponent

def calculator():
    print("Welcome to the Interactive Calculator!")

    while True:
        print("Please select an operation:")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Exponent")
        print("6. Quit")

        choice = input("Enter your choice (1-5): ")

        if choice == '5':
            print("Thank you for using the calculator. Goodbye!")
            break

        try:
            num1 = float(input("Enter the first number: "))
            num2 = float(input("Enter the second number: "))
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            continue

        if choice == '1':
            result = add(num1, num2)
            print(f"Result: {num1} + {num2} = {result}")
        elif choice == '2':
            result = subtract(num1, num2)
            print(f"Result: {num1} - {num2} = {result}")
        elif choice == '3':
            result = multiply(num1, num2)
            print(f"Result: {num1} * {num2} = {result}")
        elif choice == '4':
            try:
                result = divide(num1, num2)
                print(f"Result: {num1} / {num2} = {result}")
            except ValueError as e:
                print(str(e))
        elif choice == '5':
            try:
                result = exponent(num1, num2)
                print(f"Result: {num1} ^ {num2} = {result}")
            except ValueError as e:
                print(str(e))
        else:
            print("Invalid choice. Please select a valid operation.")

        print()  # Print an empty line for readability

calculator()

