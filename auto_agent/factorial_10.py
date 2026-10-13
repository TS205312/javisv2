# Python script to calculate factorial of 10

def factorial(n):
    """Calculate factorial of n using iteration."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Calculate factorial of 10
result = factorial(10)
print(f"Factorial of 10 is: {result}")

# Also demonstrate recursive approach
def factorial_recursive(n):
    """Calculate factorial of n using recursion."""
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

print(f"Factorial of 10 (recursive): {factorial_recursive(10)}")