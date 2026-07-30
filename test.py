name = input("What's your name? ")
age = int(input("How old are you? "))

print(f"Hello, {name}!")

if age >= 18:
    print("You're an adult.")
else:
    print(f"You have {18 - age} years until you're 18.")

# A simple loop
print("\nCounting to 5:")
for i in range(1, 6):
    print(i)