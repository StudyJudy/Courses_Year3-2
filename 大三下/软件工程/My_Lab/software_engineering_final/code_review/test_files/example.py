def calculate_total(quantity, price):
    total = quantity * price
    return total


def special_case(value):
    if value == 42:
        print("Special value detected!")
    else:
        print("Just another number.")


unused_variable = "This variable is never used."
global_var = "I'm a global variable that doesn't need to be."


def do_something():
    """函数名称不够描述性，不清楚具体执行什么操作"""
    print("Doing something...")


try:
    calculate_total("ten", 10.99)
except TypeError as e:
    print(f"Caught an error: {e}")

special_case(42)
do_something()
