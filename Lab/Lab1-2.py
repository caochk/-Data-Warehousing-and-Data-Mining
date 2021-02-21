import math


def f(x):
    return x * math.log(x) - 16.0

def fprime(x):
    return 1.0 + math.log(x)

def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    x_i = x_0

    for i in range(MAX_ITER):
        x_j = x_i - f(x_i)/fprime(x_i)
        if abs(x_i - x_j) < EPSILON:
            return x_j
        x_i = x_j

    return -1


