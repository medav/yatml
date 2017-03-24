import matplotlib.pyplot as plt
import numpy as np
import random

plt.xkcd()

# This is adapted from Andrej Karpathy's blog:
# http://karpathy.github.io/neuralnets/
# written in Python instead of JS (Obviously)

# Define out unknown function we want to maximize
def unknown_function(a, b):
    return a*b

# Define our initial "guess" for a point.
start_a = -2
start_b = 3

def numeric_gradient_search(func, start_a, start_b, h, step_size):
    # Initilize our point
    a = start_a
    b = start_b

    # Keep a list of all the points we visit so we can
    # plot it later for fun!
    al = []
    bl = []
    
    for i in range(100):
        # Record our current point.
        al.append(a)
        bl.append(b)

        # How did we do?
        out = func(a, b)

        # Approximate derivative of f in terms of x and y
        dfda = (func(a + h, b) - out) / h
        dfdb = (func(a, b + h) - out) / h

        # Update our point based on the approximated
        # gradients
        a = a - step_size * dfda
        b = b - step_size * dfdb

    return (a, b), al, bl

(best_a, best_b), a_list, b_list = numeric_gradient_search(unknown_function, start_a, start_b, h=0.0001, step_size=0.01)

print('Best points: ({}, {})'.format(best_a, best_b))
plt.plot(a_list, b_list)
plt.show()