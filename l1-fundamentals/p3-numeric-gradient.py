############################################################
# Part 3: Numeric Gradient Search.
############################################################

import matplotlib.pyplot as plt
import numpy as np
import random

# This is adapted from Andrej Karpathy's blog:
# http://karpathy.github.io/neuralnets/
# written in Python instead of JS (Obviously)

# Define out unknown function we want to maximize
def unknown_function(x, y):
    return x*y

# Define our initial "guess" for a point.
start_x = -2
start_y = 3

def numeric_gradient_search(func, start_x, start_y, h, step_size):
    # Initilize our point
    x = start_x
    y = start_y

    # Keep a list of all the points we visit so we can
    # plot it later for fun!
    xl = []
    yl = []
    
    for i in range(100):
        # Record our current point.
        xl.append(x)
        yl.append(y)

        # How did we do?
        out = func(x, y)

        # Approximate derivative of f in terms of x and y
        dfdx = (func(x + h, y) - out) / h
        dfdy = (func(x, y + h) - out) / h

        # Update our point based on the approximated
        # gradients
        x = x + step_size * dfdx
        y = y + step_size * dfdy

    return (x, y), xl, yl

(best_x, best_y), x_list, y_list = numeric_gradient_search(unknown_function, start_x, start_y, h=0.0001, step_size=0.01)

print('Best points: ({}, {})'.format(best_x, best_y))
plt.plot(x_list, y_list)
plt.show()