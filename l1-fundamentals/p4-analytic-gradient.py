############################################################
# Part 2: Analytic Gradient Search.
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

# Here we define our analytic gradients
def dfdx(x, y):
    return y

def dfdy(x, y):
    return x

# Define our initial "guess" for a point.
start_x = -2
start_y = 3

def analytic_gradient_search(func, start_x, start_y, step_size):
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

        # Update our point based on analytic gradients
        x = x + step_size * dfdx(x, y)
        y = y + step_size * dfdy(x, y)

    return (x, y), xl, yl

(best_x, best_y), x_list, y_list = analytic_gradient_search(unknown_function, start_x, start_y, step_size=0.01)

print('Best points: ({}, {})'.format(best_x, best_y))
plt.plot(x_list, y_list)
plt.show()