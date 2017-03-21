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

def random_local_search(func, start_x, start_y, tweak_amt):
    # Store the best output
    best = -999999

    # Initilize our point
    x = start_x
    y = start_y

    # Keep a list of all the points we visit so we can
    # plot it later for fun!
    xl = []
    yl = []
    
    for i in range(100):
        # Pick a new x and y to try randomly
        x_try = x + tweak_amt * (random.random() * 2 - 1)
        y_try = y + tweak_amt * (random.random() * 2 - 1)

        # How did we do?
        out = func(x_try, y_try)

        # If our output is better, update x and y and
        # record it in our lists.
        if out > best:
            x = x_try
            y = y_try
            best = out

            xl.append(x)
            yl.append(y)

    return (x, y), xl, yl


(best_x, best_y), x_list, y_list = random_local_search(unknown_function, start_x, start_y, tweak_amt=0.01)

print('Best points: ({}, {})'.format(best_x, best_y))
plt.plot(x_list, y_list)
plt.show()