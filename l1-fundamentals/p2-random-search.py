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

def random_local_search(func, start_a, start_b, tweak_amt):
    # Store the best output
    best = -999999

    # Initilize our point
    a = start_a
    b = start_b

    # Keep a list of all the points we visit so we can
    # plot it later for fun!
    al = []
    bl = []
    
    for i in range(100):
        # Pick a new a and b to try randomly
        a_try = a + tweak_amt * (random.random() * 2 - 1)
        b_try = b + tweak_amt * (random.random() * 2 - 1)

        # How did we do?
        out = func(a_try, b_try)

        # If our output is better, update a and b and
        # record it in our lists.
        if out > best:
            a = a_try
            b = b_try
            best = out

            al.append(a)
            bl.append(b)

    return (a, b), al, bl


(best_a, best_b), a_list, b_list = random_local_search(unknown_function, start_a, start_b, tweak_amt=0.01)

print('Best points: ({}, {})'.format(best_a, best_b))
plt.plot(a_list, b_list)
plt.show()