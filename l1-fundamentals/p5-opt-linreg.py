import matplotlib.pyplot as plt
import numpy as np

# Those same random points
x = [1,1,2,3.5,5,6,7.5,8,8,10]
y = [2,1,5,6.5,11,12,15,15,16,21]

num_pts = len(x)

# First, let's get a baseline from a well known 
# regression algorithm.
a, b = np.polyfit(x,y,1)
print('Baseline Values: a = {}, b = {}'.format(a, b))

# Define our "Unknown function": the squared
# difference between the prediction and target
def func(a, b, x, y):
    return (a*x + b - y) ** 2

# Define our gradient functions, which tell us
# which direction of a and b yield an increase
# or decrease in f. 
#
# N.B. We don't need gradients for x and y 
#      because they are non-trainable variables!

def dfda(a, b, x, y):
    return 2*a*x*x + 2*b*x - 2*x*y

def dfdb(a, b, x, y):
    return 2*a*x - 2*y + 2*b

# Define our initial "guess". We are going to
# guess a flat line with intercept at y-bar.
a = 0
b = np.mean(y)

# Our step size tells us how much to tweak a and b
# by each iteration
step_size = 0.001

for i in range(10000):
    # Here we pick a point on the list to "train"
    # on. This loop will simply go through all the 
    # points in our list over and over.
    xp, yp = x[i % num_pts], y[i % num_pts]
    
    # Find the output of our "f" for the given point
    # and our current values of a and b.
    out = func(a, b, xp, yp)

    # Update our a and b values based on their gradients.
    #
    # N.B. We negate the gradient because we want to 
    #      **Minimize** the difference, not maximize.
    a = a - step_size * dfda(a, b, xp, yp)
    b = b - step_size * dfdb(a, b, xp, yp)


print('Gradient Search: a = {}, b = {}'.format(a, b))