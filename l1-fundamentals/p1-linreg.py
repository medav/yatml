import matplotlib.pyplot as plt
import numpy as np

plt.xkcd()

# Some random points
x = [1,1,2,3.5,5,6,7.5,8,8,10]
y = [2,1,5,6.5,11,12,15,15,16,21]

# Let's see what they look like
plt.scatter(x, y)
plt.show()

# Fit a degree 1 polynomial (line)
# to the data.
a, b = np.polyfit(x,y,1)

# Let's plot the line y = y-bar
plt.scatter(x, y)
plt.plot([0,10],[np.mean(y), np.mean(y)])
plt.show()


# Let's see how the regression did
plt.scatter(x, y)
plt.plot([0,10],[0,a*10+b])
plt.show()
