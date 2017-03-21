import matplotlib.pyplot as plt
import numpy as np

x = [1,1,2,3.5,5,6,7.5,8,8,10]
y = [2,1,5,6.5,11,12,15,15,16,21]

plt.scatter(x, y)
plt.show()

# y = ax + b
a, b = np.polyfit(x,y,1)

plt.scatter(x, y)
plt.plot([0,10],[0,a*10+b])
plt.show()
