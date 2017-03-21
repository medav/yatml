import tensorflow as tf
import numpy as np

# Our list of "random" points from the previous lesson
x = [1,1,2,3.5,5,6,7.5,8,8,10]
y = [2,1,5,6.5,11,12,15,15,16,21]

num_pts = len(x)

# Once again, we compute the regression with numpy
# to get a baseline for a and b
a, b = np.polyfit(x,y,1)
print('Baseline Values: a = {}, b = {}'.format(a, b))

# Here we make a placeholder for the input to our
# computational graph.
x_in = tf.placeholder(tf.float32, [None])

# Create **trainable** TensorFlow variables. TensorFlow
# knows since these are created with Variable() that
# they can be trained by the optimizer.
a = tf.Variable(0, dtype=tf.float32, name="weight")
b = tf.Variable(np.mean(y), dtype=tf.float32, name="bias")

# y_pred is the predicted y-value from our graph
y_pred = a * x_in + b

# y_act is a placeholder for the actual y-value
# associated with x_in
y_act = tf.placeholder(tf.float32, [None])

# Define our "loss" function, which is the same
# squared-difference as before. Recall that we do
# this instead of absolute value because it's
# easier to differentiate.
squared_diff = tf.square(y_pred - y_act)

# Create a SGD optimizer (built-in to TensorFlow)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(squared_diff)

# Setup the TensorFlow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# This is the train loop. Here we pick a point, and
# direct TensorFlow to train the graph off that point.
for i in range(10000):
    xp, yp = x[i % num_pts], y[i % num_pts]
    sess.run(train_step, feed_dict={x_in: [xp], y_act: [yp]})

# Grap the final values for a and b after training 
# and print them for the user.
trained_a = sess.run(a)
trained_b = sess.run(b)
print('Trained Values: a = {}, b = {}'.format(trained_a, trained_b))