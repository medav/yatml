import numpy as np
import tensorflow as tf

# First, let's create a "placeholder", which is
# the value we as the user supply to the graph.
x = tf.placeholder(tf.float32, shape=None)

# Here we create two variables: a and b. We can
# give them default values. 
a = tf.Variable(1, dtype=tf.float32, name="weight")
b = tf.Variable(2, dtype=tf.float32, name="bias")

# Now we form the computational graph:
f = a*x
y = f + b

# N.B. We could just as easily write:
# y = a*x + b

# Now we ask TensorFlow for a new session to run
# things on and initialize variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Here is the input we will feed to the graph.
our_x = 1.0

# Now run the computational graph.
y_out = sess.run(y, feed_dict={x: our_x})

# Print the result.
print('y_out = {}'.format(y_out))