import numpy as np
import tensorflow as tf

# First, let's create a "placeholder", which is
# the value we as the user supply to the graph.
x = tf.placeholder(tf.float32, [2, 2])

# Now let's create a constant matrix in the graph
mat = tf.constant([[0,1],[1,0]], dtype=tf.float32)

# The output of our graph is x*mat as a matrix
# multiplication.
result = tf.matmul(x, mat)

# Now we ask TensorFlow for a new session to run
# things on.
sess = tf.InteractiveSession()

# Here is the input we will feed to the graph.
our_x = np.array([[1,2],[3,4]])

# Now run the computational graph.
tf_out = sess.run(result, feed_dict={x: our_x})

# Print the result.
print(tf_out)