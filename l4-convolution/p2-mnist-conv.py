import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop

# This is adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

# Some "hyperparameters"
batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The Conv-Net takes images in their original shape. The MNIST dataset
# gives us 28x28 pixel images where each pixel only has one color channel,
# hence each image is a 28x28x1 array.
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# Convert the labels to "one-hot" encoded vectors
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

# Here we construct a Convolutional Network with 2 convolutions,
# one fully connected layer and one classification layer. We start
# with an input of shape (28, 28, 1) for a 28x28 image with one
# color channel:
input = Input(shape=(28, 28, 1))

# Next, we create a convolution layer. It will contain 32 filters, each
# of which is a 3x3 patch that slides over the input image. The result
# of applying one filter to the input is an output of the same shape 
# where each pixel of output corresponds to the same pixel of input convolved 
# with its 8 neighbors (for 3x3 filter). The 32 filters then generate 32
# output channels. I.e. the output shape of this layer is (28, 28, 32)
conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input)

# Here we perform a max pool with 2x2 filters. Here we break down the previous
# image into squares of 2x2 and choose the max value to represent the whole 
# patch. This effectively reduces each spacial dimension by a factor of 2.
# The output shape here, then, would be (14, 14, 32)
pool1 = MaxPooling2D((2, 2), border_mode='same')(conv1)

# Now we perform a second convolution with 3x3 filters. Note that here a 
# filter operates over 9 pixels of the pooled image, which effecively means
# the filter can capture features from a 18x18 patch of original pixels. This
# great because it allows later convolutions to learn more abstract features.
# The resulting shape here is (14, 14, 64).
conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)

# Then we perform another max pool with 2x2, reducing the image by another
# factor of 2 in each dimension, resulting in a shape (7, 7, 64)
pool2 = MaxPooling2D((2, 2), border_mode='same')(conv2)

# Now we flatten the above layer to be shape (7*7*64) = (3136)
flat = Flatten()(pool2)

# Then we make one hidden fully connected layer
fc1 = Dense(128, activation='relu')(flat)

# Finally, we create a classification layer which is just a fully connected
# layer with a softmax activation function.
classifier = Dense(10, activation='softmax')(fc1)

model = Model(input=input, output=classifier)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Now train!
history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=epochs,
                    verbose=1, validation_data=(x_test, y_test))

# Finally, evaluate the model with the provided testing data.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])