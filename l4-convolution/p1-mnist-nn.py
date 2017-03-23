import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop

# This is adapted from https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

# Some "hyperparameters"
batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# A simple NN takes a flat vector (instead of 28x28 square patches)
# so we reshape it and convert it to floats between 0 and 1.
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

# Convert the labels to "one-hot" encoded vectors.
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

# Here we construct our simple 2-layer fully connected network.
input = Input(shape=(784, ), dtype='float32')
fc1 = Dense(512, activation='relu')(input)
fc2 = Dense(512, activation='relu')(fc1)
classifier = Dense(10, activation='softmax')(fc2)

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