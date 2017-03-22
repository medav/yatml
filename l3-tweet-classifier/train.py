import keras
import keras.utils
from keras.preprocessing.text import Tokenizer
from tweet_model import tweet_classifier_model
import six.moves.cPickle
import numpy as np
import os
import json

# First, read the settings.json into a dict
with open('settings.json') as settings_file:   
    settings = json.load(settings_file)

# Our tweets as text and labels as integers
tweets = []
labels = []

idx = 0

# Here we open up all the downloaded tweets and read
# them into memory. Each entry in tweets is a string
# containing the text, and the corresponding entry
# in labels contains an integer 0 through N where N-1
# is the number of handles.
for handle in settings['handles']:
    with open('./tweets/{}.txt'.format(handle), 'r') as f:
        lines = f.readlines()
        tweets = tweets + lines
        labels = labels + [idx] * len(lines)
        idx = idx + 1

# Next we create a Tokenizer (from Keras's built in 
# preprocessing library) and "fit" it on our text. 
# The Tokenizer will read all of the tweets in our list
# and pick the top max_words words observed from them. 
# It will then construct a mapping from these words to
# indices in a vector of length max_words.
tokenizer = Tokenizer(settings['train']['max_words'])
num_classes = len(settings['handles'])
tokenizer.fit_on_texts(tweets)

# After the Tokenizer as been constructed, dump it to
# a file for use in testing/deployment
six.moves.cPickle.dump(tokenizer, open('tokenizer.bin', 'wb'))

# The "X" (input data) to our network is our list of tweets transformed
# into a list of vectors where each entry corresponds to the count of the
# occurrences of the word matching that index. We first convert the tweets
# to sequences of indices (corresponding to the words) then convert those
# sequences to vectors.
#
# The result is a MxN matrix (array) where M is the number of tweets and
# N is max_words.
x_train = tokenizer.texts_to_sequences(tweets)
x_train = np.array(tokenizer.sequences_to_matrix(x_train, mode='count'))

# The "Y" (output classification) of our network is the label integers
# converted by "one-hot" encoding to a vector of length N for N classes.
# For example, if there were 4 classes, each label would map as:
#     0 -> [1, 0, 0, 0]
#     1 -> [0, 1, 0, 0]
#     2 -> [0, 0, 1, 0]
#     3 -> [0, 0, 0, 1]
#
# Note that these are all **valid probability distributions**. This is 
# intentional because classifier networds **learn a probability distribution**.
y_train = keras.utils.np_utils.to_categorical(labels, num_classes)

# Now we create our model with the given information in settings.json.
model = tweet_classifier_model(settings['train']['max_words'], num_classes)

# This prints a summary of the model to stdout.
model.summary()

# Here we compile our model and set it up for training.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Now, we train. This is where the magic happens.
print('Training {} class test classifier...'.format(num_classes))
model.fit(x_train, y_train, nb_epoch=settings['train']['num_epochs'], batch_size=settings['train']['batch_size'], verbose=1, validation_split=0.1)

# Finally, save what we learned to a file.
model.save_weights('classifier_weights.h5')


