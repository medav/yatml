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

# Load the Tokenizer from disk. This lets us take 
# NEW text and convert it to feature vectors.
tokenizer = six.moves.cPickle.load(open('tokenizer.bin', 'rb'))

# Create the Keras model again
model = tweet_classifier_model(settings['train']['max_words'], len(settings['handles']))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Here we load the learned weights from the 
# training process.
model.load_weights('classifier_weights.h5')

# Our test program will just read in lines of
# text then predict the matching class with Keras.

line=''
print('Type stuff!')

while line != 'exit':
    line = input()

    seq = tokenizer.texts_to_sequences([line])
    seq = np.array(tokenizer.sequences_to_matrix(seq, mode='binary'))

    pred = model.predict(seq)
    idx = np.argmax(pred[0])
    
    print('I am {:0.2f}% sure this is from "{}"'.format(pred[0][idx]*100, settings['handles'][idx]))
    print('Others:')
    for j in range(len(settings['handles'])):
        print('"{}": {:0.2f}%'.format(settings['handles'][j], pred[0][j]*100))

    print()