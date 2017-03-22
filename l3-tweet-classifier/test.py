import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
import keras.utils
from model import tweet_classifier_model
import os
import json
import six.moves.cPickle

with open('settings.json') as settings_file:   
    settings = json.load(settings_file)

num_classes = len(settings['handles'])

tokenizer = six.moves.cPickle.load(open('tokenizer.bin', 'rb'))

model = tweet_classifier_model(settings['train']['max_words'], num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights('classifier_weights.h5')

line=''

print('Type stuff!')

while line != 'exit':
    line = input()

    seq = tokenizer.texts_to_sequences([line])
    seq = np.array(tokenizer.sequences_to_matrix(seq, mode='binary'))

    pred = model.predict(seq)
    
    idx = np.argmax(pred[0])
    print('I am {:0.2f}% sure this is from "{}"'.format(pred[0][idx]*100, settings['handles'][idx]))
    print()