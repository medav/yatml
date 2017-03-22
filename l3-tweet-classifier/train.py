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

tweets = []
labels = []
idx = 0
all_text = ""

for handle in settings['handles']:
    with open('./tweets/{}.txt'.format(handle), 'r') as f:
        lines = f.readlines()
        tweets = tweets + lines
        labels = labels + [idx] * len(lines)
        idx = idx + 1

tokenizer = Tokenizer(settings['train']['max_words'])
num_classes = len(settings['handles'])
tokenizer.fit_on_texts(tweets)

six.moves.cPickle.dump(tokenizer, open('tokenizer.bin', 'wb'))

x_train = tokenizer.texts_to_sequences(tweets)
x_train = np.array(tokenizer.sequences_to_matrix(x_train, mode='binary'))
y_train = keras.utils.np_utils.to_categorical(labels, num_classes)

model = tweet_classifier_model(settings['train']['max_words'], num_classes)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)

print('Training {} class test classifier...'.format(num_classes))
model.fit(x_train, y_train, nb_epoch=settings['train']['num_epochs'], batch_size=settings['train']['batch_size'], verbose=1, validation_split=0.1)

model.save_weights('classifier_weights.h5')


