from keras.layers import Dense, Dropout, Activation
from keras.models import Model
from keras.layers import Input


def tweet_classifier_model(max_words, num_classes):
    input = Input(shape=(max_words, ), dtype='float32')

    fcn1 = Dense(512)(input)
    relu = Activation('relu')(fcn1)
    fcn2 = Dense(4)(relu)
    softmax = Activation('softmax')(fcn2)

    return Model(input=input, output=softmax)
