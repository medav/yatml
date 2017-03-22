from keras.layers import Dense, Dropout, Activation
from keras.models import Model
from keras.layers import Input


def tweet_classifier_model(max_words, num_classes):
    """
    Creates a Keras Model that represents our neural-network structure.

    You can print a summary of this model by calling .Summary() on
    the output from this function.
    """

    # We start with an input that has a length of max_words
    input = Input(shape=(max_words, ), dtype='float32')

    # Here we build our network. It consists of:
    #   1. Fully connected layer with 512 hidden units
    #   2. Restricted linear unit (ReLU) activation
    #   3. A fully connected layer with 4 hidden units
    #   4. Softmax activation
    #
    # Note that #3 above is our "classifier" layer as it 
    # takes our current network output and reduces it down 
    # to the final bins that our classification is based on
    # 
    # Also, these calls use two sets of parenthesis. This
    # is because the Keras layer functions return a lambda
    # that takes one argument - the input tensor to the 
    # constructed layer. 

    fc1 = Dense(512)(input)
    relu = Activation('relu')(fc1)
    fc2 = Dense(4)(relu)
    softmax = Activation('softmax')(fc2)

    # Once we construct the network, we just ask
    # Keras to make it a model and return the result.
    return Model(input=input, output=softmax)
