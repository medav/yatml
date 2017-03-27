# Tweet Classifier DNN

## What is this?
This is an end-to-end example of how to build a neural network in Keras to classify text. Specifically, this example takes tweets pulled from some public accounts and trains a network to classify text as those tweets. 

## Usage
### Training
To train the network, simply run
~~~
PS> python .\train.py
~~~

### Testing
To test the network, run
~~~
PS> python .\test.py
~~~

## Tweet Collection
You can add more handles in `settings.json`. You will need to setup a Twitter API keys (I will add instructions on this later) then run the `collector.py` script:
~~~
PS> python .\collector.py
~~~

This will download ~400 tweets from each handle in the settings file. You can then retrain the network and test it!
