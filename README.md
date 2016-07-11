Adversarial Machine Learning
============================

MSc thesis code for:


- feature extraction from spam datasets:

  - from the [TREC 2007 dataset](http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html): **extract.py**

  the features for an email are the (binary) presence or absence of a token (a word)


- poisoning attacks on the training data:

  - empty attack: **empty.py**

    all the features of the poisoning emails are set to zero


- training and testing of binary classification models:

  - [ADALINE](https://en.wikipedia.org/wiki/ADALINE) model: **adaline.py**

    like the better known [perceptron](), a single layer neural network that calculates a weighted sum of inputs. the difference is that it trains on this weighted sum, and outputs the thresholded weighted sum

  - [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) model: **naivebayes.py**


A few [ipython notebooks](https://ipython.org/notebook.html) try the implementations out and can serve as examples: [**notebooks/**](https://github.com/galvanic/adversML/tree/master/notebooks)
