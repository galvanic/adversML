Adversarial Machine Learning
============================

MSc thesis project code

Getting Started
---------------

1. clone the repo and `cd` into the folder
2. create a [YAML](http://yaml.org/) config file for the experiment setup.

  - **name**: the name of the index column in the dataframe, a shorter
    value will make it easier to manipulate
  - **key**: a dimension along which the experiment varies
  - **values**: varied for the experiment

  For example, you can have one experiment using the adaline classifier and
  another using the logistic regression classifier. You would express that as:

  ```yaml
  - name: classifier
    key: [classifier, type]
    values: [adaline, logistic regression]
  ```

  Where `[adaline, logistic regression]` is a list of the
  values that `classifier` can take.

  A more general example:

  ```yaml
  - name: classifier
    key: [classifier, type]
    values: [adaline, logistic regression]

  - name: attack
    key: [attack, type]
    values: [dictionary, focussed, empty, ham]

  - name: '% poisoned'
    key: [attack, parameters, percentage_samples_poisoned]
    values: [.0, .1, .2, .5]
  ```

  The order of the (name, key, values) group counts, as that
  is the order the columns will be in the DataFrame results (but
  the order can then be changed).

3. Check the values in [**default_spec.yaml**](https://github.com/galvanic/adversarialML/blob/master/default_spec.yaml), especially the `dataset_filename` key (although this can also
  be a key that is varied.

    Here is a full example:

    ```yaml
    dataset_filename: trec2007-1607252257
    label_type:
      ham_label: -1
      spam_label: 1

    classifier:
      type: none
      training_parameters: {}
      testing_parameters: {}

    attack:
      type: none
      parameters: {}
    ```

4. decide how many threads to run your code on. Given the size of the dataset
  in memory, allocate at least "double" the amount of RAM. For example, if
  you run on 8 cores, make sure you have 16GB of RAM otherwise you will get a
  `MemoryError`.

3. Run the pipeline, here with 4 cores for example:

    ```shell
    python3 main.py ~/path/to/experiment/config.yaml ~/folder/where/dataset/is/ ~/folder/to/save/results/to/ 4
    ```

Details
-------

This repo includes code for:


- feature extraction from spam datasets:

  - from the [TREC 2007 dataset](http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html): **extract.py**

  the features for an email are the (binary) presence or absence of a token (a word)


- poisoning attacks on the training data:

  - dictionary attack: [**dictionary.py**](https://github.com/galvanic/adversarialML/blob/master/attacks/dictionary.py)

    all the features of the poisoned emails are set to one

  - empty attack: [**empty.py**](https://github.com/galvanic/adversarialML/blob/master/attacks/empty.py)

    all the features of the poisoned emails are set to zero

  - ham attack: [**ham.py**](https://github.com/galvanic/adversarialML/blob/master/attacks/ham.py)

    contaminating emails contain features indicative of the ham class

  - focussed attack: [**focussed.py**](https://github.com/galvanic/adversarialML/blob/master/attacks/focussed.py)


- training and testing of binary classification models:

  - [ADALINE](https://en.wikipedia.org/wiki/ADALINE) model: [**adaline.py**](https://github.com/galvanic/adversarialML/blob/master/classifiers/adaline.py)

    like the better known [perceptron](), a single layer neural network that calculates a weighted sum of inputs. the difference is that it trains on this weighted sum, and outputs the thresholded weighted sum

  - [Logistic regression]() model: [**logistic_regression**](https://github.com/galvanic/adversarialML/blob/master/classifiers/logistic_regression.py)

  - [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) model: [**naivebayes.py**](https://github.com/galvanic/adversarialML/blob/master/classifiers/naivebayes.py)


A few [ipython notebooks](https://ipython.org/notebook.html) try the implementations out and can serve as examples: [**notebooks/**](https://github.com/galvanic/adversML/tree/master/notebooks)

TODO
----

- implement stochastic and batch training: refactor out gradient descent from training functions
  and take stochastic or batch as argument

### software eng (ie. not directly important for this project) stuff:

- implement how to store experiment files, prob grouped in batches
- assert all matrix shapes and types
- implement logging
- implement other termination conditions: detect convergence
- implement data loading from different filetypes
- add tests

### optimisations:

- ? optimise pipeline for experiments where the same dataset, same attacks, etc. are used
    or not worth the time ?
- profile code
- ? [bit arrays](https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries)
- ? [explicitly free memory](https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python)
- ? make ipython notebook on MI and feature selection for ham
