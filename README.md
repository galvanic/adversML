Adversarial Machine Learning
============================

MSc thesis project code

Getting Started
---------------

1. clone the repo and `cd` into the folder
2. modify `parameter_ranges` and check the values in `fixed_parameters`, especially
   the `dataset_filename` key, in the `main` function of **main.py**.

  - `parameter_ranges` is a list of 2-tuples, where the first value is the key, a
    dimension along which the experiment varies. For example, you can have one
    experiment using the adaline classifier and another using the logistic
    regression classifier. You would express that as:

    ```python
    parameter_ranges = [
        ('classifier', ['adaline', 'logistic regression']),
    ]
    ```

    Where `['adaline', 'logistic regression']` is a list of the values that
    `'classifier'` can take.

    A more general example:

    ```python
    parameter_ranges = [
        ('classifier', ['adaline', 'logistic regression', 'naive bayes']),
        ('attack', ['dictionary', 'focussed', 'empty', 'none']),
        (('attack_parameters', 'percentage_samples_poisoned'), [.0, .1, .2, .5]),
        ('repetition', range(1, 20+1)),
    ]
    ```

  - `fixed_parameters` is a dictionary with default values for the experiment.
    For example, it specifies the filename of the dataset to use.

    Here's a full example:

    ```python
    fixed_parameters = {
        'dataset_filename': 'trec2007-1607201347',
        'label_type': {
            'ham_label': -1,
            'spam_label': 1,
        },
        'training_parameters': {},
        'testing_parameters': {},
        'attack_parameters': {},
        'attack': None,
    }
    ```

3. Run the pipeline:

    ```shell
    python3 main.py ~/folder/where/dataset/is/ ~/folder/to/save/results/to/
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

- implement stochastic and batch training

### software eng (ie. not directly important for this project) stuff:

- decide and implement how to store experiment spec and results (+logging), prob grouped in batches
- assert all matrix shapes and types
- seek feedback to refine pipeline
- implement logging
- implement data loading from different filetypes

### optimisations:

- ? optimise pipeline for experiments where the same dataset, same attacks, etc. are used
    or not worth the time ?
- profile code
- ? [bit arrays](https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries)
- ? [explicitly free memory](https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python)
- ? make ipython notebook on MI and feature selection for ham
