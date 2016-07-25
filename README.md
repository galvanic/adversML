Adversarial Machine Learning
============================

MSc thesis project code.

Getting Started
---------------

1. clone the repo and `cd` into the folder
2. modify `experiments`, a list of experiment specs in the `main` function.

    You can add more experiments, create their specs automatically in a loop, etc.
    Especially check the `dataset_filename` key.

    For example:

    ```python
    experiments = [
        {
            'dataset': 'trec2007',
            'dataset_filename': 'trec2007-1607201347',
            'feature_extraction_parameters': {
            },
            'label_type': {
                'ham_label': -1,
                'spam_label': 1,
            },
            'attack': None,
            'attack_parameters': {
                'percentage_samples_poisoned': 0.1,
            },
            'classifier': adaline,
            'training_parameters': {
                'learning_rate': 0.06,
                'initial_weights': None,
                'termination_condition': max_iters(40),
                'verbose': False,
            },
            'testing_parameters': {
            },
        },

        {
            'dataset': 'trec2007',
            'dataset_filename': 'trec2007-1607201347',
            'feature_extraction_parameters': {
            },
            'label_type': {
                'ham_label': -1,
                'spam_label': 1,
            },
            'attack': None,
            'attack_parameters': {
                'percentage_samples_poisoned': 0.1,
            },
            'classifier': naivebayes,
            'training_parameters': {
            },
            'testing_parameters': {
            },
        },
    ]
    ```

3. Run the pipeline:

    ```shell
    python3 pipeline.py
    ```

Details
-------

This repo includes code for:


- feature extraction from spam datasets:

  - from the [TREC 2007 dataset](http://plg.uwaterloo.ca/~gvcormac/treccorpus07/about.html): **extract.py**

  the features for an email are the (binary) presence or absence of a token (a word)


- poisoning attacks on the training data:

  - empty attack: **empty.py**

    all the features of the poisoning emails are set to zero

  - ham attack: **hamattack.py**

    contaminating emails contain features indicative of the ham class


- training and testing of binary classification models:

  - [ADALINE](https://en.wikipedia.org/wiki/ADALINE) model: **adaline.py**

    like the better known [perceptron](), a single layer neural network that calculates a weighted sum of inputs. the difference is that it trains on this weighted sum, and outputs the thresholded weighted sum

  - [Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) model: **naivebayes.py**


A few [ipython notebooks](https://ipython.org/notebook.html) try the implementations out and can serve as examples: [**notebooks/**](https://github.com/galvanic/adversML/tree/master/notebooks)

TODO
----

- implement mutual information
- add other dataset extraction, classifier models, and attacks
- assert all matrix shapes and types
- decide on what to do for the different ham labels
- decide on appropriate format/type for experiment and how to post-process it
- seek feedback to refine pipeline
- implement data loading from different filetypes
- implement only part of features are ham in ham attack (does this make sense since
  we are in a poisoning and not an evasion attack?)
- organise code into folders

optimisations:

- implement stochastic and batch training
- ? optimise pipeline for experiments where the same dataset, same attacks, etc. are used
    or not worth the time ?
- profile code
- ? [bit arrays](https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries)
- ? [explicitly free memory](https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python)
- ? worth making an ipython notebook on MI and feature selection for ham
- implement logging
