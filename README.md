Adversarial Machine Learning
============================

MSc thesis project code

Getting Started
---------------

1. clone the repo and `cd` into the folder
2. create a [YAML](http://yaml.org/) configuration file for the experiment setup.

  - **name**: the name of the index column in the dataframe, a shorter
    value will make it easier to manipulate
  - **key**: a dimension along which the experiment varies
  - **values**: varied for the experiment

  For example, you can have one experiment using the adaline classifier and
  another using the logistic regression classifier. You would express that as:

  ```yaml
  - name: classifier
    key: [classifier, type]
    values:
      - adaline
      - logistic regression
  ```

  Where `[adaline, logistic regression]` is a list of the
  values that `classifier` can take.

  A more general example:

  ```yaml
  - name: classifier
    key: [classifier, type]
    values:
      - adaline
      - logistic regression
      - naive bayes

  - name: attack
    key: [attack, type]
    values:
      - dictionary
      - empty
      - ham
      - focussed

  - name: '% poisoned'
    key: [attack, parameters, percentage_samples_poisoned]
    values:
      - .0
      - .1
      - .2
      - .5
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

3. Run the pipeline, here with 4 threads for example:

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


A few [ipython notebooks](https://ipython.org/notebook.html) showcase the results and brief initial observations/interpretations:

  - analysis of experiment batch [1608232142](https://nbviewer.jupyter.org/urls/gist.github.com/galvanic/2719ae005a16a4f71139fcaa3c4e0eb3/raw/8b4584878bbb6592813b156eb7e6daf9e85aae15/1608232142.ipynb) and corresponding [specifications](https://gist.githubusercontent.com/galvanic/2719ae005a16a4f71139fcaa3c4e0eb3/raw/8b4584878bbb6592813b156eb7e6daf9e85aae15/1608232142.yml)

  - analysis of experiment batches [1608310218](https://gist.github.com/galvanic/7f7a3233fbb52893a63008be9643c8ac#file-1608310218-adaptive-yaml) and [1608302248](https://gist.github.com/galvanic/7f7a3233fbb52893a63008be9643c8ac#file-1608302248-adaptive-yaml), where I looked at the effect of varying the adaptive rate: [online notebook](https://nbviewer.jupyter.org/urls/gist.github.com/galvanic/7f7a3233fbb52893a63008be9643c8ac/raw/5b056187f229f8cc3267e4ebeb9c191d15078090/1608310218.ipynb)

TODO
----

- implement attacker knowledge
- prepare batch test specs to find good learning rates depending on classifier and dataset
- implement different attacks in adaptive experiment pipeline
- write extract functions for:

  - enron
  - MNIST

- test experiments on MNIST
- brainstorm attacks for adaptive convex combination experiment
- implement regret measure

### software eng (ie. not directly important for this project) stuff:

- implement how to store experiment files, prob grouped in batches
- assert all matrix shapes and types
- implement data loading from different filetypes (automatically detect npy, dat, csv, etc.)
- add tests

### optimisations:

- ? optimise pipeline for experiments where the same dataset, same attacks, etc. are used
    or not worth the time ?
  -> look into [Makefile](https://bost.ocks.org/mike/make/) to manage dependencies between files
- profile code
- re-implement logging of intermediate results, but maybe only first few characters, or statistics
  or info on the array (contains nan, something like that), would need to see what is actually useful
- ? [bit arrays](https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries)
- ? [explicitly free memory](https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python)
- ? make ipython notebook on MI and feature selection for ham
