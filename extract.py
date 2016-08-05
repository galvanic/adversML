#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import re
from email.parser import Parser
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import time
import pickle
from itertools import starmap


def process_trec_dataset(trec_folderpath, countvectorizer_params, verbose=True):
    '''
    Returns features and labels for the TREC dataset, given the
    filepath of the index (where the labels are).

    Inputs:
    - trec_folderpath: assumes absolute filepath and that the TREC
                       folder structure has not been changed
    - options: dictionary of values for sklearn's CountVectorizer
               arguments
    - max_emails: for debugging

    Outputs:
    - X: sparse matrix of features
    - Y: N * 1 matrix of binary labels, 1 for spam, 0 for ham
         with N: number of samples

    TODO yield labels then features one after the other so as
         to not have to keep everything in memory ?
    TODO strip html stuff ? or valuable features ?
    TODO stemming
    TODO use vocabulary and stop words to not keep all features
    '''
    index_filepath = os.path.join(trec_folderpath, 'full', 'index')

    if verbose: print('Getting email labels')
    ## Get the email filepaths from the file where the
    ## corresponding email spam/ham label is (see regex)
    with open(index_filepath, 'r') as ifile:
        raw_labels = ifile.readlines()

    labels = []
    for label in raw_labels:
        pattern = '((?:sp|h)am) .*?inmail\\.(\\d{1,5})'
        match = re.search(pattern, label)
        if match:
            category = match.group(1)
            email_num = match.group(2)
            labels.append((email_num, category))

    indices, labels = zip(*labels)

    ## format the labels into desired matrix
    d = {'spam': 1, 'ham': -1}
    Y = np.fromiter(map(lambda l: d[l], labels), dtype=np.int8)
    Y = Y.reshape(len(Y), 1)

    if verbose: print('Getting email contents')
    ## Get email contents to later extract features from
    parser = Parser()

    corpus = []
    for email_num in indices:
        filepath = os.path.join(trec_folderpath, 'data', 'inmail.%s' % email_num)

        with open(filepath, 'r', encoding='ISO-8859-1') as ifile:
            email = parser.parse(ifile)

        ## TODO also keep the subject header

        ## code and details from:
        ## https://stackoverflow.com/questions/17874360/python-how-to-parse-the-body-from-a-raw-email-given-that-raw-email-does-not/32840516#32840516
        if email.is_multipart():
            for part in email.walk():
                content_type = part.get_content_type()
                content_dispo = str(part.get('Content-Disposition'))

                # skip any text/plain (txt) attachments
                ## include both 'text/plain' and 'text/html'
                if 'text' in content_type and 'attachment' not in content_dispo:
                    body = part.get_payload(decode=False)
                    break ## only keep the first email
        else:
            body = email.get_payload(decode=False)

        corpus.append((email_num, body))

    indices, corpus = zip(*corpus)

    if verbose: print('Getting email features')
    ## Extract features from emails
    ## we are only interested in presence of a word not frequency
    vectorizer = CountVectorizer(
                    dtype=np.bool_,
                    **countvectorizer_params)
    X = vectorizer.fit_transform(corpus) ## X: features
    ## keep X as a sparse matrix to take up less memory space

    if verbose: print(X.shape)

    return X, Y, vectorizer.get_feature_names()


def main(infolder, outfolder, sparse_X=False):
    '''
    Process dataset to get features and labels, including option
    to save these to a file.

    Inputs:
    - infolder: path of the trec dataset folder
    - outfolder: path of the folder to save processed dataset to
    - sparse_X: whether to save the X features data as a sparse array or a
                regular numpy array
    '''
    infolder, outfolder = map(os.path.abspath, [infolder, outfolder])

    ## set up options
    countvectorizer_params = {
        'encoding': 'utf-8',
        'decode_error': 'strict',
        'strip_accents': 'unicode',
        'stop_words': 'english',
        'lowercase': True,
        'min_df': 1,
        'max_features': 10000,
        'vocabulary': None,
        'binary': True,
    }

    ## Get the data
    X, Y, feature_names = process_trec_dataset(
        trec_folderpath=infolder,
        countvectorizer_params=countvectorizer_params,
        verbose=True)

    if not sparse_X:
        try:
            X = X.toarray()
        except MemoryError:
            print('MemoryError: could not save as Numpy array, saving as sparse array instead')
            pass

    ## save the data
    saved_at = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    ## TODO make folder if doesn't exist yet
    outfilepath = os.path.join(outfolder, 'trec2007-%s' % saved_at)

    with open('%s-features.dat' % outfilepath, 'wb') as outfile:
        pickle.dump(X, outfile)

    with open('%s-labels.dat' % outfilepath, 'wb') as outfile:
        pickle.dump(Y, outfile)

    ## make readme file with details of processing
    with open('%s-readme.md' % outfilepath, 'w') as outfile:

        title = time.strftime('Processed on %y/%m/%d at %H:%M',
            time.localtime(time.time()))
        outfile.write('%s\n' % title)

        outfile.write('TREC 2007 dataset\n')

        outfile.write('X: (%d x %d) \t%s \tdtype: %s\n' % (*X.shape, type(X).__name__, X.dtype))
        outfile.write('Y: (%d x %d) \t%s \tdtype: %s\n' % (*Y.shape, type(Y).__name__, Y.dtype))

        subtitle = 'CountVectorizer params'
        outfile.write('\n%s\n%s\n\n%s\n' % (
            subtitle,
            '-'*len(subtitle),
            '\n'.join(list(starmap(lambda k,v: '%s: %s' % (k,v),
                countvectorizer_params.items()))) ))

        subtitle = 'Features'
        outfile.write('\n%s\n%s\n\n%s\n' % (
            subtitle,
            '-'*len(subtitle),
            '\n'.join(feature_names) ))

    return


if __name__ == '__main__':

    infolder = sys.argv[1]
    outfolder = sys.argv[2]
    sys.exit(main(infolder, outfolder))

