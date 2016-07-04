#!/usr/bin/env python
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


def process_trec_dataset(trec_folderpath, options,
         max_emails=None, verbose=True):
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
    '''
    index_filepath = os.path.join(trec_folderpath, 'full', 'index')

    if verbose: print('Getting email labels')
    ## Get the email filepaths from the file where the
    ## corresponding email spam/ham label is (see regex)
    with open(index_filepath, 'r') as ifile:
        raw_labels = ifile.readlines()
        if max_emails: raw_labels = raw_labels[:max_emails]

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
    d = {'spam': 1, 'ham': 0}
    Y = np.fromiter(map(lambda l: d[l], labels), dtype=np.uint8)
    Y = np.matrix(Y.reshape(len(Y), 1))

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
                if content_type == 'text/plain' and 'attachment' not in content_dispo:
                    body = part.get_payload(decode=False)
                    break ## only keep the first email
        else:
            body = email.get_payload(decode=False)

        ## TODO strip html stuff ? or valuable features ?
        try:
            corpus.append((email_num, body))
        except NameError:
            pass

    indices, corpus = zip(*corpus)

    if verbose: print('Getting email features')
    ## Extract features from emails
    ## we are only interested in presence of a word not frequency
    vectorizer = CountVectorizer(
                    binary=True,
                    dtype=np.bool_,
                    **options)
    X = vectorizer.fit_transform(corpus) ## X: features
    ## keep X as a sparse matrix to take up less memory space

    return X, Y, vectorizer.get_feature_names()


def main(ifilepath, outfolder):
    '''
    Process dataset to get features and labels, including option
    to save these to a file.
    '''
    ifilepath, outfolder = map(os.path.abspath, [ifilepath, outfolder])

    ## set up options
    options = {
        'min_df': 1,
        'max_features': 10000,
    }

    ## Get the data
    X, Y, feature_names = process_trec_dataset(
        trec_folderpath=ifilepath,
        options=options,
        max_emails=100,
        verbose=True)

    ## Save the data
    saved_at = time.strftime('%y%m%d%H%M', time.localtime(time.time()))
    ## TODO make folder if doesn't exist yet
    outfilepath = os.path.join(outfolder, 'trec2007-%s' % saved_at)

    with open('%s-features.dat' % outfilepath, 'wb') as outfile:
        pickle.dump(X, outfile)

    with open('%s-labels.dat' % outfilepath, 'wb') as outfile:
        pickle.dump(Y, outfile)

    ## make readme file with details of processing
    with open('%s-readme.md' % outfilepath, 'w') as outfile:

        title = time.strftime('Data processed on %y/%m/%d at %H:%M',
            time.localtime(time.time()))
        outfile.write('%s\n%s\n' % (title, '='*len(title)))

        outfile.write('TREC 2007 dataset\n')

        subtitle = 'Options'
        outfile.write('\n%s\n%s\n\n%s\n' % (
            subtitle,
            '-'*len(subtitle),
            '\n'.join(list(starmap(lambda k,v: '%s: %s' % (k,v), options.items()))) ))

        subtitle = 'Features'
        outfile.write('\n%s\n%s\n\n%s\n' % (
            subtitle,
            '-'*len(subtitle),
            '\n'.join(feature_names) ))

    return


if __name__ == '__main__':

    ifilepath = sys.argv[1]
    outfolder = sys.argv[2]
    sys.exit(main(ifilepath, outfolder))

