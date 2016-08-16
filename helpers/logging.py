# coding: utf-8
'''
Put as much of the logging(anything recorded for posteriority) logic in here
'''
import numpy as np
np.set_printoptions(precision=2, threshold=10e6, linewidth=10e10)

import threading
tls = threading.local()

import os
import os.path
import subprocess
import yaml

import logging
import logging.config
tls.logger = logging.getLogger()

## load logging configuration settings
code_dirpath = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
log_config_filepath = os.path.join(code_dirpath, 'config', 'logging.yaml')
with open(log_config_filepath, 'r') as infile:
    LOGGING_CONFIG = yaml.safe_load(infile)

## logging git commit hash in which code is running
cwd = os.getcwd()
os.chdir(code_dirpath)
COMMIT_HASH = subprocess.getoutput('git log -1 --format=%H')
os.chdir(cwd)


def log(get_experiment_id=None, from_argument='args'):

    def decorator(function):
        def wrapper(*args, **kwargs):

            if get_experiment_id.__name__ == '<lambda>':
                tls.experiment_id = get_experiment_id(locals()[from_argument])

            tls.logger = logging.getLogger(tls.experiment_id)
            tls.logger.propagate = True

            value = function(*args, **kwargs)
            return value
        return wrapper

    if get_experiment_id.__name__ == '<lambda>':
        return decorator

    else:
        function = get_experiment_id
        return decorator(function)

