# coding: utf-8
'''
'''
import numpy as np
np.set_printoptions(precision=2, threshold=10e6, linewidth=10e10)

import threading
tls = threading.local()

import yaml
import logging
import logging.config
tls.logger = logging.getLogger()

## load logging configuration settings
with open('config/logging.yaml', 'r') as infile:
    LOGGING_CONFIG = yaml.load(infile)

'''
DEFAULT_FORMAT = LOGGING_CONFIG['formatters']['default']['format']
DEFAULT_DATEFMT = LOGGING_CONFIG['formatters']['default']['datefmt']
CONCISE_FORMAT = LOGGING_CONFIG['formatters']['concise']['format']
CONCISE_DATEFMT = LOGGING_CONFIG['formatters']['concise']['datefmt']
'''


def log(get_experiment_id=None, from_argument='args'):

    def decorator(function):
        def wrapper(*args, **kwargs):

            if get_experiment_id.__name__ == '<lambda>':
                tls.experiment_id = get_experiment_id(locals()[from_argument])

            tls.logger = logging.getLogger(tls.experiment_id)
            tls.logger.propagate = True
            '''
            tls.logger.propagate = False

            if not tls.logger.handlers:

                ## log to console
                new_format = '%s [experiment #%s]' % (DEFAULT_FORMAT, tls.experiment_id)
                new_formatter = logging.Formatter(fmt=new_format, datefmt=DEFAULT_DATEFMT)

                console_handler = logging.StreamHandler()
                console_handler.setFormatter(new_formatter)
                console_handler.setLevel('INFO')
                tls.logger.addHandler(console_handler)

                ## log to file
                new_format = '%s [experiment #%s]' % (CONCISE_FORMAT, tls.experiment_id)
                new_formatter = logging.Formatter(fmt=new_format, datefmt=CONCISE_DATEFMT)

                log_filepath = tls.logger.parent.handlers[-1].baseFilename
                file_handler = logging.FileHandler(log_filepath)
                file_handler.setFormatter(new_formatter)
                file_handler.setLevel('DEBUG')
                tls.logger.addHandler(file_handler)
            '''

            value = function(*args, **kwargs)
            return value
        return wrapper

    if get_experiment_id.__name__ == '<lambda>':
        return decorator

    else:
        function = get_experiment_id
        return decorator(function)

