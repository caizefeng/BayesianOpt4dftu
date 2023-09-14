import logging
import sys


class BoLogging:

    def __init__(self):
        self._root_name = "BayesOpt4dftu"
        self._root_logger = logging.getLogger(self._root_name)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        self._ch = logging.StreamHandler(sys.stdout)
        self._ch.setFormatter(formatter)
        self._root_logger.addHandler(self._ch)
        self._root_logger.setLevel(logging.INFO)

    def get_logger(self, subname) -> logging.Logger:
        return logging.getLogger('.'.join((self._root_name, subname)))  # child logger
