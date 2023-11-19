import inspect
import logging
import sys


class BoLoggerGenerator:
    _loggers = {}
    _root_name = "BayesOpt4dftu"
    _root_logger = logging.getLogger(_root_name)
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s]: %(message)s', '%Y-%m-%d %H:%M:%S')
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(formatter)
    _root_logger.addHandler(_ch)
    _root_logger.setLevel(logging.INFO)

    @staticmethod
    def get_logger(subname=None) -> logging.Logger:
        if subname is None:
            # Inspect the stack to find the name of the caller's file
            frame = inspect.currentframe()
            file_name = frame.f_back.f_globals["__name__"].split('.')[-1]
            subname = file_name

        if subname not in BoLoggerGenerator._loggers:
            # A logger named "A.B" is considered a child of the logger named "A"
            logger = logging.getLogger(f"{BoLoggerGenerator._root_name}.{subname}")
            BoLoggerGenerator._loggers[subname] = logger

        return BoLoggerGenerator._loggers[subname]
