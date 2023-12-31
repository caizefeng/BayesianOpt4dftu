import os
import shutil
import sys
import time
import warnings
from contextlib import contextmanager

from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.logging import BoLoggerGenerator


class TempFileManager:
    _logger = BoLoggerGenerator.get_logger("TempFileManager")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def setup_temp_files(self):
        # Temporary config
        shutil.copyfile(self._config.config_path, self._config.tmp_config_path)

        # Temporary Bayesian Optimization log
        header = []
        for i, u in enumerate(self._config.which_u):
            header.append(f"U_ele_{str(i + 1)}")

        if os.path.exists(self._config.tmp_u_path):
            os.remove(self._config.tmp_u_path)

        if self._config.alpha_mag:
            with open(self._config.tmp_u_path, 'w+') as f:
                f.write(f"{(' '.join(header))} "
                        f"{self._config.column_names['band_gap']} "
                        f"{self._config.column_names['delta_gap']} "
                        f"{self._config.column_names['delta_band']} "
                        f"{self._config.column_names['delta_mag']} \n")
        else:
            with open(self._config.tmp_u_path, 'w+') as f:
                f.write(f"{(' '.join(header))} "
                        f"{self._config.column_names['band_gap']} "
                        f"{self._config.column_names['delta_gap']} "
                        f"{self._config.column_names['delta_band']} \n")

        self._logger.info("Temporary files initiated.")

    def clean_up(self):
        shutil.move(self._config.tmp_u_path, self._config.u_path)
        os.remove(self._config.tmp_config_path)

        self._logger.info("Temporary files removed.")


@contextmanager
def task_timer(label, logger):
    logger.info(f"{label} begins.")
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed_time = end - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        hours_text = f"{int(hours)} Hour{'s' if hours != 1 else ''}"
        minutes_text = f"{int(minutes)} Minute{'s' if minutes != 1 else ''}"
        seconds_text = f"{seconds:.2f} Second{'s' if seconds != 1 else ''}"
        logger.info(f"{label} completed in {hours_text} {minutes_text} {seconds_text}.")


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def find_and_readlines_first(directory, file_list):
    for filename in file_list:
        if os.path.exists(os.path.join(directory, filename)):
            with open(os.path.join(directory, filename), 'r') as file:
                return file.readlines()
    raise FileNotFoundError(f"None of these files ({file_list}) were found.")


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper  # noqa
