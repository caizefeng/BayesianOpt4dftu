import os
import shutil
import sys
import warnings

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

        # Temporary Bayesian optimization log
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

    def clean_up_temp_files(self):
        shutil.move(self._config.tmp_u_path, self._config.u_path)
        os.remove(self._config.tmp_config_path)

        self._logger.info("Temporary files removed.")


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__name__} is deprecated", DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return wrapper  # noqa
