import logging
import os
import shutil
import sys

from BayesOpt4dftu.configuration import Config


class BoLoggerGenerator:

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


class TempFileManager:
    def __init__(self, config):
        self._config = config  # type: Config

    def setup_temp_files(self):
        # Temporary config
        shutil.copyfile(self._config.config_path, self._config.tmp_config_path)

        # Temporary Bayesian optimization log
        header = []
        for i, u in enumerate(self._config.which_u):
            header.append(f"U_ele_{str(i + 1)}")

        if os.path.exists(self._config.tmp_u_path):
            os.remove(self._config.tmp_u_path)

        if self._config.delta_mag_weight:
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

    def clean_up_temp_files(self):
        shutil.move(self._config.tmp_u_path, self._config.u_path)
        os.remove(self._config.tmp_config_path)


class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
