import os
import shutil

from BayesOpt4dftu.common.configuration import Config
from BayesOpt4dftu.common.logger import BoLoggerGenerator
from BayesOpt4dftu.utils.file_utils import format_log_file_pd


class TempFileManager:
    _logger = BoLoggerGenerator.get_logger("TempFileManager")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def setup_temp_files(self):
        if self._config.resume_checkpoint:
            if not os.path.exists(self._config.tmp_u_path) or os.path.getsize(self._config.tmp_u_path) == 0:
                self._logger.error(f"Missing or damaged checkpoint file: {self._config.tmp_u_path}.")
                raise RuntimeError

            elif not os.path.exists(self._config.tmp_config_path) or os.path.getsize(self._config.tmp_config_path) == 0:
                self._logger.error(f"Missing or damaged checkpoint file: {self._config.tmp_config_path}.")
                raise RuntimeError

            else:
                self._logger.info("Previous temporary files will continue to be used.")

        else:
            # Temporary config
            shutil.copyfile(self._config.config_path, self._config.tmp_config_path)

            # Temporary Bayesian Optimization log
            if os.path.exists(self._config.tmp_u_path):
                os.remove(self._config.tmp_u_path)

            with open(self._config.tmp_u_path, 'w+') as f:
                f.write(f"{(' '.join(self._config.headers))} \n")

            self._logger.info("Temporary files initiated.")

    def clean_up(self):
        shutil.move(self._config.tmp_u_path, self._config.u_path)
        format_log_file_pd(input_file=self._config.u_path, output_file=self._config.formatted_u_path,
                           decimals=4, width=15, logger=self._logger)
        os.remove(self._config.tmp_config_path)

        self._logger.info("Temporary files removed.")
