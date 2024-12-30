import numpy as np
from vaspvis.utils import BandGap

from BayesOpt4dftu.common.configuration import Config
from BayesOpt4dftu.common.logger import BoLoggerGenerator


class DeltaGap:
    _logger = BoLoggerGenerator.get_logger("DeltaGap")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def __init__(self):
        self._dftu_gap: float = 0.0
        self._baseline_gap: float = 0.0
        self._delta_gap: float = 0.0

    def get_delta_gap(self):
        return self._delta_gap

    def get_baseline_gap(self):
        return self._baseline_gap

    def get_dftu_gap(self):
        return self._dftu_gap

    def compute_delta_gap(self):
        self._dftu_gap = float(BandGap(folder=self._config.combined_path_dict['dftu']['band'],
                                       method=1, spin='both').bg)

        if self._config.baseline == 'hse':
            self._baseline_gap = float(BandGap(folder=self._config.combined_path_dict['hse']['band'],
                                               method=1, spin='both').bg)

        # TODO: band gap from GW calc
        # Now we only deal with metals in GW calc so it's fine
        elif self._config.baseline == 'gw':
            self._baseline_gap = 0.0

        elif self._config.baseline == 'dft':
            self._baseline_gap = float(BandGap(folder=self._config.combined_path_dict['dft']['band'],
                                               method=1, spin='both').bg)

        else:
            self._logger.error("Unsupported baseline calculation: only 'hse', 'gw' or 'dft' are accepted.")
            raise ValueError

        self._delta_gap = np.sqrt(np.mean((self._dftu_gap - self._baseline_gap) ** 2))
