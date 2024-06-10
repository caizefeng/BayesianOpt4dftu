import os

from pymatgen.io.vasp import Incar

from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.delta_band import DeltaBand
from BayesOpt4dftu.delta_gap import DeltaGap
from BayesOpt4dftu.delta_mag import DeltaMag
from BayesOpt4dftu.logging import BoLoggerGenerator


class DeltaAll:
    _logger = BoLoggerGenerator.get_logger("DeltaAll")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config
        DeltaGap.init_config(config)
        DeltaBand.init_config(config)
        if DeltaAll._config.include_mag:
            DeltaMag.init_config(config)

    def __init__(self):
        self.dg = DeltaGap()
        self.db = DeltaBand()
        if self._config.include_mag:
            self.dm = DeltaMag()

    def compute_delta(self):
        self.dg.compute_delta_gap()
        self.db.compute_delta_band()
        if self._config.include_mag:
            self.dm.compute_delta_mag(component=self._config.mag_axis)

    def write_delta(self, na_padding=False):
        # U values
        incar = Incar.from_file(os.path.join(self._config.combined_path_dict['dftu']['band'], 'INCAR'))
        u = incar['LDAUU']

        # Band gap
        u.append(self.dg.get_dftu_gap())

        # Delta Gap
        u.append(self.dg.get_delta_gap())

        # Delta band
        u.append(self.db.get_delta_band())

        # Delta magnetization
        if self._config.include_mag:
            u.append(self.dm.get_delta_mag())

        if na_padding:
            output = " ".join(str(x) for x in u) + " N/A" * 2
        else:
            output = " ".join(str(x) for x in u)

        with open(self._config.tmp_u_path, 'a') as f:
            f.write(output + '\n')

    def report_baseline_gap(self):
        self._logger.info(
            f"Band gap from "
            f"{self._config.method_name_dict[self._config.baseline]} calculation: {self.dg.get_baseline_gap()} eV")

    def report_baseline_magnetization(self):
        self._logger.info(
            f"Magnetic moment ('{self._config.mag_axis}' component) from "
            f"{self._config.method_name_dict[self._config.baseline]} calculation: {self.dm.get_baseline_mag()}")

    def report_optimal_dftu_gap(self):
        self._logger.info(
            f"Band gap from "
            f"optimal DFT+U calculation: {self.dg.get_dftu_gap()} eV")

    def report_optimal_dftu_magnetization(self):
        self._logger.info(
            f"Magnetic moment ('{self._config.mag_axis}' component) from "
            f"optimal DFT+U calculation: {self.dm.get_dftu_mag()}")
