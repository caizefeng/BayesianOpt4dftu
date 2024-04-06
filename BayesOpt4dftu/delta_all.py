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
            self.dm.compute_delta_mag()

    def write_delta(self):
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

        output = ' '.join(str(x) for x in u)
        with open(self._config.tmp_u_path, 'a') as f:
            f.write(output + '\n')
