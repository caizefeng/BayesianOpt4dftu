import os
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from pymatgen.io.vasp import Outcar

from BayesOpt4dftu.common.configuration import Config
from BayesOpt4dftu.common.logger import BoLoggerGenerator


class DeltaMag:
    _logger = BoLoggerGenerator.get_logger("DeltaMag")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def __init__(self):
        self._outcar_with_mag: Dict[str, str] = {
            'dftu': os.path.join(self._config.combined_path_dict['dftu']['scf'], 'OUTCAR'),
            'hse': os.path.join(self._config.combined_path_dict['hse']['band'], 'OUTCAR'),  # ['hse']['scf'] is a preliminary PBE calculation
            'gw': os.path.join(self._config.combined_path_dict['gw']['scf'], 'OUTCAR'),
            'dft': os.path.join(self._config.combined_path_dict['dft']['scf'], 'OUTCAR')
        }
        self._noncollinear: bool = DeltaMag.read_lnoncollinear(self._outcar_with_mag[self._config.baseline])
        self._delta_mag: float = 0.0
        self._baseline_mag: Optional[NDArray] = None
        self._dftu_mag: Optional[NDArray] = None

    def get_delta_mag(self):
        return self._delta_mag

    def get_baseline_mag(self):
        return self._baseline_mag

    def get_dftu_mag(self):
        return self._dftu_mag

    def compute_delta_mag(self, component='all', mode='total'):

        # Does not matter if collinear
        assert component in ['all', 'x', 'y', 'z'], 'Unsupported magnetization component.'
        axis_map = dict(zip(['all', 'x', 'y', 'z'], [-1, 0, 1, 2]))
        axis = axis_map[component]

        assert mode in ['orbit', 'total'], 'Unsupported magnetization mode.'

        outcar_dftu = Outcar(self._outcar_with_mag['dftu'])
        outcar_baseline = Outcar(self._outcar_with_mag[self._config.baseline])

        mag_dftu = outcar_dftu.magnetization
        mag_baseline = outcar_baseline.magnetization

        self._dftu_mag = DeltaMag.mag2array(mag_dftu, noncollinear=self._noncollinear, axis=axis, mode=mode)
        self._baseline_mag = DeltaMag.mag2array(mag_baseline, noncollinear=self._noncollinear, axis=axis, mode=mode)

        self._delta_mag = np.sqrt(np.mean((self._dftu_mag - self._baseline_mag) ** 2))  # RMSE

    @staticmethod
    def read_lnoncollinear(outcar_path):
        """
        Check if LNONCOLLINEAR is set to T (True) in a VASP OUTCAR file.

        :param outcar_path: Path to the OUTCAR file.
        :return: True if LNONCOLLINEAR is set to T, False otherwise.
        """
        with open(outcar_path, 'r') as file:
            for line in file:
                if 'LNONCOLLINEAR' in line and 'T' in line.split():
                    # Check if LNONCOLLINEAR is followed by T
                    return True
                else:
                    continue
        return False

    @staticmethod
    def mag2array(mag, noncollinear=True, axis=-1, mode='total') -> NDArray:
        num_ions = len(mag)

        orbits = mag[0].keys()
        num_orbits = len(orbits) - 1

        if noncollinear:
            if mode == 'orbit':
                mag_array = np.zeros((num_ions, num_orbits, 3 if axis == -1 else 1))
            else:
                mag_array = np.zeros((num_ions, 3 if axis == -1 else 1))

            for i in range(num_ions):
                for j, orb in enumerate(orbits):
                    if mode == 'orbit' and orb != 'tot':
                        if axis == -1:
                            mag_array[i][j] = list(mag[i][orb])
                        else:
                            mag_array[i][j] = list(mag[i][orb])[axis]

                    elif mode == 'total' and orb == 'tot':
                        if axis == -1:
                            mag_array[i] = list(mag[i][orb])
                        else:
                            mag_array[i] = list(mag[i][orb])[axis]
        else:
            if mode == 'orbit':
                mag_array = np.zeros((num_ions, num_orbits))
            else:
                mag_array = np.zeros((num_ions, 1))

            for i in range(num_ions):
                for j, orb in enumerate(orbits):
                    if mode == 'orbit' and orb != 'tot':
                        mag_array[i][j] = mag[i][orb]

                    elif mode == 'total' and orb == 'tot':
                        mag_array[i] = mag[i][orb]

        return mag_array

    @staticmethod
    def mag2string(mag_array: NDArray):
        return np.array2string(mag_array.reshape(-1), formatter={'float_kind': lambda x: "%.3f" % x}, separator=',')
