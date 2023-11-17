import os

import numpy as np
from pymatgen.io.vasp import Outcar


class DeltaMag:
    def __init__(self, path='./', baseline='hse'):
        self.path = path
        self.baseline = baseline
        self.noncollinear = DeltaMag.read_lnoncollinear(os.path.join(self.path, 'dftu/band/OUTCAR'))
        self._delta_mag = 0.0

    def get_delta_mag(self):
        return self._delta_mag

    @staticmethod
    def read_lnoncollinear(outcar_path):
        """
        Check if LNONCOLLINEAR is set to T (True) in a VASP OUTCAR file.

        :param outcar_path: Path to the OUTCAR file.
        :return: True if LNONCOLLINEAR is set to T, False otherwise.
        """
        with open(outcar_path, 'r') as file:
            for line in file:
                if 'LNONCOLLINEAR' in line:
                    # Check if LNONCOLLINEAR is followed by T
                    return 'T' in line.split()
        return False

    @staticmethod
    def mag2array(mag, noncollinear=True, axis=2, mode='total'):
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

    def compute_delta_mag(self, component='all', mode='total'):

        # Does not matter if collinear
        assert component in ['x', 'y', 'z', 'all'], 'Unsupported magnetization component!'
        axis_map = dict(zip(['x', 'y', 'z', 'all'], [-1, 0, 1, 2]))
        axis = axis_map[component]

        assert mode in ['orbit', 'total'], 'Unsupported magnetization mode!'

        outcar_dftu = Outcar(os.path.join(self.path, 'dftu/scf/OUTCAR'))
        outcar_path = {'hse': 'hse/band/OUTCAR', 'gw': 'gw/scf/OUTCAR'}  # HSE calcs are always SCF
        outcar_baseline = Outcar(os.path.join(self.path, outcar_path[self.baseline]))

        mag_dftu = outcar_dftu.magnetization
        mag_baseline = outcar_baseline.magnetization

        mag_array_dftu = DeltaMag.mag2array(mag_dftu, noncollinear=self.noncollinear, axis=axis, mode=mode)
        mag_array_gw = DeltaMag.mag2array(mag_baseline, noncollinear=self.noncollinear, axis=axis, mode=mode)

        self._delta_mag = np.sqrt(np.mean((mag_array_dftu - mag_array_gw) ** 2))  # RMSE
