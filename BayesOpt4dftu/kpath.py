import os.path
import re
from typing import Optional, List, Dict, Any

import numpy as np
from ase import Atoms
from ase.dft.kpoints import get_special_points
from pymatgen.io.vasp import Kpoints


class BoBandPath:

    def __init__(self, is_auto=True, baseline_path=None, num_kpoints=None, k_labels=None, custom_kpoints=False):
        self._is_auto: bool = is_auto
        self._baseline_path: Optional[str] = baseline_path
        self._num_kpoints: Optional[int] = num_kpoints
        self._k_labels: Optional[str] = k_labels
        self._custom_kpoints: bool = custom_kpoints
        self._atoms: Optional[Atoms] = None
        self._k_labels_list: Optional[List[str]] = None
        self._special_kpoints: Optional[Dict[str, Any]] = None
        self._k_path: Optional[Kpoints] = None
        self._k_path_with_scf_grid: Optional[Kpoints] = None

    def set_atoms(self, atoms: Atoms):
        self._atoms = atoms

    def generate(self):
        if not self._is_auto:
            if self._custom_kpoints:
                self._special_kpoints = special_kpoints_dict
            else:
                self._special_kpoints = get_special_points(self._atoms.cell)
            self._k_labels_list = re.split(r'\s+', self._k_labels)
            self.generate_line_mode()
        else:
            # TODO: Read kpoints files to achieve auto (both HSE and GW format)
            pass

    def write_kpoints(self, directory, concat_ibzkpt=False):
        if concat_ibzkpt:
            if self._k_path_with_scf_grid is None:
                self.concatenate_with_ibzkpt(directory)
            self._k_path_with_scf_grid.write_file(os.path.join(directory, 'KPOINTS'))
        else:
            self._k_path.write_file(os.path.join(directory, 'KPOINTS'))

    def generate_line_mode(self):
        kptset = list()
        lbs = list()
        for i in range(len(self._k_labels_list)):
            if self._k_labels_list[i] in self._special_kpoints.keys():
                kptset.append(self._special_kpoints[self._k_labels_list[i]])
                lbs.append(self._k_labels_list[i])
                if i in range(1, len(self._k_labels_list) - 1):
                    kptset.append(self._special_kpoints[self._k_labels_list[i]])
                    lbs.append(self._k_labels_list[i])

        self._k_path = Kpoints(comment='band',
                               kpts=kptset,
                               num_kpts=self._num_kpoints,
                               style='Line_mode',  # noqa
                               coord_type="Reciprocal",
                               labels=lbs)

    def concatenate_with_ibzkpt(self, directory):
        with open(directory + '/IBZKPT', 'r') as ibz:
            kpoints_lines = ibz.readlines()

        kpoints_lines[1] = str(
            self._num_kpoints * (len(self._k_labels_list) - 1) + int(kpoints_lines[1].split('\n')[0])) + '\n'

        for i in range(len(self._k_labels_list) - 1):
            k_head = self._special_kpoints[self._k_labels_list[i]]
            k_tail = self._special_kpoints[self._k_labels_list[i + 1]]
            increment = (k_tail - k_head) / (self._num_kpoints - 1)
            kpoints_lines.append(' '.join(map(str, k_head)) + ' 0 ' + self._k_labels_list[i] + '\n')
            for j in range(1, self._num_kpoints - 1):
                k_next = k_head + increment * j
                kpoints_lines.append(' '.join(map(str, k_next)) + ' 0\n')
            kpoints_lines.append(' '.join(map(str, k_tail)) + ' 0 ' + self._k_labels_list[i + 1] + '\n')

        self._k_path_with_scf_grid = Kpoints.from_string(''.join(kpoints_lines))


# Deprecated
special_kpoints_dict = {"F": np.array([0.5, 0.5, 0]),
                        "G": np.array([0, 0, 0]),
                        "T": np.array([0.5, 0.5, 0.5]),
                        "K": np.array([0.8, 0.35, 0.35]),
                        "L": np.array([0.5, 0, 0])}
