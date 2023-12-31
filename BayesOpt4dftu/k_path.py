import os.path
import re
from typing import Optional, List, Dict, Any

import numpy as np
from ase import Atoms
from ase.dft.kpoints import get_special_points
from pymatgen.io.vasp import Kpoints

from BayesOpt4dftu.io_utils import find_and_readlines_first


class BoBandPath:

    def __init__(self, is_auto=True, baseline_type=None, baseline_path=None, num_kpoints=None, k_labels=None,
                 custom_kpoints=False):
        self._is_auto: bool = is_auto
        self._baseline_path: Optional[str] = baseline_path
        self._baseline_type: Optional[str] = baseline_type
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
            self._k_path = self.from_line_mode()
        else:
            if self._baseline_type == 'hse':
                self._k_path = self.from_baseline_reciprocal()
            elif self._baseline_type == 'gw':
                self._k_path = self.from_baseline_gw()

    def write_kpoints(self, directory, concat_ibzkpt=False):
        if self._k_path is None:
            raise RuntimeError("Cannot execute 'write_kpoints' method before 'generate' method.")
        if concat_ibzkpt:
            if self._k_path_with_scf_grid is None:
                self.concat_with_ibzkpt(directory)
            self._k_path_with_scf_grid.write_file(os.path.join(directory, 'KPOINTS'))
        else:
            self._k_path.write_file(os.path.join(directory, 'KPOINTS'))

    def from_line_mode(self):
        kptset = list()
        lbs = list()
        for i in range(len(self._k_labels_list)):
            if self._k_labels_list[i] in self._special_kpoints.keys():
                kptset.append(self._special_kpoints[self._k_labels_list[i]])
                lbs.append(self._k_labels_list[i])
                if i in range(1, len(self._k_labels_list) - 1):
                    kptset.append(self._special_kpoints[self._k_labels_list[i]])
                    lbs.append(self._k_labels_list[i])

        return Kpoints(comment="BayesOpt4dftu: K-path from user input",
                       kpts=kptset,
                       num_kpts=self._num_kpoints,
                       style=Kpoints.supported_modes.Line_mode,
                       coord_type="Reciprocal",
                       labels=lbs)

    def concat_with_ibzkpt(self, directory):
        with open(directory + '/IBZKPT', 'r') as ibz:
            kpoints_contents = ibz.readlines()

        kpoints_contents[1] = str(
            self._num_kpoints * (len(self._k_labels_list) - 1) + int(kpoints_contents[1].split('\n')[0])) + '\n'

        for i in range(len(self._k_labels_list) - 1):
            k_head = self._special_kpoints[self._k_labels_list[i]]
            k_tail = self._special_kpoints[self._k_labels_list[i + 1]]
            increment = (k_tail - k_head) / (self._num_kpoints - 1)
            kpoints_contents.append(' '.join(map(str, k_head)) + ' 0 ' + self._k_labels_list[i] + '\n')
            for j in range(1, self._num_kpoints - 1):
                k_next = k_head + increment * j
                kpoints_contents.append(' '.join(map(str, k_next)) + ' 0\n')
            kpoints_contents.append(' '.join(map(str, k_tail)) + ' 0 ' + self._k_labels_list[i + 1] + '\n')

        self._k_path_with_scf_grid = Kpoints.from_string(''.join(kpoints_contents))
        self._k_path_with_scf_grid.comment = "BayesOpt4dftu: Kpoints from user input and scf K-grid"

    def from_baseline_reciprocal(self):
        k_path = Kpoints.from_file(os.path.join(self._baseline_path, 'KPOINTS'))

        filtered_kpts = []
        filtered_weights = []
        filtered_labels = []
        for kpt, weight, label in zip(k_path.kpts, k_path.kpts_weights, k_path.labels):
            if weight == 0:
                filtered_kpts.append(kpt)
                filtered_weights.append(1)  # Sum of weights can't be zero
                filtered_labels.append(label)

        k_path.comment = "BayesOpt4dftu: K-path from hybrid band structure"
        k_path.num_kpts = len(filtered_kpts)
        k_path.kpts = filtered_kpts
        k_path.kpts_weights = filtered_weights
        k_path.labels = filtered_labels
        return k_path

    def from_baseline_gw(self):
        kpt_contents = find_and_readlines_first(self._baseline_path,
                                                ['wannier90_band.kpt',
                                                 'wannier90.1_band.kpt',
                                                 'wannier90.2_band.kpt'])
        labelinfo_contents = find_and_readlines_first(self._baseline_path,
                                                      ['wannier90_band.labelinfo.dat',
                                                       'wannier90.1_band.labelinfo.dat',
                                                       'wannier90.2_band.labelinfo.dat'])
        # Processing the kpt file to extract k-points and weights
        num_kpts = int(kpt_contents[0].strip())
        kpts = []
        kpts_weights = []
        for line in kpt_contents[1:]:
            parts = line.split()
            kpt = [float(parts[i]) for i in range(3)]
            weight = float(parts[3])
            kpts.append(kpt)
            kpts_weights.append(weight)

        if len(kpts) != num_kpts:
            raise ValueError("Inconsistency of the number of kpoints detected in the GW kpt file.")

        # Processing the labelinfo file to extract labels
        labels = [None] * num_kpts
        for line in labelinfo_contents:
            parts = line.split()
            label = parts[0].strip()
            index = int(parts[1]) - 1  # Adjusting index to 0-based
            labels[index] = label

        # Creating Kpoints instance
        return Kpoints(
            comment="BayesOpt4dftu: K-path from GW band structure",
            style=Kpoints.supported_modes.Reciprocal,
            num_kpts=num_kpts,
            kpts=kpts,
            kpts_weights=kpts_weights,
            labels=labels
        )


# Deprecated
special_kpoints_dict = {"F": np.array([0.5, 0.5, 0]),
                        "G": np.array([0, 0, 0]),
                        "T": np.array([0.5, 0.5, 0.5]),
                        "K": np.array([0.8, 0.35, 0.35]),
                        "L": np.array([0.5, 0, 0])}
