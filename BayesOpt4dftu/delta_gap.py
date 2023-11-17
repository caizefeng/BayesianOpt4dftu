import os

import numpy as np
from vaspvis.utils import BandGap


class DeltaGap:
    def __init__(self, path='./', baseline='hse'):
        self.path = path
        self.baseline = baseline
        self._dftu_gap_value = 0.0
        self._baseline_gap_value = 0.0
        self._delta_gap_value = 0.0

    def get_delta_gap(self):
        return self._delta_gap_value

    def get_baseline_gap(self):
        return self._baseline_gap_value

    def get_dftu_gap(self):
        return self._dftu_gap_value

    def compute_delta_gap(self):
        self._dftu_gap_value = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both').bg

        if self.baseline == 'hse':
            self._baseline_gap_value = BandGap(folder=os.path.join(self.path, 'hse/band'), method=1, spin='both').bg

        # TODO: band gap from GW calc
        elif self.baseline == 'gw':
            self._baseline_gap_value = None

        else:
            raise Exception('Unsupported baseline calculation!')

        self._delta_gap_value = np.sqrt(np.mean((self._dftu_gap_value - self._baseline_gap_value) ** 2))
