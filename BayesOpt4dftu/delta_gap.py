import os

import numpy as np
from vaspvis.utils import BandGap


class DeltaGap:
    def __init__(self, path='./', baseline='hse'):
        self.path = path
        self.baseline = baseline
        self._dftu_gap = 0.0
        self._baseline_gap = 0.0
        self._delta_gap = 0.0

    def get_delta_gap(self):
        return self._delta_gap

    def get_baseline_gap(self):
        return self._baseline_gap

    def get_dftu_gap(self):
        return self._dftu_gap

    def compute_delta_gap(self):
        self._dftu_gap = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both').bg

        if self.baseline == 'hse':
            self._baseline_gap = BandGap(folder=os.path.join(self.path, 'hse/band'), method=1, spin='both').bg

        # TODO: band gap from GW calc
        # Now we only deal with metals in GW calc so it's fine
        elif self.baseline == 'gw':
            self._baseline_gap = 0.0

        else:
            raise Exception('Unsupported baseline calculation!')

        self._delta_gap = np.sqrt(np.mean((self._dftu_gap - self._baseline_gap) ** 2))
