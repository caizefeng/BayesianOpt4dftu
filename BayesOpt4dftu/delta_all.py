import os

from pymatgen.io.vasp import Incar

from BayesOpt4dftu.delta_band import DeltaBand
from BayesOpt4dftu.delta_gap import DeltaGap
from BayesOpt4dftu.delta_mag import DeltaMag


class DeltaAll:
    def __init__(self, path='./', baseline='hse', bandrange=None):
        if bandrange is None:
            bandrange = [5, 5]

        self.path = path
        self.dg = DeltaGap(path=path, baseline=baseline)
        self.db = DeltaBand(bandrange=bandrange, path=path, baseline=baseline)
        self.dm = DeltaMag(path=path, baseline=baseline)

    def compute_delta(self):
        self.dg.compute_delta_gap()
        self.db.compute_delta_band()
        self.dm.compute_delta_mag()

    def write_delta(self):
        # U values
        incar = Incar.from_file(os.path.join(self.path, 'dftu/band/INCAR'))
        u = incar['LDAUU']

        # Band gap
        u.append(self.dg.get_dftu_gap())

        # Delta Gap
        u.append(self.dg.get_delta_gap())

        # Delta band
        u.append(self.db.get_delta_band())

        # Delta magnetization
        u.append(self.dm.get_delta_mag())

        output = ' '.join(str(x) for x in u)
        with open('u_tmp.txt', 'a') as f:
            f.write(output + '\n')
            f.close()
