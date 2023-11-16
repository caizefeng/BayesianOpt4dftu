import os

from pymatgen.io.vasp import Incar
from vaspvis.utils import BandGap

from BayesOpt4dftu.delta_band import DeltaBand


class DeltaAll:
    def __init__(self, path='./', baseline='hse', bandrange=None):
        if bandrange is None:
            bandrange = [5, 5]

        self.path = path
        self.db = DeltaBand(bandrange=bandrange, path=path, baseline=baseline)

        # self.dm =

    def calculate(self):
        self.db.delta_band()

    def write_delta(self):
        bg = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both', ).bg

        # U values
        incar = Incar.from_file(os.path.join(self.path, 'dftu/band/INCAR'))
        u = incar['LDAUU']

        # Band gap
        u.append(bg)

        # Delta band
        u.append(self.db.get_delta_band())

        output = ' '.join(str(x) for x in u)
        with open('u_tmp.txt', 'a') as f:
            f.write(output + '\n')
            f.close()
