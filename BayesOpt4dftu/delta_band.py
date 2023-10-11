import os
from xml.etree import ElementTree as ET

import numpy as np
from pymatgen.io.vasp import Incar
from vaspvis import Band
from vaspvis.utils import BandGap


class DeltaBand(object):
    def __init__(self, bandrange=(5, 5), path='./', iteration=1, interpolate=False):
        self.path = path
        self.br_vb = bandrange[0]
        self.br_cb = bandrange[1]
        self.interpolate = interpolate
        self.vasprun_hse = os.path.join(path, 'hse/band/vasprun.xml')
        self.kpoints_hse = os.path.join(path, 'hse/band/KPOINTS')
        self.vasprun_dftu = os.path.join(path, 'dftu/band/vasprun.xml')
        self.kpoints_dftu = os.path.join(path, 'dftu/band/KPOINTS')
        self.iteration = iteration

    @staticmethod
    def read_ispin_nbands_nkpts(filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ispin = int(root.findall(
            './parameters/separator/.[@name="electronic"]/separator/.[@name="electronic spin"]/i/.[@name="ISPIN"]')[
                        0].text)
        nbands = int(root.findall(
            './parameters/separator/.[@name="electronic"]/i/.[@name="NBANDS"]')[0].text)
        nkpts = len(root.findall('./kpoints/varray/.[@name="kpointlist"]/v'))

        return ispin, nbands, nkpts

    @staticmethod
    def access_eigen(b: Band, interpolate=False):
        wave_vectors = b._get_k_distance()
        eigenvalues = b.eigenvalues

        if interpolate:
            _, eigenvalues_interp = b._get_interpolated_data(
                wave_vectors=wave_vectors,
                data=eigenvalues
            )

        if interpolate:
            return eigenvalues_interp
        else:
            return eigenvalues

    def locate_and_shift_bands(self, eigenvalues):
        band_mean = eigenvalues.mean(axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.max(eigenvalues[below_index])
        cbm = np.min(eigenvalues[above_index])

        if cbm < vbm:
            vbm = 0.0
            cbm = 0.0

        valence_bands = eigenvalues[below_index[-self.br_vb:]]
        conduction_bands = eigenvalues[above_index[:self.br_cb]]

        valence_bands -= vbm
        conduction_bands -= cbm

        shifted_bands = np.r_[conduction_bands, valence_bands]

        return shifted_bands

    def delta_band(self):
        ispin_hse, nbands_hse, nkpts_hse = DeltaBand.read_ispin_nbands_nkpts(self.vasprun_hse)
        ispin_dftu, nbands_dftu, nkpts_dftu = DeltaBand.read_ispin_nbands_nkpts(self.vasprun_dftu)

        if nbands_hse != nbands_dftu:
            raise Exception('The band number of HSE and GGA+U are not match!')

        kpoints = [line for line in open(self.kpoints_hse) if line.strip()]
        kpts_diff = 0
        for ii, line in enumerate(kpoints[3:]):
            if line.split()[3] != '0':
                kpts_diff += 1

        if nkpts_hse - kpts_diff != nkpts_dftu:
            raise Exception(
                'The kpoints number of HSE and GGA+U are not match!')

        new_n = 500

        if ispin_hse == 1 and ispin_dftu == 1:
            band_hse = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )
            band_dftu = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            eigenvalues_hse = DeltaBand.access_eigen(band_hse, interpolate=self.interpolate)
            eigenvalues_dftu = DeltaBand.access_eigen(band_dftu, interpolate=self.interpolate)

            shifted_hse = self.locate_and_shift_bands(eigenvalues_hse)
            shifted_dftu = self.locate_and_shift_bands(eigenvalues_dftu)

            n = shifted_hse.shape[0] * shifted_hse.shape[1]
            delta_band = sum((1 / n) * sum((shifted_hse - shifted_dftu) ** 2)) ** (1 / 2)

            bg = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both', ).bg

            incar = Incar.from_file('./dftu/band/INCAR')
            u = incar['LDAUU']
            u.append(bg)
            u.append(delta_band)
            output = ' '.join(str(x) for x in u)

            with open('u_tmp.txt', 'a') as f:
                f.write(output + '\n')
                f.close()

            return delta_band

        elif ispin_hse == 2 and ispin_dftu == 2:
            band_hse_up = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            band_dftu_up = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False
            )

            band_hse_down = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            band_dftu_down = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            eigenvalues_hse_up = DeltaBand.access_eigen(band_hse_up, interpolate=self.interpolate)
            eigenvalues_dftu_up = DeltaBand.access_eigen(band_dftu_up, interpolate=self.interpolate)

            shifted_hse_up = self.locate_and_shift_bands(eigenvalues_hse_up)
            shifted_dftu_up = self.locate_and_shift_bands(eigenvalues_dftu_up)

            n_up = shifted_hse_up.shape[0] * shifted_hse_up.shape[1]
            delta_band_up = sum((1 / n_up) * sum((shifted_hse_up - shifted_dftu_up) ** 2)) ** (1 / 2)

            eigenvalues_hse_down = DeltaBand.access_eigen(band_hse_down, interpolate=self.interpolate)
            eigenvalues_dftu_down = DeltaBand.access_eigen(band_dftu_down, interpolate=self.interpolate)

            shifted_hse_down = self.locate_and_shift_bands(eigenvalues_hse_down)
            shifted_dftu_down = self.locate_and_shift_bands(eigenvalues_dftu_down)

            n_down = shifted_hse_down.shape[0] * shifted_hse_down.shape[1]
            delta_band_down = sum((1 / n_down) * sum((shifted_hse_down - shifted_dftu_down) ** 2)) ** (1 / 2)

            delta_band = np.mean([delta_band_up, delta_band_down])

            bg = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both', ).bg

            incar = Incar.from_file('./dftu/band/INCAR')
            u = incar['LDAUU']

            u.append(bg)
            u.append(delta_band)
            output = ' '.join(str(x) for x in u)

            with open('u_tmp.txt', 'a') as f:
                f.write(output + '\n')
                f.close()

            return delta_band
        else:
            raise Exception('The spin number of HSE and GGA+U are not match!')
