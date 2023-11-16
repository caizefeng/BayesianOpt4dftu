import os
from xml.etree import ElementTree as ET

import numpy as np
from pymatgen.io.vasp import Incar, Outcar
from vaspvis import Band
from vaspvis.utils import BandGap


class DeltaBand(object):
    def __init__(self, bandrange=(5, 5), path='./', baseline='hse', interpolate=False):
        self.path = path
        self.br_vb = bandrange[0]
        self.br_cb = bandrange[1]
        self.interpolate = interpolate
        self.baseline = baseline
        self.vasprun_hse = os.path.join(path, 'hse/band/vasprun.xml')
        self.kpoints_hse = os.path.join(path, 'hse/band/KPOINTS')
        self.vasprun_dftu = os.path.join(path, 'dftu/band/vasprun.xml')
        self.kpoints_dftu = os.path.join(path, 'dftu/band/KPOINTS')
        self._delta_band_value = 0.0  # type: float

    def get_delta_band(self):
        return self._delta_band_value

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

    @staticmethod
    def access_eigen_gw(gw_folder_path, ispin):
        efermi_gw = Outcar(os.path.join(gw_folder_path, 'OUTCAR_band')).efermi
        win = open(os.path.join(gw_folder_path, 'wannier90.win'), 'r+').readlines()

        nbands = 0  # type: int
        for line in win:
            split_line = line.split('\n')[:-1][0]
            if 'num_wann' in split_line:
                nbands = int(split_line.split('=')[-1].strip())

        if ispin == 1:
            data = open(os.path.join(gw_folder_path, 'wannier90.band.dat'), 'r+').readlines()
            eigenvalues = []
            DeltaBand.clean_wannier_data(data, eigenvalues)

            eigenvalues = np.array(eigenvalues).reshape((nbands, -1, 2))[:, :, 1] - efermi_gw
            return eigenvalues

        elif ispin == 2:
            data_up = open(os.path.join(gw_folder_path, 'wannier90.up_band.dat'), 'r+').readlines()
            data_dn = open(os.path.join(gw_folder_path, 'wannier90.dn_band.dat'), 'r+').readlines()
            eigenvalues_up = []
            eigenvalues_down = []
            DeltaBand.clean_wannier_data(data_up, eigenvalues_up)
            DeltaBand.clean_wannier_data(data_dn, eigenvalues_down)

            eigenvalues_up = np.array(eigenvalues_up).reshape((nbands, -1, 2))[:, :, 1] - efermi_gw
            eigenvalues_down = np.array(eigenvalues_down).reshape((nbands, -1, 2))[:, :, 1] - efermi_gw
            return eigenvalues_up, eigenvalues_down

    @staticmethod
    def clean_wannier_data(raw_data, eigenvalues):
        for line in raw_data:
            split_line = line.split('\n')[:-1][0].split(' ')
            filter_line = list(filter(None, split_line))
            if not filter_line:
                continue
            else:
                eigenvalues.append([float(x) for x in filter_line])

    def check_hse_compatibility(self, ispin_dftu, nbands_dftu, nkpts_dftu):
        ispin_hse, nbands_hse, nkpts_hse = DeltaBand.read_ispin_nbands_nkpts(self.vasprun_hse)

        if nbands_hse != nbands_dftu:
            raise Exception('The band number of HSE and GGA+U do not match!')

        kpoints = [line for line in open(self.kpoints_hse) if line.strip()]
        kpts_diff = 0
        for ii, line in enumerate(kpoints[3:]):
            if line.split()[3] != '0':
                kpts_diff += 1
        if nkpts_hse - kpts_diff != nkpts_dftu:
            raise Exception(
                'The kpoints number of HSE and GGA+U do not match!')

        if ispin_hse != ispin_dftu:
            raise Exception('The spin number of HSE and GGA+U do not match!')

    def delta_band(self):
        ispin_dftu, nbands_dftu, nkpts_dftu = DeltaBand.read_ispin_nbands_nkpts(self.vasprun_dftu)

        if self.baseline == 'hse':
            self.check_hse_compatibility(ispin_dftu, nbands_dftu, nkpts_dftu)

        if self.baseline == 'hse':
            new_n = 500
        else:
            new_n = 200

        if ispin_dftu == 1:
            band_dftu = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )
            eigenvalues_dftu = DeltaBand.access_eigen(band_dftu, interpolate=self.interpolate)
            shifted_dftu = self.locate_and_shift_bands(eigenvalues_dftu)
            n = shifted_dftu.shape[0] * shifted_dftu.shape[1]

            if self.baseline == 'hse':
                band_hse = Band(
                    folder=os.path.join(self.path, 'hse/band'),
                    spin='up',
                    interpolate=self.interpolate,
                    new_n=new_n,
                    projected=False,
                )
                eigenvalues_hse = DeltaBand.access_eigen(band_hse, interpolate=self.interpolate)
                shifted_baseline = self.locate_and_shift_bands(eigenvalues_hse)
            elif self.baseline == 'gw':
                eigenvalues_gw = DeltaBand.access_eigen_gw('gw', ispin=ispin_dftu)
                shifted_baseline = self.locate_and_shift_bands(eigenvalues_gw)
            else:
                raise Exception('Unsupported baseline calculation!')

            self._delta_band_value = sum((1 / n) * sum((shifted_baseline - shifted_dftu) ** 2)) ** (1 / 2)

        elif ispin_dftu == 2:
            band_dftu_up = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False
            )
            band_dftu_down = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )
            eigenvalues_dftu_up = DeltaBand.access_eigen(band_dftu_up, interpolate=self.interpolate)
            eigenvalues_dftu_down = DeltaBand.access_eigen(band_dftu_down, interpolate=self.interpolate)
            shifted_dftu_up = self.locate_and_shift_bands(eigenvalues_dftu_up)
            shifted_dftu_down = self.locate_and_shift_bands(eigenvalues_dftu_down)
            n_up = shifted_dftu_up.shape[0] * shifted_dftu_up.shape[1]
            n_down = shifted_dftu_down.shape[0] * shifted_dftu_down.shape[1]

            if self.baseline == 'hse':
                band_hse_up = Band(
                    folder=os.path.join(self.path, 'hse/band'),
                    spin='up',
                    interpolate=self.interpolate,
                    new_n=new_n,
                    projected=False,
                )
                band_hse_down = Band(
                    folder=os.path.join(self.path, 'hse/band'),
                    spin='down',
                    interpolate=self.interpolate,
                    new_n=new_n,
                    projected=False,
                )
                eigenvalues_hse_up = DeltaBand.access_eigen(band_hse_up, interpolate=self.interpolate)
                eigenvalues_hse_down = DeltaBand.access_eigen(band_hse_down, interpolate=self.interpolate)
                shifted_baseline_up = self.locate_and_shift_bands(eigenvalues_hse_up)
                shifted_baseline_down = self.locate_and_shift_bands(eigenvalues_hse_down)

            elif self.baseline == 'gw':
                eigenvalues_gw_up, eigenvalues_gw_down = DeltaBand.access_eigen_gw('gw', ispin=ispin_dftu)
                shifted_baseline_up = self.locate_and_shift_bands(eigenvalues_gw_up)
                shifted_baseline_down = self.locate_and_shift_bands(eigenvalues_gw_down)
            else:
                raise Exception('Unsupported baseline calculation!')

            delta_band_up = sum((1 / n_up) * sum((shifted_baseline_up - shifted_dftu_up) ** 2)) ** (1 / 2)
            delta_band_down = sum((1 / n_down) * sum((shifted_baseline_down - shifted_dftu_down) ** 2)) ** (1 / 2)
            self._delta_band_value = np.mean([delta_band_up, delta_band_down])

        else:
            raise Exception('Incorrect ISPIN value!')
