import inspect
import os
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from pymatgen.io.vasp import Outcar
from vaspvis import Band

from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.logging import BoLoggerGenerator


class DeltaBand:
    _logger = BoLoggerGenerator.get_logger("DeltaBand")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def __init__(self, interpolate=False):

        self._interpolate: bool = interpolate
        self._br_vb: int
        self._br_cb: int
        self._br_vb, self._br_cb = self._config.br
        self._baseline: str = self._config.baseline

        self._hse_band_path: str = self._config.combined_path_dict['hse']['band']
        self._dftu_band_path: str = self._config.combined_path_dict['dftu']['band']
        self._gw_band_path: str = self._config.combined_path_dict['gw']['band']
        self._vasprun_hse: str = os.path.join(self._hse_band_path, 'vasprun.xml')
        self._kpoints_hse: str = os.path.join(self._hse_band_path, 'KPOINTS')
        self._vasprun_dftu: str = os.path.join(self._dftu_band_path, 'vasprun.xml')
        self._kpoints_dftu: str = os.path.join(self._dftu_band_path, 'KPOINTS')

        self._delta_band: float = 0.0
        self._auto_kpath = self._config.auto_kpath
        self._line_mode_kpath = self._config.line_mode_kpath
        self._is_first_run: bool = True
        self._slice_length: Optional[NDArray] = None
        self._slice_weight: Optional[NDArray] = None
        self._num_slices: int = 0
        self._num_kpts_each_slice: int = 0

    def get_delta_band(self):
        return self._delta_band

    def compute_delta_band(self):
        ispin_dftu, nbands_dftu, nkpts_dftu = DeltaBand.read_ispin_nbands_nkpts(self._vasprun_dftu)

        if self._is_first_run and self._baseline == 'hse':
            self.check_hse_compatibility(ispin_dftu, nbands_dftu, nkpts_dftu)

        if self._baseline == 'hse':
            new_n = 500
        else:
            new_n = inspect.signature(Band.__init__).parameters["new_n"]  # default value for this parameter

        if ispin_dftu == 1:
            band_dftu = Band(
                folder=self._dftu_band_path,
                spin='up',
                interpolate=self._interpolate,
                new_n=new_n,
                projected=False,
            )  # Shifted to E-fermi = 0 by default
            eigenvalues_dftu = self.access_eigen(band_dftu, interpolate=self._interpolate)
            shifted_dftu = self.locate_and_shift_bands(eigenvalues_dftu)
            n = shifted_dftu.shape[0] * shifted_dftu.shape[1]

            if self._baseline == 'hse':
                band_hse = Band(
                    folder=self._hse_band_path,
                    spin='up',
                    interpolate=self._interpolate,
                    new_n=new_n,
                    projected=False,
                )
                eigenvalues_hse = self.access_eigen(band_hse, interpolate=self._interpolate)
                shifted_baseline = self.locate_and_shift_bands(eigenvalues_hse)
            elif self._baseline == 'gw':
                eigenvalues_gw = self.access_eigen_gw(self._gw_band_path, ispin=ispin_dftu)
                shifted_baseline = self.locate_and_shift_bands(eigenvalues_gw)
            else:
                self._logger.error("Unsupported baseline calculation: only 'hse' or 'gw' are accepted.")
                raise ValueError

            self._delta_band = self.scale_delta_band(n, shifted_baseline, shifted_dftu)

        elif ispin_dftu == 2:
            band_dftu_up = Band(
                folder=self._dftu_band_path,
                spin='up',
                interpolate=self._interpolate,
                new_n=new_n,
                projected=False
            )
            band_dftu_down = Band(
                folder=self._dftu_band_path,
                spin='down',
                interpolate=self._interpolate,
                new_n=new_n,
                projected=False,
            )
            eigenvalues_dftu_up = self.access_eigen(band_dftu_up, interpolate=self._interpolate)
            eigenvalues_dftu_down = self.access_eigen(band_dftu_down, interpolate=self._interpolate)
            shifted_dftu_up = self.locate_and_shift_bands(eigenvalues_dftu_up)
            shifted_dftu_down = self.locate_and_shift_bands(eigenvalues_dftu_down)
            n_up = shifted_dftu_up.shape[0] * shifted_dftu_up.shape[1]
            n_down = shifted_dftu_down.shape[0] * shifted_dftu_down.shape[1]

            if self._baseline == 'hse':
                band_hse_up = Band(
                    folder=self._hse_band_path,
                    spin='up',
                    interpolate=self._interpolate,
                    new_n=new_n,
                    projected=False,
                )
                band_hse_down = Band(
                    folder=self._hse_band_path,
                    spin='down',
                    interpolate=self._interpolate,
                    new_n=new_n,
                    projected=False,
                )
                eigenvalues_hse_up = self.access_eigen(band_hse_up, interpolate=self._interpolate)
                eigenvalues_hse_down = self.access_eigen(band_hse_down, interpolate=self._interpolate)
                shifted_baseline_up = self.locate_and_shift_bands(eigenvalues_hse_up)
                shifted_baseline_down = self.locate_and_shift_bands(eigenvalues_hse_down)

            elif self._baseline == 'gw':
                eigenvalues_gw_up, eigenvalues_gw_down = self.access_eigen_gw(
                    self._gw_band_path, ispin=ispin_dftu)
                shifted_baseline_up = self.locate_and_shift_bands(eigenvalues_gw_up)
                shifted_baseline_down = self.locate_and_shift_bands(eigenvalues_gw_down)
            else:
                self._logger.error("Unsupported baseline calculation: only 'hse' or 'gw' are accepted.")
                raise ValueError

            delta_band_up = self.scale_delta_band(n_up, shifted_baseline_up, shifted_dftu_up)
            delta_band_down = self.scale_delta_band(n_down, shifted_baseline_down, shifted_dftu_down)
            self._delta_band = np.mean([delta_band_up, delta_band_down])

        else:
            self._logger.error("Incorrect ISPIN value.")
            raise ValueError

        self._is_first_run = False

    def access_eigen(self, b: Band, interpolate=False):
        eigenvalues = b.eigenvalues

        if not self._auto_kpath:
            wave_vectors = b._get_k_distance()

            # Compute k-space length and corresponding weight of each slice
            if self._line_mode_kpath and self._is_first_run:
                self._num_slices, self._num_kpts_each_slice = wave_vectors.shape
                self._slice_length = wave_vectors[:, -1] - wave_vectors[:, 0]
                max_length = np.max(self._slice_length)
                self._slice_weight = self._slice_length / max_length  # normalize so that the maximum weight equals 1

            if interpolate:
                _, eigenvalues_interp = b._get_interpolated_data(
                    wave_vectors=wave_vectors,
                    data=eigenvalues
                )
                return eigenvalues_interp

        return eigenvalues

    def scale_delta_band(self, n_bands, shifted_baseline, shifted_dftu):
        """
        Calculate RMSE between two set of bands.
        Weight Δband(k) based on each k-path slice's length (i.e. density), needed only when type(num_kpts) = int.
        """
        unweighted_delta_band_k = (1 / n_bands) * sum((shifted_baseline - shifted_dftu) ** 2)

        if self._line_mode_kpath:
            weighted_delta_band_k_2d = unweighted_delta_band_k.reshape(-1, self._num_kpts_each_slice).copy()
            weighted_delta_band_k_2d[:, 1:-1] *= self._slice_weight[:, np.newaxis]

            # Trim duplicated high symmetry points, @formatter:off
            weighted_delta_band_k = np.delete(
                weighted_delta_band_k_2d,
                np.s_[
                    self._num_kpts_each_slice - 1:
                    self._num_slices * self._num_kpts_each_slice - 1:
                    self._num_kpts_each_slice
                ]
            )  # @formatter:on
            delta_band_k = weighted_delta_band_k
        else:
            delta_band_k = unweighted_delta_band_k

        return sum(delta_band_k) ** (1 / 2)

    def locate_and_shift_bands(self, eigenvalues):
        band_mean = eigenvalues.mean(axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.max(eigenvalues[below_index])
        cbm = np.min(eigenvalues[above_index])

        if cbm < vbm:
            vbm = 0.0
            cbm = 0.0

        valence_bands = eigenvalues[below_index[-self._br_vb:]]
        conduction_bands = eigenvalues[above_index[:self._br_cb]]

        valence_bands -= vbm
        conduction_bands -= cbm

        shifted_bands = np.r_[conduction_bands, valence_bands]

        return shifted_bands

    def check_hse_compatibility(self, ispin_dftu, nbands_dftu, nkpts_dftu):
        ispin_hse, nbands_hse, nkpts_hse = DeltaBand.read_ispin_nbands_nkpts(self._vasprun_hse)

        if nbands_hse != nbands_dftu:
            self._logger.error("The band number of HSE and DFT+U do not match.")
            raise RuntimeError

        kpoints = [line for line in open(self._kpoints_hse) if line.strip()]
        kpts_diff = 0
        for ii, line in enumerate(kpoints[3:]):
            if line.split()[3] != '0':
                kpts_diff += 1
        if nkpts_hse - kpts_diff != nkpts_dftu:
            self._logger.error("The kpoints number of HSE and DFT+U do not match.")
            raise RuntimeError

        if ispin_hse != ispin_dftu:
            self._logger.error("The spin number of HSE and DFT+U do not match.")
            raise RuntimeError

    def access_eigen_gw(self, gw_band_dir, ispin):
        # TODO: Accurate E-fermi for METAL should be from fine-grid SCF calculation
        efermi_gw = Outcar(os.path.join(gw_band_dir, 'OUTCAR')).efermi
        win = open(os.path.join(gw_band_dir, 'wannier90.win'), 'r+').readlines()

        nbands: int = 0
        for line in win:
            split_line = line.split('\n')[:-1][0]
            if 'num_wann' in split_line:
                nbands = int(split_line.split('=')[-1].strip())

        if ispin == 1:
            data = open(os.path.join(gw_band_dir, 'wannier90_band.dat'), 'r+').readlines()
            concatenated_k_e = []
            DeltaBand.clean_wannier_data(data, concatenated_k_e)

            eigenvalues = np.array(concatenated_k_e).reshape((nbands, -1, 2))[:, :, 1] - efermi_gw

            if self._line_mode_kpath:
                eigenvalues = self.pad_gw_band(eigenvalues)

            return eigenvalues

        elif ispin == 2:
            data_up = open(os.path.join(gw_band_dir, 'wannier90.1_band.dat'), 'r+').readlines()
            data_dn = open(os.path.join(gw_band_dir, 'wannier90.2_band.dat'), 'r+').readlines()
            concatenated_k_e_up = []
            concatenated_k_e_down = []
            DeltaBand.clean_wannier_data(data_up, concatenated_k_e_up)
            DeltaBand.clean_wannier_data(data_dn, concatenated_k_e_down)

            eigenvalues_up = np.array(concatenated_k_e_up).reshape((nbands, -1, 2))[:, :, 1] - efermi_gw
            eigenvalues_down = np.array(concatenated_k_e_down).reshape((nbands, -1, 2))[:, :, 1] - efermi_gw

            if self._line_mode_kpath:
                eigenvalues_up, eigenvalues_down = self.pad_gw_band(eigenvalues_up), self.pad_gw_band(eigenvalues_down)

            return eigenvalues_up, eigenvalues_down

    def pad_gw_band(self, eigenvalues):
        """
        Duplicate the columns of GW band data [n, k] to align with DFT+U band data
        """

        # Number of columns in the original array
        n_col = eigenvalues.shape[1]

        if n_col != self._num_slices * (self._num_kpts_each_slice - 1):
            raise ValueError(
                "For GW baseline, `_num_kpts_each_slice` must equal to `(k_gw + 1)`. "
                "`k_gw` is the number of kpoints in each kpath segment of the GW calculation."
            )

        # Calculate the number of columns to be added
        num_new_cols = n_col // (self._num_kpts_each_slice - 1)

        # Create an empty array with the new shape
        padded_eigenvalues = np.empty((eigenvalues.shape[0], n_col + num_new_cols), dtype=eigenvalues.dtype)

        # Iterate over the original array and copy/duplicate columns
        old_col = 0
        new_col = 0
        while old_col < n_col:
            padded_eigenvalues[:, new_col] = eigenvalues[:, old_col]
            new_col += 1
            # Duplicate every m-th column
            if (old_col + 1) % (self._num_kpts_each_slice - 1) == 0:
                padded_eigenvalues[:, new_col] = eigenvalues[:, old_col]
                new_col += 1
            old_col += 1

        return padded_eigenvalues

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
    def clean_wannier_data(raw_data, concatenated_k_e):
        for line in raw_data:
            split_line = line.split('\n')[:-1][0].split(' ')
            filter_line = list(filter(None, split_line))
            if not filter_line:
                continue
            else:
                concatenated_k_e.append([float(x) for x in filter_line])
