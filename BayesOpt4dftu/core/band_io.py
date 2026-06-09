"""VASP band-structure parsing utilities (vendored from vaspvis).

This module contains the (small) subset of vaspvis that BayesOpt4dftu relies on, vendored so
that BayesOpt4dftu has no dependency on vaspvis or its heavy visualization stack
(pyprocar / pychemia / pyvista / vtk), none of which is installable on Python >= 3.12.

Derived from vaspvis @ commit 3ca2126 (https://github.com/caizefeng/vaspvis, MIT License,
author Derek Dardzinski, maintainer Zefeng Cai):

  * ``Band``                -- unprojected eigenvalue extraction only (vaspvis/band.py).
                               Projected bands, unfolding, plotting and SOC-axis projections
                               are intentionally not vendored.
  * ``BandGap``             -- band-gap extraction incl. the GW/wannier90 variant
                               (vaspvis/utils.py). The pyprocar-based ``soc_axis`` masking is
                               intentionally not vendored.
  * ``clean_wannier_data``  -- wannier90 ``*_band.dat`` parsing helper (vaspvis/utils.py).

The eigenvalue caching behaviour (``eigenvalues.npy``, same array layout) is kept identical so
existing caches written by vaspvis remain valid, and ``DftManager.remove_old_eigenvalues``
keeps working unchanged. The only deliberate deviation from the source is that the E-fermi
value is read from OUTCAR in pure Python instead of shelling out to ``grep`` (same semantics:
first matching line, third whitespace-separated token).
"""

import os
from copy import deepcopy

import numpy as np
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Eigenval
from pymatgen.symmetry.bandstructure import HighSymmKpath
from scipy.interpolate import interp1d


def _read_efermi(outcar_path, shift=0.0):
    """Return E-fermi from an OUTCAR (first 'E-fermi' line, third token), plus ``shift``."""
    try:
        with open(outcar_path, "r") as f:
            for line in f:
                if "E-fermi" in line:
                    return float(line.split()[2]) + shift
    except OSError as e:
        raise ValueError(f"Error reading E-fermi value from {outcar_path}: {e}")
    raise ValueError(f"No E-fermi value found in {outcar_path}")


def clean_wannier_data(raw_data, concatenated_k_e):
    for line in raw_data:
        split_line = line.split('\n')[:-1][0].split(' ')
        filter_line = list(filter(None, split_line))
        if not filter_line:
            continue
        else:
            concatenated_k_e.append([float(x) for x in filter_line])


class Band:
    """Unprojected band-structure data extracted from a VASP band calculation folder.

    Parameters mirror the vaspvis ``Band`` class (only the subset BayesOpt4dftu uses):

        folder (str): folder containing EIGENVAL/OUTCAR/POSCAR/INCAR/KPOINTS
        projected (bool): must stay False; projected parsing is not vendored
        spin (str): 'up' or 'down'
        shift_efermi (float): optional shift added to the Fermi energy
        interpolate (bool): whether ``_get_interpolated_data`` will be used
        new_n (int): number of interpolated points per k-path segment
        efermi_folder (str | None): folder whose OUTCAR provides E-fermi (defaults to ``folder``)
    """

    def __init__(
        self,
        folder,
        projected=False,
        spin="up",
        shift_efermi=0,
        interpolate=True,
        new_n=200,
        custom_kpath=None,
        efermi_folder=None,
    ):
        if projected:
            raise NotImplementedError(
                "Projected band parsing is not vendored into BayesOpt4dftu; "
                "use the standalone vaspvis package for projected band structures."
            )
        if custom_kpath is not None:
            raise NotImplementedError("custom_kpath is not vendored into BayesOpt4dftu.")

        self.interpolate = interpolate
        self.new_n = new_n
        self.eigenval = Eigenval(os.path.join(folder, "EIGENVAL"))

        outcar_path = os.path.join(efermi_folder or folder, "OUTCAR")
        self.efermi = _read_efermi(outcar_path, shift=shift_efermi)

        self.poscar = Poscar.from_file(
            os.path.join(folder, "POSCAR"),
            check_for_POTCAR=False,
            read_velocities=False,
        )
        self.incar = Incar.from_file(os.path.join(folder, "INCAR"))
        if "LSORBIT" in self.incar:
            if self.incar["LSORBIT"]:
                self.lsorbit = True
            else:
                self.lsorbit = False
        else:
            self.lsorbit = False

        if "ISPIN" in self.incar:
            if self.incar["ISPIN"] == 2:
                self.ispin = True
            else:
                self.ispin = False
        else:
            self.ispin = False

        if "LHFCALC" in self.incar:
            if self.incar["LHFCALC"]:
                self.hse = True
            else:
                self.hse = False
        else:
            self.hse = False

        self.kpoints_file = Kpoints.from_file(os.path.join(folder, "KPOINTS"))

        self.folder = folder
        self.spin = spin
        self.spin_dict = {"up": Spin.up, "down": Spin.down}
        self.pre_loaded_bands = os.path.isfile(
            os.path.join(folder, "eigenvalues.npy")
        )
        self.eigenvalues, self.kpoints = self._load_bands()

    def _load_bands(self):
        """Load eigenvalues (shifted to E-fermi = 0) and k-points.

        Identical logic and on-disk cache format ('eigenvalues.npy') as vaspvis.
        """

        if self.spin == "up":
            spin = 0
        if self.spin == "down":
            spin = 1

        if self.pre_loaded_bands:
            with open(
                os.path.join(self.folder, "eigenvalues.npy"), "rb"
            ) as eigenvals:
                band_data = np.load(eigenvals)

            if self.ispin and not self.lsorbit:
                eigenvalues = band_data[:, :, [0, 2]]
                kpoints = band_data[0, :, 4:]
            else:
                eigenvalues = band_data[:, :, 0]
                kpoints = band_data[0, :, 2:]
        else:
            if len(self.eigenval.eigenvalues.keys()) > 1:
                eigenvalues_up = np.transpose(
                    self.eigenval.eigenvalues[Spin.up], axes=(1, 0, 2)
                )
                eigenvalues_down = np.transpose(
                    self.eigenval.eigenvalues[Spin.down], axes=(1, 0, 2)
                )
                eigenvalues_up[:, :, 0] = eigenvalues_up[:, :, 0] - self.efermi
                eigenvalues_down[:, :, 0] = (
                    eigenvalues_down[:, :, 0] - self.efermi
                )
                eigenvalues = np.concatenate(
                    [eigenvalues_up, eigenvalues_down], axis=2
                )
            else:
                eigenvalues = np.transpose(
                    self.eigenval.eigenvalues[Spin.up], axes=(1, 0, 2)
                )
                eigenvalues[:, :, 0] = eigenvalues[:, :, 0] - self.efermi

            kpoints = np.array(self.eigenval.kpoints)

            if self.hse:
                kpoint_weights = np.array(self.eigenval.kpoints_weights)
                zero_weight = np.where(kpoint_weights == 0)[0]
                eigenvalues = eigenvalues[:, zero_weight]
                kpoints = kpoints[zero_weight]

            band_data = np.append(
                eigenvalues,
                np.tile(kpoints, (eigenvalues.shape[0], 1, 1)),
                axis=2,
            )

            np.save(os.path.join(self.folder, "eigenvalues.npy"), band_data)

            if len(self.eigenval.eigenvalues.keys()) > 1:
                eigenvalues = eigenvalues[:, :, [0, 2]]
            else:
                eigenvalues = eigenvalues[:, :, 0]

        if len(self.eigenval.eigenvalues.keys()) > 1:
            eigenvalues = eigenvalues[:, :, spin]

        return eigenvalues, kpoints

    def _get_slices(self, hse=False):
        if not hse:
            high_sym_points = self.kpoints_file.kpts
            num_kpts = self.kpoints_file.num_kpts
            num_slices = int(len(high_sym_points) / 2)
            slices = [
                slice(i * num_kpts, (i + 1) * num_kpts, None)
                for i in range(num_slices)
            ]

        if hse:
            structure = self.poscar.structure
            kpath_obj = HighSymmKpath(structure)
            kpath_coords = np.array(list(kpath_obj._kpath["kpoints"].values()))
            index = np.where(
                np.isclose(
                    self.kpoints[:, None],
                    kpath_coords,
                )
                .all(-1)
                .any(-1)
                == True  # noqa: E712
            )[0]

            segements = []
            for i in range(0, len(index) - 1):
                if not i % 2:
                    segements.append([index[i], index[i + 1]])

            slices = [slice(i[0], i[1] + 1, None) for i in segements]

        return slices

    def _get_k_distance(self):
        slices = self._get_slices(hse=self.hse)
        kdists = []

        for j, i in enumerate(range(len(slices))):
            inv_cell = deepcopy(self.poscar.structure.lattice.inv_matrix)
            inv_cell_norms = np.linalg.norm(inv_cell, axis=1)
            inv_cell /= inv_cell_norms.min()

            # Compare only identical relative cell lengths
            kpt_c = np.dot(self.kpoints[slices[i]], inv_cell.T)

            kdist = np.r_[
                0, np.cumsum(np.linalg.norm(np.diff(kpt_c, axis=0), axis=1))
            ]
            if j == 0:
                kdists.append(kdist)
            else:
                kdists.append(kdist + kdists[-1][-1])

        return kdists

    def _get_interpolated_data_segment(
        self, wave_vectors, data, crop_zero=False, kind="cubic"
    ):
        data_shape = data.shape

        if len(data_shape) == 1:
            fs = interp1d(wave_vectors, data, kind=kind, axis=0)
        else:
            fs = interp1d(wave_vectors, data, kind=kind, axis=1)

        new_wave_vectors = np.linspace(
            wave_vectors.min(), wave_vectors.max(), self.new_n
        )
        data = fs(new_wave_vectors)

        if crop_zero:
            data[np.where(data < 0)] = 0

        return new_wave_vectors, data

    def _get_interpolated_data(
        self, wave_vectors, data, crop_zero=False, kind="cubic"
    ):
        slices = self._get_slices(hse=self.hse)
        data_shape = data.shape
        if len(data_shape) == 1:
            data = [data[i] for i in slices]
        else:
            data = [data[:, i] for i in slices]

        wave_vectors = [wave_vectors[i] for i in slices]

        if len(data_shape) == 1:
            fs = [
                interp1d(i, j, kind=kind, axis=0)
                for (i, j) in zip(wave_vectors, data)
            ]
        else:
            fs = [
                interp1d(i, j, kind=kind, axis=1)
                for (i, j) in zip(wave_vectors, data)
            ]

        new_wave_vectors = [
            np.linspace(wv.min(), wv.max(), self.new_n) for wv in wave_vectors
        ]
        data = np.hstack([f(wv) for (f, wv) in zip(fs, new_wave_vectors)])
        wave_vectors = np.hstack(new_wave_vectors)

        if crop_zero:
            data[np.where(data < 0)] = 0

        return wave_vectors, data


class BandGap:
    """Band-gap extraction from a VASP band folder (or a wannier90 GW band folder).

    Mirrors the vaspvis ``BandGap`` class for the variants BayesOpt4dftu uses:
    method 0 (values closest to E-fermi) / method 1 (band-mean classification),
    spin 'up'/'down'/'both', and ``is_gw=True`` (eigenvalues from wannier90 ``*_band.dat``).
    The pyprocar-based ``soc_axis`` masking is not vendored.
    """

    def __init__(self, folder, spin='both', soc_axis=None, method=0, is_gw=False) -> None:
        if soc_axis is not None:
            raise NotImplementedError(
                "soc_axis-resolved band gaps are not vendored into BayesOpt4dftu; "
                "use the standalone vaspvis package."
            )
        self.folder = folder
        self.method = method
        self.spin = spin
        self.soc_axis = soc_axis
        self.is_gw = is_gw
        self.eigenval = Eigenval(os.path.join(folder, 'EIGENVAL'))
        # For the purpose of band gap determination, an accurate E-fermi from the full-BZ
        # calculation is not necessary, so the band-calculation OUTCAR value is used.
        self.efermi = _read_efermi(os.path.join(folder, 'OUTCAR'))

        self.incar = Incar.from_file(
            os.path.join(folder, 'INCAR')
        )
        if 'LSORBIT' in self.incar:
            if self.incar['LSORBIT']:
                self.lsorbit = True
            else:
                self.lsorbit = False
        else:
            self.lsorbit = False

        if 'ISPIN' in self.incar:
            if self.incar['ISPIN'] == 2:
                self.ispin = True
            else:
                self.ispin = False
        else:
            self.ispin = False

        if 'LHFCALC' in self.incar:
            if self.incar['LHFCALC']:
                self.hse = True
            else:
                self.hse = False
        else:
            self.hse = False

        self.pre_loaded_bands = os.path.isfile(os.path.join(folder, 'eigenvalues.npy'))

        self.bg, self.vbm, self.cbm = self._get_bandgap(method=self.method)

    def _load_eigenvals(self):
        if self.pre_loaded_bands:
            with open(os.path.join(self.folder, 'eigenvalues.npy'), 'rb') as eigenvals:
                band_data = np.load(eigenvals)

            if self.ispin and not self.lsorbit:
                eigenvalues_up = band_data[:, :, [0, 1]]
                eigenvalues_down = band_data[:, :, [2, 3]]
                if self.spin == 'both':
                    eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
                elif self.spin == 'up':
                    eigenvalues_bg = eigenvalues_up
                elif self.spin == 'down':
                    eigenvalues_bg = eigenvalues_down
            else:
                eigenvalues_bg = band_data[:, :, [0, 1]]

        elif self.is_gw:
            win = open(os.path.join(self.folder, 'wannier90.win'), 'r+').readlines()
            nbands: int = 0
            for line in win:
                split_line = line.split('\n')[:-1][0]
                if 'num_wann' in split_line:
                    nbands = int(split_line.split('=')[-1].strip())

            if len(self.eigenval.eigenvalues.keys()) > 1:
                data_up = open(os.path.join(self.folder, 'wannier90.1_band.dat'), 'r+').readlines()
                data_dn = open(os.path.join(self.folder, 'wannier90.2_band.dat'), 'r+').readlines()
                concatenated_k_e_up = []
                concatenated_k_e_down = []
                clean_wannier_data(data_up, concatenated_k_e_up)
                clean_wannier_data(data_dn, concatenated_k_e_down)
                eigenvalues_up = np.array(concatenated_k_e_up).reshape((nbands, -1, 2))[:, :, 1] - self.efermi
                eigenvalues_down = np.array(concatenated_k_e_down).reshape((nbands, -1, 2))[:, :, 1] - self.efermi

                if self.spin == 'both':
                    eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
                elif self.spin == 'up':
                    eigenvalues_bg = eigenvalues_up
                elif self.spin == 'down':
                    eigenvalues_bg = eigenvalues_down

            else:
                data = open(os.path.join(self.folder, 'wannier90_band.dat'), 'r+').readlines()
                concatenated_k_e = []
                clean_wannier_data(data, concatenated_k_e)
                eigenvalues = np.array(concatenated_k_e).reshape((nbands, -1, 2))[:, :, 1] - self.efermi

                eigenvalues_bg = eigenvalues

            return eigenvalues_bg

        else:
            if len(self.eigenval.eigenvalues.keys()) > 1:
                eigenvalues_up = np.transpose(self.eigenval.eigenvalues[Spin.up], axes=(1, 0, 2))
                eigenvalues_down = np.transpose(self.eigenval.eigenvalues[Spin.down], axes=(1, 0, 2))
                eigenvalues_up[:, :, 0] = eigenvalues_up[:, :, 0] - self.efermi
                eigenvalues_down[:, :, 0] = eigenvalues_down[:, :, 0] - self.efermi
                eigenvalues = np.concatenate(
                    [eigenvalues_up, eigenvalues_down],
                    axis=2
                )
                if self.spin == 'both':
                    eigenvalues_bg = np.vstack([eigenvalues_up, eigenvalues_down])
                elif self.spin == 'up':
                    eigenvalues_bg = eigenvalues_up
                elif self.spin == 'down':
                    eigenvalues_bg = eigenvalues_down
            else:
                eigenvalues = np.transpose(self.eigenval.eigenvalues[Spin.up], axes=(1, 0, 2))
                eigenvalues[:, :, 0] = eigenvalues[:, :, 0] - self.efermi
                eigenvalues_bg = eigenvalues

            kpoints = np.array(self.eigenval.kpoints)

            if self.hse:
                kpoint_weights = np.array(self.eigenval.kpoints_weights)
                zero_weight = np.where(kpoint_weights == 0)[0]
                eigenvalues = eigenvalues[:, zero_weight]
                eigenvalues_bg = eigenvalues_bg[:, zero_weight]
                kpoints = kpoints[zero_weight]

            band_data = np.append(
                eigenvalues,
                np.tile(kpoints, (eigenvalues.shape[0], 1, 1)),
                axis=2,
            )
            np.save(os.path.join(self.folder, 'eigenvalues.npy'), band_data)

        return eigenvalues_bg

    @staticmethod
    def _method_0(eigenvalues):
        if len(eigenvalues.shape) == 3:
            eigenvalues = eigenvalues[:, :, 0]

        occupied = eigenvalues[np.where(eigenvalues < 0)]
        unoccupied = eigenvalues[np.where(eigenvalues > 0)]

        vbm = np.nanmax(occupied)
        cbm = np.nanmin(unoccupied)

        if np.nansum(np.abs(np.diff(np.sign(eigenvalues))) > 0) == 0:
            bg = cbm - vbm
        else:
            bg = 0

        return bg, vbm, cbm

    @staticmethod
    def _method_1(eigenvalues):
        if len(eigenvalues.shape) == 3:
            eigenvalues = eigenvalues[:, :, 0]

        band_mean = np.nanmean(eigenvalues, axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.nanmax(eigenvalues[below_index])
        cbm = np.nanmin(eigenvalues[above_index])

        if np.nansum(np.abs(np.diff(np.sign(eigenvalues))) > 0) == 0:
            bg = cbm - vbm
        else:
            bg = 0

        return bg, vbm, cbm

    def _get_bandgap(self, method=0):
        bg, vbm, cbm = np.nan, np.nan, np.nan
        eigenvalues = self._load_eigenvals()

        if method == 0:
            bg, vbm, cbm = self._method_0(eigenvalues)
        elif method == 1:
            bg, vbm, cbm = self._method_1(eigenvalues)

        return bg, vbm, cbm
