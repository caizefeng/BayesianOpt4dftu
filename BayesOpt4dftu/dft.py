import json
import os
import shutil
import subprocess
from collections import defaultdict

import numpy as np
from ase import Atoms, Atom
from ase.calculators.vasp import Vasp
from ase.dft.kpoints import get_special_points
from pymatgen.io.vasp import Kpoints, Incar, Poscar

from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.kpath import BoKpath


class VaspInit:
    _config = None  # type: Config

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def __init__(self):
        if self._config.dry_run:
            config_path = self._config.config_path
        else:
            config_path = self._config.tmp_config_path
        with open(config_path, 'r') as f:
            self._input_dict = json.load(f)
        self._struct_info = self._input_dict['structure_info']
        self._general_flags = self._input_dict['general_flags']
        self._import_kpath = self._config.import_kpath
        self._atoms = None

    def init_atoms(self):
        lattice_param = self._struct_info['lattice_param']
        cell = np.array(self._struct_info['cell'])
        self._atoms = Atoms(cell=cell * lattice_param)
        for atom in self._struct_info['atoms']:
            self._atoms.append(Atom(atom[0], atom[1], magmom=atom[2]))

        return self._atoms

    # def kpt4band(self, method):
    #     num_kpts = self.struct_info['num_kpts']
    #     labels = self.struct_info['kpath']
    #
    #     if isinstance(num_kpts, list):
    #         if all(isinstance(elem, int) for elem in num_kpts) and len(num_kpts) == len(labels) - 1:
    #             return "Variable is a list of integers with the required length"
    #         else:
    #             raise ValueError("The number of elements in `num_kpts` should be ONE less than the length of `kpath`!")
    #     elif isinstance(num_kpts, int):
    #         num_kpts = [num_kpts] * (len(labels) - 1)
    #     elif num_kpts == 'auto':
    #         if self._baseline
    #             return "Variable is the special string 'auto'"
    #     else:
    #         raise TypeError("Unsupported `num_kpts` type")
    #
    #     if method == 'pbe':
    #         self.kpt4pbeband(directory, import_kpath)
    #     elif method == 'hse':
    #         self.kpt4hseband(directory, import_kpath)

    def kpt4pbeband(self, path, import_kpath):
        if import_kpath:
            special_kpoints = BoKpath.special_kpoints_dict
        else:
            special_kpoints = get_special_points(self._atoms.cell)

        num_kpts = self._struct_info['num_kpts']
        labels = self._struct_info['kpath']
        kptset = list()
        lbs = list()
        if labels[0] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[0]])
            lbs.append(labels[0])

        for i in range(1, len(labels) - 1):
            if labels[i] in special_kpoints.keys():
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
        if labels[-1] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[-1]])
            lbs.append(labels[-1])

        # Hardcoded for EuS and EuTe since one of the k-point is not in the special kpoints list.
        if 'EuS' in self._atoms.symbols or 'EuTe' in self._atoms.symbols:
            kptset[0] = np.array([0.5, 0.5, 1])

        kpt = Kpoints(comment='band',
                      kpts=kptset,
                      num_kpts=num_kpts,
                      style='Line_mode',
                      coord_type="Reciprocal",
                      labels=lbs)
        kpt.write_file(path + '/KPOINTS')

    def kpt4hseband(self, path, import_kpath):
        ibz = open(path + '/IBZKPT', 'r')
        num_kpts = self._struct_info['num_kpts']
        labels = self._struct_info['kpath']
        ibzlist = ibz.readlines()
        ibzlist[1] = str(num_kpts * (len(labels) - 1) + int(ibzlist[1].split('\n')[0])) + '\n'
        if import_kpath:
            special_kpoints = BoKpath.special_kpoints_dict
        else:
            special_kpoints = get_special_points(self._atoms.cell)
        for i in range(len(labels) - 1):
            k_head = special_kpoints[labels[i]]
            k_tail = special_kpoints[labels[i + 1]]
            increment = (k_tail - k_head) / (num_kpts - 1)
            ibzlist.append(' '.join(map(str, k_head)) + ' 0 ' + labels[i] + '\n')
            for j in range(1, num_kpts - 1):
                k_next = k_head + increment * j
                ibzlist.append(' '.join(map(str, k_next)) + ' 0\n')
            ibzlist.append(' '.join(map(str, k_tail)) + ' 0 ' + labels[i + 1] + '\n')
        with open(path + '/KPOINTS', 'w') as f:
            f.writelines(ibzlist)

    def generate_input(self, directory, step, method, import_kpath):
        flags = {}
        flags.update(self._general_flags)
        flags.update(self._input_dict[step])
        if step == 'scf':
            # `scf` dir of `hse` is only used to generate IBZKPT, so it's always `input_dict['pbe']`.
            flags.update(self._input_dict['pbe'])
            calc = Vasp(self._atoms,
                        directory=directory,
                        kpts=self._struct_info['kgrid_' + method],
                        gamma=True,
                        setups='recommended',
                        **flags)
            calc.write_input(self._atoms)

            # Corner case: Ni2O2
            if str(self._atoms.symbols) in ['Ni2O2']:
                mom_list = {'Ni': 2, 'Mn': 5, 'Co': 3, 'Fe': 4}
                s = str(self._atoms.symbols[0])
                incar_scf = Incar.from_file(directory + '/INCAR')
                incar_scf['MAGMOM'] = '%s -%s 0 0' % (mom_list[s], mom_list[s])
                incar_scf.write_file(directory + '/INCAR')

            VaspInit.modify_poscar_direct(path=directory)
        elif step == 'band':
            flags.update(self._input_dict[method])
            calc = Vasp(self._atoms,
                        directory=directory,
                        gamma=True,
                        setups='recommended',
                        **flags)

            # INCAR POSCAR POTCAR
            calc.write_input(self._atoms)
            VaspInit.modify_poscar_direct(path=directory)

            # KPOINTS
            if method == 'pbe':
                self.kpt4pbeband(directory, import_kpath)
            elif method == 'hse':
                self.kpt4hseband(directory, import_kpath)

        # Rewrite LDAU flags to reflect correct numerical precision
        if method == 'pbe':
            self.rewrite_ldau(directory)

    def rewrite_ldau(self, directory):

        # Load the JSON data
        params = self._input_dict["pbe"]

        # Check if LDAU should be applied
        if params["ldau"]:

            # Load the current INCAR and POSCAR
            incar = Incar.from_file(directory + '/INCAR')
            poscar = Poscar.from_file(directory + '/POSCAR')

            # Extract LDAU parameters from JSON
            ldau_luj = params["ldau_luj"]
            elements = [site.specie.name for site in poscar.structure]

            # Defaults
            ldaul = defaultdict(lambda: -1)
            ldauu = defaultdict(lambda: 0)
            ldauj = defaultdict(lambda: 0)

            # Update defaults with provided values
            for element, values in ldau_luj.items():
                ldaul[element] = values["L"]
                ldauu[element] = values["U"]
                ldauj[element] = values["J"]

            # Order values by appearance of elements in POSCAR
            incar["LDAUL"] = [ldaul[el] for el in elements]
            incar["LDAUU"] = [ldauu[el] for el in elements]
            incar["LDAUJ"] = [ldauj[el] for el in elements]

            # Write the modified INCAR back to file
            incar.write_file(directory + '/INCAR')

        else:
            print("LDAU not set in the provided JSON data.")

    @staticmethod
    def modify_poscar_direct(path='./'):
        with open(path + '/POSCAR', 'r') as f:
            poscar = f.readlines()
            poscar[7] = 'Direct\n'

        with open(path + '/POSCAR', 'w') as d:
            d.writelines(poscar)

    @staticmethod
    def remove_old_eigenvalues(method):
        eigenvalues_file = os.path.join(VaspInit._config.combined_path_dict[method]['band'],
                                        VaspInit._config.eigen_cache_file_name)
        if os.path.isfile(eigenvalues_file):
            os.remove(eigenvalues_file)


class DftExecutor:

    def __init__(self, config):
        self._config = config  # type: Config

    def calculate(self, method: str):
        calc = VaspInit()

        calc.init_atoms()

        # Recursive directory creation; it won't raise an error if the directory already exists
        os.makedirs(self._config.combined_path_dict[method]['scf'], exist_ok=True)
        os.makedirs(self._config.combined_path_dict[method]['band'], exist_ok=True)
        VaspInit.remove_old_eigenvalues(method)

        if method == 'dftu':
            calc.generate_input(self._config.combined_path_dict[method]['scf'], 'scf', 'pbe',
                                self._config.import_kpath)
            calc.generate_input(self._config.combined_path_dict[method]['band'], 'band', 'pbe',
                                self._config.import_kpath)
        elif method == 'hse':
            # `scf` dir of `hse` is only used to generate IBZKPT
            calc.generate_input(self._config.combined_path_dict[method]['scf'], 'scf', 'pbe',
                                self._config.import_kpath)

        # Exit if dry run
        if self._config.dry_run:
            return

        # Calc in `scf` dir
        os.chdir(self._config.combined_path_dict[method]['scf'])
        errorcode_scf = subprocess.call('%s > %s' % (self._config.vasp_run_command, self._config.out_file_name),
                                        shell=True)
        os.chdir(self._config.abs_root_dir)

        # Copy necessary files from `scf` to `band`
        if method == 'dftu':
            for filename in ["CHG", "CHGCAR", "WAVECAR"]:
                shutil.copy(os.path.join(self._config.combined_path_dict[method]['scf'], filename),
                            self._config.combined_path_dict[method]['band'])
        elif method == 'hse':
            for filename in ["IBZKPT"]:
                shutil.copy(os.path.join(self._config.combined_path_dict[method]['scf'], filename),
                            self._config.combined_path_dict[method]['band'])
            calc.generate_input(self._config.combined_path_dict[method]['band'], 'band', 'hse',
                                self._config.import_kpath)

        # Calc in `band` dir
        os.chdir(self._config.combined_path_dict[method]['band'])
        errorcode_band = subprocess.call('%s > %s' % (self._config.vasp_run_command, self._config.out_file_name),
                                         shell=True)
        os.chdir(self._config.abs_root_dir)
