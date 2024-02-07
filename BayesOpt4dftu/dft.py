import json
import os
import shutil
import subprocess
from collections import defaultdict
from typing import Optional, Dict, Any

import numpy as np
from ase import Atoms, Atom
from ase.calculators.vasp import Vasp
from pymatgen.io.vasp import Incar, Poscar

from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.io_utils import deprecated
from BayesOpt4dftu.k_path import BoBandPath
from BayesOpt4dftu.logging import BoLoggerGenerator


class VaspInit:
    _logger = BoLoggerGenerator.get_logger("VaspInit")
    _config: Config = None

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
            self._input_dict: Dict[Any, Any] = json.load(f)
        self._struct_info: Dict[Any, Any] = self._input_dict['structure_info']
        self._general_flags: Dict[Any, Any] = self._input_dict['general_flags']

        self._atoms: Optional[Atoms] = None
        self._k_path: Optional[BoBandPath] = None

    def init_atoms(self):
        lattice_param = self._struct_info['lattice_param']
        cell = np.array(self._struct_info['cell'])
        self._atoms = Atoms(cell=cell * lattice_param)
        for atom in self._struct_info['atoms']:
            self._atoms.append(Atom(atom[0], atom[1], magmom=atom[2]))

    def init_k_path(self):
        if self._config.auto_kpath:
            self._k_path = BoBandPath(is_auto=True,
                                      baseline_type=self._config.baseline,
                                      baseline_path=self._config.combined_path_dict[self._config.baseline]['band'])
        else:
            self._k_path = BoBandPath(is_auto=False,
                                      num_kpoints=self._struct_info['num_kpts'],
                                      k_labels=self._struct_info['kpath'],
                                      custom_kpoints=False)
        self._k_path.set_atoms(self._atoms)
        self._k_path.generate()

    def generate_input(self, directory, step, method):
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
            VaspInit.modify_poscar_direct(path=directory)

        elif step == 'band':
            flags.update(self._input_dict[method])
            calc = Vasp(self._atoms,
                        directory=directory,
                        gamma=True,
                        setups='recommended',
                        **flags)
            calc.write_input(self._atoms)  # INCAR POSCAR POTCAR
            VaspInit.modify_poscar_direct(path=directory)

            # KPOINTS
            if method == 'pbe':
                self._k_path.write_kpoints(directory)
            elif method == 'hse':
                self._k_path.write_kpoints(directory, concat_ibzkpt=True)

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

    def run_vasp(self, working_directory):
        original_directory = os.getcwd()
        os.chdir(working_directory)
        errorcode = subprocess.call(f"{self._config.vasp_run_command} > {self._config.out_file_name}", shell=True)
        os.chdir(original_directory)
        return errorcode

    @staticmethod
    def modify_poscar_direct(path='./'):
        with open(path + '/POSCAR', 'r') as f:
            poscar = f.readlines()
            poscar[7] = 'Direct\n'

        with open(path + '/POSCAR', 'w') as d:
            d.writelines(poscar)

    @deprecated
    def corner_case_kpath(self, kptset):
        # Hardcoded for EuS and EuTe since one of the k-point is not in the special kpoints list.
        if 'EuS' in self._atoms.symbols or 'EuTe' in self._atoms.symbols:
            kptset[0] = np.array([0.5, 0.5, 1])

    @deprecated
    def corner_case_magmom(self, directory):
        if str(self._atoms.symbols) in ['Ni2O2']:
            mom_list = {'Ni': 2, 'Mn': 5, 'Co': 3, 'Fe': 4}
            s = str(self._atoms.symbols[0])
            incar_scf = Incar.from_file(directory + '/INCAR')
            incar_scf['MAGMOM'] = '%s -%s 0 0' % (mom_list[s], mom_list[s])
            incar_scf.write_file(directory + '/INCAR')


class DftManager:
    _logger = BoLoggerGenerator.get_logger("DftManager")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def __init__(self):
        self._logger.info("DFT calculations begin.")
        self._dftu_task_counter: int = 0

    def run_task(self, method: str):
        # Logging
        if not self._config.dry_run:
            if method == 'hse':
                self._logger.info("Baseline hybrid DFT calculation begins.")
            elif method == 'dftu' and self._dftu_task_counter == 0:
                self._logger.info("Consecutive DFT+U calculations begin.")

        if not self._config.dry_run:
            if method == 'dftu':
                self._dftu_task_counter += 1

        calc = VaspInit()
        calc.init_atoms()
        calc.init_k_path()

        # Recursive directory creation; it won't raise an error if the directory already exists
        os.makedirs(self._config.combined_path_dict[method]['scf'], exist_ok=True)
        os.makedirs(self._config.combined_path_dict[method]['band'], exist_ok=True)
        DftManager.remove_old_eigenvalues(method)

        if method == 'dftu':
            calc.generate_input(self._config.combined_path_dict[method]['scf'], 'scf', 'pbe')
            calc.generate_input(self._config.combined_path_dict[method]['band'], 'band', 'pbe')
        elif method == 'hse':
            # `scf` dir of `hse` is only used to generate IBZKPT
            calc.generate_input(self._config.combined_path_dict[method]['scf'], 'scf', 'pbe')

        # Exit if dry run
        if self._config.dry_run:
            if method == 'hse':
                self._logger.info(
                    "No actual hybrid DFT calculation was performed. Review the input files before proceeding.")
            elif method == 'dftu':
                self._logger.info(
                    "No actual DFT+U calculations were performed. Review the input files before proceeding.")
            self._logger.info("Dry run executed.")
            return

        # Calc in `scf` dir
        errorcode_scf = calc.run_vasp(self._config.combined_path_dict[method]['scf'])

        # Copy necessary files from `scf` to `band`
        if method == 'dftu':
            for filename in ["CHG", "CHGCAR", "WAVECAR"]:
                shutil.copy(os.path.join(self._config.combined_path_dict[method]['scf'], filename),
                            self._config.combined_path_dict[method]['band'])
        elif method == 'hse':
            for filename in ["IBZKPT"]:
                shutil.copy(os.path.join(self._config.combined_path_dict[method]['scf'], filename),
                            self._config.combined_path_dict[method]['band'])
            calc.generate_input(self._config.combined_path_dict[method]['band'], 'band', 'hse')

        # Calc in `band` dir
        errorcode_band = calc.run_vasp(self._config.combined_path_dict[method]['band'])

        if method == 'hse':
            self._logger.info("Baseline hybrid DFT calculation finished.")

    def finalize(self):
        self._logger.info("All DFT calculations finished.")

    @staticmethod
    def remove_old_eigenvalues(method):
        eigenvalues_file = os.path.join(DftManager._config.combined_path_dict[method]['band'],
                                        DftManager._config.eigen_cache_file_name)
        if os.path.isfile(eigenvalues_file):
            os.remove(eigenvalues_file)
