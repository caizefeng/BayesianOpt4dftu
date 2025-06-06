import importlib.resources as resources
import json
import os

import numpy as np
from jsonschema import validate, ValidationError

from BayesOpt4dftu.common.logger import BoLoggerGenerator


class Config:
    _logger = BoLoggerGenerator.get_logger("Config")
    _instance = None

    def __new__(cls, config_file="input.json"):
        if not cls._instance:
            cls._instance = super().__new__(cls)

            cls._instance._logger.info("Loading configuration ...")

            with resources.path("BayesOpt4dftu.schemas", "input_schema.json") as schema_path:
                cls._instance._validate_config(config_file, str(schema_path))

            cls._instance._load_config(config_file)

        return cls._instance

    def _validate_config(self, config_file, schema_file):
        # Load the schema
        with open(schema_file, 'r') as schema_f:
            schema = json.load(schema_f)

        # Load the data to be validated
        with open(config_file, 'r') as config_f:
            data = json.load(config_f)

        # Perform validation
        try:
            validate(instance=data, schema=schema)
            self._logger.info("JSON schema validation successful.")
        except ValidationError as e:
            self._logger.error(f"JSON schema validation failed: {e.message}")
            raise

    def _load_config(self, config_file):

        with open(config_file, "r") as config_f:
            data = json.load(config_f)

        # BO parameters
        bo_params = data['bo']
        self.resume_checkpoint = bo_params.get('resume_checkpoint', False)
        self.k = float(bo_params.get('kappa', 5))
        self.alpha_gap = bo_params.get('alpha_gap', 0.25)
        self.alpha_band = bo_params.get('alpha_band', 0.75)
        self.alpha_mag = bo_params.get('alpha_mag', 0.0)
        self.mag_axis = bo_params.get('mag_axis', 'all')
        if self.alpha_mag:
            self.include_mag = True
        else:
            self.include_mag = False
        self.which_u = tuple(bo_params.get('which_u', [1, 1]))
        self.urange = tuple(bo_params.get('urange', [-10.0, 10.0]))
        self.br = tuple(bo_params.get('br', [5, 5]))
        self.elements = bo_params.get('elements', ['In', 'As'])
        self.iteration = bo_params.get('iteration', 50)
        self.threshold = bo_params.get('threshold', 0.0001)
        self.baseline = bo_params.get('baseline', 'hse')
        self.report_optimum_interval = bo_params.get('report_optimum_interval', 10)
        self.threshold_opt_u = bo_params.get('threshold_opt_u', 0.0)
        self.print_magmom = bo_params.get('print_magmom', False)

        # File paths
        self.root_dir = './'
        self.abs_root_dir = os.path.abspath(self.root_dir)

        self.step_dir_dict = {'scf': 'scf', 'band': 'band'}
        self.method_dir_dict = {'dftu': 'dftu', 'hse': 'hse', 'gw': 'gw', 'dft': 'dft'}
        self.combined_path_dict = {}
        for key in self.method_dir_dict:
            self.combined_path_dict[key] = {k: os.path.join(self.root_dir, key, v)
                                            for k, v in self.step_dir_dict.items()}

        self.config_file_name = config_file
        self.tmp_config_file_name = f"{config_file.split('.')[0]}_tmp.{config_file.split('.')[1]}"
        self.u_file_name = f"u_kappa_{self.k}_ag_{self.alpha_gap}_ab_{self.alpha_band}_am_{self.alpha_mag}.txt"
        self.tmp_u_file_name = 'u_tmp.txt'

        self.config_path = os.path.join(self.root_dir, self.config_file_name)
        self.tmp_config_path = os.path.join(self.root_dir, self.tmp_config_file_name)
        self.u_path = os.path.join(self.root_dir, self.u_file_name)
        self.tmp_u_path = os.path.join(self.root_dir, self.tmp_u_file_name)
        self.formatted_u_path = os.path.join(self.root_dir, f"formatted_{self.u_file_name}")

        self.eigen_cache_file_name = 'eigenvalues.npy'

        # Log file headers
        self.column_names = {'band_gap': 'band_gap',
                             'delta_gap': 'delta_gap',
                             'delta_band': 'delta_band',
                             'delta_mag': 'delta_mag',
                             'mag_moment': 'mag_moment',
                             'obj_func': 'obj_func',
                             'd_obj': 'd_obj'}

        self.headers = []
        for i, u in enumerate(self.which_u):
            self.headers.append(f"U_ele_{str(i + 1)}")
        self.headers.append(self.column_names['band_gap'])
        self.headers.append(self.column_names['delta_gap'])
        self.headers.append(self.column_names['delta_band'])
        if self.include_mag:
            self.headers.append(self.column_names['delta_mag'])
        if self.print_magmom:
            self.headers.append(self.column_names['mag_moment'])
        self.headers.append(self.column_names['obj_func'])
        self.headers.append(self.column_names['d_obj'])

        self.method_name_dict = {'hse': 'hybrid DFT',
                                 'gw': 'GW',
                                 'dft': 'DFT'}

        # VASP parameters
        vasp_env_params = data['vasp_env']
        # Command to run VASP executable.
        self.vasp_run_command = vasp_env_params.get('vasp_run_command', 'srun -n 54 vasp_ncl')
        # Define the name for output file.
        self.out_file_name = os.path.expanduser(vasp_env_params.get('out_file_name', 'vasp.out'))
        # Define the path directing to the VASP pseudopotential.
        self.vasp_pp_path = os.path.expanduser(vasp_env_params.get('vasp_pp_path', '/home/xxx/pseudopotentials/'))
        os.environ['VASP_PP_PATH'] = self.vasp_pp_path
        self.dry_run = vasp_env_params.get('dry_run', False)
        self.dftu_only = vasp_env_params.get('dftu_only', False)
        self.get_optimal_band = vasp_env_params.get('get_optimal_band', True)

        # Additional messages to users
        if self.baseline == 'gw':
            self._logger.info("Detected baseline calculation setting: GW.")
            self._logger.info("Ensure an OUTCAR file with ACCURATE E-fermi and magnetic moment, obtained using high-precision BZ integration, is placed in the `gw/scf` folder.")
        elif self.baseline == 'hse':
            self._logger.info("Detected baseline calculation setting: HSE.")
        elif self.baseline == 'dft':
            self._logger.info("Detected baseline calculation setting: Semi-local DFT.")

        # K-path parameters
        self.num_kpts = data['structure_info']['num_kpts']
        if self.baseline == 'gw' and self.num_kpts != "auto":
            self._logger.error("Baseline GW currently only supports automatic K-path. "
                               "Ensure `num_kpts` is set to 'auto'.")
            raise ValueError

        if self.dftu_only is False and self.num_kpts == "auto":
            self._logger.error(f"`num_kpts` must be an integer instead of 'auto' when `dftu_only` is set to false.")
            raise ValueError

        if isinstance(self.num_kpts, int) and self.num_kpts > 0:
            self._logger.info("K-path for band manually set.")
            self.line_mode_kpath = True
            self.auto_kpath = False
        elif self.num_kpts == "auto":
            self._logger.info("K-path for band will be automatically deduced from the baseline calculation.")
            self._logger.info("Ensure uniform density of sampling points along the path in the baseline calculation.")
            self.auto_kpath = True
            self.line_mode_kpath = False
        else:
            self._logger.error("Unsupported `num_kpts` type: only positive integers or 'auto' are accepted.")
            raise ValueError

        self.custom_kpoints = data['structure_info'].get('custom_kpoints', None)
        if self.custom_kpoints is not None and isinstance(self.custom_kpoints, dict):
            self.custom_kpoints = {key: np.array(value) for key, value in self.custom_kpoints.items()}  # convert to numpy array

        # Path to custom POTCAR
        self.custom_POTCAR_path = data['structure_info'].get('custom_POTCAR_path', None)

        self._logger.info(f"Configuration loaded from file {self.config_file_name}.")


