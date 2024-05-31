import json
import os

import importlib.resources as resources
import shutil

from jsonschema import validate, ValidationError
from BayesOpt4dftu.logging import BoLoggerGenerator


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

        # File paths
        self.root_dir = './'
        self.abs_root_dir = os.path.abspath(self.root_dir)

        self.step_dir_dict = {'scf': 'scf', 'band': 'band'}
        self.method_dir_dict = {'dftu': 'dftu', 'hse': 'hse', 'gw': 'gw'}
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

        self.eigen_cache_file_name = 'eigenvalues.npy'

        self.column_names = {'band_gap': 'band_gap',
                             'delta_gap': 'delta_gap',
                             'delta_band': 'delta_band',
                             'delta_mag': 'delta_mag',
                             'obj_func': 'obj_func',
                             'd_obj': 'd_obj'}

        # VASP parameters
        vasp_env_params = data['vasp_env']
        # Command to run VASP executable.
        self.vasp_run_command = vasp_env_params.get('vasp_run_command', 'srun -n 54 vasp_ncl')
        # Define the name for output file.
        self.out_file_name = os.path.expanduser(vasp_env_params.get('out_file_name', 'vasp.out'))
        # Define the path directing to the VASP pseudopotential.
        self.vasp_pp_path = os.path.expanduser(vasp_env_params.get('vasp_pp_path', '/home/maituoy/pp_vasp/'))
        os.environ['VASP_PP_PATH'] = self.vasp_pp_path
        self.dry_run = vasp_env_params.get('dry_run', False)
        self.dftu_only = vasp_env_params.get('dftu_only', False)
        self.get_optimal_band = vasp_env_params.get('get_optimal_band', False)

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

        self._logger.info(f"Configuration loaded from file {self.config_file_name}.")


class TempFileManager:
    _logger = BoLoggerGenerator.get_logger("TempFileManager")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def setup_temp_files(self):
        if self._config.resume_checkpoint:
            if not os.path.exists(self._config.tmp_u_path) or os.path.getsize(self._config.tmp_u_path) == 0:
                self._logger.error(f"Missing or damaged checkpoint file: {self._config.tmp_u_path}.")
                raise RuntimeError

            elif not os.path.exists(self._config.tmp_config_path) or os.path.getsize(self._config.tmp_config_path) == 0:
                self._logger.error(f"Missing or damaged checkpoint file: {self._config.tmp_config_path}.")
                raise RuntimeError

            else:
                self._logger.info("Previous temporary files will continue to be used.")

        else:
            # Temporary config
            shutil.copyfile(self._config.config_path, self._config.tmp_config_path)

            # Temporary Bayesian Optimization log
            header = []
            for i, u in enumerate(self._config.which_u):
                header.append(f"U_ele_{str(i + 1)}")

            if os.path.exists(self._config.tmp_u_path):
                os.remove(self._config.tmp_u_path)

            if self._config.alpha_mag:
                with open(self._config.tmp_u_path, 'w+') as f:
                    f.write(f"{(' '.join(header))} "
                            f"{self._config.column_names['band_gap']} "
                            f"{self._config.column_names['delta_gap']} "
                            f"{self._config.column_names['delta_band']} "
                            f"{self._config.column_names['delta_mag']} "
                            f"{self._config.column_names['obj_func']} "
                            f"{self._config.column_names['d_obj']} \n")
            else:
                with open(self._config.tmp_u_path, 'w+') as f:
                    f.write(f"{(' '.join(header))} "
                            f"{self._config.column_names['band_gap']} "
                            f"{self._config.column_names['delta_gap']} "
                            f"{self._config.column_names['delta_band']} "
                            f"{self._config.column_names['obj_func']} "
                            f"{self._config.column_names['d_obj']} \n")

            self._logger.info("Temporary files initiated.")

    def clean_up(self):
        shutil.move(self._config.tmp_u_path, self._config.u_path)
        os.remove(self._config.tmp_config_path)

        self._logger.info("Temporary files removed.")
