import json
import os


class Config:
    _instance = None

    def __new__(cls, config_file="input.json"):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file):
        with open(config_file, "r") as f:
            data = json.load(f)

        self.config_file_name = config_file
        self.tmp_config_file_name = f"{config_file.split('.')[0]}_tmp.{config_file.split('.')[1]}"
        vasp_env_params = data['vasp_env']
        # Command to run VASP executable.
        self.vasp_run_command = vasp_env_params.get('vasp_run_command', 'srun -n 54 vasp_ncl')
        # Define the name for output file.
        self.out_file_name = vasp_env_params.get('out_file_name', 'vasp.out')
        # Define the path direct to the VASP pseudopotential.
        self.vasp_pp_path = vasp_env_params.get('vasp_pp_path', '/home/maituoy/pp_vasp/')
        self.dry_run = vasp_env_params.get('dry_run', False)
        self.dftu_only = vasp_env_params.get('dftu_only', False)
        self.get_optimal_band = vasp_env_params.get('get_optimal_band', False)

        os.environ['VASP_PP_PATH'] = self.vasp_pp_path

        bo_params = data['bo']
        self.k = float(bo_params.get('kappa', 5))
        self.a1 = bo_params.get('alpha1', 0.25)
        self.a2 = bo_params.get('alpha2', 0.75)
        self.which_u = tuple(bo_params.get('which_u', [1, 1]))
        self.urange = tuple(bo_params.get('urange', [-10, 10]))
        self.br = tuple(bo_params.get('br', [5, 5]))
        self.import_kpath = bo_params.get('import_kpath', False)
        self.elements = bo_params.get('elements', ['In', 'As'])
        self.iteration = bo_params.get('iteration', 50)
        self.threshold = bo_params.get('threshold', 0.0001)
