import numpy as np


class BoKpath:
    special_kpoints_dict = {"F": np.array([0.5, 0.5, 0]),
                            "G": np.array([0, 0, 0]),
                            "T": np.array([0.5, 0.5, 0.5]),
                            "K": np.array([0.8, 0.35, 0.35]),
                            "L": np.array([0.5, 0, 0])}
