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

from BayesOpt4dftu.special_kpath import kpath_dict


class VaspInit(object):
    def __init__(self, input_path):
        with open(input_path, 'r') as f:
            self.input_dict = json.load(f)
        self.struct_info = self.input_dict['structure_info']
        self.general_flags = self.input_dict['general_flags']
        self.atoms = None

    def init_atoms(self):
        lattice_param = self.struct_info['lattice_param']
        cell = np.array(self.struct_info['cell'])
        self.atoms = Atoms(cell=cell * lattice_param)
        for atom in self.struct_info['atoms']:
            self.atoms.append(Atom(atom[0], atom[1], magmom=atom[2]))

        return self.atoms

    @staticmethod
    def modify_poscar_direct(path='./'):
        with open(path + '/POSCAR', 'r') as f:
            poscar = f.readlines()
            poscar[7] = 'Direct\n'
            f.close()

        with open(path + '/POSCAR', 'w') as d:
            d.writelines(poscar)
            d.close()

    def kpt4pbeband(self, path, import_kpath):
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)

        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
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
        if 'EuS' in self.atoms.symbols or 'EuTe' in self.atoms.symbols:
            kptset[0] = np.array([0.5, 0.5, 1])

        kpt = Kpoints(comment='band', kpts=kptset, num_kpts=num_kpts,
                      style='Line_mode', coord_type="Reciprocal", labels=lbs)
        kpt.write_file(path + '/KPOINTS')

    def kpt4hseband(self, path, import_kpath):
        ibz = open(path + '/IBZKPT', 'r')
        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
        ibzlist = ibz.readlines()
        ibzlist[1] = str(num_kpts * (len(labels) - 1) +
                         int(ibzlist[1].split('\n')[0])) + '\n'
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)
        for i in range(len(labels) - 1):
            k_head = special_kpoints[labels[i]]
            k_tail = special_kpoints[labels[i + 1]]
            increment = (k_tail - k_head) / (num_kpts - 1)
            ibzlist.append(' '.join(map(str, k_head)) +
                           ' 0 ' + labels[i] + '\n')
            for j in range(1, num_kpts - 1):
                k_next = k_head + increment * j
                ibzlist.append(' '.join(map(str, k_next)) + ' 0\n')
            ibzlist.append(' '.join(map(str, k_tail)) +
                           ' 0 ' + labels[i + 1] + '\n')
        with open(path + '/KPOINTS', 'w') as f:
            f.writelines(ibzlist)

    def generate_input(self, directory, step, xc, import_kpath):
        flags = {}
        flags.update(self.general_flags)
        flags.update(self.input_dict[step])
        if step == 'scf':
            # `scf` dir under `hse` is just used to generate IBZKPT, so xc=pbe is enough.
            flags.update(self.input_dict['pbe'])
            calc = Vasp(self.atoms,
                        directory=directory,
                        kpts=self.struct_info['kgrid_' + xc],
                        gamma=True,
                        setups='recommended',
                        **flags)
            calc.write_input(self.atoms)
            # random exception (Ni2O2)
            if str(self.atoms.symbols) in ['Ni2O2']:
                mom_list = {'Ni': 2, 'Mn': 5, 'Co': 3, 'Fe': 4}
                s = str(self.atoms.symbols[0])
                incar_scf = Incar.from_file(directory + '/INCAR')
                incar_scf['MAGMOM'] = '%s -%s 0 0' % (mom_list[s], mom_list[s])
                incar_scf.write_file(directory + '/INCAR')

            VaspInit.modify_poscar_direct(path=directory)
        elif step == 'band':
            flags.update(self.input_dict[xc])
            calc = Vasp(self.atoms,
                        directory=directory,
                        gamma=True,
                        setups='recommended',
                        **flags)
            calc.write_input(self.atoms)
            VaspInit.modify_poscar_direct(path=directory)
            if xc == 'pbe':
                self.kpt4pbeband(directory, import_kpath)
            elif xc == 'hse':
                self.kpt4hseband(directory, import_kpath)
        # Rewrite LDAU flags to reflect correct numerical precision
        if xc == 'pbe':
            self.rewrite_ldau(directory)

    def rewrite_ldau(self, directory):

        # Load the JSON data
        params = self.input_dict["pbe"]

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
    def remove_old_eigenvalues(root_dir, method_dir):
        if os.path.isfile(f'{root_dir}/{method_dir}/band/eigenvalues.npy'):
            os.remove(f'{root_dir}/{method_dir}/band/eigenvalues.npy')


def calculate(command: str, config_file_name: str, outfilename: str, method: str, import_kpath: bool, is_dry: bool):
    olddir = os.getcwd()
    calc = VaspInit(f"{olddir}/{config_file_name}")
    calc.init_atoms()

    # Recursive directory creation; it won't raise an error if the directory already exists
    os.makedirs(olddir + '/%s/scf' % method, exist_ok=True)
    os.makedirs(olddir + '/%s/band' % method, exist_ok=True)
    VaspInit.remove_old_eigenvalues(olddir, method)

    if method == 'dftu':
        calc.generate_input(olddir + '/%s/scf' % method, 'scf', 'pbe', import_kpath)
        calc.generate_input(olddir + '/%s/band' % method, 'band', 'pbe', import_kpath)
    elif method == 'hse':
        calc.generate_input(olddir + '/%s/scf' % method, 'scf', 'pbe', import_kpath)

    # exit if dry run
    if is_dry:
        return

    # calc in `scf` dir
    os.chdir(olddir + '/%s/scf' % method)
    errorcode_scf = subprocess.call('%s > %s' % (command, outfilename), shell=True)

    if method == 'dftu':
        for filename in ["CHG", "CHGCAR", "WAVECAR"]:
            shutil.copy(filename, olddir + '/%s/band' % method)
    elif method == 'hse':
        # `scf` dir under `hse` is just used to generate IBZKPT
        for filename in ["IBZKPT"]:
            shutil.copy(filename, olddir + '/%s/band' % method)
        calc.generate_input(olddir + '/%s/band' % method, 'band', 'hse', import_kpath)

    # calc in `band` dir
    os.chdir(olddir + '/%s/band' % method)
    errorcode_band = subprocess.call('%s > %s' % (command, outfilename), shell=True)
    os.chdir(olddir)
