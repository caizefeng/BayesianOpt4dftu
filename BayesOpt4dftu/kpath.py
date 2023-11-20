import numpy as np

# class BoKpath:


    def generate_line_mode(self):
        kptset = list()
        lbs = list()
        for i in range(len(self._k_labels_list)):
            if self._k_labels_list[i] in self._special_kpoints.keys():
                kptset.append(self._special_kpoints[self._k_labels_list[i]])
                lbs.append(self._k_labels_list[i])
                if i in range(1, len(self._k_labels_list) - 1):
                    kptset.append(self._special_kpoints[self._k_labels_list[i]])
                    lbs.append(self._k_labels_list[i])

        self._k_path = Kpoints(comment='band',
                               kpts=kptset,
                               num_kpts=self._num_kpoints,
                               style='Line_mode',  # noqa
                               coord_type="Reciprocal",
                               labels=lbs)

    def concatenate_with_ibzkpt(self, directory):
        with open(directory + '/IBZKPT', 'r') as ibz:
            kpoints_lines = ibz.readlines()

        kpoints_lines[1] = str(
            self._num_kpoints * (len(self._k_labels_list) - 1) + int(kpoints_lines[1].split('\n')[0])) + '\n'

        for i in range(len(self._k_labels_list) - 1):
            k_head = self._special_kpoints[self._k_labels_list[i]]
            k_tail = self._special_kpoints[self._k_labels_list[i + 1]]
            increment = (k_tail - k_head) / (self._num_kpoints - 1)
            kpoints_lines.append(' '.join(map(str, k_head)) + ' 0 ' + self._k_labels_list[i] + '\n')
            for j in range(1, self._num_kpoints - 1):
                k_next = k_head + increment * j
                kpoints_lines.append(' '.join(map(str, k_next)) + ' 0\n')
            kpoints_lines.append(' '.join(map(str, k_tail)) + ' 0 ' + self._k_labels_list[i + 1] + '\n')

        self._k_path_with_scf_grid = Kpoints.from_string(''.join(kpoints_lines))


# Deprecated
special_kpoints_dict = {"F": np.array([0.5, 0.5, 0]),
                        "G": np.array([0, 0, 0]),
                        "T": np.array([0.5, 0.5, 0.5]),
                        "K": np.array([0.8, 0.35, 0.35]),
                        "L": np.array([0.5, 0, 0])}
