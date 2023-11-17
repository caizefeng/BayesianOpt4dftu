import json
import os

import pandas as pd
from ase.dft.kpoints import *
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt

from BayesOpt4dftu.io_helpers import SuppressPrints


class OptimizerGenerator:
    def __init__(self, utxt_path, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa):
        data = pd.read_csv(utxt_path, header=0, delimiter=r"\s+", engine="python")
        self.opt_u_index = opt_u_index
        self.u_range = u_range
        self.a1 = a1
        self.a2 = a2
        self.delta_mag_weight = delta_mag_weight
        self.kappa = kappa
        self.n_obs, _ = data.shape
        self.data = data
        self.utility_function = UtilityFunction(kind="ucb", kappa=kappa, xi=0)

    def loss(self, delta_gap=0.0, delta_band=0.0, delta_mag=0.0, alpha_1=0.5, alpha_2=0.5, delta_mag_weight=0.0):
        return -alpha_1 * delta_gap ** 2 - alpha_2 * delta_band ** 2 - delta_mag_weight * delta_mag ** 2

    def set_bounds(self):
        # Set up the indices of variables that are going to be optimized.
        variables_string = ['u_' + str(i) for i, o in enumerate(self.opt_u_index) if o]

        # Set up the U ranges for each variable.
        pbounds = {}
        for variable in variables_string:
            pbounds[variable] = self.u_range
        return pbounds

    def optimizer(self):
        pbounds = self.set_bounds()
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
            allow_duplicate_points=True
        )

        v_strings = list(pbounds.keys())
        opt_index = [int(v.split('_')[1]) for v in v_strings]

        for i in range(self.n_obs):
            values = list()
            for j in opt_index:
                values.append(self.data.iloc[i][j])
            params = {}
            for (value, variable) in zip(values, v_strings):
                params[variable] = value

            if self.delta_mag_weight:
                target = self.loss(delta_gap=self.data.iloc[i].delta_gap,
                                   delta_band=self.data.iloc[i].delta_band,
                                   delta_mag=self.data.iloc[i].delta_mag,
                                   alpha_1=self.a1,
                                   alpha_2=self.a2,
                                   delta_mag_weight=self.delta_mag_weight)
            else:
                target = self.loss(delta_gap=self.data.iloc[i].delta_gap,
                                   delta_band=self.data.iloc[i].delta_band,
                                   alpha_1=self.a1,
                                   alpha_2=self.a2)

            # Suppress non-unique data point registration messages
            with SuppressPrints():
                optimizer.register(
                    params=params,
                    target=target,
                )

        return optimizer, target


class PlotBO(OptimizerGenerator):
    def __init__(self, utxt_path, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa, elements):
        super().__init__(utxt_path, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa)
        optimizer, target = self.optimizer()
        self.optimizer = optimizer
        self.target = target
        self.elements = elements
        self.optimal = 0

    def get_optimal(self, x, mu):
        best_obj = mu.max()
        best_index = np.where(mu == mu.max())[0][0]
        best_u = x[best_index]
        optimal = (best_u, best_obj)
        return optimal

    def predict(self, ratio=1):
        u = list(self.optimizer.res[0]["params"].keys())
        dim = len(u)
        plot_size = len(self.optimizer.res) * ratio
        if dim == 1:
            x = np.linspace(self.u_range[0], self.u_range[1], 10000).reshape(-1, 1)
            x_obs = np.array([res["params"][u[0]] for res in self.optimizer.res]).reshape(-1, 1)[:plot_size]
            y_obs = np.array([res["target"] for res in self.optimizer.res])[:plot_size]

            self.optimizer._gp.fit(x_obs, y_obs)
            mu, sigma = self.optimizer._gp.predict(x, return_std=True)
            self.optimal = self.get_optimal(x, mu)

            data4plot = {'mu': mu,
                         'sigma': sigma,
                         'x': x,
                         'x_obs': x_obs,
                         'y_obs': y_obs}

            return data4plot

        if dim == 2:
            x = y = np.linspace(self.u_range[0], self.u_range[1], 300)
            X, Y = np.meshgrid(x, y)
            x = X.ravel()
            y = Y.ravel()
            X = np.vstack([x, y]).T

            x1_obs = np.array([[res["params"][u[0]]] for res in self.optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"][u[1]]] for res in self.optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in self.optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs))

            self.optimizer._gp.fit(obs, y_obs)
            mu, sigma = self.optimizer._gp.predict(X, eval)
            self.optimal = self.get_optimal(X, mu)

            data4plot = {'mu': mu,
                         'sigma': sigma,
                         'obs': obs,
                         'x1_obs': x1_obs,
                         'x2_obs': x2_obs,
                         'x': x,
                         'y': y,
                         'X': X}

            return data4plot

        if dim == 3:
            x = y = z = np.linspace(self.u_range[0], self.u_range[1], 100)
            X, Y, Z = np.meshgrid(x, y, z)
            x = X.ravel()
            y = Y.ravel()
            z = Z.ravel()
            X = np.vstack([x, y, z]).T

            x1_obs = np.array([[res["params"][u[0]]] for res in self.optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"][u[1]]] for res in self.optimizer.res])[:plot_size]
            x3_obs = np.array([[res["params"][u[2]]] for res in self.optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in self.optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs, x3_obs))

            self.optimizer._gp.fit(obs, y_obs)
            mu, sigma = self.optimizer._gp.predict(X, eval)
            self.optimal = self.get_optimal(X, mu)

            return mu, sigma

    def plot(self, ratio=1):
        u = list(self.optimizer.res[0]["params"].keys())
        dim = len(u)
        plot_size = len(self.optimizer.res) * ratio
        opt_eles = [ele for i, ele in enumerate(self.elements) if self.opt_u_index[i]]

        if dim == 1:
            d = self.predict()
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1)
            axis = plt.subplot(gs[0])
            acq = plt.subplot(gs[1])
            axis.plot(d['x_obs'].flatten(), d['y_obs'], 'D', markersize=8, label=u'Observations', color='r')
            axis.plot(d['x'], d['mu'], '--', color='k', label='Prediction')
            axis.fill(np.concatenate([d['x'], d['x'][::-1]]),
                      np.concatenate([d['mu'] - 1.9600 * d['sigma'], (d['mu'] + 1.9600 * d['sigma'])[::-1]]),
                      alpha=.6, fc='c', ec='None', label='95% confidence interval')

            axis.set_xlim(self.u_range)
            axis.set_ylim((None, None))
            axis.set_ylabel('f(x)')

            utility = self.utility_function.utility(d['x'], self.optimizer._gp, 0)
            acq.plot(d['x'], utility, label='Acquisition Function', color='purple')
            acq.plot(d['x'][np.argmax(utility)], np.max(utility), '*', markersize=15,
                     label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
            acq.set_xlim(self.u_range)
            acq.set_ylim((np.min(utility) - 0.5, np.max(utility) + 0.5))
            acq.set_ylabel('Acquisition')
            acq.set_xlabel('U (eV)')
            axis.legend(loc=4, borderaxespad=0.)
            acq.legend(loc=4, borderaxespad=0.)

            plt.savefig('1D_kappa_%s_a1_%s_a2_%s.png' % (self.kappa, self.a1, self.a2), dpi=400)

        if dim == 2:
            d = self.predict()
            fig, axis = plt.subplots(1, 2, figsize=(15, 5))
            plt.subplots_adjust(wspace=0.2)

            axis[0].plot(d['x1_obs'], d['x2_obs'], 'D', markersize=4, color='k', label='Observations')
            axis[0].set_title('Gaussian Process Predicted Mean', pad=10)
            im1 = axis[0].hexbin(d['x'], d['y'], C=d['mu'], cmap=cm.jet, bins=None)
            axis[0].axis([d['x'].min(), d['x'].max(), d['y'].min(), d['y'].max()])
            axis[0].set_xlabel(r'U_%s (eV)' % opt_eles[0], labelpad=5)
            axis[0].set_ylabel(r'U_%s (eV)' % opt_eles[1], labelpad=10, va='center')
            cbar1 = plt.colorbar(im1, ax=axis[0])

            utility = self.utility_function.utility(d['X'], self.optimizer._gp, self.optimizer.max)
            axis[1].plot(d['x1_obs'], d['x2_obs'], 'D', markersize=4, color='k', label='Observations')
            axis[1].set_title('Acquisition Function', pad=10)
            axis[1].set_xlabel(r'U_%s (eV)' % opt_eles[0], labelpad=5)
            axis[1].set_ylabel(r'U_%s (eV)' % opt_eles[1], labelpad=10, va='center')
            im2 = axis[1].hexbin(d['x'], d['y'], C=utility, cmap=cm.jet, bins=None)
            axis[1].axis([d['x'].min(), d['x'].max(), d['y'].min(), d['y'].max()])
            cbar2 = plt.colorbar(im2, ax=axis[1])

            plt.savefig('2D_kappa_%s_a1_%s_a2_%s.png' % (self.kappa, self.a1, self.a2), dpi=400)


class BayesOptDftu(PlotBO):
    def __init__(self,
                 path,
                 config_file_name,
                 opt_u_index=(1, 1, 0),
                 u_range=(0, 10),
                 a1=0.25,
                 a2=0.75,
                 delta_mag_weight=0.0,
                 kappa=2.5,
                 elements=['ele1', 'ele2', 'ele3'],
                 plot=False):
        self.path = path
        self.config_file_name = config_file_name

        if plot:
            upath = "./u_kappa_%s_a1_%s_a2_%s.txt" % (kappa, a1, a2)
        if not plot:
            upath = './u_tmp.txt'
        super().__init__(upath, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa, elements)

    def get_gap_baseline(self):
        return self.gap_baseline

    def bo(self):
        next_point_to_probe = self.optimizer.suggest(self.utility_function)

        U = list(next_point_to_probe.values())

        self.update_u_config(U)

        return self.target

    def update_u_config(self, U):
        os.chdir(self.path)
        with open(self.config_file_name, 'r') as f:
            data = json.load(f)
            u_pointer = 0
            for i, opt_u_on in enumerate(self.opt_u_index):
                if opt_u_on:
                    data["pbe"]["ldau_luj"][self.elements[i]]["U"] = U[u_pointer]
                    u_pointer += 1
            f.close()
        with open(self.config_file_name, 'w') as f:
            json.dump(data, f, indent=4)
            f.close()
