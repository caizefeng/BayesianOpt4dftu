import json
from typing import Optional, Tuple, Any, List, Dict

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt

from BayesOpt4dftu.configuration import Config
from BayesOpt4dftu.io_utils import SuppressPrints
from BayesOpt4dftu.logging import BoLoggerGenerator


class OptimizerGenerator:
    def __init__(self, utxt_path, column_names, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa):
        self._utility_function = UtilityFunction(kind="ucb", kappa=kappa, xi=0)
        self._opt_u_index: List[float] = opt_u_index
        self._u_range: List[float] = u_range
        self._a1: float = a1
        self._a2: float = a2
        self._delta_mag_weight: float = delta_mag_weight
        self._kappa: float = kappa

        self._utxt_path: str = utxt_path
        self._column_names: Dict[Any, str] = column_names
        self._n_obs: Optional[int] = None
        self._data: Optional[pd.DataFrame] = None

    def loss(self, delta_gap=0.0, delta_band=0.0, delta_mag=0.0, alpha_1=0.5, alpha_2=0.5, delta_mag_weight=0.0):
        return -alpha_1 * delta_gap ** 2 - alpha_2 * delta_band ** 2 - delta_mag_weight * delta_mag ** 2

    def set_bounds(self):
        # Set up the indices of variables that are going to be optimized.
        variables_string = ['u_' + str(i) for i, o in enumerate(self._opt_u_index) if o]

        # Set up the U ranges for each variable.
        pbounds = {}
        for variable in variables_string:
            pbounds[variable] = self._u_range
        return pbounds

    def set_data(self):
        data = pd.read_csv(self._utxt_path, header=0, delimiter=r"\s+", engine="python")
        self._n_obs, _ = data.shape
        self._data = data

    def get_optimizer(self):
        self.set_data()
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

        for i in range(self._n_obs):
            values = list()
            for j in opt_index:
                values.append(self._data.iloc[i][j])
            params = {}
            for (value, variable) in zip(values, v_strings):
                params[variable] = value

            if self._delta_mag_weight:
                target = self.loss(delta_gap=self._data.iloc[i][self._column_names['delta_gap']],
                                   delta_band=self._data.iloc[i][self._column_names['delta_band']],
                                   delta_mag=self._data.iloc[i][self._column_names['delta_mag']],
                                   alpha_1=self._a1,
                                   alpha_2=self._a2,
                                   delta_mag_weight=self._delta_mag_weight)
            else:
                target = self.loss(delta_gap=self._data.iloc[i][self._column_names['delta_gap']],
                                   delta_band=self._data.iloc[i][self._column_names['delta_band']],
                                   alpha_1=self._a1,
                                   alpha_2=self._a2)

            # Suppress non-unique data point registration messages
            with SuppressPrints():
                optimizer.register(
                    params=params,
                    target=target,
                )

        return optimizer, target

    @staticmethod
    def retrieve_optimal(x, mu):
        best_obj = mu.max()
        best_index = np.where(mu == mu.max())[0][0]
        best_x = x[best_index]
        optimal = (best_x, best_obj)
        return optimal


class BoStepExecutor(OptimizerGenerator):
    _logger = BoLoggerGenerator.get_logger("BoStepExecutor")

    def __init__(self, utxt_path, column_names, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa, elements):
        super().__init__(utxt_path, column_names, opt_u_index, u_range, a1, a2, delta_mag_weight, kappa)
        self._elements: List[str] = elements
        self._optimizer: Optional[BayesianOptimization] = None
        self._target: Optional[float] = None
        self._optimal: Optional[Tuple[Any, Any]] = None

    def get_optimal(self):
        optimal_u, optimal_obj = self._optimal
        self._logger.info(f"Optimal U value: {optimal_u}")
        self._logger.info(f"Optimal objective function: {optimal_obj}")
        return self._optimal

    def advance_step(self):
        self._optimizer, self._target = self.get_optimizer()
        return self._optimizer.suggest(self._utility_function), self._target

    def predict(self, ratio=1):
        u = list(self._optimizer.res[0]["params"].keys())
        dim = len(u)
        plot_size = len(self._optimizer.res) * ratio
        if dim == 1:
            x = np.linspace(self._u_range[0], self._u_range[1], 10000).reshape(-1, 1)
            x_obs = np.array([res["params"][u[0]] for res in self._optimizer.res]).reshape(-1, 1)[:plot_size]
            y_obs = np.array([res["target"] for res in self._optimizer.res])[:plot_size]

            self._optimizer._gp.fit(x_obs, y_obs)
            mu, sigma = self._optimizer._gp.predict(x, return_std=True)
            self._optimal = OptimizerGenerator.retrieve_optimal(x, mu)

            data4plot = {'mu': mu,
                         'sigma': sigma,
                         'x': x,
                         'x_obs': x_obs,
                         'y_obs': y_obs}

            return data4plot

        if dim == 2:
            x = y = np.linspace(self._u_range[0], self._u_range[1], 300)
            x_mesh, y_mesh = np.meshgrid(x, y)
            x = x_mesh.ravel()
            y = y_mesh.ravel()
            x_mesh = np.vstack([x, y]).T

            x1_obs = np.array([[res["params"][u[0]]] for res in self._optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"][u[1]]] for res in self._optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in self._optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs))

            self._optimizer._gp.fit(obs, y_obs)
            mu, sigma = self._optimizer._gp.predict(x_mesh, eval)
            self._optimal = OptimizerGenerator.retrieve_optimal(x_mesh, mu)

            data4plot = {'mu': mu,
                         'sigma': sigma,
                         'obs': obs,
                         'x1_obs': x1_obs,
                         'x2_obs': x2_obs,
                         'x': x,
                         'y': y,
                         'X': x_mesh}

            return data4plot

        if dim == 3:
            x = y = z = np.linspace(self._u_range[0], self._u_range[1], 100)
            x_mesh, y_mesh, z_mesh = np.meshgrid(x, y, z)
            x = x_mesh.ravel()
            y = y_mesh.ravel()
            z = z_mesh.ravel()
            x_mesh = np.vstack([x, y, z]).T

            x1_obs = np.array([[res["params"][u[0]]] for res in self._optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"][u[1]]] for res in self._optimizer.res])[:plot_size]
            x3_obs = np.array([[res["params"][u[2]]] for res in self._optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in self._optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs, x3_obs))

            self._optimizer._gp.fit(obs, y_obs)
            mu, sigma = self._optimizer._gp.predict(x_mesh, eval)
            self._optimal = OptimizerGenerator.retrieve_optimal(x_mesh, mu)

            return mu, sigma

    def plot(self, ratio=1):
        u = list(self._optimizer.res[0]["params"].keys())
        dim = len(u)
        plot_size = len(self._optimizer.res) * ratio
        opt_eles = [ele for i, ele in enumerate(self._elements) if self._opt_u_index[i]]

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

            axis.set_xlim(self._u_range)
            axis.set_ylim((None, None))
            axis.set_ylabel('f(x)')

            utility = self._utility_function.utility(d['x'], self._optimizer._gp, 0)
            acq.plot(d['x'], utility, label='Acquisition Function', color='purple')
            acq.plot(d['x'][np.argmax(utility)], np.max(utility), '*', markersize=15,
                     label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
            acq.set_xlim(self._u_range)
            acq.set_ylim((np.min(utility) - 0.5, np.max(utility) + 0.5))
            acq.set_ylabel('Acquisition')
            acq.set_xlabel('U (eV)')
            axis.legend(loc=4, borderaxespad=0.)
            acq.legend(loc=4, borderaxespad=0.)

            plt.savefig('1D_kappa_%s_a1_%s_a2_%s.png' % (self._kappa, self._a1, self._a2), dpi=400)

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

            utility = self._utility_function.utility(d['X'], self._optimizer._gp, self._optimizer.max)
            axis[1].plot(d['x1_obs'], d['x2_obs'], 'D', markersize=4, color='k', label='Observations')
            axis[1].set_title('Acquisition Function', pad=10)
            axis[1].set_xlabel(r'U_%s (eV)' % opt_eles[0], labelpad=5)
            axis[1].set_ylabel(r'U_%s (eV)' % opt_eles[1], labelpad=10, va='center')
            im2 = axis[1].hexbin(d['x'], d['y'], C=utility, cmap=cm.jet, bins=None)
            axis[1].axis([d['x'].min(), d['x'].max(), d['y'].min(), d['y'].max()])
            cbar2 = plt.colorbar(im2, ax=axis[1])

            plt.savefig('2D_kappa_%s_a1_%s_a2_%s.png' % (self._kappa, self._a1, self._a2), dpi=400)


class BoDftuIterator(BoStepExecutor):
    _logger = BoLoggerGenerator.get_logger("BoDftuIterator")
    _config: Config = None

    @classmethod
    def init_config(cls, config: Config):
        if cls._config is None:
            cls._config = config

    def __init__(self, plot=False):
        if plot:
            upath = self._config.u_path
        else:
            upath = self._config.tmp_u_path
        super().__init__(upath, self._config.column_names,
                         self._config.which_u, self._config.urange,
                         self._config.a1, self._config.a2, self._config.delta_mag_weight, self._config.k,
                         self._config.elements)
        self._logger.info("Bayesian Optimization begins.")
        self._i_step: int = 0
        self._obj_current: Optional[float] = None
        self._obj_next: Optional[float] = None
        self._exit_converged: bool = False

    def next(self):
        self._obj_current = self._obj_next
        next_point_to_probe, self._obj_next = self.advance_step()
        u_next = list(next_point_to_probe.values())
        self.update_u_config(u_next)
        self._i_step += 1

    def update_u_config(self, u_new):
        with open(self._config.tmp_config_path, 'r') as f:
            data = json.load(f)
            u_pointer = 0
            for i, opt_u_on in enumerate(self._opt_u_index):
                if opt_u_on:
                    data["pbe"]["ldau_luj"][self._config.elements[i]]["U"] = u_new[u_pointer]
                    u_pointer += 1
        with open(self._config.tmp_config_path, 'w') as f:
            json.dump(data, f, indent=4)

    def converge(self):
        if self._obj_current is not None:  # Can't be the 1st step
            is_converged = (self._config.threshold != 0
                            and abs(self._obj_next - self._obj_current) <= self._config.threshold)
        else:
            is_converged = False

        if is_converged:
            self._logger.info(f"Convergence reached at iteration {self._i_step + 1}, exiting.")
            self._exit_converged = True
        return is_converged

    def finalize(self):
        if not self._exit_converged:
            self._logger.info("Bayesian Optimization exited without reaching convergence: "
                              "maximum number of steps reached.")

        self._logger.info("Bayesian Optimization finished.")
