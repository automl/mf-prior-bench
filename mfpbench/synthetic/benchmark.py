from __future__ import annotations

from abc import ABC
from typing import Any

from pathlib import Path

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from mfpbench.benchmark import Benchmark
from mfpbench.download import DATAROOT


# Empirically obtained hyperparameters for the hartmann function - have not tested ranking correlation yet
# The function flattens with increasing fidelity bias. Along with increasing noise, that obviously makes
# one config harder to distinguish from another.
# Moreover, this works with any number of fidelitiy levels
DEFAULT_FIDELITY_QUALITY = {
    'terrible': {'fidelity_bias': 4.0, 'fidelity_noise': 0.8},
    'bad': {'fidelity_bias': 2.0, 'fidelity_noise': 0.4},
    'moderate': {'fidelity_bias': 1.0, 'fidelity_noise':  0.2},
    'good': {'fidelity_bias': 0.5, 'fidelity_noise': 0.1}
}

# @Eddie: 5 fidelities (1, 2, 3, 4, 5) - see self.z_min, self.z_max = 1, num_fidelities would make it have the desired behaviour
# DEFAULT_FIDELITY_QUALITY determines how good the fidelity correlations are, and do not have anything to do with the number of fidelities.

class MultiFidelityHartmann3:

    def __init__(self, num_fidelities, fidelity_quality=None, fidelity_bias=0.5, fidelity_noise=0.1):
        """A multifidelity version of the Hartmann3 function. Carried a bias term, which flattens the objective, and a noise term. The impact of both
        terms decrease with increasing fidelity, meaning that (num_fidelities) is the best fidelity. This fidelity level also constitutes a noiseless,
        true evaluation of the Hartmann function.

        Args:
            num_fidelities ([list]): The number of fidelities to use.
            fidelity_quality ([type], optional): [description]. General quality of the fidelity. Will update with ranking correlation.
            fidelity_bias (float, optional): The amount of bias introduced, realized as a flattening of the objective. Defaults to 0.5 (arbitrary number).
            fidelity_noise (float, optional): The amount of noise introduced, decreasing linearly (in st.dev.) with the fidelity. Defaults to 0.1.
        """
        self.z_min, self.z_max = 1, num_fidelities

        if fidelity_quality is not None:
            try:
                self.fidelity_bias = DEFAULT_FIDELITY_QUALITY[fidelity_quality]['fidelity_bias']
                self.fidelity_noise = DEFAULT_FIDELITY_QUALITY[fidelity_quality]['fidelity_noise']
            except KeyError:
                (f'No such default fidelity setting. Available options are {DEFAULT_FIDELITY_QUALITY.keys()}. Running default (good).')

                self.fidelity_bias = DEFAULT_FIDELITY_QUALITY['good']['fidelity_bias']
                self.fidelity_noise = DEFAULT_FIDELITY_QUALITY['good']['fidelity_noise']

        else:
            self.fidelity_bias = fidelity_bias
            self.fidelity_noise = fidelity_noise

        self.optimum = (0.114614, 0.555649, 0.852547)
        self.dim = 3

    def __call__(self, z, X_0, X_1, X_2):
        norm_z = (z - self.z_min) / (self.z_max - self.z_min)
        # Highest fidelity (1) accounts for the regular Hartmann
        X = np.array([X_0, X_1, X_2]).reshape(1, -1)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])

        alpha_prime = alpha - self.fidelity_bias * np.power(1 - norm_z, 2)
        A = np.array([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ])
        P = np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ])

        inner_sum = np.sum(
            A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))
        H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        # and add some noise
        noise = np.random.normal(size=H.size) * \
            self.fidelity_noise * (1 - norm_z)
        return (H + noise)[0]


class MultiFidelityHartmann6:

    def __init__(self, num_fidelities, fidelity_quality=None, fidelity_bias=0.5, fidelity_noise=0.1):
        """A multifidelity version of the Hartmann6 function. Carried a bias term, which flattens the objective, and a noise term. The impact of both
        terms decrease with increasing fidelity, meaning that (num_fidelities) is the best fidelity. This fidelity level also constitutes a noiseless,
        true evaluation of the Hartmann function.

        Args:
            num_fidelities ([list]): The number of fidelities to use.
            fidelity_quality ([type], optional): [description]. General quality of the fidelity. Will update with ranking correlation.
            fidelity_bias (float, optional): The amount of bias introduced, realized as a flattening of the objective. Defaults to 0.5 (arbitrary number).
            fidelity_noise (float, optional): The amount of noise introduced, decreasing linearly (in st.dev.) with the fidelity. Defaults to 0.1.
        """
        self.z_min, self.z_max = 1, num_fidelities

        if fidelity_quality is not None:
            try:
                self.fidelity_bias = DEFAULT_FIDELITY_QUALITY[fidelity_quality]['fidelity_bias']
                self.fidelity_noise = DEFAULT_FIDELITY_QUALITY[fidelity_quality]['fidelity_noise']
            except KeyError:
                (f'No such default fidelity setting. Available options are {DEFAULT_FIDELITY_QUALITY.keys()}. Running default (good).')

                self.fidelity_bias = DEFAULT_FIDELITY_QUALITY['good']['fidelity_bias']
                self.fidelity_noise = DEFAULT_FIDELITY_QUALITY['good']['fidelity_noise']

        else:
            self.fidelity_bias = fidelity_bias
            self.fidelity_noise = fidelity_noise

        self.optimum = (0.20169, 0.150011, 0.476874,
                        0.275332, 0.311652, 0.6573)
        self.dim = 6

    def __call__(self, z, X_0, X_1, X_2, X_3, X_4, X_5):
        norm_z = (z - self.z_min) / (self.z_max - self.z_min)
        # Highest fidelity (1) accounts for the regular Hartmann
        X = np.array([X_0, X_1, X_2,  X_3, X_4, X_5]).reshape(1, -1)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        alpha_prime = alpha - self.fidelity_bias * np.power(1 - norm_z, 2)
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ])
        P = np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ])

        inner_sum = np.sum(
            A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))
        H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        # and add some noise
        noise = np.random.normal(size=H.size) * \
            self.fidelity_noise * (1 - norm_z)
        return (H + noise)[0]
