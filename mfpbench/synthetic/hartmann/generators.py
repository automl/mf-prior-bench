"""This module extends Hartmann functions to incorporate fidelities"""
from __future__ import annotations

from abc import ABC

import numpy as np


class MFHartmannGenerator(ABC):
    """A multifidelity version of the Hartmann3 function.

    Carried a bias term, which flattens the objective, and a noise term.
    The impact of both terms decrease with increasing fidelity, meaning that
    ``num_fidelities`` is the best fidelity. This fidelity level also constitutes
    a noiseless, true evaluation of the Hartmann function.
    """

    optimum: tuple[float, ...]

    def __init__(self, n_fidelities: int, fidelity_bias: float, fidelity_noise: float):
        """
        Parameters
        ----------
        n_fidelities: int
            The fidelity at which the function is evalated

        fidelity_bias: float
            Amount of bias, realized as a flattening of the objective.

        fidelity_noise: float
            Amount of noise, decreasing linearly (in st.dev.) with fidelity.
        """
        self.z_min, self.z_max = (1, n_fidelities)
        self.bias = fidelity_bias
        self.noise = fidelity_noise

    def __call__(self, z: int, Xs: tuple[float, ...]) -> float:
        """
        Parameters
        ----------
        z : int
            The fidelity at which to query

        Xs : tuple[float, ...]
            The Xs as input to the function, in the correct order

        Returns
        -------
        float
        """
        ...

    @property
    def dims(self) -> int:
        """The dimension of the input"""
        return len(self.optimum)


class MFHartmann3(MFHartmannGenerator):
    optimum = (0.114614, 0.555649, 0.852547)

    def __call__(self, z: int, Xs: tuple[float, ...]) -> float:
        """
        Parameters
        ----------
        z: int
            The fidelity

        X_0, X_1, X_2: float
            Parameters of the function.

        Returns
        -------
        float
            The function value
        """
        assert self.z_min <= z <= self.z_max
        assert len(Xs) == self.dims
        X_0, X_1, X_2 = Xs

        norm_z = (z - self.z_min) / (self.z_max - self.z_min)
        # Highest fidelity (1) accounts for the regular Hartmann
        X = np.array([X_0, X_1, X_2]).reshape(1, -1)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])

        alpha_prime = alpha - self.bias * np.power(1 - norm_z, 2)
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )

        inner_sum = np.sum(A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))

        # TODO: Didn't seem used
        # H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        # and add some noise
        noise = np.random.normal(size=H.size) * self.noise * (1 - norm_z)
        return (H + noise)[0]


class MFHartmann6(MFHartmannGenerator):

    optimum = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    def __call__(self, z: int, Xs: tuple[float, ...]) -> float:
        """
        Parameters
        ----------
        z: int
            The fidelity it's evaluated at

        X_0, ..., X_5: float
            Parameters of the function

        Returns
        -------
        float
            The function value
        """
        assert self.z_min <= z <= self.z_max
        assert len(Xs) == self.dims
        X_0, X_1, X_2, X_3, X_4, X_5 = Xs

        norm_z = (z - self.z_min) / (self.z_max - self.z_min)
        # Highest fidelity (1) accounts for the regular Hartmann
        X = np.array([X_0, X_1, X_2, X_3, X_4, X_5]).reshape(1, -1)
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        alpha_prime = alpha - self.bias * np.power(1 - norm_z, 2)
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        inner_sum = np.sum(A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))

        # TODO: Doesn't seem to be used?
        # H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        # and add some noise
        noise = np.random.normal(size=H.size) * self.noise * (1 - norm_z)
        return (H + noise)[0]
