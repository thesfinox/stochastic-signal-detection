# -*- coding: utf-8 -*-
"""
Base classes

Base abstract classes for the distributions
"""
from inspect import signature
from typing import Tuple

import numpy as np
from scipy.integrate import quad


class BaseDistribution:
    """A base class for probability density functions."""

    def __init__(self):
        self.name = self.__class__.__name__  # name of the distribution

    def __repr__(self):
        arguments = signature(self.__init__).parameters
        return f'{self.name}({", ".join([f"{k}" for k in arguments.keys()])})'

    def __str__(self):
        return self.__repr__()

    def __call__(self, x: float):
        return NotImplementedError(
            'All probability density functions must implement the __call__ method.'
        )

    @property
    def max(self):
        return None

    @property
    def min(self):
        return None

    def integrate(self,
                  a: float = -np.inf,
                  b: float = np.inf,
                  moment: float = 0) -> Tuple[float, float]:
        """
        Integrate the probability density function from a to b.

        Parameters
        ----------
        a : float, optional
            The lower bound of the integration (default is -inf)
        b : float, optional
            The upper bound of the integration (default is inf)
        moment : float, optional
            The moment to compute (default is 0)

        Returns
        -------
        Tuple[float, float]
            The value of the integral and the error
        """
        return quad(lambda x: x**moment * self(x), a, b)

    def grad(self, x: float, dx: float = 1e-9) -> float:
        """
        Compute the derivative of the probability density function at a given point.

        Parameters
        ----------
        x : float
            The point at which to evaluate the derivative
        dx : float, optional
            The step size (default is 1e-9)

        Returns
        -------
        float
            The value of the derivative at the given point
        """
        # Compute the derivative using the central difference method
        dy = self(x + dx) - self(x - dx)
        return dy / (2.0*dx)
