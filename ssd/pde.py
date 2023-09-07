# -*- coding: utf-8 -*-
"""
SSD - PDE

The PDE module contains the class for the differential equation encoding the behaviour of the renormalization group.
"""
from pde import PDEBase, ScalarField
from .base import BaseDistribution
from pde.grids.boundaries.axes import BoundariesData
import numpy as np


class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self,
                 dist: BaseDistribution,
                 k2max: float,
                 k2min: float = 0.0,
                 noise: float = 0.0,
                 epsilon: float = 1.e-9,
                 bc: BoundariesData = 'auto_periodic_neumann'):
        """
        Parameters
        ----------
        dist : BaseDistribution
            Distribution of the signal
        k2max : float
            Energy scale (``k^2_{max}``)
        k2min : float
            Lower bound of the energy scale (default is 0.0)
        noise : float
            Noise intensity (default is 0.0)
        epsilon : float
            Small number to avoid division by zero (default is 1.e-9)
        bc : BoundariesData
            Boundary conditions (default is 'auto_periodic_neumann')

        Raises
        ------
        ValueError
            If the lower bound of the energy scale is negative
        """
        super().__init__(noise=noise)
        self.dist = dist
        self.k2max = k2max
        self.k2min = k2min
        if self.k2min < 0:
            raise ValueError(
                f'The lower bound of the energy scale must be positive. Found k2min = {self.k2min} < 0 instead.'
            )
        self.epsilon = epsilon
        self.bc = bc

        # Compute constants
        self._I = self.dist.integrate(self.k2min, self.k2max)[0]
        self._dimU = self._I / self.k2max / self.dist(self.k2max)
        P = self.k2max * self.dist.grad(self.k2max) / self.dist(self.k2max)
        self._dimChi = 2 - self._dimU * (P+2)

    @property
    def expression(self) -> str:
        """
        Return the expression for the PDE.

        .. math::

            \\dot{\\bar{\\mathcal{U_k^\\prime}}}[\\bar{\\chi}] = - \\mathrm{dim}_{\\tau}(\\bar{\\mathcal{U_k^\\prime}})\\, \\bar{\\mathcal{U_k^\\prime}}[\\bar{\\chi}] + \\mathrm{dim}_{\\tau}(\\chi)\\, \\bar{\\chi}\\, \\bar{\\mathcal{U_k^{\\prime\\prime}}}[\\bar{\\chi}] -2 \\frac{3\\, \\bar{\\mathcal{U_k^{\\prime\\prime}}}[\\bar{\\chi}] + 2\\, \\bar{\\chi}\\, \\bar{\\mathcal{U_k^{\\prime\\prime\\prime}}}[\\bar{\\chi}]}{(1 + \\bar{\\mu}^2)^2}
        """
        return r"\dot{\bar{\mathcal{U_k^\prime}}}[\bar{\chi}] = - \mathrm{dim}_{\tau}(\bar{\mathcal{U_k^\prime}})\, \bar{\mathcal{U_k^\prime}}[\bar{\chi}] + \mathrm{dim}_{\tau}(\chi)\, \bar{\chi}\, \bar{\mathcal{U_k^{\prime\prime}}}[\bar{\chi}] -2 \frac{3\, \bar{\mathcal{U_k^{\prime\prime}}}[\bar{\chi}] + 2\, \bar{\chi}\, \bar{\mathcal{U_k^{\prime\prime\prime}}}[\bar{\chi}]}{(1 + \bar{\mu}^2)^2}"

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:
        """
        Compute the evolution rate of the differential equation.

        Parameters
        ----------
        state : ScalarField
            The state of the field
        t : float
            The current time (default is 0)

        Returns
        -------
        ScalarField
            The evolution rate of the field at the given time
        """
        # Get the coordinates
        x = state.grid.axes_coords[0]
        U = state

        # Compute the derivatives
        grad = U.gradient(bc=self.bc)[0]
        grad.data = np.nan_to_num(grad.data)
        grad2 = grad.gradient(bc=self.bc)[0]
        grad2.data = np.nan_to_num(grad2.data)

        # Compute auxiliary quantities
        block = x * grad
        block.data = np.nan_to_num(block.data)
        block2 = x * grad2
        block2.data = np.nan_to_num(block2.data)

        # Compute the adimensional mass
        mu2 = U + 2*block
        mu2.data = np.nan_to_num(mu2.data)

        # Sum the components (first term)
        Q1 = -self._dimU * U + self._dimChi * block
        Q1.data = np.nan_to_num(Q1.data)

        # Second term
        num = 3*grad + 2*block2
        num.data = np.nan_to_num(num.data)
        den = (1 + mu2**2)**2
        den.data = np.nan_to_num(den.data)
        Q2 = -2 * num / (den + self.epsilon)
        Q2.data = np.nan_to_num(Q2.data)

        # Compute the final result
        result = -(Q1 + Q2)
        result.data = np.nan_to_num(result.data)
        result.label = 'SSD'

        return result
