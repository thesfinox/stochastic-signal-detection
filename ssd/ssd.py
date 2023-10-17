# -*- coding: utf-8 -*-
"""
SSD - PDE

The PDE module contains the class for the differential equation encoding the behaviour of the renormalization group.
"""
from pde import PDEBase, ScalarField
from pde.grids.boundaries.axes import BoundariesData

from .base import BaseDistribution


class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self,
                 dist: BaseDistribution,
                 noise: float = 0.0,
                 epsilon: float = 1.e-9,
                 bc: BoundariesData = 'auto_periodic_neumann'):
        """
        Parameters
        ----------
        dist : BaseDistribution
            Distribution of the signal
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
        self.epsilon = epsilon
        self.bc = bc

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
        k = t
        k2 = t**2
        x = state.grid.axes_coords[0]
        U = state

        # Compute constants
        _I = self.dist.integrate(0, k, moment=1, power=2)[0]
        _dimU = 2 * _I / (k2 + self.epsilon) / (self.dist(k2) + self.epsilon)
        _P = k2 * self.dist.grad(k2) / (self.dist(k2) + self.epsilon)
        _dimChi = 2 - _dimU * (_P+2)

        # Compute the derivatives
        grad = U.gradient(bc=self.bc)[0]
        grad2 = grad.gradient(bc=self.bc)[0]

        # Compute auxiliary quantities
        block = x * grad
        block2 = x * grad2

        # Compute the adimensional mass
        mu2 = U + 2*block

        # Sum the components (first term)
        Q1 = -_dimU * U + _dimChi*block

        # Second term
        num = 3*grad + 2*block2
        den = (1 + mu2**2)**2
        Q2 = -2 * num / (den + self.epsilon)

        # Compute the final result
        result = k * self.dist(k2) * (Q1+Q2) / (_I + self.epsilon)
        result.label = 'SSD'

        return result
