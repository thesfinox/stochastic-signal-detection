# -*- coding: utf-8 -*-
"""
SSD - PDE

The PDE module contains the class for the differential equation encoding the behaviour of the renormalization group.
"""
from pde import PDEBase, ScalarField
from .base import BaseDistribution
from pde.grids.boundaries.axes import BoundariesData


class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self,
                 dist: BaseDistribution,
                 k2: float,
                 noise: float = 0.0,
                 epsilon: float = 1.e-9,
                 bc: BoundariesData = 'auto_periodic_neumann'):
        """
        Parameters
        ----------
        dist : BaseDistribution
            Distribution of the signal
        k2 : float
            Renormalization scale (``k^2``)
        noise : float
            Noise intensity (default is 0.0)
        epsilon : float
            Small number to avoid division by zero (default is 1.e-9)
        bc : BoundariesData
            Boundary conditions (default is 'auto_periodic_neumann')
        """
        super().__init__(noise=noise)
        self.dist = dist
        self.k2 = k2
        self.epsilon = epsilon
        self.bc = bc

    @property
    def expression(self) -> str:
        """Return the expression for the PDE"""
        return r"\dot{\overline{\mathcal{U_k^\prime}}}[\overline{\chi}] = - \mathrm{dim}_{\tau}(\overline{\mathcal{U_k^\prime}})\, \overline{\mathcal{U_k^\prime}}[\overline{\chi}] + \mathrm{dim}_{\tau}(\chi)\, \overline{\chi}\, \overline{\mathcal{U_k^{\prime\prime}}}[\overline{\chi}] -2 \frac{3\, \overline{\mathcal{U_k^{\prime\prime}}}[\overline{\chi}] + 2\, \overline{\chi}\, \overline{\mathcal{U_k^{\prime\prime\prime}}}[\overline{\chi}]}{(1 + \overline{\mu}^2)^2}"

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:

        # Get the coordinates
        x = state.grid.axes_coords[0]
        U = state

        # Compute the main objects
        I = self.dist.integrate(0, self.k2)[0]

        # Compute the dimensions
        dimU = I / self.k2 / self.dist(self.k2)

        P = self.k2 * self.dist.grad(self.k2) / self.dist(self.k2)
        dimChi = 2 - dimU * (P+2)

        # Compute the derivatives
        grad = U.gradient(bc=self.bc)[0]
        grad2 = grad.gradient(bc=self.bc)[0]
        block = x * grad
        block2 = x * grad2

        # Compute the adimensional mass
        mu2 = U + 2*block

        # Sum the components
        Q1 = -dimU * U + dimChi*block

        num = 3*grad + 2*block2
        den = (1 + mu2**2)**2
        Q2 = -2 * num / (den + self.epsilon)

        result = Q1 + Q2
        result.label = 'SSD'

        return result
