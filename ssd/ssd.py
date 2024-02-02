# -*- coding: utf-8 -*-
"""
Partial Differential Equations

The PDE module contains the class for the differential equation encoding the behaviour of the renormalization group.
"""
from pde import PDEBase, ScalarField
from pde.grids.boundaries.axes import BoundariesData
from scipy.integrate import simpson

from .base import BaseDistribution


class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self,
                 dist: BaseDistribution,
                 noise: float = 0.0,
                 epsilon: float = 1.e-9,
                 sign: int = 1,
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
        sign : int
            Sign of the evolution rate (default is +1)
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
        self.sign = float(int(sign > 0) - int(sign < 0))
        self.bc = bc

        # Collect the dimensions of the operators
        self.dimUp_ = []
        self.dimChi_ = []

        # Collect the operators
        self.kappa_bar_ = []
        self.mu_4_bar_ = []
        self.mu_6_bar_ = []
        self.mu_8_bar_ = []

        # Collect the fiels
        self.Up_ = []

        # Collect the energy scales
        self.k2_ = []

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
        k2 = t
        chi = state.grid.axes_coords[0]
        Up = state

        # Store the fields and the energy scales
        self.Up_.append(list(Up.data))
        self.k2_.append(k2)

        # Store the operators:
        #  - kappa_bar: first zero of Up
        #  - mu_4_bar: derivative of Up at kappa_bar
        #  - mu_6_bar: second derivative of Up at kappa_bar
        #  - mu_8_bar: third derivative of Up at kappa_bar
        grad = Up.gradient(bc=self.bc)[0]
        grad2 = grad.gradient(bc=self.bc)[0]
        grad3 = grad2.gradient(bc=self.bc)[0]

        zero = (Up.data[3:] >= 0.0).argmax() + 3
        kappa_bar = chi[zero]
        self.kappa_bar_.append(kappa_bar)
        mu_4_bar = round(grad.data[zero], 3)
        self.mu_4_bar_.append(mu_4_bar)
        mu_6_bar = round(grad2.data[zero] / 2.0, 3)
        self.mu_6_bar_.append(mu_6_bar)
        mu_8_bar = round(grad3.data[zero] / 6.0, 3)
        self.mu_8_bar_.append(mu_8_bar)

        # Compute constants
        _I = self.dist.integrate(0, k2)[0]
        _R = _I / (self.dist(k2) + self.epsilon)

        _dimUp = _R / (k2 + self.epsilon)
        _dimChi = 2 - _dimUp * (k2 * self.dist.grad(k2) + 2)

        # Store the dimensions
        self.dimUp_.append(_dimUp)
        self.dimChi_.append(_dimChi)

        # Compute the derivatives
        grad = Up.gradient(bc=self.bc)[0]
        grad2 = grad.gradient(bc=self.bc)[0]

        # Prefactor
        pref = 1 / (_R + self.epsilon)

        # Compute auxiliary quantities
        block = chi * grad
        block2 = chi * grad2

        # Compute the adimensional mass
        mu2 = Up + 2*block

        # Sum the components (first term)
        Q1 = -_dimUp * Up + _dimChi*block

        # Second term
        num = 3*grad + 2*block2
        den = (1 + mu2**2)**2
        Q2 = -2 * num / (den + self.epsilon)

        # Compute the final result
        result = self.sign * pref * (Q1+Q2)
        result.label = 'SSD'

        return result
