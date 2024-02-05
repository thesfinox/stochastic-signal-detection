# -*- coding: utf-8 -*-
"""
Partial Differential Equations

The PDE module contains the class for the differential equation encoding the behaviour of the renormalization group.
"""
from typing import Optional

import numpy as np
from pde import PDEBase, ScalarField
from pde.grids.boundaries.axes import BoundariesData

from .base.base import BaseDistribution


class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self,
                 dist: BaseDistribution,
                 noise: float = 0.0,
                 epsilon: float = 1.e-9,
                 sign: int = 1,
                 uv_scale: Optional[float] = None,
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
        uv_scale : float
            Energy scale chosen in the UV (default is None, e.g. if evolution starts in the IR)
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
        self.uv_scale = uv_scale
        self.bc = bc

        # Collect the dimensions of the operators
        self.dimdU_ = []
        self.dimChi_ = []

        # Collect the operators
        self.kappa_bar_ = []
        self.mu_4_bar_ = []
        self.mu_6_bar_ = []
        self.mu_8_bar_ = []

        # Collect the fiels
        self.dU_ = []

        # Collect the energy scales
        self.k2_ = []

    @property
    def k2(self) -> float:
        """Return the list of energy scales."""
        return np.array(self.k2_)

    @property
    def dU(self) -> float:
        """Return the list of values of the derivative of the field."""
        return np.array(self.dU_)

    @property
    def dimdU(self) -> float:
        """Return the list of values of the dimension of the derivative of the field."""
        return np.array(self.dimdU_)

    @property
    def dimChi(self) -> float:
        """Return the list of values of the dimension of the field :math:`\chi`."""
        return np.array(self.dimChi_)

    @property
    def kappa_bar(self) -> float:
        """Return the list of values of the first zero of the derivative of the field."""
        return np.array(self.kappa_bar_)

    @property
    def mu_4_bar(self) -> float:
        """Return the list of values of the derivative of the field at the first zero (quartic coupling)."""
        return np.array(self.mu_4_bar_)

    @property
    def mu_6_bar(self) -> float:
        """Return the list of values of the second derivative of the field at the first zero (sextic coupling)."""
        return np.array(self.mu_6_bar_)

    @property
    def mu_8_bar(self) -> float:
        """Return the list of values of the third derivative of the field at the first zero (octic coupling)."""
        return np.array(self.mu_8_bar_)

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
        if self.uv_scale is None:
            # When going from IR to UV, the energy scale is increasing towards
            # the UV scale.
            k2 = t
        else:
            # When going from UV to IR, the energy scale is decreasing towards
            # the IR scale.
            k2 = self.uv_scale - t
        chi = state.grid.axes_coords[0]
        dU = state

        # Store the fields and the energy scales
        self.dU_.append(dU.data.copy())
        self.k2_.append(k2)

        # Store the operators:
        #  - kappa_bar: first zero of Up
        #  - mu_4_bar: derivative of Up at kappa_bar
        #  - mu_6_bar: second derivative of Up at kappa_bar
        #  - mu_8_bar: third derivative of Up at kappa_bar
        grad = dU.gradient(bc=self.bc)[0]
        grad2 = dU.laplace(bc=self.bc)
        grad3 = grad.laplace(bc=self.bc)

        zero = (dU.data[3:] >= 0.0).argmax() + 3
        kappa_bar = chi[zero]
        self.kappa_bar_.append(kappa_bar)
        mu_4_bar = grad.data[zero]
        self.mu_4_bar_.append(mu_4_bar)
        mu_6_bar = grad2.data[zero] / 2.0
        self.mu_6_bar_.append(mu_6_bar)
        mu_8_bar = grad3.data[zero] / 6.0
        self.mu_8_bar_.append(mu_8_bar)

        # Compute the constants
        # N.B.: see the comments for the integration and the energy scales.
        if self.uv_scale is None:
            # When going from IR to UV, we use the distribution as it is. Hence,
            # there is no need to inverse the integration or the gradient.
            _I = self.dist.integrate(0, t)[0]
            _grad = self.dist.grad(t)
        else:
            # When going from UV to IR, we take the specular reflection of the
            # distribution. Hence, we need to reverse the integration to go
            # from the scale to the max energy scale. Moreove, the sign of the
            # gradient is also reversed.
            _I = self.dist.integrate(t, self.uv_scale)[0]
            _grad = -self.dist.grad(t)
        _dimdU = _I / (k2 * self.dist(t) + self.epsilon)
        _dimChi = 2 - _dimdU * (k2 * _grad / self.dist(t) + 2)
        self.dimdU_.append(_dimdU)
        self.dimChi_.append(_dimChi)

        # Compute the evolution
        C1 = -1 / (k2 + self.epsilon)  # dtau/dk2 * (-dimUp)
        C2_1 = 2 * self.dist(t) / (_I + self.epsilon)  # dtau/dk2 * dimChi (p.1)
        C2_2 = -_grad / (self.dist(t) + self.epsilon)  # dtau/dk2 * dimChi (p.2)
        C2_3 = -2 / (k2 + self.epsilon)  # dtau/dk2 * dimChi (p.3)
        C2 = C2_1 * C2_2 * C2_3
        C3 = -2 * self.dist(t) / (_I + self.epsilon)  # dtaudk2 * (-2)

        _mu2 = dU + 2*chi*grad
        P1 = C1 * dU
        P2 = C2 * chi * grad
        P3 = C3 * (3*dU + 2*chi*grad2) / (1 + _mu2)**2

        result = self.sign * (P1+P2+P3)
        result.label = 'SSD'

        return result
