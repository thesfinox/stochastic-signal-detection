# -*- coding: utf-8 -*-
"""
Stochastic Signal Detection (SSD)

Perform a simulation of the evolution of a stochastic field.
"""
import sys
import argparse
import numpy as np
from pde import MemoryStorage, PDEBase, ScalarField, CartesianGrid, plot_kymograph, PlotTracker, PDE
from pde.grids.boundaries.axes import BoundariesData
from pde.pdes.base import expr_prod
from pde.visualization.movies import movie
from ssd.base import BaseDistribution
from ssd.distributions import TranslatedInverseMarchenkoPastur
from scipy import constants as C

__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'Perform a simulation of the evolution of a stochastic field'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'


class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self,
                 dist: BaseDistribution,
                 k2: float,
                 noise: float = 0.0,
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
        bc : BoundariesData
            Boundary conditions (default is 'auto_periodic_neumann')
        """
        super().__init__(noise=noise)
        self.dist = dist
        self.k2 = k2
        self.bc = bc

    @property
    def expression(self) -> str:
        """Return the expression for the PDE"""
        block = "χ * d_dχ(U')"
        mu2 = f"U' + 2 * {block}"

        part_1 = "-dim(U') * U'"
        part_2 = f"dim(χ) * {block}"

        num = "3 * d_dχ(U') + 2 * χ * d_dχ(d_dχ(U'))"
        den = f"(1 + ({mu2})**2)**2"

        return f"{part_1} + {part_2} - 2 * ({num}) / ({den})"

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:

        Chi = state.grid.axes_coords[0]
        Uprime = state

        integral = self.dist.integrate(0, self.k2)[0]
        grad = self.dist.grad(self.k2)
        value = self.dist(self.k2)

        dimUprime = integral / self.k2 / value

        dimChi_1 = integral / self.k2 / value
        dimChi_2 = grad * integral / np.sqrt(self.k2) / value**2
        dimChi = 2 - 2 * dimChi_1 - dimChi_2 / 2

        block = Chi * Uprime.gradient(bc=self.bc)[0]
        mu2 = Uprime + 2*block

        part_1 = -dimUprime * Uprime
        part_2 = dimChi * block

        num_1 = Uprime.gradient(bc=self.bc)[0]
        num_2 = Chi * Uprime.gradient(bc=self.bc)[0].gradient(bc=self.bc)[0]
        num = 3 * num_1 + 2 * num_2
        den = (1 + mu2**2)**2

        result = part_1 + part_2 - 2*num/den
        result.label = 'stochastic evolution'

        return result


def main(args):

    # Define the grid
    grid = CartesianGrid([[-C.pi, C.pi]], [32], periodic=True)
    # state = ScalarField.random_normal(grid, mean=0.0, std=10)
    state = ScalarField.from_expression(grid, 'x**2')
    bc = 'periodic'

    # Initialize a storage
    storage = MemoryStorage()
    trackers = [
        'progress',
        'steady_state',
        storage.tracker(interval=1),
        PlotTracker(show=True)
    ]

    # Define the PDE and solve
    dist = TranslatedInverseMarchenkoPastur(L=0.25, sigma=1.5)
    eq = SSD(dist=dist, k2=1.1, noise=0.0, bc=bc)
    eq.solve(state, t_range=100, dt=0.01, tracker=trackers)

    movie(storage, 'ssd.mp4', show_time=True, movie_args={'framerate': 10})

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('-v',
                        dest='verb',
                        action='count',
                        default=0,
                        help='Verbosity level')
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
