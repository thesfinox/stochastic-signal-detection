# -*- coding: utf-8 -*-
"""
Stochastic Signal Detection (SSD)

Perform a simulation of the evolution of a stochastic field.
"""
import sys
import argparse
from pde import MemoryStorage, PDEBase, ScalarField, CartesianGrid, plot_kymograph, PlotTracker, PDE
from pde.grids.boundaries.axes import BoundariesData
from pde.pdes.base import expr_prod
from pde.visualization.movies import movie
from scipy import constants as C



__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'Perform a simulation of the evolution of a stochastic field'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'

class SSD(PDEBase):
    """Stochastic Signal Detection (SSD)"""

    def __init__(self, k: float, noise: float = 0.0, bc: BoundariesData = 'auto_periodic_neumann'):
        """
        Parameters
        ----------
        k : float
            Renormalization scale
        noise : float
            Noise intensity (default is 0.0)
        bc : BoundariesData
            Boundary conditions (default is 'auto_periodic_neumann')
        """
        super().__init__(noise=noise)
        self.k = k
        self.bc = bc

    @property
    def expression(self) -> str:
        return f'-dim(f) * f + dim(x) * d_dx(f) - 2 * (3 d_dx(f) + 2 * x * laplace(f)) / (1 + f + 2 * x * laplace(f)**2)**2'

    def evolution_rate(self, state: ScalarField, t: float = 0) -> ScalarField:

        #TODO compute the correct values of the coefficients
        dim_f = 3e-1
        dim_x = 2e-1

        # Parse the state
        x = state.grid.axes_coords[0]
        f = state
        grad_f = state.gradient(bc=self.bc)[0]
        lapl_f = state.laplace(bc=self.bc)

        # Compute each factor
        fac_1 = -dim_f * f
        fac_2 = dim_x * grad_f

        # Compute the fraction
        num = 3 * grad_f + 2 * x * grad_f
        den = (1 + f + 2 * x * lapl_f**2)**2
        fac_3 = -2 * num / den

        # Assemble the result
        result = fac_1 + fac_2 + fac_3
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
    trackers = ['progress',
                'steady_state',
                storage.tracker(interval=1),
                PlotTracker(show=True)
                ]

    # Define the PDE and solve
    # eq = SSD(k=1.0, noise=0.0, bc=bc)
    eq = PDE({'f': f'-0.01 * f + 0.01 * d_dx(f) - 2 * (3 * d_dx(f) + 2 * x * laplace(f)) / (1 + f + 2 * x * laplace(f)**2)**2'})
    eq.solve(state, t_range=1000, dt=0.01, tracker=trackers)

    movie(storage,
          'ssd.mp4',
          show_time=True,
          plot_args={

          },
          movie_args={'framerate': 10})

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__, epilog=__epilog__)
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('-v', dest='verb', action='count', default=0, help='Verbosity level')
    args = parser.parse_args()

    code = main(args)


    sys.exit(code)
