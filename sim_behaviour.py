# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Study the behaviour of a Marchenko-Pastur distribution in the presence of a deterministic signal.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pde import CartesianGrid, MemoryStorage, ScalarField
from tabulate import tabulate

from ssd import (SSD,
                 InterpolateDistribution,
                 MarchenkoPastur,
                 TranslatedInverseMarchenkoPastur)
from ssd.utils.matrix import create_bulk, create_signal
from ssd.utils.plots import (plot_inverse_mp_distribution,
                             plot_mp_distribution,
                             plot_potential)

__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'Study the behaviour of a Marchenko-Pastur distribution in the presence of a deterministic signal.'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'


def main(args):

    # Print the command line arguments
    table = tabulate(vars(args).items(),
                     tablefmt='fancy_grid',
                     headers=['Argument', 'Value'])
    print(table)

    # Create the output directory
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Define the prefix of the output files
    prefix = (f'L={args.L}' + '_' + f'kmax={args.kmax}' + '_'
              + f'mu1={args.mu1}' + '_' + f'mu2={args.mu2}' + '_'
              + f'mu3={args.mu3}')

    # Set the log
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(output / f'{prefix}.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.info(f'Initial setup:\n{table}')

    # Visualize the starting point of the potential
    with plt.style.context('fast', after_reset=True):
        plot_potential(args.x_inf,
                       args.x_sup,
                       args.n_values,
                       args.mu1,
                       args.mu2,
                       args.mu3,
                       output,
                       prefix)

    # Create a random matrix (bulk distribution) and a signal matrix
    Z = create_bulk(rows=args.N, ratio=args.L, random_state=args.seed)
    S = create_signal(rows=args.N,
                      ratio=args.L,
                      rank=args.rank,
                      random_state=args.seed)
    logger.debug(f'Z.shape = {Z.shape}')
    logger.debug(f'S.shape = {S.shape}')

    # Tune a parameter beta to increate the signal-to-noise ratio
    beta_list = np.linspace(0, args.beta_max, num=args.n_beta)
    if args.n_beta == 1:
        beta_list = [args.beta_max]
    logger.debug(f'beta_list = {beta_list}')

    # Compute the eigenvalues of the covariance matrix and its inverse
    critical_position = []
    for n, beta in enumerate(beta_list):

        logger.info(f'beta = {beta:.2f}')
        subprefix = f'beta={beta:.2f}' + '_' + prefix

        # Compute the full matrix
        X = Z + beta*S

        # Compute the covariance matrix and its inverse
        C = np.cov(X, rowvar=False)
        logger.debug(f'C.shape = {C.shape}')

        # Compute the eigenvalues of the covariance matrix and its inverse
        E = np.linalg.eigvalsh(C)
        logger.debug(f'E.max (mass) = {E.max()}')
        logger.debug(f'E.min = {E.min()}')

        E_inv = np.flip(1 / E)
        logger.debug(f'E_inv.max = {E_inv.max()}')
        logger.debug(f'E_inv.min = {E_inv.min()}')

        # Remove the mass scale to translate to 0
        E_inv -= E_inv.min()

        # Plot the Marchenko-Pastur distribution
        mp = MarchenkoPastur(L=args.L)
        if n % 25 == 0:
            x = np.linspace(0, E.max() * 1.1, num=10000)
            y = np.array([mp(xi) for xi in x])
            plot_mp_distribution(E, x, y, args.n_bins, output, subprefix)

        # Find the mass scale of the noise
        mass_scale = (E >= mp.max).argmax()
        mass_scale_bottom = (E >= mp.max + args.a).argmax()
        mass_scale_top = (E >= mp.max - args.a).argmax()

        # Identify the scale in the inverse distribution of eigenvalues
        mass_scale = E_inv[-mass_scale]
        mass_scale_bottom = E_inv[-mass_scale_bottom]
        mass_scale_top = E_inv[-mass_scale_top]
        logger.debug(f'Bulk mass = {mass_scale}')
        logger.debug(f'Interval = [{mass_scale_bottom}, {mass_scale_top}]')

        # Draw the inverse Marchenko-Pastur distribution
        bins = args.n_bins**2  # increase number of bins for resolution
        dist = InterpolateDistribution(bins=bins)  # empirical distribution
        dist = dist.fit(E_inv, n=2, s=0.5, force_origin=True)

        # Define the grid
        grid = CartesianGrid(
            [[args.x_inf, args.x_sup]],  # range of x coordinates
            [args.n_values],  # number of points in x direction
            periodic=args.periodic,  # periodicity in x direction
        )
        expression = (f'{args.mu1}' + ' + ' + f'{args.mu2} * x' + ' + '
                      + f'{args.mu3} * x**2')
        state = ScalarField.from_expression(grid, expression)  # initial state
        bc = 'periodic' if args.periodic else 'auto_periodic_neumann'

        # Initialize a storage
        storage = MemoryStorage()
        trackers = [
            'steady_state',
            storage.tracker(interval=1),
        ]

        # Define the PDE and solve
        k2min = 0.0 if args.kmax > 0 else mass_scale_bottom
        k2max = args.kmax if args.kmax > 0 else mass_scale_top
        eq = SSD(dist=dist, k2max=k2max, k2min=k2min, noise=0.0, bc=bc)
        _ = eq.solve(state, t_range=args.t_range, dt=args.dt, tracker=trackers)

        # Plot the inverse distribution with lines over the integration interval
        if n % 25 == 0:
            mp_inv = TranslatedInverseMarchenkoPastur(L=args.L)
            x = np.linspace(0, E_inv.max() * 1.1, num=10000)
            y_dist = np.array([dist(xi) for xi in x])
            y = np.array([mp_inv(xi) for xi in x])
            plot_inverse_mp_distribution(E_inv,
                                         x,
                                         y,
                                         y_dist,
                                         k2min,
                                         k2max,
                                         bins,
                                         output,
                                         subprefix)

        # Visualize the simulation at fixed time steps
        if n % 25 == 0:
            x = storage[0].grid.axes_coords[0]
            y = [storage[t].data for t in args.viz]
            y = np.nan_to_num(y)
            My = np.log10(y.max(axis=1) + 1).astype('int')
            y = y / y.max(axis=1, keepdims=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            C = ['r-', 'b-', 'g-', 'k-']
            for i, y_i in enumerate(y):
                ax.plot(
                    x,
                    y_i,
                    C[i],
                    label=rf'$\tau$ = {args.viz[i]} ($\times 10^{{{My[i]:d}}}$)'
                )
            ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
            ax.set_xlabel(r'$\overline{\chi}$')
            ax.set_ylabel(r'$\overline{\mathcal{U}}^{~\prime}$')

            # Create an inset axis to zoom around some values
            ax_inset_1 = ax.inset_axes([0.08, 0.15, 0.3, 0.3])
            ax_inset_1.plot(
                x,
                y[1],
                C[1],
                label=rf'$\tau$ = {args.viz[1]} ($\times 10^{{{My[1]:d}}}$)')
            left, right = (x >= 0.45).argmax(), (x >= 0.60).argmax()
            ymax = np.abs(y[1][left:right]).max()
            ax_inset_1.set_xlim([0.45, 0.60])
            ax_inset_1.set_ylim([-1.1 * ymax / 10, 1.1 * ymax / 10])
            ax_inset_1.ticklabel_format(axis='y',
                                        style='sci',
                                        scilimits=(0, 0),
                                        useMathText=True)
            ax.indicate_inset_zoom(ax_inset_1)

            ax_inset_2 = ax.inset_axes([0.57, 0.07, 0.3, 0.3])
            ax_inset_2.plot(
                x,
                y[2],
                C[2],
                label=rf'$\tau$ = {args.viz[2]} ($\times 10^{{{My[2]:d}}}$)')
            left, right = (x >= 0.70).argmax(), (x >= 0.85).argmax()
            ymax = np.abs(y[2][left:right]).max()
            ax_inset_2.set_xlim([0.70, 0.85])
            ax_inset_2.set_ylim([-1.1 * ymax / 10, 1.1 * ymax / 10])
            ax_inset_2.ticklabel_format(axis='y',
                                        style='sci',
                                        scilimits=(0, 0),
                                        useMathText=True)
            ax.indicate_inset_zoom(ax_inset_2)

            plt.savefig(output / f'{subprefix}_sim.pdf')
            plt.close(fig)

        # Display the evolution of the field in a given position
        t = []
        y = []

        x = storage[0].grid.axes_coords[0]
        idx = (x >= args.chi).argmax()
        for time, field in storage.items():
            data = np.nan_to_num(field.data)
            value = data[idx]
            t.append(time)
            y.append(value)
            if int(np.log10(np.abs(value) + 1)) >= 308:
                break

        t = np.asarray(t)
        y = np.asarray(y)
        My = int(np.log10(np.abs(y).max() + 1))
        y /= np.abs(y).max()

        if n % 25 == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(t, y, 'r-')
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel(
                rf'$\overline{{\mathcal{{U}}}}^{{~\prime}}[{{\chi_0}}]$ [$\times 10^{{{My}}}$]'
            )
            ax.ticklabel_format(axis='y',
                                style='sci',
                                scilimits=(0, 0),
                                useMathText=True)
            # ax.set_yscale('symlog')

            plt.savefig(output / f'{subprefix}_sim_time.pdf')
            plt.close(fig)

        # Compute the critical position
        pos = np.abs(np.diff(y)).argmax()
        critical_position.append(pos)

    # Plot the critical position as a function of beta
    fig, ax = plt.subplots()
    ax.plot(beta_list, critical_position, 'k-', label='critical position')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\tau$')
    plt.tight_layout()
    plt.savefig(output / f'{prefix}_criticality.pdf')
    plt.close(fig)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument('-o',
                        dest='output',
                        type=str,
                        default='sim_behaviour',
                        help='Output directory')
    parser.add_argument('-N',
                        type=int,
                        default=7500,
                        help='Number of rows (data vectors) of the matrix')
    parser.add_argument(
        '-L',
        type=float,
        default=0.8,
        help=
        'Ratio between the number of columns (variables) and the number of rows (data vectors)'
    )
    parser.add_argument('--kmax', type=float, default=0.01, help='IR cutoff')
    parser.add_argument('--kmin', type=float, default=0.00, help='IR cutoff')
    parser.add_argument('--a',
                        type=float,
                        default=0.5,
                        help='Neighborhood of the mass scale (symmetric)')
    parser.add_argument('--mu1',
                        type=float,
                        default=0.0,
                        help='Mass parameter (quadratic term)')
    parser.add_argument('--mu2', type=float, default=1.0, help='Quartic term')
    parser.add_argument('--mu3', type=float, default=0.0, help='6th power term')
    parser.add_argument('--rank',
                        type=int,
                        default=2500,
                        help='Rank of the signal matrix')
    parser.add_argument('--beta-max',
                        type=float,
                        default=0.5,
                        help='Maximum value of beta (signal-to-noise ratio)')
    parser.add_argument(
        '--n-beta',
        type=int,
        default=100,
        help='Number of beta values in the interval [0, beta_max]')
    parser.add_argument('--n-bins',
                        type=int,
                        default=100,
                        help='Number of bins for the histogram')
    parser.add_argument('--x-inf',
                        type=float,
                        default=0.0,
                        help='Lower bound of the domain')
    parser.add_argument('--x-sup',
                        type=float,
                        default=1.0,
                        help='Upper bound of the domain')
    parser.add_argument('--n-values',
                        type=int,
                        default=1000,
                        help='Number of grid points')
    parser.add_argument('--periodic',
                        action='store_true',
                        help='Periodic boundary conditions')
    parser.add_argument('--t-range', type=float, default=300, help='Time range')
    parser.add_argument('--dt', type=float, default=1, help='Time step')
    parser.add_argument('--viz',
                        nargs=4,
                        type=float,
                        default=[0, 10, 30, 50],
                        help='Time steps to visualize')
    parser.add_argument('--chi',
                        type=float,
                        default=0.5,
                        help='Position of the potential to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('-v',
                        dest='verb',
                        action='count',
                        default=0,
                        help='Verbosity level')
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
