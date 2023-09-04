# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Study the behaviour of a Marchenko-Pastur distribution in the presence of a deterministic signal.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from pde import CartesianGrid, MemoryStorage, ScalarField
from tabulate import tabulate
from tqdm import tqdm

from ssd import (SSD,
                 InterpolateDistribution,
                 MarchenkoPastur,
                 TranslatedInverseMarchenkoPastur)
from ssd.utils.matrix import create_bulk, create_signal

mpl.use('agg')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 6)

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
              + f'mu3={args.mu3}' + '_' + f'mu4={args.mu4}')

    # Visualize the starting point of the potential
    with plt.style.context('fast', after_reset=True):

        fig, ax = plt.subplots()
        x = np.linspace(args.x_inf, args.x_sup, args.n_values)
        y = args.mu1 * x + args.mu2 * x**2 + args.mu3 * x**3 + args.mu4 * x**4

        ax.plot(x, y, 'k-', label='potential')
        ax.set_xlabel(r'$\chi$')
        ax.set_ylabel(r'$\overline{\mathcal{U}}$')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.plot(1,
                0,
                ">k",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                markersize=20)
        ax.plot(0,
                1,
                "^k",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                markersize=20)

        plt.tight_layout()
        plt.savefig(output / f'{prefix}_potential.pdf')
        plt.close(fig)

    # Create a random matrix (bulk distribution) and a signal matrix
    Z = create_bulk(rows=args.N, ratio=args.L, random_state=args.seed)
    S = create_signal(rows=args.N,
                      ratio=args.L,
                      rank=args.rank,
                      random_state=args.seed)

    # Tune a parameter beta to increate the signal-to-noise ratio
    beta_list = np.linspace(0, args.beta_max, num=args.n_beta)
    beta_space = tqdm(beta_list,
                      desc='signal-to-noise',
                      leave=False,
                      unit='beta')

    # Compute the eigenvalues of the covariance matrix and its inverse
    critical_position = []
    for beta in beta_space:

        # Update the progress bar
        beta_space.set_postfix(beta=beta)
        subprefix = f'beta={beta:.2f}' + '_' + prefix

        # Compute the full matrix
        X = Z + beta*S

        # Compute the covariance matrix and its inverse
        C = np.cov(X, rowvar=False)

        # Compute the eigenvalues of the covariance matrix and its inverse
        E = np.linalg.eigvalsh(C)
        E_inv = np.flip(1 / E)
        E_inv -= E_inv.min()

        # Draw the Marchenko-Pastur distribution
        mp = MarchenkoPastur(L=args.L)
        x = np.linspace(0, E.max() * 1.1, num=10000)
        y = np.array([mp(xi) for xi in x])

        fig, ax = plt.subplots()
        ax.hist(E,
                bins=args.n_bins,
                density=True,
                label='empirical',
                alpha=0.5,
                color='b')
        ax.plot(x, y, 'r-', label='MP distribution')
        ax.set_xlabel('eigenvalues')
        ax.set_ylabel(r'$\mu$')

        ax_inset_1 = fig.add_axes([0.5, 0.24, 0.35, 0.25])
        ax_inset_1.set_xlim([2.5, 1.05 * E.max()])
        ax_inset_1.set_ylim([0.0, 0.2])
        ax_inset_1.hist(E,
                        bins=args.n_bins,
                        density=True,
                        alpha=0.5,
                        label='empirical',
                        color='b')
        ax_inset_1.plot(x, y, 'r-', label='theoretical')

        ax_inset_2 = fig.add_axes([0.25, 0.55, 0.35, 0.25])
        ax_inset_2.set_xlim([-0.1, 1.5])
        ax_inset_2.set_ylim([0.2, 0.9])
        ax_inset_2.hist(E,
                        bins=args.n_bins,
                        density=True,
                        alpha=0.5,
                        label='empirical',
                        color='b')
        ax_inset_2.plot(x, y, 'r-', label='theoretical')

        ax.indicate_inset_zoom(ax_inset_1)
        ax.indicate_inset_zoom(ax_inset_2)

        ax.legend(loc='best')
        plt.savefig(output / f'{subprefix}_mp_dist.pdf')
        plt.close(fig)

        # Find the mass scale of the noise
        mass_scale = (E >= mp.max).argmax()
        if mass_scale == 0:
            mass_scale = len(E) - 1

        # Identify the scale in the inverse distribution of eigenvalues
        mass_scale = E_inv[-mass_scale]

        # Draw the inverse Marchenko-Pastur distribution
        bins=args.n_bins**2
        dist = InterpolateDistribution(bins=bins)
        dist = dist.fit(E_inv, n=2, s=0.5, force_origin=True)
        mp_inv = TranslatedInverseMarchenkoPastur(L=args.L)
        x = np.linspace(0, E_inv.max() * 1.1, num=10000)
        y_dist = np.array([dist(xi) for xi in x])
        y = np.array([mp_inv(xi) for xi in x])

        fig, ax = plt.subplots()
        ax.hist(E_inv,
                bins=bins,
                density=True,
                label='empirical',
                alpha=0.5,
                color='b')
        ax.plot(x, y_dist, 'k-', label='interpolated')
        ax.plot(x, y, 'r-', label='theoretical')

        ax.set_xlabel(r'$k^2$')
        ax.set_ylabel(r'$\rho$')

        ax_inset_1 = fig.add_axes([0.35, 0.25, 0.35, 0.55])
        ax_inset_1.set_xlim([-0.1, 1.0])
        ax_inset_1.set_ylim([0.0, 0.9])
        ax_inset_1.hist(E_inv,
                        bins=bins,
                        density=True,
                        alpha=0.5,
                        label='empirical',
                        color='b')
        ax_inset_1.plot(x, y_dist, 'k-', label='interpolated')
        ax_inset_1.plot(x, y, 'r-', label='theoretical')

        ax.indicate_inset_zoom(ax_inset_1)

        ax.legend(loc='best')
        plt.savefig(output / f'{subprefix}_mp_inv_dist.pdf')
        plt.close(fig)

        # Define the grid
        grid = CartesianGrid(
            [[args.x_inf, args.x_sup]],
            [args.n_values],
            periodic=args.periodic,
        )
        expression = (f'{args.mu1}' + ' + ' + f'{args.mu2} * x' + ' + '
                      + f'{args.mu3} * x**2' + ' + ' + f'{args.mu4} * x**3')
        state = ScalarField.from_expression(grid, expression)
        bc = 'periodic' if args.periodic else 'auto_periodic_neumann'

        # Initialize a storage
        storage = MemoryStorage()
        trackers = [
            'progress',
            'steady_state',
            storage.tracker(interval=1),
        ]

        # Define the PDE and solve
        if args.kmax > 0:
            eq = SSD(dist=dist, k2=args.kmax, noise=0.0, bc=bc)
        else:
            eq = SSD(dist=dist, k2=mass_scale, noise=0.0, bc=bc)
        _ = eq.solve(state,
                          t_range=args.t_range,
                          dt=args.dt,
                          tracker=trackers)


        # Visualize the simulation at fixed time steps
        x = storage[0].grid.axes_coords[0]
        y = [storage[t].data for t in args.viz]
        y = np.nan_to_num(y)
        My = [int(np.log10(y_i.max() + 1)) for y_i in y]

        y = [y_i / y_i.max() for y_i in y]

        fig, ax = plt.subplots(figsize=(8, 6))
        C = ['r-', 'b-', 'g-', 'k-']
        for i, y_i in enumerate(y):
            ax.plot(
                x,
                y_i,
                C[i],
                label=rf'$\tau$ = {args.viz[i]} ($\times 10^{{{My[i]:d}}}$)')
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
        ax_inset_1.set_ylim([-1.1*ymax/10, 1.1*ymax/10])
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
        ax_inset_2.set_ylim([-1.1*ymax/10, 1.1*ymax/10])
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

        fig, ax = plt.subplots(figsize=(8, 6))
        for time, field in storage.items():

            x = field.grid.axes_coords[0]
            idx = (x >= args.chi).argmax()
            t.append(time)
            y.append(field.data[idx])

        ax.plot(t, y, 'r-')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(rf'$\overline{{\mathcal{{U}}}}^{{~\prime}}[{{\chi_0}}]$')
        ax.ticklabel_format(axis='y',
                            style='sci',
                            scilimits=(0, 0),
                            useMathText=True)
        ax.set_yscale('symlog')

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
    parser.add_argument('--mu1',
                        type=float,
                        default=0.0,
                        help='Mass parameter (quadratic term)')
    parser.add_argument('--mu2', type=float, default=1.0, help='Quartic term')
    parser.add_argument('--mu3', type=float, default=0.0, help='6th power term')
    parser.add_argument('--mu4', type=float, default=0.0, help='8th power term')
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
