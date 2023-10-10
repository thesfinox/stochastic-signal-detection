# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Study the behaviour of a Marchenko-Pastur distribution in the presence of a deterministic signal.
"""
import argparse
import logging
import sqlite3
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pde import CartesianGrid, MemoryStorage, ScalarField
from tabulate import tabulate
from matplotlib.patches import Ellipse

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
    parameters = vars(args)
    keys = list(parameters.keys())[3:]
    values = [parameters[key] for key in keys]
    table = tabulate(parameters.items(),
                     tablefmt='fancy_grid',
                     headers=['Argument', 'Value'])
    print(table)

    # Create the output directory
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Define the prefix of the output files
    prefix = '_'.join([f'{key}={value}' for key, value in zip(keys, values)])

    # Set the log
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(output / f'{prefix}.log', mode='w')
    handler.setLevel(logging.DEBUG)
    form = '[%(asctime)s] %(levelname)s: %(message)s'
    handler.setFormatter(logging.Formatter(form))

    logger.addHandler(handler)
    logger.info(f'Initial setup:\n{table}')

    ##################################
    #                                #
    # PRELIMINARY ANALYSIS           #
    #                                #
    ##################################

    # Visualize the starting point of the potential
    kappa_0 = args.params[0]
    mu0_0 = args.params[1]
    mu1_0 = args.params[2]
    a = -mu0_0 * kappa_0 + mu1_0 * kappa_0**2
    b = mu0_0 - 2*mu1_0*kappa_0
    c = mu1_0
    logger.debug(f'Initial potential: {a:.2f} + {b:.2f} x + {c:.2f} x^2')
    if args.debug:
        with plt.style.context('fast', after_reset=True):
            plot_potential(args.xinf,
                           args.xsup,
                           args.nval,
                           a,
                           b,
                           c,
                           output,
                           prefix)

    # Create a random matrix (bulk distribution) and a signal matrix
    Z = create_bulk(rows=args.rows, ratio=args.ratio, random_state=args.seed)
    S = create_signal(rows=args.rows,
                      ratio=args.ratio,
                      rank=args.rank,
                      random_state=args.seed)
    logger.debug(f'Z.shape = {Z.shape}')
    logger.debug(f'S.shape = {S.shape}')
    logger.debug(f'beta = {args.beta:.2f}')

    # Compute the full matrix
    X = Z + args.beta * S
    C = np.cov(X, rowvar=False)
    logger.debug(f'C.shape = {C.shape}')

    E = np.linalg.eigvalsh(C)
    logger.debug(f'E.max = {E.max()}')
    logger.debug(f'E.min = {E.min()}')

    E_inv = np.flip(1 / E)
    logger.debug(f'E_inv.max = {E_inv.max()}')
    logger.debug(f'E_inv.min = {E_inv.min()}')
    E_inv -= E_inv.min()

    # Plot the Marchenko-Pastur distribution
    mp = MarchenkoPastur(L=args.ratio)
    if args.debug:
        x = np.linspace(0, E.max() * 1.1, num=10000)
        y = np.array([mp(xi) for xi in x])
        plot_mp_distribution(E, x, y, args.nbins, output, prefix)

    # Find the mass scale of the noise
    if args.a is not None:
        mass_scale = (E >= mp.max).argmax()
        mass_scale_bottom = (E >= mp.max + args.a).argmax()
        mass_scale_top = (E >= mp.max - args.a).argmax()

        mass_scale = E_inv[-mass_scale]
        mass_scale_bottom = E_inv[-mass_scale_bottom]
        mass_scale_top = E_inv[-mass_scale_top]
    else:
        mass_scale = 0.0
        mass_scale_bottom = min(args.bounds)
        mass_scale_top = max(args.bounds)
    logger.debug(f'Mass scale = {mass_scale}')
    logger.debug(f'Interval = [{mass_scale_bottom}, {mass_scale_top}]')

    # Draw the inverse Marchenko-Pastur distribution
    bins = args.nbins**2  # increase number of bins for resolution
    dist = InterpolateDistribution(bins=bins)  # empirical distribution
    dist = dist.fit(E_inv, n=2, s=args.smooth, force_origin=True)
    if args.debug:
        mp_inv = TranslatedInverseMarchenkoPastur(L=args.ratio)
        x = np.linspace(0, E_inv.max() * 1.1, num=25000)
        y_dist = np.array([dist(xi) for xi in x])
        y = np.array([mp_inv(xi) for xi in x])
        plot_inverse_mp_distribution(E_inv,
                                     x,
                                     y,
                                     y_dist,
                                     mass_scale_bottom,
                                     mass_scale_top,
                                     bins,
                                     output,
                                     prefix)

    ##################################
    #                                #
    # SIMULATION                     #
    #                                #
    ##################################

    # Define the grid
    grid = CartesianGrid(
        [[args.xinf, args.xsup]],  # range of x coordinates
        [args.nval],  # number of points in x direction
        periodic=args.periodic,  # periodicity in x direction
    )
    expression = f'{mu0_0} * (x - {kappa_0}) + {mu1_0} * (x - {kappa_0})**2'
    state = ScalarField.from_expression(grid, expression)  # initial state
    bc = 'periodic' if args.periodic else 'auto_periodic_neumann'

    # Initialize a storage
    t_range = [np.sqrt(mass_scale_bottom), np.sqrt(mass_scale_top)]
    dt = (t_range[1] - t_range[0]) / args.nsteps
    dt_viz = dt * args.nsteps / 5
    storage = MemoryStorage()
    storage_viz = MemoryStorage()
    trackers = [
        'progress',
        'steady_state',
        storage.tracker(interval=dt),
        storage_viz.tracker(interval=dt_viz),
    ]

    # Define the PDE and solve
    eq = SSD(dist=dist, noise=0.0, bc=bc)
    _ = eq.solve(state, t_range=t_range, dt=dt, tracker=trackers)

    ##################################
    #                                #
    # POST-PROCESSING                #
    #                                #
    ##################################

    # Visualize the simulation at fixed time steps
    if args.debug:
        fig, ax = plt.subplots(nrows=len(storage_viz), figsize=(8, 6), sharex=True)
        cmap = plt.get_cmap('tab10')
        for n, (time, field) in enumerate(storage_viz.items()):

            # Collect data
            chi = field.grid.axes_coords[0]
            Up = field.data

            # Plot the field
            ax[n].plot(chi, Up, color=cmap(n), label=f'k = {time:.3f}')
            ax[n].set_xlabel(r'$\overline{\chi}$')
            ax[n].set_ylabel(r'$\overline{\mathcal{U}}^{~\prime}$')
            ax[n].legend(loc='best')
            ax[n].ticklabel_format(axis='y',
                                   style='sci',
                                   scilimits=(0, 0),
                                   useMathText=True)
        plt.savefig(output / f'{prefix}_sim.pdf')
        plt.close(fig)

    # Visualize the evolution of the field in a given position
    def find_params(chi, Up):

        # Find the position of the minimum
        idx = np.argmin(Up)
        kappa = chi[idx]

        # Compute the derivative of the potential and compute in the minimum
        dUp = np.gradient(Up, chi)
        mu0 = dUp[idx]

        # Compute the second der. of the potential and compute in the minimum
        d2Up = np.gradient(dUp, chi)
        mu1 = d2Up[idx]

        return kappa, mu0, mu1

    k = []
    kappas = []
    kappas_bar = []
    mu0s = []
    mu1s = []
    Up_starts = []
    Up_ends = []
    Up_ratios = []
    for time, field in storage.items():

        # Collect data
        k.append(time)
        Up_start = field.data[0]
        Up_end = field.data[-1]
        Up_ratio = (Up_end-Up_start) / Up_start

        Up_starts.append(Up_start)
        Up_ends.append(Up_end)
        Up_ratios.append(np.abs(Up_ratio))
        logger.debug(
            f'k = {time:.3f}, U\'[{args.xinf}] = {Up_start:.3f}, U\'[{args.xsup}] = {Up_end:.3f}'
        )

        # Find the parameters
        xdata = field.grid.axes_coords[0]
        ydata = field.data
        kappa, mu0, mu1 = find_params(xdata, ydata)
        logger.debug(
            f'k = {time:.3f} => kappa = {kappa:.3f}, mu0 = {mu0:.3f}, mu1 = {mu1:.3f}'
        )

        # Store the parameters
        k2 = time**2
        I = dist.integrate(0, time, moment=1, power=2)[0]
        dimU = 2 * I / (k2 + 1.e-9) / (dist(k2) + 1.e-9)
        P = k2 * dist.grad(k2) / (dist(k2) + 1.e-9)
        dimChi = 2 - dimU * (P+2)
        kappas.append(kappa)
        kappas_bar.append(kappa / dimChi / 2.0)
        mu0s.append(mu0)
        mu1s.append(mu1)

    # Find the centroid of the distribution of the final values of mu0 and mu1
    mu0_avg = np.mean(mu0s[-50:])
    mu0_std = np.std(mu0s[-50:])
    mu1_avg = np.mean(mu1s[-50:])
    mu1_std = np.std(mu1s[-50:])

    if args.debug:
        # Plot the evolution of the field
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        ax[0].plot(k, Up_starts, 'k-')
        ax[0].set_xlabel(r'$\longmapsto~UV~\longrightarrow~k~\longrightarrow~IR~\longrightarrow$')
        ax[0].invert_xaxis()
        ax[0].set_ylabel(
            rf'$\overline{{\mathcal{{U}}}}^{{~\prime}}[{args.xinf}]$')
        ax[0].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)

        ax[1].plot(k, Up_ends, 'k-')
        ax[1].set_xlabel(r'$\longmapsto~UV~\longrightarrow~k~\longrightarrow~IR~\longrightarrow$')
        ax[1].invert_xaxis()
        ax[1].set_ylabel(
            rf'$\overline{{\mathcal{{U}}}}^{{~\prime}}[{args.xsup}]$')
        ax[1].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        plt.tight_layout()
        plt.savefig(output / f'{prefix}_sim_time.pdf')
        plt.close(fig)

        # Plot the evolution of the parameters
        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(32, 12))
        ax = ax.flatten()

        ax[0].plot(k[25:],
                   kappas_bar[25:],
                   'k-',
                   label='simulated')
        ax[0].invert_xaxis()
        ax[0].set_xlabel(r'$\longmapsto~UV~\longrightarrow~k~\longrightarrow~IR~\longrightarrow$')
        ax[0].set_ylabel(r'$\overline{\kappa}$')
        ax[0].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)

        ax[1].plot(k[25:], kappas[25:], 'k-')
        ax[1].invert_xaxis()
        ax[1].set_xlabel(r'$\longmapsto~UV~\longrightarrow~k~\longrightarrow~IR~\longrightarrow$')
        ax[1].set_ylabel(r'$\kappa$')
        ax[1].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        ax[2].plot(k[25:], mu0s[25:], 'k-')
        ax[2].invert_xaxis()
        ax[2].set_xlabel(r'$\longmapsto~UV~\longrightarrow~k~\longrightarrow~IR~\longrightarrow$')
        ax[2].set_ylabel(r'$\mu_0$')
        ax[2].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        # ax[2].set_yscale('symlog')

        ax[3].plot(k[25:], mu1s[25:], 'k-')
        ax[3].invert_xaxis()
        ax[3].set_xlabel(r'$\longmapsto~UV~\longrightarrow~k~\longrightarrow~IR~\longrightarrow$')
        ax[3].set_ylabel(r'$\mu_1$')
        ax[3].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        # ax[3].set_yscale('symlog')

        ax[4].plot(kappas[25:], mu0s[25:], 'k-')
        ax[4].set_xlabel(r'$\kappa$')
        ax[4].set_ylabel(r'$\mu_0$')
        ax[4].ticklabel_format(axis='both',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        # ax[4].set_xscale('symlog')
        # ax[4].set_yscale('symlog')

        ax[5].plot(kappas[25:], mu1s[25:], 'k-')
        ax[5].set_xlabel(r'$\kappa$')
        ax[5].set_ylabel(r'$\mu_1$')
        ax[5].ticklabel_format(axis='both',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        # ax[5].set_xscale('symlog')
        # ax[5].set_yscale('symlog')

        ax[6].plot(mu0s[25:], mu1s[25:], 'k-')
        ax[6].set_xlabel(r'$\mu_0$')
        ax[6].set_ylabel(r'$\mu_1$')
        ax[6].ticklabel_format(axis='both',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        # ax[6].set_xscale('symlog')
        # ax[6].set_yscale('symlog')

        # Draw an ellipse around the centroid
        ell = Ellipse(xy=(mu0_avg, mu1_avg),
                        width=mu0_std * 2,
                        height=mu1_std * 2,
                        angle=0,
                        color='r',
                        alpha=0.25)
        ax[6].add_artist(ell)

        plt.tight_layout()
        plt.savefig(output / f'{prefix}_sim_params_time.pdf')
        plt.close(fig)

    # Save the data of the simulation
    nmax_chi_0 = int(np.argmax(np.abs(Up_starts)))
    nmax_chi_1 = int(np.argmax(np.abs(Up_ends)))
    data = {
        'rows': int(args.rows),
        'ratio': float(args.ratio),
        'rank': int(args.rank),
        'beta': float(args.beta),
        'mass_scale': float(mass_scale),
        'mass_scale_bottom': float(mass_scale_bottom),
        'mass_scale_top': float(mass_scale_top),
        'kappa_0': float(kappa_0),
        'mu_0_0': float(mu0_0),
        'mu_1_0': float(mu1_0),
        'bkappas': json.dumps(list(map(float, kappas_bar))),
        'kappas': json.dumps(list(map(float, kappas))),
        'mu_0s': json.dumps(list(map(float, mu0s))),
        'mu_1s': json.dumps(list(map(float, mu1s))),
        'mu_0_avg': float(mu0_avg),
        'mu_0_std': float(mu0_std),
        'mu_1_avg': float(mu1_avg),
        'mu_1_std': float(mu1_std),
        'nmax_chi_0': nmax_chi_0,
        'max_chi_0': float(Up_starts[nmax_chi_0]),
        'argmax_chi_0': float(k[nmax_chi_0]),
        'nmax_chi_1': nmax_chi_1,
        'max_chi_1': float(Up_ends[nmax_chi_1]),
        'argmax_chi_1': float(k[nmax_chi_1]),
        'min_frac': float(np.min(Up_ratios)),
        'nmin_frac': int(np.argmin(Up_ratios)),
        'argmin_frac': float(k[np.argmin(Up_ratios)]),
    }

    # Store the data in a SQLite database
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(args.db) as conn:

            # Create a cursor
            cursor = conn.cursor()

            # Create the table if it does not exist
            sql_query = f"""CREATE TABLE IF NOT EXISTS {args.table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                seed INTEGER,
                rows INTEGER,
                ratio REAL,
                rank INTEGER,
                beta REAL,
                mass_scale REAL,
                mass_scale_bottom REAL,
                mass_scale_top REAL,
                kappa_0 REAL,
                mu_0_0 REAL,
                mu_1_0 REAL,
                bkappas TEXT,
                kappas TEXT,
                mu_0s TEXT,
                mu_1s TEXT,
                mu_0_avg REAL,
                mu_0_std REAL,
                mu_1_avg REAL,
                mu_1_std REAL,
                nmax_chi_0 INTEGER,
                max_chi_0 REAL,
                argmax_chi_0 REAL,
                nmax_chi_1 INTEGER,
                max_chi_1 REAL,
                argmax_chi_1 REAL,
                min_frac REAL,
                nmin_frac INTEGER,
                argmin_frac REAL
            )"""
            cursor.execute(sql_query)

            # Insert the data from the dictionary
            sql_query = f"""INSERT INTO {args.table} (
                timestamp,
                seed,
                rows,
                ratio,
                rank,
                beta,
                mass_scale,
                mass_scale_bottom,
                mass_scale_top,
                kappa_0,
                mu_0_0,
                mu_1_0,
                bkappas,
                kappas,
                mu_0s,
                mu_1s,
                mu_0_avg,
                mu_0_std,
                mu_1_avg,
                mu_1_std,
                nmax_chi_0,
                max_chi_0,
                argmax_chi_0,
                nmax_chi_1,
                max_chi_1,
                argmax_chi_1,
                min_frac,
                nmin_frac,
                argmin_frac
            ) VALUES (
                '{date}',
                {int(args.seed)},
                {data['rows']},
                {data['ratio']},
                {data['rank']},
                {data['beta']},
                {data['mass_scale']},
                {data['mass_scale_bottom']},
                {data['mass_scale_top']},
                {data['kappa_0']},
                {data['mu_0_0']},
                {data['mu_1_0']},
                '{data['bkappas']}',
                '{data['kappas']}',
                '{data['mu_0s']}',
                '{data['mu_1s']}',
                {data['mu_0_avg']},
                {data['mu_0_std']},
                {data['mu_1_avg']},
                {data['mu_1_std']},
                {data['nmax_chi_0']},
                {data['max_chi_0']},
                {data['argmax_chi_0']},
                {data['nmax_chi_1']},
                {data['max_chi_1']},
                {data['argmax_chi_1']},
                {data['min_frac']},
                {data['nmin_frac']},
                {data['argmin_frac']}
            )"""
            cursor.execute(sql_query)

    except sqlite3.Error as e:
        logger.error(e)
        return e

    finally:
        conn.close()

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument('--output',
                        type=str,
                        default='sim_behaviour',
                        help='Output directory')
    parser.add_argument('--db',
                        type=str,
                        default='simulation.sqlite',
                        help='Database file')
    parser.add_argument('--table',
                        type=str,
                        default='simulation',
                        help='Table name')
    parser.add_argument('--rows',
                        type=int,
                        default=7500,
                        help='Number of rows (data vectors) of the matrix')
    parser.add_argument('--ratio',
                        type=float,
                        default=0.8,
                        help='Ratio between columns and rows')
    scale = parser.add_mutually_exclusive_group(required=True)
    scale.add_argument('--a',
                       type=float,
                       default=None,
                       help='Neighborhood of the mass scale (symmetric)')
    scale.add_argument('--bounds',
                       nargs=2,
                       type=float,
                       default=None,
                       help='Interval of the mass scale')
    temp = parser.add_mutually_exclusive_group(required=True)
    temp.add_argument('--params',
                      type=float,
                      nargs=3,
                      default=None,
                      help='Parameters (kappa, mu0, mu1)')
    parser.add_argument('--rank',
                        type=int,
                        default=2500,
                        help='Rank of the signal matrix')
    parser.add_argument('--beta',
                        type=float,
                        default=0.5,
                        help='Signal-to-noise ratio')
    parser.add_argument('--nbins',
                        type=int,
                        default=100,
                        help='Number of bins for the histogram')
    parser.add_argument('--xinf',
                        type=float,
                        default=0.0,
                        help='Lower bound of the domain')
    parser.add_argument('--xsup',
                        type=float,
                        default=1.0,
                        help='Upper bound of the domain')
    parser.add_argument('--nval',
                        type=int,
                        default=1000,
                        help='Number of grid points')
    parser.add_argument('--periodic',
                        action='store_true',
                        help='Periodic boundary conditions')
    parser.add_argument('--nsteps', type=int, default=1000, help='Time steps')
    parser.add_argument(
        '--smooth',
        type=float,
        default=0.3,
        help='Smoothing parameter for the empirical distribution')
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
