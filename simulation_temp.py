# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Study the behaviour of a Marchenko-Pastur distribution in the presence of a deterministic signal.
"""
import argparse
import logging
import sqlite3
import sys
from datetime import datetime
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

    # Generate the values if not provided
    if args.mu is None:
        Tc = args.temp[0]
        T = args.temp[-1]
        args.mu = [T - Tc, T, T**2]
    if args.temp is None:
        Tc = None
        T = None

    # Visualize the starting point of the potential
    if args.debug:
        with plt.style.context('fast', after_reset=True):
            plot_potential(args.xinf,
                           args.xsup,
                           args.nval,
                           args.mu[0],
                           args.mu[1],
                           args.mu[2],
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
        y_0 = np.array([mp(xi) for xi in x])
        plot_mp_distribution(E, x, y_0, args.nbins, output, prefix)

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
        y_0 = np.array([mp_inv(xi) for xi in x])
        plot_inverse_mp_distribution(E_inv,
                                     x,
                                     y_0,
                                     y_dist,
                                     mass_scale_bottom,
                                     mass_scale_top,
                                     bins,
                                     output,
                                     prefix)

    # Define the grid
    grid = CartesianGrid(
        [[args.xinf, args.xsup]],  # range of x coordinates
        [args.nval],  # number of points in x direction
        periodic=args.periodic,  # periodicity in x direction
    )
    expression = f'{args.mu[0]} + {args.mu[1]} * x + {args.mu[2]} * x**2'
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

    # Visualize the simulation at fixed time steps
    if args.debug:
        fig, ax = plt.subplots(nrows=len(storage_viz), figsize=(8, 6), sharex=True)
        cmap = plt.get_cmap('tab10')
        for n, (time, field) in enumerate(storage_viz.items()):

            # Collect data
            x = field.grid.axes_coords[0]
            y_0 = field.data

            # Plot the field
            ax[n].plot(x, y_0, color=cmap(n), label=f'k = {time:.3f}')
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
    t = []
    y_0 = []
    y_1 = []
    frac = []
    for time, field in storage.items():

        # Collect data
        t.append(time)
        y_0.append(field.data[0])
        y_1.append(field.data[-1])
        frac.append(1 - field.data[0] / field.data[-1])
        logger.debug(
            f't = {time:.3f}, y0 = {field.data[0]:.3f}, y1 = {field.data[-1]:.3f}'
        )

    if args.debug:
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        ax[0].plot(t, y_0, 'k-')
        ax[0].set_xlabel('k')
        ax[0].invert_xaxis()
        ax[0].set_ylabel(
            rf'$\overline{{\mathcal{{U}}}}^{{~\prime}}[{args.xinf}]$')
        ax[0].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        ax[1].plot(t, y_1, 'k-')
        ax[1].set_xlabel('k')
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

    # Save the data of the simulation
    nmax_chi_0 = int(np.argmax(np.abs(y_0)))
    nmax_chi_1 = int(np.argmax(np.abs(y_1)))
    data = {
        'rows': int(args.rows),
        'ratio': float(args.ratio),
        'rank': int(args.rank),
        'beta': float(args.beta),
        'mass_scale': float(mass_scale),
        'mass_scale_bottom': float(mass_scale_bottom),
        'mass_scale_top': float(mass_scale_top),
        'mu_0': float(args.mu[0]),
        'mu_1': float(args.mu[1]),
        'mu_2': float(args.mu[2]),
        'Tc': None if Tc is None else float(Tc),
        'T': None if T is None else float(T),
        'nmax_chi_0': nmax_chi_0,
        'max_chi_0': float(y_0[nmax_chi_0]),
        'argmax_chi_0': float(t[nmax_chi_0]),
        'nmax_chi_1': nmax_chi_1,
        'max_chi_1': float(y_1[nmax_chi_1]),
        'argmax_chi_1': float(t[nmax_chi_1]),
        'min_frac': float(np.min(frac)),
        'nmin_frac': int(np.argmin(frac)),
        'argmin_frac': float(t[np.argmin(frac)]),
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
                mu_0 REAL,
                mu_1 REAL,
                mu_2 REAL,
                Tc REAL,
                T REAL,
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
                mu_0,
                mu_1,
                mu_2,
                Tc,
                T,
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
                {data['mu_0']},
                {data['mu_1']},
                {data['mu_2']},
                {data['Tc']},
                {data['T']},
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
        return 1

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
    temp.add_argument('--mu',
                      type=float,
                      nargs=3,
                      default=None,
                      help='Mass parameters (quadratic, quartic, 6th power)')
    temp.add_argument('--temp',
                      type=float,
                      nargs=2,
                      default=None,
                      help='Temperature (critical, considered)')
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
