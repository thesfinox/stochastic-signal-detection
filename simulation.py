# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Simulation of the stochastic signal detection mechanism.
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from pde import CartesianGrid, MemoryStorage, ScalarField
from PIL import Image
from scipy.integrate import simpson

from ssd import __version__
from ssd.distributions import (InterpolateDistribution,
                               MarchenkoPastur,
                               SpecularReflection,
                               TranslatedInverseMarchenkoPastur)
from ssd.ssd import SSD
from ssd.utils.cfg import get_params, logger
from ssd.utils.matrix import create_bulk, create_signal

mpl.use('agg')
mpl.rcParams['figure.figsize'] = (8, 6)
plt.style.use('ggplot')

__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'Simulation of the stochastic signal detection mechanism.'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'


def main(args):

    # Organize the arguments
    log = logger(args.log)
    cfg = get_params(args.config, args.arguments)

    # Create the output directory
    output_dir = Path(cfg.OUTPUT.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the initial conditions
    log.info("Defining the initial conditions...")
    init = cfg.INPUT.INIT
    if init.BY_INIT.ENABLED:
        kappa_bar_0 = init.BY_INIT.KAPPA_0
        mu_bar_0_0 = init.BY_INIT.MU_0
        mu_bar_1_0 = init.BY_INIT.MU_1
        expr = f'{mu_bar_0_0} * (x - {kappa_bar_0}) + {mu_bar_1_0} * (x - {kappa_bar_0})**2'
    elif init.BY_PARAMS.ENABLED:
        mu_bar_0_0 = init.BY_PARAMS.MU_0
        mu_bar_1_0 = init.BY_PARAMS.MU_1
        mu_bar_2_0 = init.BY_PARAMS.MU_2
        expr = f'{mu_bar_0_0} + {mu_bar_1_0} * x + {mu_bar_2_0} * x**2'
    elif init.BY_TEMP.ENABLED:
        T = init.BY_TEMP.T
        TC = init.BY_TEMP.TC
        mu_bar_0_0 = T - TC
        mu_bar_1_0 = T
        mu_bar_2_0 = T**2
        expr = f'{mu_bar_0_0} + {mu_bar_1_0} * x + {mu_bar_2_0} * x**2'
    else:
        raise ValueError('No valid initial conditions defined!')
    log.debug(f"Initial conditions: {expr}")

    # Define the bulk distribution
    log.info('Defining the bulk distribution...')
    matrix = cfg.INPUT.MATRIX
    Z = create_bulk(rows=matrix.ROWS,
                    columns=matrix.COLUMNS,
                    random_state=matrix.SEED)

    # Define the signal
    log.info('Defining the signal distribution...')
    signal = cfg.INPUT.SIGNAL
    if signal.BY_DET.ENABLED:
        S = create_signal(rows=matrix.ROWS,
                          columns=matrix.COLUMNS,
                          rank=signal.BY_DET.RANK,
                          random_state=matrix.SEED)
    elif signal.BY_IMG.ENABLED:
        image = Image.open(signal.BY_IMG.FILE)
        image = image.convert('L')
        image = image.resize(Z.shape[::-1])
        S = np.array(image) / 255.0
    else:
        raise ValueError('No valid signal defined!')

    # Create the full distribution and compute the eigenvalues of the
    # covariance matrix
    X = Z + signal.RATIO * S
    C = np.cov(X, rowvar=False)
    log.debug(f'Covariance shape: {C.shape}')

    log.debug("Computing the eigenvalues of the covariance matrix...")
    E = np.linalg.eigvalsh(C)
    log.debug(f"Eigenvalues:\nlambda_max = {E.max()}\nlambda_min = {E.min()}")

    E_inv = np.flip(1 / E)
    log.debug(f"Momenta:\nk2_max = {E_inv.max()}\nk2_min = {E_inv.min()}")
    E_inv -= E_inv.min()

    # Define the MP distribution associated to the background
    log.info("Defining the MP distribution associated to the background...")
    ratio = matrix.COLUMNS / matrix.ROWS
    mp = MarchenkoPastur(L=ratio)

    # Plot the distribution of the eigenvalues
    log.info("Plotting the distribution of the eigenvalues...")
    x = np.linspace(0.0, 1.15 * E.max(), 2500)
    y = np.array([mp(x_) for x_ in x])
    fig, ax = plt.subplots()
    ax.hist(E,
            bins=cfg.INPUT.BINNING.BINS,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y, color='r', label='MP')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\mu(\lambda)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues.png')
    plt.close(fig)

    # Define the energy scale
    log.info("Defining the energy scale...")
    e_scale = cfg.INPUT.E_SCALE
    if e_scale.BY_MASS_SCALE.ENABLED:
        width = e_scale.BY_MASS_SCALE.WIDTH
        m2_top = 1 / (mp.max - width)
        m2_bot = 1 / (mp.max + width)
    elif e_scale.BY_VALUE.ENABLED:
        m2_top = e_scale.BY_VALUE.MAX
        m2_bot = e_scale.BY_VALUE.MIN
    log.debug(
        f"Energy scale of the integration:\nk2_max = {m2_top}\nk2_min = {m2_bot}"
    )

    # Define the distribution of the simulation
    log.info("Defining the distribution of the simulation...")
    dist = InterpolateDistribution(bins=cfg.INPUT.BINNING.BINS)
    dist = dist.fit(E_inv,
                    n=2,
                    s=cfg.INPUT.BINNING.SMOOTHING,
                    force_origin=True)
    dist_th = TranslatedInverseMarchenkoPastur(L=ratio)

    # Plot the distribution (inverse)
    log.info("Plotting the distribution of the simulation...")
    x = np.linspace(-0.1, 1.15 * E_inv.max(), 2500)
    y = np.array([dist(x_) for x_ in x])
    y_th = np.array([dist_th(x_) for x_ in x])
    fig, ax = plt.subplots()
    ax.hist(E_inv,
            bins=cfg.INPUT.BINNING.BINS**2,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y_th, color='r', label='inverse MP')
    ax.plot(x, y, color='k', label='interpolated')
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv.png')
    plt.close(fig)

    x = np.linspace(-0.1, 0.1 * E_inv.max(), 2500)
    y = np.array([dist(x_) for x_ in x])
    y_th = np.array([dist_th(x_) for x_ in x])
    fig, ax = plt.subplots()
    ax.hist(E_inv,
            bins=cfg.INPUT.BINNING.BINS**2,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y_th, color='r', label='inverse MP')
    ax.plot(x, y, color='k', label='interpolated')
    ax.set_xlim(0.0, 0.1 * E_inv.max())
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv_zoom.png')
    plt.close(fig)

    # If the simulation goes from IR to UV, then the distribution is in the
    # good direction. However, if the simulation goes from UV to IR, then we
    # need to reflect the distribution to go from high k (energy) to low k.
    if not cfg.SIM.IR_TO_UV:
        dist = SpecularReflection(dist, shift=np.sqrt(m2_top))

    # Define the grid of the simulation
    log.info("Defining the grid of the simulation...")
    grid = CartesianGrid(
        [[cfg.SIM.INF, cfg.SIM.SUP]],
        [cfg.SIM.N_VALUES],
        periodic=cfg.SIM.PERIODIC,
    )
    state = ScalarField.from_expression(grid, expr)
    bc = 'periodic' if cfg.SIM.PERIODIC else 'auto_periodic_neumann'

    # Storage
    if cfg.SIM.IR_TO_UV:
        t_range = [np.sqrt(m2_bot), np.sqrt(m2_top)]
    else:
        t_range = [0.0, np.sqrt(m2_top) - np.sqrt(m2_bot)]
    dt = (t_range[1] - t_range[0]) / cfg.SIM.N_STEPS
    storage = MemoryStorage()
    trackers = [storage.tracker(interval=dt)]

    # Run the simulation
    log.info("Running the simulation...")
    eq = SSD(dist, noise=0.0, bc=bc)
    _ = eq.solve(state, t_range=t_range, dt=dt, tracker=trackers)

    # Collect the results
    k_bar_list = []
    k_list = []
    Up_start_list = []
    Up_end_list = []
    kappa_bar_list = []
    mu_bar_0_list = []
    mu_bar_1_list = []
    kappa_list = []
    mu_0_list = []
    mu_1_list = []
    for k, Up_data in storage.items():

        # "Time" of the simulation
        if cfg.SIM.IR_TO_UV:
            k_bar_list.append(k)
        else:
            k_bar_list.append(np.sqrt(m2_top) - k)

        # Values of the potential
        Up = Up_data.data
        Up_start_list.append(Up[0])
        Up_end_list.append(Up[-1])
        log.debug(
            f"k = {k_bar_list[-1]:.5f} => U\'[{cfg.SIM.INF}] = {Up[0]:.3f}, U\'[{cfg.SIM.SUP}] = {Up[-1]:.3f}"
        )

        # Compute the parameters of the potential, using the expression
        #  $$
        #    \bar{\mathcal{U}}^{\prime}[\bar{\chi}]
        #    =
        #    \bar{\mu_0} (\bar{\chi} - \bar{\kappa})
        #    +
        #    \bar{\mu_1} (\bar{\chi} - \bar{\kappa})^2,
        #  $$
        #  where $\bar{\kappa}$ is the location of the minimum of the potential
        #  $\bar{\mathcal{U}}$.
        U = simpson(Up, grid.axes_coords[0])
        kappa_bar = grid.axes_coords[0][np.argmin(U)]
        kappa_bar_list.append(kappa_bar)

        dUp = np.gradient(Up, grid.axes_coords[0], edge_order=2)
        mu_bar_0 = dUp[np.argmin(U)]
        mu_bar_0_list.append(mu_bar_0)

        d2Up = np.gradient(dUp, grid.axes_coords[0], edge_order=2)
        mu_bar_1 = d2Up[np.argmin(U)] / 2
        mu_bar_1_list.append(mu_bar_1)

        log.debug(
            f"k = {k_bar_list[-1]:.5f} => kappa_bar = {kappa_bar:.3f}, mu_bar_0 = {mu_bar_0:.3f}, mu_bar_1 = {mu_bar_1:.3f}"
        )

        # Now convert to dimensional quantities
        # $$
        #   \kappa = \rho(k^2) \dot{s}^2 \bar{\kappa},
        #  \quad
        #  \mu_0 = \frac{k^2}{\rho(k^2) \dot{s}^2} \bar{\mu_0},
        #  \quad
        #  \mu_1 = \frac{k^2}{\rho(k^2)^2 \dot{s}^4} \bar{\mu_1}.
        # $$
        I = dist.integrate(0, k, moment=1, power=2)[0]

        kappa = kappa_bar * I**2 / k**4 / dist(k**2)
        mu_0 = mu_bar_0 * k**6 * dist(k**2) / I**2
        mu_1 = mu_bar_1 * k**10 * dist(k**2)**2 / I**4

        log.debug(
            f"k = {k_bar_list[-1]:.5f} => kappa = {kappa:.3f}, mu_0 = {mu_0:.3f}, mu_1 = {mu_1:.3f}"
        )

        if not np.isnan([kappa, mu_0, mu_1]).any():
            kappa_list.append(kappa)
            mu_0_list.append(mu_0)
            mu_1_list.append(mu_1)
            if cfg.SIM.IR_TO_UV:
                k_list.append(k)
            else:
                k_list.append(np.sqrt(m2_top) - k)
        else:
            log.warning("Found NaN values! Skipping...")

    # Add information to the sqlite database
    log.info("Filling database...")
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(cfg.OUTPUT.DB.OUTPUT_FILE) as conn:

            # Create a cursor
            cursor = conn.cursor()

            # Create the table if it does not exist
            table = cfg.OUTPUT.DB.TABLE
            sql_query = f"""CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATETIME,
            seed INTEGER,
            rows INTEGER,
            columns INTEGER,
            beta REAL,
            m2_bot REAL,
            m2_top REAL,
            Up_start TEXT,
            Up_end TEXT,
            kappa_bar TEXT,
            mu_bar_0 TEXT,
            mu_bar_1 TEXT,
            kappa TEXT,
            mu_0 TEXT,
            mu_1 TEXT
            )"""
            cursor.execute(sql_query)

            # Insert values
            sql_query = f"""INSERT INTO {table} (
            date,
            seed,
            rows,
            columns,
            beta,
            m2_bot,
            m2_top,
            Up_start,
            Up_end,
            kappa_bar,
            mu_bar_0,
            mu_bar_1,
            kappa,
            mu_0,
            mu_1
            ) VALUES (
            '{now}',
            {cfg.INPUT.MATRIX.SEED},
            {cfg.INPUT.MATRIX.ROWS},
            {cfg.INPUT.MATRIX.COLUMNS},
            {cfg.INPUT.SIGNAL.RATIO},
            {m2_bot},
            {m2_top},
            '{json.dumps(list(Up_start_list))}',
            '{json.dumps(list(Up_end_list))}',
            '{json.dumps(list(kappa_bar_list))}',
            '{json.dumps(list(mu_bar_0_list))}',
            '{json.dumps(list(mu_bar_1_list))}',
            '{json.dumps(list(kappa_list))}',
            '{json.dumps(list(mu_0_list))}',
            '{json.dumps(list(mu_1_list))}'
            )"""
            cursor.execute(sql_query)

    except sqlite3.Error as e:
        log.error(e)

    finally:
        conn.close()

    # Visualize the results
    log.info("Visualizing the results...")
    fig, ax = plt.subplots(ncols=3, nrows=5, figsize=(40, 18))
    ax[0, 0].plot(k_bar_list, Up_start_list, 'k-')
    ax[0, 0].plot([k_bar_list[0]], [Up_start_list[0]], 'ro')
    ax[0, 0].plot([k_bar_list[-1]], [Up_start_list[-1]], 'bo')
    ax[0, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[0, 0].set_xlabel('k')
    ax[0, 0].set_ylabel(
        rf'$\bar{{\mathcal{{U}}}}^{{\prime}}[\bar{{{cfg.SIM.INF}}}]$')
    ax[0, 1].plot(k_bar_list, Up_end_list, 'k-')
    ax[0, 1].plot([k_bar_list[0]], [Up_end_list[0]], 'ro')
    ax[0, 1].plot([k_bar_list[-1]], [Up_end_list[-1]], 'bo')
    ax[0, 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[0, 1].set_xlabel('k')
    ax[0, 1].set_ylabel(
        rf'$\bar{{\mathcal{{U}}}}^{{\prime}}[\bar{{{cfg.SIM.SUP}}}]$')
    ax[0, 2].plot(Up_start_list, Up_end_list, 'k-')
    ax[0, 2].plot([Up_start_list[0]], [Up_end_list[0]], 'ro')
    ax[0, 2].plot([Up_start_list[-1]], [Up_end_list[-1]], 'bo')
    ax[0, 2].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[0, 2].set_xlabel(
        rf'$\bar{{\mathcal{{U}}}}^{{\prime}}[\bar{{{cfg.SIM.INF}}}]$')
    ax[0, 2].set_ylabel(
        rf'$\bar{{\mathcal{{U}}}}^{{\prime}}[\bar{{{cfg.SIM.SUP}}}]$')
    ax[1, 0].plot(k_bar_list, kappa_bar_list, 'k-')
    ax[1, 0].plot([k_bar_list[0]], [kappa_bar_list[0]], 'ro')
    ax[1, 0].plot([k_bar_list[-1]], [kappa_bar_list[-1]], 'bo')
    ax[1, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[1, 0].set_xlabel('k')
    ax[1, 0].set_ylabel(r'$\bar{\kappa}$')
    ax[1, 1].plot(k_bar_list, mu_bar_0_list, 'k-')
    ax[1, 1].plot([k_bar_list[0]], [mu_bar_0_list[0]], 'ro')
    ax[1, 1].plot([k_bar_list[-1]], [mu_bar_0_list[-1]], 'bo')
    ax[1, 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[1, 1].set_xlabel('k')
    ax[1, 1].set_ylabel(r'$\bar{\mu}_0$')
    ax[1, 2].plot(k_bar_list, mu_bar_1_list, 'k-')
    ax[1, 2].plot([k_bar_list[0]], [mu_bar_1_list[0]], 'ro')
    ax[1, 2].plot([k_bar_list[-1]], [mu_bar_1_list[-1]], 'bo')
    ax[1, 2].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[1, 2].set_xlabel('k')
    ax[1, 2].set_ylabel(r'$\bar{\mu}_1$')
    ax[2, 0].plot(k_list, kappa_list, 'k-')
    ax[2, 0].plot([k_list[0]], [kappa_list[0]], 'ro')
    ax[2, 0].plot([k_list[-1]], [kappa_list[-1]], 'bo')
    ax[2, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[2, 0].set_xlabel('k')
    ax[2, 0].set_ylabel(r'$\kappa$')
    ax[2, 1].plot(k_list, mu_0_list, 'k-')
    ax[2, 1].plot([k_list[0]], [mu_0_list[0]], 'ro')
    ax[2, 1].plot([k_list[-1]], [mu_0_list[-1]], 'bo')
    ax[2, 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[2, 1].set_xlabel('k')
    ax[2, 1].set_ylabel(r'$\mu_0$')
    ax[2, 2].plot(k_list, mu_1_list, 'k-')
    ax[2, 2].plot([k_list[0]], [mu_1_list[0]], 'ro')
    ax[2, 2].plot([k_list[-1]], [mu_1_list[-1]], 'bo')
    ax[2, 2].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[2, 2].set_xlabel('k')
    ax[2, 2].set_ylabel(r'$\mu_1$')
    ax[3, 0].plot(kappa_bar_list, mu_bar_0_list, 'k-')
    ax[3, 0].plot([kappa_bar_list[0]], [mu_bar_0_list[0]], 'ro')
    ax[3, 0].plot([kappa_bar_list[-1]], [mu_bar_0_list[-1]], 'bo')
    ax[3, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[3, 0].set_xlabel(r'$\bar{\kappa}$')
    ax[3, 0].set_ylabel(r'$\bar{\mu}_0$')
    ax[3, 1].plot(kappa_bar_list, mu_bar_1_list, 'k-')
    ax[3, 1].plot([kappa_bar_list[0]], [mu_bar_1_list[0]], 'ro')
    ax[3, 1].plot([kappa_bar_list[-1]], [mu_bar_1_list[-1]], 'bo')
    ax[3, 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[3, 1].set_xlabel(r'$\bar{\kappa}$')
    ax[3, 1].set_ylabel(r'$\bar{\mu}_1$')
    ax[3, 2].plot(mu_bar_0_list, mu_bar_1_list, 'k-')
    ax[3, 2].plot([mu_bar_0_list[0]], [mu_bar_1_list[0]], 'ro')
    ax[3, 2].plot([mu_bar_0_list[-1]], [mu_bar_1_list[-1]], 'bo')
    ax[3, 2].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[3, 2].set_xlabel(r'$\bar{\mu}_0$')
    ax[3, 2].set_ylabel(r'$\bar{\mu}_1$')
    ax[4, 0].plot(kappa_list, mu_0_list, 'k-')
    ax[4, 0].plot([kappa_list[0]], [mu_0_list[0]], 'ro')
    ax[4, 0].plot([kappa_list[-1]], [mu_0_list[-1]], 'bo')
    ax[4, 0].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[4, 0].set_xlabel(r'$\kappa$')
    ax[4, 0].set_ylabel(r'$\mu_0$')
    ax[4, 1].plot(kappa_list, mu_1_list, 'k-')
    ax[4, 1].plot([kappa_list[0]], [mu_1_list[0]], 'ro')
    ax[4, 1].plot([kappa_list[-1]], [mu_1_list[-1]], 'bo')
    ax[4, 1].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[4, 1].set_xlabel(r'$\kappa$')
    ax[4, 1].set_ylabel(r'$\mu_1$')
    ax[4, 2].plot(mu_0_list, mu_1_list, 'k-')
    ax[4, 2].plot([mu_0_list[0]], [mu_1_list[0]], 'ro')
    ax[4, 2].plot([mu_0_list[-1]], [mu_1_list[-1]], 'bo')
    ax[4, 2].ticklabel_format(axis='both', style='sci', scilimits=(0, 0))
    ax[4, 2].set_xlabel(r'$\mu_0$')
    ax[4, 2].set_ylabel(r'$\mu_1$')
    plt.tight_layout()
    plt.savefig(output_dir / 'simulation.png')
    plt.close(fig)

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument('arguments',
                        nargs='*',
                        metavar='ARG',
                        help='configuration list')
    parser.add_argument('--log', type=str, default=None, help='log file')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='configuration file')
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s - v{__version__}')
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
