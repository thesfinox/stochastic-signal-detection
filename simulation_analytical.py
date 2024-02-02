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
from pde import CartesianGrid, ScalarField
from PIL import Image

from ssd import __version__
from ssd.distributions import (InterpolateDistribution,
                               MarchenkoPastur,
                               SpecularReflection,
                               TranslatedInverseMarchenkoPastur)
from ssd.ssd import SSD
from ssd.utils.cfg import get_params, logger, print_config
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
    cfg = get_params(args.config, args.arguments)
    if args.print_config:
        print_config(cfg)
        sys.exit(0)
    log = logger(args.log)

    # Create the output directory
    output_dir = Path(cfg.OUTPUT.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the actual configuration in the output directory
    log.info("Saving the configuration...")
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        f.write(cfg.dump())

    # Define the initial conditions
    log.info("Defining the initial conditions...")
    init = cfg.INPUT.INIT
    if init.BY_INIT.ENABLED:
        log.debug("Init by initial conditions...")
        kappa_bar_0 = init.BY_INIT.KAPPA_0
        mu_bar_0_0 = init.BY_INIT.MU_4
        mu_bar_1_0 = init.BY_INIT.MU_6
        mu_bar_2_0 = init.BY_INIT.MU_8
        expr = f'{mu_bar_0_0} * (x - {kappa_bar_0}) + {mu_bar_1_0} * (x - {kappa_bar_0})**2 + {mu_bar_2_0} * (x - {kappa_bar_0})**3'
        pot = f'{mu_bar_0_0} * (x - {kappa_bar_0})**2 / 2 + {mu_bar_1_0} * (x - {kappa_bar_0})**3 / 3 + {mu_bar_2_0} * (x - {kappa_bar_0})**4 / 4'
    elif init.BY_PARAMS.ENABLED:
        log.debug("Init by parameters...")
        mu_bar_0 = init.BY_PARAMS.MU_0
        mu_bar_0 = init.BY_PARAMS.MU_1
        mu_bar_0 = init.BY_PARAMS.MU_2
        mu_bar_0 = init.BY_PARAMS.MU_3
        expr = f'{mu_bar_0_0} + {mu_bar_1_0} * x + {mu_bar_2_0} * x**2 + {mu_bar_3_0} * x**3'
        pot = f'{mu_bar_0_0} * x + {mu_bar_1_0} * x**2 / 2 + {mu_bar_2_0} * x**3 / 3 + {mu_bar_3_0} * x**4 / 4'
    elif init.BY_TEMP.ENABLED:
        log.debug("Init by temperature...")
        T = init.BY_TEMP.T
        TC = init.BY_TEMP.TC
        mu_bar_0_0 = T - TC
        mu_bar_1_0 = T
        mu_bar_2_0 = T**2
        mu_bar_3_0 = T**3
        expr = f'{mu_bar_0_0} + {mu_bar_1_0} * x + {mu_bar_2_0} * x**2 + {mu_bar_3_0} * x**3'
        pot = f'{mu_bar_0_0} * x + {mu_bar_1_0} * x**2 / 2 + {mu_bar_2_0} * x**3 / 3 + {mu_bar_3_0} * x**4 / 4'
    else:
        raise ValueError('No valid initial conditions defined!')
    log.debug(f"Initial conditions:\nU' = {expr}\nU = {pot}")

    # Define the MP distribution associated to the background
    log.info("Defining the MP distribution associated to the background...")
    matrix = cfg.INPUT.MATRIX
    ratio = matrix.COLUMNS / matrix.ROWS
    mp = MarchenkoPastur(L=ratio)

    # Plot the distribution of the eigenvalues
    log.info("Plotting the distribution of the eigenvalues...")
    x = np.linspace(0.0, 4.0, 2500)
    y = np.array([mp(x_) for x_ in x])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'r-', label='MP')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\mu(\lambda)$')
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues.png')
    plt.close(fig)

    # Define the energy scale
    log.info("Defining the energy scale...")
    e_scale = cfg.INPUT.E_SCALE
    if e_scale.BY_MASS_SCALE.ENABLED:
        log.debug("Using the mass scale with fixed width as energy scale...")
        width = e_scale.BY_MASS_SCALE.WIDTH
        m2_top = 1 / (mp.max - width)
        m2_bot = 1 / (mp.max + width)
    elif e_scale.BY_VALUE.ENABLED:
        log.debug("Using a fixed energy scale...")
        m2_top = e_scale.BY_VALUE.MAX
        m2_bot = e_scale.BY_VALUE.MIN
    elif e_scale.BY_ENDPOINT.ENABLED:
        log.debug("Using the endpoint as energy scale...")
        width = e_scale.BY_ENDPOINT.WIDTH
        eps = e_scale.BY_ENDPOINT.EPSILON
        m2_top = 1 / (mp.max + width)
        m2_bot = eps
    elif e_scale.BY_MAX.ENABLED:
        log.debug("Using the maximum of the inverse MP as energy scale...")
        m2_top = 1 / mp.max
        m2_bot = 0.0
    else:
        raise ValueError('No valid energy scale defined!')

    # Define the distribution of the simulation
    log.info("Defining the distribution of the simulation...")
    dist = TranslatedInverseMarchenkoPastur(L=ratio)

    # Plot the distribution (inverse)
    log.info("Plotting the distribution of the simulation...")
    x = np.linspace(0.0, 3.0, 2500)
    y = np.array([dist(x_) for x_ in x])

    # Redefine the mass scale if the simulations uses the max of the inverse
    # MP distribution as top energy scale
    if e_scale.BY_MAX.ENABLED:
        log.debug(
            'Redefining the energy scale (by the max of the inverse MP)...')
        x_zoom = np.linspace(0.0, 1.0, 2500)
        y_zoom = np.array([dist(x_) for x_ in x_zoom])
        m2_top = x_zoom[y_zoom.argmax()]
        m2_bot = e_scale.BY_MAX.EPSILON
    log.debug(
        f"Energy scale of the integration:\nk2_max = {m2_top}\nk2_min = {m2_bot}"
    )

    fig, ax = plt.subplots()
    ax.plot(x, y, 'r-', label='inverse MP')
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv.png')
    plt.close(fig)

    x = np.linspace(0.0, 1.0, 2500)
    y = np.array([dist(x_) for x_ in x])

    fig, ax = plt.subplots()
    ax.plot(x, y, 'r-', label='inverse MP')
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv_zoom.png')
    plt.close(fig)

    x = np.linspace(0.0, 1.0, 2500)
    y = np.array([dist(x_) for x_ in x])

    fig, ax = plt.subplots()
    ax.plot(x, y, 'r-', label='inverse MP')
    ax.axvspan(m2_bot,
               m2_top,
               color='g',
               alpha=0.25,
               label='integration region')
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv_zoom2.png')
    plt.close(fig)

    # If the simulation goes from IR to UV, then the distribution is in the
    # good direction. However, if the simulation goes from UV to IR, then we
    # need to reflect the distribution to go from high k (energy) to low k.
    if not cfg.SIM.IR_TO_UV:
        dist = SpecularReflection(dist, shift=m2_top)

        # Plot the distribution (reflected)
        log.info("Plotting the distribution of the simulation (reflected)...")
        x = np.linspace(-m2_top, m2_top, 2500)
        y = np.array([dist(x_) for x_ in x])

        fig, ax = plt.subplots()
        ax.plot(x, y, 'k-', label='distribution')
        ax.set_xlim(-m2_top, m2_top)
        ax.set_xlabel(r'$k^2$')
        ax.set_ylabel(r'$\rho$')
        ax.axvspan(0.0,
                   m2_top - m2_bot,
                   color='g',
                   alpha=0.25,
                   label='integration region')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'mp_eigenvalues_reflected.png')
        plt.close(fig)

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
        t_range = [m2_bot, m2_top]
    else:
        t_range = [0.0, m2_top - m2_bot]
    dt = (t_range[1] - t_range[0]) / cfg.SIM.N_STEPS

    # Run the simulation
    log.info("Running the simulation...")
    sign = int(cfg.SIM.SIGN > 0) - int(cfg.SIM.SIGN < 0)
    eq = SSD(dist, noise=0.0, sign=sign, bc=bc)
    _ = eq.solve(state, t_range=t_range, dt=dt, tracker=['progress'])

    # Save the results
    Up_list = eq.Up_
    k2_list = eq.k2_
    if not cfg.SIM.IR_TO_UV:
        k2_list = list(m2_top - np.array(k2_list))

    # Make a list of the initial points and the end points of Up
    Up_start_list = [Up[0] for Up in Up_list]
    Up_end_list = [Up[-1] for Up in Up_list]

    # Collect the lists of dimensions
    dimUp = eq.dimUp_
    dimChi = eq.dimChi_

    # Compute the dimensions of the operators
    log.info("Computing the dimensions of the operators...")
    dim_kappa_bar_list = list(-np.array(dimChi))
    dim_mu_4_bar_list = list(np.array(dimChi) - np.array(dimUp))
    dim_mu_6_bar_list = list(2 * np.array(dimChi) - np.array(dimUp))
    dim_mu_8_bar_list = list(3 * np.array(dimChi) - np.array(dimUp))

    # Collect the lists of couplings
    kappa_bar_list = eq.kappa_bar_
    mu_4_bar_list = eq.mu_4_bar_
    mu_6_bar_list = eq.mu_6_bar_
    mu_8_bar_list = eq.mu_8_bar_

    # Add information to the sqlite database
    log.info("Filling database...")
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with sqlite3.connect(cfg.OUTPUT.DB.OUTPUT_FILE) as conn:

            # Create a cursor
            cursor = conn.cursor()

            # Create the table if it does not exist
            log.debug("Creating the table if it does not exist...")
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
            mu_bar_2 TEXT,
            dim_kappa_bar TEXT,
            dim_mu_4_bar TEXT,
            dim_mu_6_bar TEXT,
            dim_mu_8_bar TEXT,
            k2_list TEXT,
            Up TEXT
            )"""
            cursor.execute(sql_query)

            # Insert values
            log.debug("Inserting values...")
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
            mu_bar_2,
            dim_kappa_bar,
            dim_mu_4_bar,
            dim_mu_6_bar,
            dim_mu_8_bar,
            k2_list,
            Up
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
            '{json.dumps(list(mu_4_bar_list))}',
            '{json.dumps(list(mu_6_bar_list))}',
            '{json.dumps(list(mu_8_bar_list))}',
            '{json.dumps(list(dim_kappa_bar_list))}',
            '{json.dumps(list(dim_mu_4_bar_list))}',
            '{json.dumps(list(dim_mu_6_bar_list))}',
            '{json.dumps(list(dim_mu_8_bar_list))}',
            '{json.dumps(list(k2_list))}',
            '{json.dumps(list(Up_list))}'
            )"""
            cursor.execute(sql_query)

    except sqlite3.Error as e:
        log.error(e)

    finally:
        conn.close()

    # Visualize the results
    log.info("Visualizing the results...")
    fig, ax = plt.subplots(ncols=4, nrows=5, figsize=(40, 24))

    log.debug("Plotting the potential at starting point...")
    ax[0, 0].plot(k2_list, Up_start_list, 'k-')
    ax[0, 0].plot([k2_list[0]], [Up_start_list[0]], 'bo')
    ax[0, 0].plot([k2_list[-1]], [Up_start_list[-1]], 'ro')
    ax[0, 0].set_xlabel(r'$k^2$')
    ax[0, 0].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$')

    log.debug("Plotting the potential at ending point...")
    ax[0, 1].plot(k2_list, Up_end_list, 'k-')
    ax[0, 1].plot([k2_list[0]], [Up_end_list[0]], 'bo')
    ax[0, 1].plot([k2_list[-1]], [Up_end_list[-1]], 'ro')
    ax[0, 1].set_xlabel(r'$k^2$')
    ax[0, 1].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}]$')

    log.debug("Plotting potential curve at starting and ending points...")
    ax[0, 2].plot(Up_start_list, Up_end_list, 'k-')
    ax[0, 2].plot([Up_start_list[0]], [Up_end_list[0]], 'bo')
    ax[0, 2].plot([Up_start_list[-1]], [Up_end_list[-1]], 'ro')
    ax[0, 2].set_xlabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$')
    ax[0, 2].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}]$')

    log.debug("Plotting ratio of starting and ending points...")
    ax[0, 3].plot(k2_list,
                  np.array(Up_end_list) / np.array(Up_start_list),
                  'k-')
    ax[0, 3].plot([k2_list[0]], [Up_end_list[0] / Up_start_list[0]], 'bo')
    ax[0, 3].plot([k2_list[-1]], [Up_end_list[-1] / Up_start_list[-1]], 'ro')
    ax[0, 3].set_xlabel(r'$k^2$')
    ax[0, 3].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}] / \overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$'
    )

    log.debug("Plotting \\overline{\\kappa}...")
    ax[1, 0].plot(k2_list, kappa_bar_list, 'k-')
    ax[1, 0].plot([k2_list[0]], [kappa_bar_list[0]], 'bo')
    ax[1, 0].plot([k2_list[-1]], [kappa_bar_list[-1]], 'ro')
    ax[1, 0].set_xlabel(r'$k^2$')
    ax[1, 0].set_ylabel(r'$\overline{\kappa}$')

    log.debug("Plotting \\overline{\\mu}_0...")
    ax[1, 1].plot(k2_list, mu_4_bar_list, 'k-')
    ax[1, 1].plot([k2_list[0]], [mu_4_bar_list[0]], 'bo')
    ax[1, 1].plot([k2_list[-1]], [mu_4_bar_list[-1]], 'ro')
    ax[1, 1].set_xlabel(r'$k^2$')
    ax[1, 1].set_ylabel(r'$\overline{\mu}_4$')

    log.debug("Plotting \\overline{\\mu}_1...")
    ax[1, 2].plot(k2_list, mu_6_bar_list, 'k-')
    ax[1, 2].plot([k2_list[0]], [mu_6_bar_list[0]], 'bo')
    ax[1, 2].plot([k2_list[-1]], [mu_6_bar_list[-1]], 'ro')
    ax[1, 2].set_xlabel(r'$k^2$')
    ax[1, 2].set_ylabel(r'$\overline{\mu}_6$')

    log.debug("Plotting \\overline{\\mu}_2...")
    ax[1, 3].plot(k2_list, mu_8_bar_list, 'k-')
    ax[1, 3].plot([k2_list[0]], [mu_8_bar_list[0]], 'bo')
    ax[1, 3].plot([k2_list[-1]], [mu_8_bar_list[-1]], 'ro')
    ax[1, 3].set_xlabel(r'$k^2$')
    ax[1, 3].set_ylabel(r'$\overline{\mu}_8$')

    log.debug("Plotting dimension of \\kappa...")
    ax[2, 0].plot(k2_list, dim_kappa_bar_list, 'k-')
    ax[2, 0].plot([k2_list[0]], [dim_kappa_bar_list[0]], 'bo')
    ax[2, 0].plot([k2_list[-1]], [dim_kappa_bar_list[-1]], 'ro')
    ax[2, 0].set_xlabel(r'$k^2$')
    ax[2, 0].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\kappa}$')

    log.debug("Plotting dimension of \\mu_4...")
    ax[2, 1].plot(k2_list, dim_mu_4_bar_list, 'k-')
    ax[2, 1].plot([k2_list[0]], [dim_mu_4_bar_list[0]], 'bo')
    ax[2, 1].plot([k2_list[-1]], [dim_mu_4_bar_list[-1]], 'ro')
    ax[2, 1].set_xlabel(r'$k^2$')
    ax[2, 1].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')

    log.debug("Plotting dimension of \\mu_6...")
    ax[2, 2].plot(k2_list, dim_mu_6_bar_list, 'k-')
    ax[2, 2].plot([k2_list[0]], [dim_mu_6_bar_list[0]], 'bo')
    ax[2, 2].plot([k2_list[-1]], [dim_mu_6_bar_list[-1]], 'ro')
    ax[2, 2].set_xlabel(r'$k^2$')
    ax[2, 2].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')

    log.debug("Plotting dimension of \\mu_8...")
    ax[2, 3].plot(k2_list, dim_mu_8_bar_list, 'k-')
    ax[2, 3].plot([k2_list[0]], [dim_mu_8_bar_list[0]], 'bo')
    ax[2, 3].plot([k2_list[-1]], [dim_mu_8_bar_list[-1]], 'ro')
    ax[2, 3].set_xlabel(r'$k^2$')
    ax[2, 3].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_8$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{{\\mu}}_0..."
    )
    ax[3, 0].plot(kappa_bar_list, mu_4_bar_list, 'k-')
    ax[3, 0].plot([kappa_bar_list[0]], [mu_4_bar_list[0]], 'bo')
    ax[3, 0].plot([kappa_bar_list[-1]], [mu_4_bar_list[-1]], 'ro')
    ax[3, 0].set_xlabel(r'$\overline{\kappa}$')
    ax[3, 0].set_ylabel(r'$\overline{\mu}_4$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{{\\mu}}_1..."
    )
    ax[3, 1].plot(kappa_bar_list, mu_6_bar_list, 'k-')
    ax[3, 1].plot([kappa_bar_list[0]], [mu_6_bar_list[0]], 'bo')
    ax[3, 1].plot([kappa_bar_list[-1]], [mu_6_bar_list[-1]], 'ro')
    ax[3, 1].set_xlabel(r'$\overline{\kappa}$')
    ax[3, 1].set_ylabel(r'$\overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{{\\mu}}_4 vs \\overline{{\\mu}}_6..."
    )
    ax[3, 2].plot(mu_4_bar_list, mu_6_bar_list, 'k-')
    ax[3, 2].plot([mu_4_bar_list[0]], [mu_6_bar_list[0]], 'bo')
    ax[3, 2].plot([mu_4_bar_list[-1]], [mu_6_bar_list[-1]], 'ro')
    ax[3, 2].set_xlabel(r'$\overline{\mu}_4$')
    ax[3, 2].set_ylabel(r'$\overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{{\\mu}}_4 vs \\overline{{\\mu}}_8..."
    )
    ax[3, 3].plot(mu_4_bar_list, mu_8_bar_list, 'k-')
    ax[3, 3].plot([mu_4_bar_list[0]], [mu_8_bar_list[0]], 'bo')
    ax[3, 3].plot([mu_4_bar_list[-1]], [mu_8_bar_list[-1]], 'ro')
    ax[3, 3].set_xlabel(r'$\overline{\mu}_4$')
    ax[3, 3].set_ylabel(r'$\overline{\mu}_8$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{\\mu_4}..."
    )
    ax[4, 0].plot(dim_kappa_bar_list, dim_mu_4_bar_list, 'k-')
    ax[4, 0].plot([dim_kappa_bar_list[0]], [dim_mu_4_bar_list[0]], 'bo')
    ax[4, 0].plot([dim_kappa_bar_list[-1]], [dim_mu_4_bar_list[-1]], 'ro')
    ax[4, 0].set_xlabel(r'$-\mathrm{dim}_{k}\, \overline{\kappa}$')
    ax[4, 0].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{\\mu_6}..."
    )
    ax[4, 1].plot(dim_kappa_bar_list, dim_mu_6_bar_list, 'k-')
    ax[4, 1].plot([dim_kappa_bar_list[0]], [dim_mu_6_bar_list[0]], 'bo')
    ax[4, 1].plot([dim_kappa_bar_list[-1]], [dim_mu_6_bar_list[-1]], 'ro')
    ax[4, 1].set_xlabel(r'$-\mathrm{dim}_{k}\, \kappa$')
    ax[4, 1].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{\\mu_4} vs \\overline{\\mu_6}..."
    )
    ax[4, 2].plot(dim_mu_4_bar_list, dim_mu_6_bar_list, 'k-')
    ax[4, 2].plot([dim_mu_4_bar_list[0]], [dim_mu_6_bar_list[0]], 'bo')
    ax[4, 2].plot([dim_mu_4_bar_list[-1]], [dim_mu_6_bar_list[-1]], 'ro')
    ax[4, 2].set_xlabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')
    ax[4, 2].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{\\mu_4} vs \\overline{\\mu_8}..."
    )
    ax[4, 3].plot(dim_mu_4_bar_list, dim_mu_8_bar_list, 'k-')
    ax[4, 3].plot([dim_mu_4_bar_list[0]], [dim_mu_8_bar_list[0]], 'bo')
    ax[4, 3].plot([dim_mu_4_bar_list[-1]], [dim_mu_8_bar_list[-1]], 'ro')
    ax[4, 3].set_xlabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')
    ax[4, 3].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_8$')

    ax = ax.flatten()
    for i in range(len(ax)):
        ax[i].ticklabel_format(axis='both',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        if i != 4:
            ax[i].set_yscale('symlog')

    plt.tight_layout()
    plt.savefig(output_dir / 'simulation.png')
    plt.close(fig)

    # Plot the dimensions of the parameters on the same plot
    fig, ax = plt.subplots()
    # ax.plot(k_list,
    #         dim_kappa_bar_list,
    #         'k-',
    #         label=r'$\mathrm{dim}_{k}\, \overline{\kappa}$')
    # ax.plot([k_list[0]], [dim_kappa_bar_list[0]], 'bo')
    # ax.plot([k_list[-1]], [dim_kappa_bar_list[-1]], 'ro')
    ax.plot(k2_list,
            dim_mu_4_bar_list,
            'r-',
            label=r'$\mathrm{dim}_{k}\, \overline{\mu}_4$')
    ax.plot([k2_list[0]], [dim_mu_4_bar_list[0]], 'bo')
    ax.plot([k2_list[-1]], [dim_mu_4_bar_list[-1]], 'ro')
    ax.plot(k2_list,
            dim_mu_6_bar_list,
            'g-',
            label=r'$\mathrm{dim}_{k}\, \overline{\mu}_6$')
    ax.plot([k2_list[0]], [dim_mu_6_bar_list[0]], 'bo')
    ax.plot([k2_list[-1]], [dim_mu_6_bar_list[-1]], 'ro')
    ax.plot(k2_list,
            dim_mu_8_bar_list,
            'b-',
            label=r'$\mathrm{dim}_{k}\, \overline{\mu}_8$')
    ax.plot([k2_list[0]], [dim_mu_8_bar_list[0]], 'bo')
    ax.plot([k2_list[-1]], [dim_mu_8_bar_list[-1]], 'ro')
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel('-dim')
    ax.ticklabel_format(axis='both',
                        style='sci',
                        scilimits=(0, 0),
                        useMathText=True)
    ax.set_yscale('symlog')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_dim_param.png')

    # Record the last expression
    kappa_bar = kappa_bar_list[-1]
    mu_bar_0 = mu_4_bar_list[-1]
    mu_bar_1 = mu_6_bar_list[-1]
    mu_bar_2 = mu_8_bar_list[-1]
    expr = f'{mu_bar_0} * (x - {kappa_bar}) + {mu_bar_1} * (x - {kappa_bar})**2 + {mu_bar_2} * (x - {kappa_bar})**3'
    pot = f'{mu_bar_0} * (x - {kappa_bar})**2 / 2 + {mu_bar_1} * (x - {kappa_bar})**3 / 3 + {mu_bar_2} * (x - {kappa_bar})**4 / 4'
    log.info(f'Final expression:\nU\' = {expr}\nU = {pot}')

    log.info("Done!")
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__description__,
                                     epilog=__epilog__)
    parser.add_argument(
        'arguments',
        nargs='*',
        metavar='ARG',
        help='configuration arguments (KEY1 VALUE1 KEY2 VALUE2...)')
    parser.add_argument('--log', type=str, default=None, help='log file')
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument('--config', type=str, help='configuration file')
    config.add_argument('--print-config',
                        action='store_true',
                        help='print the configuration and exit')
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s - v{__version__}')
    args = parser.parse_args()

    code = main(args)

    sys.exit(code)
