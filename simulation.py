# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Simulation of the stochastic signal detection mechanism.
"""
import argparse
import json
import sqlite3
import sys
from cProfile import label
from datetime import datetime
from pathlib import Path

import ffmpeg
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from pde import CartesianGrid, MemoryStorage, ScalarField
from PIL import Image
from scipy.integrate import simpson
from tqdm import tqdm

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


def U_fit_function(x: float, kappa: float, mu0: float, mu1: float,
                   mu2: float) -> float:
    """
    The fit of the potential function.

    Parameters
    ----------
    x : float
        The point at which to evaluate the function
    kappa : float
        The location of the first minimum of the potential
    mu0 : float
        The value of the first derivative of the potential at the minimum
    mu1 : float
        The value of the second derivative of the potential at the minimum
    mu2 : float
        The value of the third derivative of the potential at the minimum

    Returns
    -------
    float
        The value of the function at x
    """
    return mu0 * (x - kappa)**2 + mu1 * (x - kappa)**3 + mu2 * (x - kappa)**4


def Up_fit_function(x: float, kappa: float, mu0: float, mu1: float,
                    mu2: float) -> float:
    """
    The fit of the first derivative of the potential function.

    Parameters
    ----------
    x : float
        The point at which to evaluate the function
    kappa : float
        The location of the first minimum of the potential
    mu0 : float
        The value of the first derivative of the potential at the minimum
    mu1 : float
        The value of the second derivative of the potential at the minimum
    mu2 : float
        The value of the third derivative of the potential at the minimum

    Returns
    -------
    float
        The value of the function at x
    """
    return mu0 * (x-kappa) + mu1 * (x - kappa)**2 + mu2 * (x - kappa)**3


def main(args):

    # Organize the arguments
    log = logger(args.log)
    cfg = get_params(args.config, args.arguments)

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
        mu_bar_0_0 = init.BY_INIT.MU_0
        mu_bar_1_0 = init.BY_INIT.MU_1
        mu_bar_2_0 = init.BY_INIT.MU_2
        expr = f'{mu_bar_0_0} * (x - {kappa_bar_0}) + {mu_bar_1_0} * (x - {kappa_bar_0})**2 + {mu_bar_2_0} * (x - {kappa_bar_0})**3'
    elif init.BY_PARAMS.ENABLED:
        log.debug("Init by parameters...")
        mu_bar_0_0 = init.BY_PARAMS.MU_0
        mu_bar_1_0 = init.BY_PARAMS.MU_1
        mu_bar_2_0 = init.BY_PARAMS.MU_2
        mu_bar_3_0 = init.BY_PARAMS.MU_3
        expr = f'{mu_bar_0_0} + {mu_bar_1_0} * x + {mu_bar_2_0} * x**2 + {mu_bar_3_0} * x**3'
    elif init.BY_TEMP.ENABLED:
        log.debug("Init by temperature...")
        T = init.BY_TEMP.T
        TC = init.BY_TEMP.TC
        mu_bar_0_0 = T - TC
        mu_bar_1_0 = T
        mu_bar_2_0 = T**2
        mu_bar_3_0 = T**4
        expr = f'{mu_bar_0_0} + {mu_bar_1_0} * x + {mu_bar_2_0} * x**2 + {mu_bar_3_0} * x**3'
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
        log.debug("Using deterministic matrix...")
        S = create_signal(rows=matrix.ROWS,
                          columns=matrix.COLUMNS,
                          rank=signal.BY_DET.RANK,
                          random_state=matrix.SEED)
    elif signal.BY_IMG.ENABLED:
        log.debug("Using image as source of signal...")
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

    # Plot the signal matrix
    log.info("Plotting the signal matrix...")
    fig, ax = plt.subplots()
    ax.imshow(X, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / 'signal_matrix.png')
    plt.close(fig)

    # Compute the eigenvalues of the covariance matrix
    log.debug("Computing the eigenvalues of the covariance matrix...")
    E = np.linalg.eigvalsh(C)
    log.debug(f"Eigenvalues:\nlambda_max = {E.max()}\nlambda_min = {E.min()}")

    # Define the MP distribution associated to the background
    log.info("Defining the MP distribution associated to the background...")
    ratio = matrix.COLUMNS / matrix.ROWS
    mp = MarchenkoPastur(L=ratio)

    # Plot the distribution of the eigenvalues
    log.info("Plotting the distribution of the eigenvalues...")
    x = np.linspace(0.0, 1.05 * E.max(), 2500)
    y = np.array([mp(x_) for x_ in x])
    fig, ax = plt.subplots()
    ax.hist(E,
            bins=cfg.INPUT.BINNING.BINS,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y, 'r-', label='MP')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\mu(\lambda)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues.png')
    plt.close(fig)

    # # Compute the distance between consecutive eigenvalues, and if it is too
    # # large (given by a user defined threshold), then remove the corresponding
    # # eigenvalues.
    # log.info("Removing spikes (PCA)...")
    # E_spikes = np.where(np.diff(E) > cfg.SIM.EIGEN_THRESH)[0] + 1
    # if len(E_spikes) > 0:
    #     log.debug(f"Found {len(E_spikes)} spikes!")
    #     E = np.delete(E, E_spikes)
    #     log.debug(
    #         f"New eigenvalues:\nlambda_max = {E.max()}\nlambda_min = {E.min()}")

    #     # Plot the distribution of the eigenvalues
    #     log.info("Plotting the distribution of the eigenvalues...")
    #     x = np.linspace(0.0, 1.05 * E.max(), 2500)
    #     y = np.array([mp(x_) for x_ in x])
    #     fig, ax = plt.subplots()
    #     ax.hist(E,
    #             bins=cfg.INPUT.BINNING.BINS,
    #             density=True,
    #             color='b',
    #             alpha=0.5,
    #             label='eigenvalues')
    #     ax.plot(x, y, 'r-', label='MP')
    #     ax.set_xlabel(r'$\lambda$')
    #     ax.set_ylabel(r'$\mu(\lambda)$')
    #     ax.legend()
    #     plt.tight_layout()
    #     plt.savefig(output_dir / 'mp_eigenvalues_no_spikes.png')
    #     plt.close(fig)

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
    log.debug(
        f"Energy scale of the integration:\nk2_max = {m2_top}\nk2_min = {m2_bot}"
    )

    # Compute the inverse of the eigenvalues
    log.info("Computing the inverse of the eigenvalues...")
    E_inv = np.flip(1 / E)
    log.debug(f"Momenta:\nk2_max = {E_inv.max()}\nk2_min = {E_inv.min()}")
    E_inv -= E_inv.min()

    # Define the distribution of the simulation
    log.info("Defining the distribution of the simulation...")
    dist = InterpolateDistribution(bins=cfg.INPUT.BINNING.BINS**2)
    dist = dist.fit(E_inv,
                    n=2,
                    s=cfg.INPUT.BINNING.SMOOTHING,
                    force_origin=True)
    dist_th = TranslatedInverseMarchenkoPastur(L=ratio)

    # Plot the distribution (inverse)
    log.info("Plotting the distribution of the simulation...")
    x = np.linspace(0.0, E_inv.max(), 2500)
    y = np.array([dist(x_) for x_ in x])
    y_th = np.array([dist_th(x_) for x_ in x])

    fig, ax = plt.subplots()
    ax.hist(E_inv,
            bins=cfg.INPUT.BINNING.BINS**2,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y_th, 'r-', label='inverse MP')
    ax.plot(x, y, 'k-', label='interpolated')
    ax.set_xlim(-0.1, E_inv.max())
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv.png')
    plt.close(fig)

    x = np.linspace(0.0, 0.1 * E_inv.max(), 2500)
    y = np.array([dist(x_) for x_ in x])
    y_th = np.array([dist_th(x_) for x_ in x])

    fig, ax = plt.subplots()
    ax.hist(E_inv,
            bins=cfg.INPUT.BINNING.BINS**2,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y_th, 'r-', label='inverse MP')
    ax.plot(x, y, 'k-', label='interpolated')
    ax.set_xlim(-0.1, 0.1 * E_inv.max())
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv_zoom.png')
    plt.close(fig)

    x = np.linspace(0.0, 1.0, 2500)
    y = np.array([dist(x_) for x_ in x])
    y_th = np.array([dist_th(x_) for x_ in x])

    fig, ax = plt.subplots()
    ax.hist(E_inv,
            bins=cfg.INPUT.BINNING.BINS**2,
            density=True,
            color='b',
            alpha=0.5,
            label='eigenvalues')
    ax.plot(x, y_th, 'r-', label='inverse MP')
    ax.plot(x, y, 'k-', label='interpolated')
    ax.axvspan(m2_bot,
               m2_top,
               color='g',
               alpha=0.25,
               label='integration region')
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
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
    mu_bar_2_list = []
    kappa_list = []
    mu_0_list = []
    mu_1_list = []
    mu_2_list = []
    Up_list = []
    U_list = []
    items = tqdm(storage.items(),
                 unit='step(s)',
                 leave=False,
                 total=cfg.SIM.N_STEPS)
    for n, (k, Up_data) in enumerate(items):

        # "Time" of the simulation
        if cfg.SIM.IR_TO_UV:
            time = k
        else:
            time = np.sqrt(m2_top) - k
        k_bar_list.append(time)

        # Values of the potential
        Up = Up_data.data
        Up_list.append(list(Up))
        Up_start_list.append(Up[0])
        Up_end_list.append(Up[-1])

        # Compute the parameters of the potential, using the expression
        #  $$
        #    \overline{\mathcal{U}}^{\prime}[\overline{\chi}]
        #    =
        #    \overline{\mu_0} (\overline{\chi} - \overline{\kappa})
        #    +
        #    \overline{\mu_1} (\overline{\chi} - \overline{\kappa})^2,
        #    +
        #    \overline{\mu_2} (\overline{\chi} - \overline{\kappa})^3,
        #  $$
        #  where $\overline{\kappa}$ is the location of the minimum of the potential
        #  $\overline{\mathcal{U}}$.
        U = np.array([
            simpson(Up[:i], grid.axes_coords[0][:i]) for i in range(1, len(Up))
        ])
        U_list.append(list(U))
        kappa_bar = grid.axes_coords[0][np.argmin(U)]
        kappa_bar_list.append(kappa_bar)

        dUp = np.gradient(Up, grid.axes_coords[0], edge_order=2)
        mu_bar_0 = dUp[np.argmin(U)]
        mu_bar_0_list.append(mu_bar_0)

        d2Up = np.gradient(dUp, grid.axes_coords[0], edge_order=2)
        mu_bar_1 = d2Up[np.argmin(U)] / 2
        mu_bar_1_list.append(mu_bar_1)

        d3Up = np.gradient(d2Up, grid.axes_coords[0], edge_order=2)
        mu_bar_2 = d3Up[np.argmin(U)] / 6
        mu_bar_2_list.append(mu_bar_2)

        # Save the potential curve and its fit in an image object
        if cfg.OUTPUT.VIDEO_OUTPUT:
            U_fit = np.array([
                U_fit_function(x_, kappa_bar, mu_bar_0, mu_bar_1, mu_bar_2)
                for x_ in grid.axes_coords[0]
            ])
            fig, ax = plt.subplots()
            ax.plot(grid.axes_coords[0], U_fit, 'k-')
            ax.set_xlabel(r'$\overline{\chi}$')
            ax.set_ylabel(r'$\overline{\mathcal{U}}$')
            ax.set_xlim(cfg.SIM.INF, cfg.SIM.SUP)
            ax.ticklabel_format(axis='y',
                                style='sci',
                                scilimits=(0, 0),
                                useMathText=True)
            ax.set_title(rf'$k = {time:.5f}$')
            ax.set_yscale('symlog')
            plt.tight_layout()
            plt.savefig(output_dir / f'u_proj_{n:04d}.png')
            plt.close(fig)

            Up_fit = np.array([
                Up_fit_function(x_, kappa_bar, mu_bar_0, mu_bar_1, mu_bar_2)
                for x_ in grid.axes_coords[0]
            ])
            fig, ax = plt.subplots()
            ax.plot(grid.axes_coords[0], Up, 'k-', label='data')
            ax.plot(grid.axes_coords[0], Up_fit, 'r--', label='fit')
            ax.set_xlabel(r'$\overline{\chi}$')
            ax.set_ylabel(r'$\overline{\mathcal{U}}^{\prime}$')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            ax.set_xlim(cfg.SIM.INF, cfg.SIM.SUP)
            ax.ticklabel_format(axis='y',
                                style='sci',
                                scilimits=(0, 0),
                                useMathText=True)
            ax.set_title(rf'$k = {time:.5f}$')
            ax.set_yscale('symlog')
            plt.tight_layout()
            plt.savefig(output_dir / f'up_{n:04d}.png')
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.plot(grid.axes_coords[0][:-1], U, 'k-')
            ax.set_xlabel(r'$\overline{\chi}$')
            ax.set_ylabel(r'$\overline{\mathcal{U}}$')
            ax.set_xlim(cfg.SIM.INF, cfg.SIM.SUP)
            ax.ticklabel_format(axis='y',
                                style='sci',
                                scilimits=(0, 0),
                                useMathText=True)
            ax.set_title(rf'$k = {time:.5f}$')
            ax.set_yscale('symlog')
            plt.tight_layout()
            plt.savefig(output_dir / f'u_{n:04d}.png')
            plt.close(fig)

        # Now convert to dimensional quantities
        # $$
        #   \kappa = \rho(k^2) \dot{s}^2 \overline{\kappa},
        #  \quad
        #  \mu_0 = \frac{k^2}{\rho(k^2) \dot{s}^2} \overline{\mu_0},
        #  \quad
        #  \mu_1 = \frac{k^2}{\rho(k^2)^2 \dot{s}^4} \overline{\mu_1}.
        # $$
        I = dist.integrate(0, k, moment=1, power=2)[0]

        kappa = kappa_bar * I**2 / k**4 / dist(k**2)
        mu_0 = mu_bar_0 * k**6 * dist(k**2) / I**2
        mu_1 = mu_bar_1 * k**10 * dist(k**2)**2 / I**4
        mu_2 = mu_bar_2 * k**14 * dist(k**2)**3 / I**6

        if not np.isnan([kappa, mu_0, mu_1]).any():
            kappa_list.append(kappa)
            mu_0_list.append(mu_0)
            mu_1_list.append(mu_1)
            mu_2_list.append(mu_2)
            k_list.append(time)
        else:
            log.warning(
                "Found NaN values in the dimensional parameters! Skipping...")

    # Convert images to videos
    if cfg.OUTPUT.VIDEO_OUTPUT:
        log.info("Converting images to videos...")
        u_proj_video = ffmpeg.input(str(output_dir / 'u_proj_*.png'),
                                    pattern_type='glob',
                                    framerate=60)
        u_proj_video = u_proj_video.output(str(output_dir / 'u_proj.mp4'))
        u_proj_video.run(quiet=True, overwrite_output=True)
        up_video = ffmpeg.input(str(output_dir / 'up_*.png'),
                                pattern_type='glob',
                                framerate=60)
        up_video = up_video.output(str(output_dir / 'up.mp4'))
        up_video.run(quiet=True, overwrite_output=True)
        u_video = ffmpeg.input(str(output_dir / 'u_*.png'),
                               pattern_type='glob',
                               framerate=60)
        u_video = u_video.output(str(output_dir / 'u.mp4'))
        u_video.run(quiet=True, overwrite_output=True)

        # Remove temporary files
        log.info("Removing temporary files...")
        for f in output_dir.glob('u_proj_*.png'):
            f.unlink()
        for f in output_dir.glob('up_*.png'):
            f.unlink()
        for f in output_dir.glob('u_*.png'):
            f.unlink()

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
            kappa TEXT,
            mu_0 TEXT,
            mu_1 TEXT,
            mu_2 TEXT,
            Up TEXT,
            U TEXT
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
            kappa,
            mu_0,
            mu_1,
            mu_2,
            Up,
            U
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
            '{json.dumps(list(mu_bar_2_list))}',
            '{json.dumps(list(kappa_list))}',
            '{json.dumps(list(mu_0_list))}',
            '{json.dumps(list(mu_1_list))}',
            '{json.dumps(list(mu_2_list))}',
            '{json.dumps(list(Up_list))}',
            '{json.dumps(list(U_list))}'
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
    ax[0, 0].plot(k_bar_list, Up_start_list, 'k-')
    ax[0, 0].plot([k_bar_list[0]], [Up_start_list[0]], 'bo')
    ax[0, 0].plot([k_bar_list[-1]], [Up_start_list[-1]], 'ro')
    ax[0, 0].set_xlabel('k')
    ax[0, 0].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$')

    log.debug("Plotting the potential at ending point...")
    ax[0, 1].plot(k_bar_list, Up_end_list, 'k-')
    ax[0, 1].plot([k_bar_list[0]], [Up_end_list[0]], 'bo')
    ax[0, 1].plot([k_bar_list[-1]], [Up_end_list[-1]], 'ro')
    ax[0, 1].set_xlabel('k')
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
    ax[0, 3].plot(k_bar_list,
                  np.array(Up_end_list) / np.array(Up_start_list),
                  'k-')
    ax[0, 3].plot([k_bar_list[0]], [Up_end_list[0] / Up_start_list[0]], 'bo')
    ax[0, 3].plot([k_bar_list[-1]], [Up_end_list[-1] / Up_start_list[-1]], 'ro')
    ax[0, 3].set_xlabel('k')
    ax[0, 3].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}] / \overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$'
    )

    log.debug("Plotting \\overline{\\kappa}...")
    ax[1, 0].plot(k_bar_list, kappa_bar_list, 'k-')
    ax[1, 0].plot([k_bar_list[0]], [kappa_bar_list[0]], 'bo')
    ax[1, 0].plot([k_bar_list[-1]], [kappa_bar_list[-1]], 'ro')
    ax[1, 0].set_xlabel('k')
    ax[1, 0].set_ylabel(r'$\overline{\kappa}$')

    log.debug("Plotting \\overline{\\mu}_0...")
    ax[1, 1].plot(k_bar_list, mu_bar_0_list, 'k-')
    ax[1, 1].plot([k_bar_list[0]], [mu_bar_0_list[0]], 'bo')
    ax[1, 1].plot([k_bar_list[-1]], [mu_bar_0_list[-1]], 'ro')
    ax[1, 1].set_xlabel('k')
    ax[1, 1].set_ylabel(r'$\overline{\mu}_0$')

    log.debug("Plotting \\overline{\\mu}_1...")
    ax[1, 2].plot(k_bar_list, mu_bar_1_list, 'k-')
    ax[1, 2].plot([k_bar_list[0]], [mu_bar_1_list[0]], 'bo')
    ax[1, 2].plot([k_bar_list[-1]], [mu_bar_1_list[-1]], 'ro')
    ax[1, 2].set_xlabel('k')
    ax[1, 2].set_ylabel(r'$\overline{\mu}_1$')

    log.debug("Plotting \\overline{\\mu}_2...")
    ax[1, 3].plot(k_bar_list, mu_bar_2_list, 'k-')
    ax[1, 3].plot([k_bar_list[0]], [mu_bar_2_list[0]], 'bo')
    ax[1, 3].plot([k_bar_list[-1]], [mu_bar_2_list[-1]], 'ro')
    ax[1, 3].set_xlabel('k')
    ax[1, 3].set_ylabel(r'$\overline{\mu}_2$')

    log.debug("Plotting \\kappa...")
    ax[2, 0].plot(k_list, kappa_list, 'k-')
    ax[2, 0].plot([k_list[0]], [kappa_list[0]], 'bo')
    ax[2, 0].plot([k_list[-1]], [kappa_list[-1]], 'ro')
    ax[2, 0].set_xlabel('k')
    ax[2, 0].set_ylabel(r'$\kappa$')

    log.debug("Plotting \\mu_0...")
    ax[2, 1].plot(k_list, mu_0_list, 'k-')
    ax[2, 1].plot([k_list[0]], [mu_0_list[0]], 'bo')
    ax[2, 1].plot([k_list[-1]], [mu_0_list[-1]], 'ro')
    ax[2, 1].set_xlabel('k')
    ax[2, 1].set_ylabel(r'$\mu_0$')

    log.debug("Plotting \\mu_1...")
    ax[2, 2].plot(k_list, mu_1_list, 'k-')
    ax[2, 2].plot([k_list[0]], [mu_1_list[0]], 'bo')
    ax[2, 2].plot([k_list[-1]], [mu_1_list[-1]], 'ro')
    ax[2, 2].set_xlabel('k')
    ax[2, 2].set_ylabel(r'$\mu_1$')

    log.debug("Plotting \\mu_2...")
    ax[2, 3].plot(k_list, mu_2_list, 'k-')
    ax[2, 3].plot([k_list[0]], [mu_2_list[0]], 'bo')
    ax[2, 3].plot([k_list[-1]], [mu_2_list[-1]], 'ro')
    ax[2, 3].set_xlabel('k')
    ax[2, 3].set_ylabel(r'$\mu_2$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{{\\mu}}_0..."
    )
    ax[3, 0].plot(kappa_bar_list, mu_bar_0_list, 'k-')
    ax[3, 0].plot([kappa_bar_list[0]], [mu_bar_0_list[0]], 'bo')
    ax[3, 0].plot([kappa_bar_list[-1]], [mu_bar_0_list[-1]], 'ro')
    ax[3, 0].set_xlabel(r'$\overline{\kappa}$')
    ax[3, 0].set_ylabel(r'$\overline{\mu}_0$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{{\\mu}}_1..."
    )
    ax[3, 1].plot(kappa_bar_list, mu_bar_1_list, 'k-')
    ax[3, 1].plot([kappa_bar_list[0]], [mu_bar_1_list[0]], 'bo')
    ax[3, 1].plot([kappa_bar_list[-1]], [mu_bar_1_list[-1]], 'ro')
    ax[3, 1].set_xlabel(r'$\overline{\kappa}$')
    ax[3, 1].set_ylabel(r'$\overline{\mu}_1$')

    log.debug(
        "Plotting phase space curve \\overline{{\\mu}}_0 vs \\overline{{\\mu}}_1..."
    )
    ax[3, 2].plot(mu_bar_0_list, mu_bar_1_list, 'k-')
    ax[3, 2].plot([mu_bar_0_list[0]], [mu_bar_1_list[0]], 'bo')
    ax[3, 2].plot([mu_bar_0_list[-1]], [mu_bar_1_list[-1]], 'ro')
    ax[3, 2].set_xlabel(r'$\overline{\mu}_0$')
    ax[3, 2].set_ylabel(r'$\overline{\mu}_1$')

    log.debug(
        "Plotting phase space curve \\overline{{\\mu}}_0 vs \\overline{{\\mu}}_2..."
    )
    ax[3, 3].plot(mu_bar_0_list, mu_bar_2_list, 'k-')
    ax[3, 3].plot([mu_bar_0_list[0]], [mu_bar_2_list[0]], 'bo')
    ax[3, 3].plot([mu_bar_0_list[-1]], [mu_bar_2_list[-1]], 'ro')
    ax[3, 3].set_xlabel(r'$\overline{\mu}_0$')
    ax[3, 3].set_ylabel(r'$\overline{\mu}_2$')

    log.debug("Plotting phase space curve \\kappa vs \\mu_0...")
    ax[4, 0].plot(kappa_list, mu_0_list, 'k-')
    ax[4, 0].plot([kappa_list[0]], [mu_0_list[0]], 'bo')
    ax[4, 0].plot([kappa_list[-1]], [mu_0_list[-1]], 'ro')
    ax[4, 0].set_xlabel(r'$\kappa$')
    ax[4, 0].set_ylabel(r'$\mu_0$')

    log.debug("Plotting phase space curve \\kappa vs \\mu_1...")
    ax[4, 1].plot(kappa_list, mu_1_list, 'k-')
    ax[4, 1].plot([kappa_list[0]], [mu_1_list[0]], 'bo')
    ax[4, 1].plot([kappa_list[-1]], [mu_1_list[-1]], 'ro')
    ax[4, 1].set_xlabel(r'$\kappa$')
    ax[4, 1].set_ylabel(r'$\mu_1$')

    log.debug("Plotting phase space curve \\mu_0 vs \\mu_1...")
    ax[4, 2].plot(mu_0_list, mu_1_list, 'k-')
    ax[4, 2].plot([mu_0_list[0]], [mu_1_list[0]], 'bo')
    ax[4, 2].plot([mu_0_list[-1]], [mu_1_list[-1]], 'ro')
    ax[4, 2].set_xlabel(r'$\mu_0$')
    ax[4, 2].set_ylabel(r'$\mu_1$')

    log.debug("Plotting phase space curve \\mu_0 vs \\mu_2...")
    ax[4, 3].plot(mu_0_list, mu_2_list, 'k-')
    ax[4, 3].plot([mu_0_list[0]], [mu_2_list[0]], 'bo')
    ax[4, 3].plot([mu_0_list[-1]], [mu_2_list[-1]], 'ro')
    ax[4, 3].set_xlabel(r'$\mu_0$')
    ax[4, 3].set_ylabel(r'$\mu_2$')

    ax = ax.flatten()
    for i in range(len(ax)):
        ax[i].ticklabel_format(axis='y',
                               style='sci',
                               scilimits=(0, 0),
                               useMathText=True)
        ax[i].set_yscale('symlog')

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
