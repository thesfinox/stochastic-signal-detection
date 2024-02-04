# -*- coding: utf-8 -*-
"""
Utilities

Utility functions for the Stochastic Signal Detection project.
"""
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from pde import CartesianGrid, ScalarField
from PIL import Image
from scipy.interpolate import splev, splrep
from yacs.config import CfgNode as CN

from ssd.base.base import BaseDistribution
from ssd.distributions import (InterpolateDistribution, MarchenkoPastur,
                               SpecularReflection,
                               TranslatedInverseMarchenkoPastur)
from ssd.ssd import SSD
from ssd.utils.cfg import get_params, logger, print_config
from ssd.utils.matrix import create_bulk, create_signal


def nan_to_num(x: np.ndarray) -> np.ndarray:
    """
    Substitute NaNs with the interpolation of the previous two values.

    Parameters
    ----------
    x : np.ndarray
        Array with NaNs.

    Returns
    -------
    np.ndarray
        Array with NaNs substituted by the interpolation.
    """
    x = np.array(x).copy()
    if np.isnan(x).any():
        if np.isnan(x[1]):
            x[1] = x[2]
            return nan_to_num(x)
        if np.isnan(x[0]):
            x[0] = x[1]
            return nan_to_num(x)
        for i in range(2, len(x)):
            if np.isnan(x[i]):
                slope = x[i-1] - x[i-2]
                x[i] = x[i-1] + slope
    return x


def fit_spline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit a cubic spline to the given data.

    Parameters
    ----------
    x : np.ndarray
        The x values.
    y : np.ndarray
        The y values.

    Returns
    -------
    np.ndarray
        The y values of the spline.
    """
    x = np.array(x).copy()
    y = np.array(y).copy()
    tck = splrep(x, y, k=3, s=0)
    return splev(x, tck)


def get_configuration(config: str, arguments: List[str], pprint_config: bool, log: Optional[Union[str, Path]]) -> Tuple[CN,logging.Logger]:
    """
    Create the configuration and the logger.

    Parameters
    ----------
    config : str
        The path to the configuration file.
    arguments : List[str]
        The list of additional command line arguments.
    pprint_config : bool
        Whether to pretty print the configuration and exit.
    log : Optional[str | Path]
        The path to the logging file (default is None, default logging).

    Returns
    -------
    Tuple[CfgNode, logging.Logger]
        The configuration nodes and the logger.
    """
    cfg = get_params(config, arguments)

    if pprint_config:
        print_config(cfg)
        sys.exit(0)

    log = logger(log)

    return cfg,log


def get_output_directory(cfg: CN, log: logging.Logger) -> Path:
    """
    Create the output directory and save the configuration to file.

    Parameters
    ----------
    cfg : CfgNode
        The configuration nodes.
    log : logging.Logger
        The logger.

    Returns
    -------
    Path
        The path to the output directory
    """
    output_dir = Path(cfg.OUTPUT.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the actual configuration in the output directory
    log.info("Saving the configuration...")
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        f.write(cfg.dump())

    return output_dir


def get_initial_expression(cfg: CN, log: logging.Logger) -> str:
    """
    Get the initial expression of the potential.

    Parameters
    ----------
    cfg : CfgNode
        The configuration nodes.
    log : logging.Logger
        The logger.

    Returns
    -------
    str
        The SciPy string of the expression.

    Raises
    ------
    ValueError
        If not valid energy scale has been selected.
    """
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
        kappa_bar_0 = init.BY_PARAMS.MU_0
        mu_bar_0_0 = init.BY_PARAMS.MU_1
        mu_bar_1_0 = init.BY_PARAMS.MU_2
        mu_bar_2_0 = init.BY_PARAMS.MU_3
        expr = f'{kappa_bar_0} + {mu_bar_0_0} * x + {mu_bar_1_0} * x**2 + {mu_bar_2_0} * x**3'
        pot = f'{kappa_bar_0} * x + {mu_bar_0_0} * x**2 / 2 + {mu_bar_1_0} * x**3 / 3 + {mu_bar_2_0} * x**4 / 4'
    elif init.BY_TEMP.ENABLED:
        log.debug("Init by temperature...")
        T = init.BY_TEMP.T
        TC = init.BY_TEMP.TC
        kappa_bar_0 = T - TC
        mu_bar_0_0 = T
        mu_bar_1_0 = T**2
        mu_bar_2_0 = T**3
        expr = f'{kappa_bar_0} + {mu_bar_0_0} * x + {mu_bar_1_0} * x**2 + {mu_bar_2_0} * x**3'
        pot = f'{kappa_bar_0} * x + {mu_bar_0_0} * x**2 / 2 + {mu_bar_1_0} * x**3 / 3 + {mu_bar_2_0} * x**4 / 4'
    else:
        raise ValueError('No valid initial conditions defined!')
    log.debug(f"Initial conditions:\nU' = {expr}\nU = {pot}")

    return expr


def create_data(cfg: CN, log: logging.Logger) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[CN, CN]]:
    """
    Create the data and noise distributions.

    Parameters
    ----------
    cfg : CfgNode
        The configuration nodes.
    log : logging.Logger
        The logger.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[CfgNode, CfgNode]]
        Two tuples containing (noise, signal, data, covariance) and (matrix configuration node, signal configuration node).

    Raises
    ------
    ValueError
        If no valid signal mode has been selected.
    """
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
        S = 2 * (np.array(image) / 255.0) - 1
    else:
        raise ValueError('No valid signal defined!')

    # Create the full distribution and compute the eigenvalues of the
    # covariance matrix
    X = Z + signal.RATIO * S
    C = np.cov(X, rowvar=False)
    log.debug(f'Covariance shape: {C.shape}')

    return (Z,S,X,C),(matrix,signal)


def plot_distributions(cfg: CN, log: logging.Logger, output_dir: Path, Z: np.ndarray, S: np.ndarray, X: np.ndarray, signal: CN) -> None:
    """
    Plot the distributions of the background, the signal, etc.

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    output_dir : Path
        The path to the output directory.
    Z : np.ndarray
        The noise distribution.
    S : np.ndarray
        The signal distribution.
    X : np.ndarray
        The full distribution (:math:`X = Z + \beta S`).
    signal : CfgNode
        The configuration node related to the signal matrix.
    """
    log.info("Plotting the signal matrix...")
    fig, ax = plt.subplots()
    ax.imshow(X, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / 'signal_matrix.png')
    plt.close(fig)

    # Compute the eigenvalues of the signal and background separately
    Cz = np.cov(Z, rowvar=False)
    Cs = np.cov(signal.RATIO * S, rowvar=False)
    Ez = np.linalg.eigvalsh(Cz)
    Es = np.linalg.eigvalsh(Cs)

    # Plot the two distributions
    fig, ax = plt.subplots()
    ax.hist(Ez,
            bins=cfg.INPUT.BINNING.BINS,
            density=True,
            color='b',
            alpha=0.5,
            label='background')
    ax.hist(Es,
            bins=cfg.INPUT.BINNING.BINS,
            density=True,
            color='r',
            alpha=0.5,
            label='signal')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\mu(\lambda)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'eval_comp.png')
    plt.close(fig)

    # Select only the eigenvalues of the signal within the bulk region
    Es = Es[Es <= Ez.max()]
    fig, ax = plt.subplots()
    ax.hist(Ez,
            bins=cfg.INPUT.BINNING.BINS,
            density=True,
            color='b',
            alpha=0.5,
            label='background')
    ax.hist(Es,
            bins=cfg.INPUT.BINNING.BINS,
            density=True,
            color='r',
            alpha=0.5,
            label='signal')
    ax.set_xlim(0.0, 1.05 * Ez.max())
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\mu(\lambda)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'eval_comp_zoom.png')
    plt.close(fig)


def compute_eigenvalues(log: logging.Logger, C: np.ndarray) -> np.ndarray:
    """
    Compute the eigenvalues of the covariance matrix.

    Parameters
    ----------
    log : logging.Logger
        The logger.
    C : np.ndarray
        The covariance matrix.

    Returns
    -------
    np.ndarray
        The eigenvalues of the covariance matrix.
    """
    log.debug("Computing the eigenvalues of the full covariance matrix...")
    E = np.linalg.eigvalsh(C)
    log.debug(f"Eigenvalues:\nlambda_max = {E.max()}\nlambda_min = {E.min()}")
    return E


def mp_distribution(cfg: CN, log: logging.Logger, output_dir: Path, matrix: CN, E: Optional[np.ndarray] = None) -> Tuple[float, MarchenkoPastur]:
    """
    Build the Marchenko-Pastur distribution and plot it (against the empirical spectrum, if available).

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    output_dir : Path
        The path to the output directory.
    matrix : CfgNode
        The configuration node related to the matrix specifications.
    E : Optional[np.ndarray]
        The empirical spectrum of eigenvalues (default is None).

    Returns
    -------
    Tuple[float, MarchenkoPastur]
        The ratio of columns / rows and the distribution.
    """
    log.info("Defining the MP distribution associated to the background...")
    ratio = matrix.COLUMNS / matrix.ROWS
    mp = MarchenkoPastur(L=ratio)

    # Plot the distribution of the eigenvalues
    log.info("Plotting the distribution of the eigenvalues...")
    if E is not None:
        x = np.linspace(0.0, 1.05 * E.max(), 2500)
    else:
        x = np.linspace(0.0, 4.0, 2500)
    y = np.array([mp(x_) for x_ in x])
    fig, ax = plt.subplots()
    if E is not None:
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
    return ratio,mp


def inverse_spectrum(log: logging.Logger, E: np.ndarray) -> np.ndarray:
    """
    Inverse the spectrum of the eigenvalues.

    Parameters
    ----------
    log : logging.Logger
        The logger.
    E : np.ndarray
        The spectrum of the sample covariance matrix.

    Returns
    -------
    np.ndarray
        The spectrum of momenta (i.e. the inverse spectrum of the sample covariance matrix).
    """
    log.info("Computing the inverse of the eigenvalues...")
    E_inv = np.flip(1 / E)
    log.debug(f"Momenta:\nk2_max = {E_inv.max()}\nk2_min = {E_inv.min()}")
    E_inv -= E_inv.min()
    return E_inv


def energy_scale(cfg: CN, log: logging.Logger, mp: MarchenkoPastur, E_inv: Optional[np.ndarray] = None) -> Tuple[Union[None, np.ndarray], Tuple[float, float], CN, bool]:
    """
    Define the energy scale of the simulation.

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    mp : MarchenkoPastur
        The Marchenko-Pastur distribution.
    E_inv : Optional[np.ndarray]
        The spectrum of momenta of the empirical distribution (default is None).

    Returns
    -------
    Tuple[Union[None, np.ndarray], Tuple[float, float], CfgNode, bool]
        A tuple containing the inverse spectrm (if provided), a tuple containing the bottom and top energy scales of the simulation, the energy scale configuration node and whether to fit the empirical distribution to the origin.

    Raises
    ------
    ValueError
        If no valid energy scale has been selected.
    """
    log.info("Defining the energy scale...")
    e_scale = cfg.INPUT.E_SCALE
    force_origin = True
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

    if (E_inv is not None) and (not e_scale.SPIKES):
        log.debug("Shifting spikes (PCA)...")

        # Find the max of the distribution
        counts, bins = np.histogram(E_inv, bins=cfg.INPUT.BINNING.BINS**2)
        E_inv_max = bins[counts.argmax()]

        # Remove spikes
        E_inv_spikes = np.where(np.diff(E_inv) >= cfg.SIM.EIGEN_THRESH)[0] + 1
        E_inv_spikes = E_inv[E_inv_spikes]
        E_inv_spikes = E_inv_spikes[E_inv_spikes < E_inv_max]

        # Shift the distribution
        if len(E_inv_spikes) > 0:
            shift = max(E_inv_spikes)
            E_inv -= shift
            force_origin = False

    return E_inv, (m2_bot, m2_top), e_scale,force_origin


def simulation_distribution(cfg: CN, log: logging.Logger, output_dir: Path, ratio: float, e_scale: CN, m2_bot: float, m2_top: float, force_origin: bool = False, E_inv: Optional[np.ndarray] = None)-> Tuple[Tuple[float, float], BaseDistribution]:
    """
    Define the distribution used in the simulation.

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    output_dir : Path
        The path to the output directory.
    ratio : float
        The ratio columns / rows of the experimental data matrix.
    e_scale : CfgNode
        The configuration node defining the energy scale.
    m2_bot : float
        The bottom limit of the energy scales used in the simulation.
    m2_top : float
        The top limit of the energy scales used in the simulation.
    force_origin : bool
        Whether to force the interpolation of the spectrum to pass through the origin (default is False).
    E_inv : Optional[np.ndarray]
        The inverse spectrum of momenta of the sample covariance matrix (default is None).

    Returns
    -------
    Tuple[Tuple[float, float], BaseDistribution]
        A tuple containing a tuple with the bottom and top values of the energy scale of the simulation, and the distribution.
    """
    log.info("Defining the distribution of the simulation...")
    if E_inv is not None:
        dist = InterpolateDistribution(bins=cfg.INPUT.BINNING.BINS**2)
        dist = dist.fit(E_inv,
                        k=3,
                        s=cfg.INPUT.BINNING.SMOOTHING,
                        force_origin=force_origin)
    dist_th = TranslatedInverseMarchenkoPastur(L=ratio)

    # Plot the distribution (inverse)
    log.info("Plotting the distribution of the simulation...")
    if E_inv is not None:
        x = np.linspace(E_inv.min(), E_inv.max(), 2500)
        y = np.array([dist(x_) for x_ in x])
    else:
        x = np.linspace(0.0, 3.0, 2500)
    y_th = np.array([dist_th(x_) for x_ in x])

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
    if E_inv is not None:
        ax.hist(E_inv,
                bins=cfg.INPUT.BINNING.BINS**2,
                density=True,
                color='b',
                alpha=0.5,
                label='eigenvalues')
        ax.plot(x, y, 'k-', label='interpolated')
        ax.set_xlim(-0.1, E_inv.max())
    ax.plot(x, y_th, 'r-', label='inverse MP')
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv.png')
    plt.close(fig)

    if E_inv is not None:
        x = np.linspace(E_inv.min(), 0.1 * E_inv.max(), 2500)
        y = np.array([dist(x_) for x_ in x])
    else:
        x = np.linspace(0.0, 1.0, 2500)
    y_th = np.array([dist_th(x_) for x_ in x])

    fig, ax = plt.subplots()
    if E_inv is not None:
        ax.hist(E_inv,
                bins=cfg.INPUT.BINNING.BINS**2,
                density=True,
                color='b',
                alpha=0.5,
                label='eigenvalues')
        ax.plot(x, y, 'k-', label='interpolated')
        ax.set_xlim(-0.1, 0.1 * E_inv.max())
    ax.plot(x, y_th, 'r-', label='inverse MP')
    ax.set_xlabel(r'$k^2$')
    ax.set_ylabel(r'$\rho$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'mp_eigenvalues_inv_zoom.png')
    plt.close(fig)

    x = np.linspace(0.0, 1.0, 2500)
    if E_inv is not None:
        y = np.array([dist(x_) for x_ in x])
    y_th = np.array([dist_th(x_) for x_ in x])

    fig, ax = plt.subplots()
    if E_inv is not None:
        ax.hist(E_inv,
                bins=cfg.INPUT.BINNING.BINS**2,
                density=True,
                color='b',
                alpha=0.5,
                label='eigenvalues')
        ax.plot(x, y, 'k-', label='interpolated')
    ax.plot(x, y_th, 'r-', label='inverse MP')
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
        if E_inv is not None:
            dist = SpecularReflection(dist, shift=m2_top)
        else:
            dist_th = SpecularReflection(dist_th, shift=m2_top)

        # Plot the distribution (reflected)
        log.info("Plotting the distribution of the simulation (reflected)...")
        x = np.linspace(-m2_top, m2_top, 2500)
        if E_inv is not None:
            y = np.array([dist(x_) for x_ in x])
        else:
            y = np.array([dist_th(x_) for x_ in x])

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

    if E_inv is not None:
        return (m2_bot,m2_top),dist
    else:
        return (m2_bot,m2_top),dist_th


def simulation(cfg: CN, log: logging.Logger, expr: str, m2_bot: float, m2_top: float, dist: BaseDistribution) -> SSD:
    """
    Run the simulation.

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    expr : str
        The SciPy string of the expression of the initial conditions.
    m2_bot : float
        The bottom limit of the energy scales used in the simulation.
    m2_top : float
        The top limit of the energy scales used in the simulation.
    dist : BaseDistribution
        The distribution used in the simulation

    Returns
    -------
    SSD
        The partial differential equation.
    """
    log.info("Defining the grid of the simulation...")
    dx = (cfg.SIM.SUP - cfg.SIM.INF) / cfg.SIM.N_VALUES
    grid = CartesianGrid(
                [[cfg.SIM.INF-dx/2.0, cfg.SIM.SUP+dx/2.0]],
        [cfg.SIM.N_VALUES],
        periodic=cfg.SIM.PERIODIC,
    )
    state = ScalarField.from_expression(grid, expr)
    bc = 'periodic' if cfg.SIM.PERIODIC else 'auto_periodic_neumann'

    # Storage
    if cfg.SIM.IR_TO_UV:
        t_range = [m2_bot, m2_top]
        uv_scale = None
    else:
        t_range = [0.0, m2_top - m2_bot]
        uv_scale = m2_top
    dt = (t_range[1] - t_range[0]) / cfg.SIM.N_STEPS

    # Run the simulation
    log.info("Running the simulation...")
    sign = int(cfg.SIM.SIGN > 0) - int(cfg.SIM.SIGN < 0)
    eq = SSD(dist, noise=0.0, sign=sign, uv_scale=uv_scale, bc=bc)
    _ = eq.solve(state, t_range=t_range, dt=dt, tracker=['progress'])

    return eq


def collect_values(cfg: CN, log: logging.Logger, eq: SSD) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Collect important quantities from the simulation.

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    eq : SSD
        The partial differential equation.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        A tuple containing a tuple with the derivative of the potential at the lower and upper bounds of the grid, and a tuple containing the anomalous dimensions of the octic projection of the potential.
    """
    dU_start = np.array([dU[0] for dU in eq.dU])
    dU_end = np.array([dU[-1] for dU in eq.dU])

    # Compute the dimensions of the operators
    log.info("Computing the dimensions of the operators...")
    dim_kappa_bar = -eq.dimChi
    dim_mu_4_bar = eq.dimChi - eq.dimdU
    dim_mu_6_bar = 2.0 * eq.dimChi - eq.dimdU
    dim_mu_8_bar = 3.0 * eq.dimChi - eq.dimdU

    # Remove NaNs
    dim_kappa_bar = nan_to_num(dim_kappa_bar)
    dim_mu_4_bar = nan_to_num(dim_mu_4_bar)
    dim_mu_6_bar = nan_to_num(dim_mu_6_bar)
    dim_mu_8_bar = nan_to_num(dim_mu_8_bar)

    # Fit a spline to the dimensions
    log.info("Fitting a spline to the dimensions...")
    if not cfg.SIM.IR_TO_UV:
        dim_kappa_bar = fit_spline(eq.k2[::-1], dim_kappa_bar[::-1])[::-1]
        dim_mu_4_bar = fit_spline(eq.k2[::-1], dim_mu_4_bar[::-1])[::-1]
        dim_mu_6_bar = fit_spline(eq.k2[::-1], dim_mu_6_bar[::-1])[::-1]
        dim_mu_8_bar = fit_spline(eq.k2[::-1], dim_mu_8_bar[::-1])[::-1]
    else:
        dim_kappa_bar = fit_spline(eq.k2, dim_kappa_bar)
        dim_mu_4_bar = fit_spline(eq.k2, dim_mu_4_bar)
        dim_mu_6_bar = fit_spline(eq.k2, dim_mu_6_bar)
        dim_mu_8_bar = fit_spline(eq.k2, dim_mu_8_bar)
    return (dU_start,dU_end),(dim_kappa_bar,dim_mu_4_bar,dim_mu_6_bar,dim_mu_8_bar)


def collect_sqlite(cfg: CN, log: logging.Logger, m2_bot: float, m2_top: float, eq: SSD, dU_start: np.ndarray, dU_end: np.ndarray, dim_kappa_bar: np.ndarray, dim_mu_4_bar:np.ndarray, dim_mu_6_bar:np.ndarray, dim_mu_8_bar:np.ndarray)-> None:
    """
    Register all important quantities in a SQLite database.

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    m2_bot : float
        The bottom limit of the energy scales used in the simulation.
    m2_top : float
        The top limit of the energy scales used in the simulation.
    eq : SSD
        The partial differential equation.
    dU_start : np.ndarray
        The values of the derivative of the potential in the lower limit of the grid.
    dU_end : np.ndarray
        The values of the derivative of the potential in the upper limit of the grid.
    dim_kappa_bar : np.ndarray
        The anomalous dimensions of the position of the first minimum of the potential.
    dim_mu_4_bar : np.ndarray
        The values of the derivative of the potential (quartic coupling).
    dim_mu_6_bar : np.ndarray
        The values of the second derivative of the potential (sextic coupling).
    dim_mu_8_bar : np.ndarray
        The values of the third derivative of the potential (octic coupling).
    """
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
            dU_start TEXT,
            dU_end TEXT,
            kappa_bar TEXT,
            mu_bar_0 TEXT,
            mu_bar_1 TEXT,
            mu_bar_2 TEXT,
            dim_kappa_bar TEXT,
            dim_mu_4_bar TEXT,
            dim_mu_6_bar TEXT,
            dim_mu_8_bar TEXT,
            k2 TEXT,
            dU TEXT
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
            dU_start,
            dU_end,
            kappa_bar,
            mu_bar_0,
            mu_bar_1,
            mu_bar_2,
            dim_kappa_bar,
            dim_mu_4_bar,
            dim_mu_6_bar,
            dim_mu_8_bar,
            k2,
            dU
            ) VALUES (
            '{now}',
            {cfg.INPUT.MATRIX.SEED},
            {cfg.INPUT.MATRIX.ROWS},
            {cfg.INPUT.MATRIX.COLUMNS},
            {cfg.INPUT.SIGNAL.RATIO},
            {m2_bot},
            {m2_top},
            '{json.dumps(dU_start.tolist())}',
            '{json.dumps(dU_end.tolist())}',
            '{json.dumps(eq.kappa_bar.tolist())}',
            '{json.dumps(eq.mu_4_bar.tolist())}',
            '{json.dumps(eq.mu_6_bar.tolist())}',
            '{json.dumps(eq.mu_8_bar.tolist())}',
            '{json.dumps(dim_kappa_bar.tolist())}',
            '{json.dumps(dim_mu_4_bar.tolist())}',
            '{json.dumps(dim_mu_6_bar.tolist())}',
            '{json.dumps(dim_mu_8_bar.tolist())}',
            '{json.dumps(eq.k2.tolist())}',
            '{json.dumps(eq.dU.tolist())}'
            )"""
            cursor.execute(sql_query)

    except sqlite3.Error as e:
        log.error(e)

    finally:
        conn.close()


def plot_results(cfg: CN, log: logging.Logger, output_dir: Path, ratio: float, m2_top: float, eq: SSD, dU_start:np.ndarray, dU_end:np.ndarray, dim_kappa_bar:np.ndarray, dim_mu_4_bar:np.ndarray, dim_mu_6_bar:np.ndarray, dim_mu_8_bar:np.ndarray):
    """
    Plot the results to visualize the evolution.]

    Parameters
    ----------
    cfg : CfgNode
        The configuration node.
    log : logging.Logger
        The logger.
    output_dir : Path
        The path to the output directory.
    ratio : float
        The ratio columns / rows of the experimental data matrix.
    m2_top : float
        The top limit of the energy scales used in the simulation.
    eq : SSD
        The partial differential equation.
    dU_start : np.ndarray
        The values of the derivative of the potential in the lower limit of the grid.
    dU_end : np.ndarray
        The values of the derivative of the potential in the upper limit of the grid.
    dim_kappa_bar : np.ndarray
        The anomalous dimensions of the position of the first minimum of the potential.
    dim_mu_4_bar : np.ndarray
        The values of the derivative of the potential (quartic coupling).
    dim_mu_6_bar : np.ndarray
        The values of the second derivative of the potential (sextic coupling).
    dim_mu_8_bar : np.ndarray
        The values of the third derivative of the potential (octic coupling).
    """
    log.info("Visualizing the results...")
    col_1 = 'ro' if cfg.SIM.IR_TO_UV else 'bo'
    col_2 = 'bo' if cfg.SIM.IR_TO_UV else 'ro'
    fig, ax = plt.subplots(ncols=4, nrows=5, figsize=(40, 24))

    log.debug("Plotting the potential at starting point...")
    ax[0, 0].plot(eq.k2, dU_start, 'k-')
    ax[0, 0].plot([eq.k2[0]], [dU_start[0]], col_1)
    ax[0, 0].plot([eq.k2[-1]], [dU_start[-1]], col_2)
    ax[0, 0].set_xlabel(r'$k^2$')
    ax[0, 0].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$')

    log.debug("Plotting the potential at ending point...")
    ax[0, 1].plot(eq.k2, dU_end, 'k-')
    ax[0, 1].plot([eq.k2[0]], [dU_end[0]], col_1)
    ax[0, 1].plot([eq.k2[-1]], [dU_end[-1]], col_2)
    ax[0, 1].set_xlabel(r'$k^2$')
    ax[0, 1].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}]$')

    log.debug("Plotting potential curve at starting and ending points...")
    ax[0, 2].plot(dU_start, dU_end, 'k-')
    ax[0, 2].plot([dU_start[0]], [dU_end[0]], col_1)
    ax[0, 2].plot([dU_start[-1]], [dU_end[-1]], col_2)
    ax[0, 2].set_xlabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$')
    ax[0, 2].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}]$')

    log.debug("Plotting ratio of starting and ending points...")
    ax[0, 3].plot(eq.k2,
                  np.array(dU_end) / np.array(dU_start),
                  'k-')
    ax[0, 3].plot([eq.k2[0]], [dU_end[0] / dU_start[0]], col_1)
    ax[0, 3].plot([eq.k2[-1]], [dU_end[-1] / dU_start[-1]], col_2)
    ax[0, 3].set_xlabel(r'$k^2$')
    ax[0, 3].set_ylabel(
        rf'$\overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.SUP}}}] / \overline{{\mathcal{{U}}}}^{{\prime}}[\overline{{{cfg.SIM.INF}}}]$'
    )

    log.debug("Plotting \\overline{\\kappa}...")
    ax[1, 0].plot(eq.k2, eq.kappa_bar, 'k-')
    ax[1, 0].plot([eq.k2[0]], [eq.kappa_bar[0]], col_1)
    ax[1, 0].plot([eq.k2[-1]], [eq.kappa_bar[-1]], col_2)
    ax[1, 0].set_xlabel(r'$k^2$')
    ax[1, 0].set_ylabel(r'$\overline{\kappa}$')

    log.debug("Plotting \\overline{\\mu}_0...")
    ax[1, 1].plot(eq.k2, eq.mu_4_bar, 'k-')
    ax[1, 1].plot([eq.k2[0]], [eq.mu_4_bar[0]], col_1)
    ax[1, 1].plot([eq.k2[-1]], [eq.mu_4_bar[-1]], col_2)
    ax[1, 1].set_xlabel(r'$k^2$')
    ax[1, 1].set_ylabel(r'$\overline{\mu}_4$')

    log.debug("Plotting \\overline{\\mu}_1...")
    ax[1, 2].plot(eq.k2, eq.mu_6_bar, 'k-')
    ax[1, 2].plot([eq.k2[0]], [eq.mu_6_bar[0]], col_1)
    ax[1, 2].plot([eq.k2[-1]], [eq.mu_6_bar[-1]], col_2)
    ax[1, 2].set_xlabel(r'$k^2$')
    ax[1, 2].set_ylabel(r'$\overline{\mu}_6$')

    log.debug("Plotting \\overline{\\mu}_2...")
    ax[1, 3].plot(eq.k2, eq.mu_8_bar, 'k-')
    ax[1, 3].plot([eq.k2[0]], [eq.mu_8_bar[0]], col_1)
    ax[1, 3].plot([eq.k2[-1]], [eq.mu_8_bar[-1]], col_2)
    ax[1, 3].set_xlabel(r'$k^2$')
    ax[1, 3].set_ylabel(r'$\overline{\mu}_8$')

    log.debug("Plotting dimension of \\kappa...")
    ax[2, 0].plot(eq.k2, dim_kappa_bar, 'k-')
    ax[2, 0].plot([eq.k2[0]], [dim_kappa_bar[0]], col_1)
    ax[2, 0].plot([eq.k2[-1]], [dim_kappa_bar[-1]], col_2)
    ax[2, 0].set_xlabel(r'$k^2$')
    ax[2, 0].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\kappa}$')

    log.debug("Plotting dimension of \\mu_4...")
    ax[2, 1].plot(eq.k2, dim_mu_4_bar, 'k-')
    ax[2, 1].plot([eq.k2[0]], [dim_mu_4_bar[0]], col_1)
    ax[2, 1].plot([eq.k2[-1]], [dim_mu_4_bar[-1]], col_2)
    ax[2, 1].set_xlabel(r'$k^2$')
    ax[2, 1].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')

    log.debug("Plotting dimension of \\mu_6...")
    ax[2, 2].plot(eq.k2, dim_mu_6_bar, 'k-')
    ax[2, 2].plot([eq.k2[0]], [dim_mu_6_bar[0]], col_1)
    ax[2, 2].plot([eq.k2[-1]], [dim_mu_6_bar[-1]], col_2)
    ax[2, 2].set_xlabel(r'$k^2$')
    ax[2, 2].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')

    log.debug("Plotting dimension of \\mu_8...")
    ax[2, 3].plot(eq.k2, dim_mu_8_bar, 'k-')
    ax[2, 3].plot([eq.k2[0]], [dim_mu_8_bar[0]], col_1)
    ax[2, 3].plot([eq.k2[-1]], [dim_mu_8_bar[-1]], col_2)
    ax[2, 3].set_xlabel(r'$k^2$')
    ax[2, 3].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_8$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{{\\mu}}_0..."
    )
    ax[3, 0].plot(eq.kappa_bar, eq.mu_4_bar, 'k-')
    ax[3, 0].plot([eq.kappa_bar[0]], [eq.mu_4_bar[0]], col_1)
    ax[3, 0].plot([eq.kappa_bar[-1]], [eq.mu_4_bar[-1]], col_2)
    ax[3, 0].set_xlabel(r'$\overline{\kappa}$')
    ax[3, 0].set_ylabel(r'$\overline{\mu}_4$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{{\\mu}}_1..."
    )
    ax[3, 1].plot(eq.kappa_bar, eq.mu_6_bar, 'k-')
    ax[3, 1].plot([eq.kappa_bar[0]], [eq.mu_6_bar[0]], col_1)
    ax[3, 1].plot([eq.kappa_bar[-1]], [eq.mu_6_bar[-1]], col_2)
    ax[3, 1].set_xlabel(r'$\overline{\kappa}$')
    ax[3, 1].set_ylabel(r'$\overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{{\\mu}}_4 vs \\overline{{\\mu}}_6..."
    )
    ax[3, 2].plot(eq.mu_4_bar, eq.mu_6_bar, 'k-')
    ax[3, 2].plot([eq.mu_4_bar[0]], [eq.mu_6_bar[0]], col_1)
    ax[3, 2].plot([eq.mu_4_bar[-1]], [eq.mu_6_bar[-1]], col_2)
    ax[3, 2].set_xlabel(r'$\overline{\mu}_4$')
    ax[3, 2].set_ylabel(r'$\overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{{\\mu}}_4 vs \\overline{{\\mu}}_8..."
    )
    ax[3, 3].plot(eq.mu_4_bar, eq.mu_8_bar, 'k-')
    ax[3, 3].plot([eq.mu_4_bar[0]], [eq.mu_8_bar[0]], col_1)
    ax[3, 3].plot([eq.mu_4_bar[-1]], [eq.mu_8_bar[-1]], col_2)
    ax[3, 3].set_xlabel(r'$\overline{\mu}_4$')
    ax[3, 3].set_ylabel(r'$\overline{\mu}_8$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{\\mu_4}..."
    )
    ax[4, 0].plot(dim_kappa_bar, dim_mu_4_bar, 'k-')
    ax[4, 0].plot([dim_kappa_bar[0]], [dim_mu_4_bar[0]], col_1)
    ax[4, 0].plot([dim_kappa_bar[-1]], [dim_mu_4_bar[-1]], col_2)
    ax[4, 0].set_xlabel(r'$-\mathrm{dim}_{k}\, \overline{\kappa}$')
    ax[4, 0].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')

    log.debug(
        "Plotting phase space curve \\overline{\\kappa} vs \\overline{\\mu_6}..."
    )
    ax[4, 1].plot(dim_kappa_bar, dim_mu_6_bar, 'k-')
    ax[4, 1].plot([dim_kappa_bar[0]], [dim_mu_6_bar[0]], col_1)
    ax[4, 1].plot([dim_kappa_bar[-1]], [dim_mu_6_bar[-1]], col_2)
    ax[4, 1].set_xlabel(r'$-\mathrm{dim}_{k}\, \kappa$')
    ax[4, 1].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{\\mu_4} vs \\overline{\\mu_6}..."
    )
    ax[4, 2].plot(dim_mu_4_bar, dim_mu_6_bar, 'k-')
    ax[4, 2].plot([dim_mu_4_bar[0]], [dim_mu_6_bar[0]], col_1)
    ax[4, 2].plot([dim_mu_4_bar[-1]], [dim_mu_6_bar[-1]], col_2)
    ax[4, 2].set_xlabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')
    ax[4, 2].set_ylabel(r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')

    log.debug(
        "Plotting phase space curve \\overline{\\mu_4} vs \\overline{\\mu_8}..."
    )
    ax[4, 3].plot(dim_mu_4_bar, dim_mu_8_bar, 'k-')
    ax[4, 3].plot([dim_mu_4_bar[0]], [dim_mu_8_bar[0]], col_1)
    ax[4, 3].plot([dim_mu_4_bar[-1]], [dim_mu_8_bar[-1]], col_2)
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
    invmp = TranslatedInverseMarchenkoPastur(L=ratio)
    x = np.linspace(0.0, m2_top, 2500)
    y = np.array([invmp(x_) for x_ in x])
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k-', label='inverse MP')
    ax.set_xlabel(r'$k^2$')
    # ax.plot(eq.k2,
    #         dim_eq.kappa_bar,
    #         'k-',
    #         label=r'$-\mathrm{dim}_{k}\, \overline{\kappa}$')
    # ax.plot([eq.k2[0]], [dim_kappa_bar[0]], col_1)
    # ax.plot([eq.k2[-1]], [dim_kappa_bar[-1]], col_2)
    ax.plot(eq.k2,
            dim_mu_4_bar,
            'r-',
            label=r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')
    ax.plot([eq.k2[0]], [dim_mu_4_bar[0]], col_1)
    ax.plot([eq.k2[-1]], [dim_mu_4_bar[-1]], col_2)
    ax.plot(eq.k2,
            dim_mu_6_bar,
            'g-',
            label=r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')
    ax.plot([eq.k2[0]], [dim_mu_6_bar[0]], col_1)
    ax.plot([eq.k2[-1]], [dim_mu_6_bar[-1]], col_2)
    ax.plot(eq.k2,
            dim_mu_8_bar,
            'b-',
            label=r'$-\mathrm{dim}_{k}\, \overline{\mu}_8$')
    ax.plot([eq.k2[0]], [dim_mu_8_bar[0]], col_1)
    ax.plot([eq.k2[-1]], [dim_mu_8_bar[-1]], col_2)
    ax.ticklabel_format(axis='both',
                        style='sci',
                        scilimits=(0, 0),
                        useMathText=True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_dim_param.png')

    # Plot the dimensions of the parameters using the MP distribution
    mp = MarchenkoPastur(L=ratio)
    x = np.linspace(0.0, mp.max+0.05, 2500)
    y = np.array([mp(x_) for x_ in x])
    k2 = (1/mp.min) * (eq.k2 - eq.k2.min()) / (eq.k2.max() - eq.k2.min()) + (1/mp.max)
    km2 = 1/k2
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k-', label='MP')
    ax.set_xlabel(r'$\lambda$')
    ax.plot(km2,
            dim_mu_4_bar,
            'r-',
            label=r'$-\mathrm{dim}_{k}\, \overline{\mu}_4$')
    ax.plot([km2[0]], [dim_mu_4_bar[0]], col_1)
    ax.plot([km2[-1]], [dim_mu_4_bar[-1]], col_2)
    ax.plot(km2,
            dim_mu_6_bar,
            'g-',
            label=r'$-\mathrm{dim}_{k}\, \overline{\mu}_6$')
    ax.plot([km2[0]], [dim_mu_6_bar[0]], col_1)
    ax.plot([km2[-1]], [dim_mu_6_bar[-1]], col_2)
    ax.plot(km2,
            dim_mu_8_bar,
            'b-',
            label=r'$-\mathrm{dim}_{k}\, \overline{\mu}_8$')
    ax.plot([km2[0]], [dim_mu_8_bar[0]], col_1)
    ax.plot([km2[-1]], [dim_mu_8_bar[-1]], col_2)
    ax.ticklabel_format(axis='both',
                        style='sci',
                        scilimits=(0, 0),
                        useMathText=True)
    ax.legend()
    ax.set_xlim(-0.05, mp.max+0.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'simulation_dim_param_mp.png')


def log_expr(log: logging.Logger, eq: SSD):
    """
    Log the projection of the potential.

    Parameters
    ----------
    log : logging.Logger
        The logger.
    eq : SSD
        The partial differential equation.
    """
    kappa_bar = eq.kappa_bar[-1]
    mu_bar_0 = eq.mu_4_bar[-1]
    mu_bar_1 = eq.mu_6_bar[-1]
    mu_bar_2 = eq.mu_8_bar[-1]
    expr = f'{mu_bar_0} * (x - {kappa_bar}) + {mu_bar_1} * (x - {kappa_bar})**2 + {mu_bar_2} * (x - {kappa_bar})**3'
    pot = f'{mu_bar_0} * (x - {kappa_bar})**2 / 2 + {mu_bar_1} * (x - {kappa_bar})**3 / 3 + {mu_bar_2} * (x - {kappa_bar})**4 / 4'
    log.info(f'Final expression:\nU\' = {expr}\nU = {pot}')
