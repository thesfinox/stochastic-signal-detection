# -*- coding: utf-8 -*-
"""
SSD - plotting utilities

Some utility functions for plotting and displaying the results of the simulation.
"""
from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


mpl.use('agg')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8, 6)

def plot_inverse_mp_distribution(evl: list,
                                 x: list,
                                 y: list,
                                 y_dist: list,
                                 k2min: Optional[float] = None,
                                 k2max: Optional[float] = None,
                                 bins: int = 1000,
                                 output: str = 'output_dir',
                                 subprefix: str = 'mp_inv_dist',
                                 ):
    """
    Plot the inverse Marchenko-Pastur distribution.

    Parameters
    ----------
    evl : list
        Inverse eigenvalues of the covariance matrix
    x : list
        x values of the distributions
    y : list
        y values of the MP distribution (inverse)
    y_dist : list
        y values of the empirical distribution
    k2min : float, optional
        Lower bound of the integration interval, by default None
    k2max : float, optional
        Upper bound of the integration interval, by default None
    bins : int, optional
        Number of bins for the histogram, by default 1000
    output : str, optional
        Output directory, by default 'output_dir'
    subprefix : str, optional
        Prefix of the output files, by default 'mp_inv_dist'
    """
    fig, ax = plt.subplots()
    ax.hist(evl,
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
    ax_inset_1.set_xlim([-0.1, 0.5])
    ax_inset_1.set_ylim([0.0, 0.9])
    ax_inset_1.hist(evl,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label='empirical',
                    color='b')
    ax_inset_1.plot(x, y_dist, 'k-', label='interpolated')
    ax_inset_1.plot(x, y, 'r-', label='theoretical')

    if (k2min is not None) and (k2max is not None):
        ax_inset_1.axvline(k2min, 0.0, 1.0, color='k', linestyle='--')
        ax_inset_1.axvline(k2max, 0.0, 1.0, color='k', linestyle='--')

    ax.indicate_inset_zoom(ax_inset_1)

    ax.legend(loc='best')
    plt.savefig(output / f'{subprefix}_mp_inv_dist.pdf')
    plt.close(fig)


def plot_mp_distribution(evl: list,
                         x: list,
                         y: list,
                         n_bins: int = 100,
                         output: str = 'output_dir',
                         subprefix: str = 'mp_dist'):
    """
    Plot the Marchenko-Pastur distribution.

    Parameters
    ----------
    evl : list
        Eigenvalues of the covariance matrix
    x : list
        x values of the MP distribution
    y : list
        y values of the MP distribution
    n_bins : int, optional
        Number of bins for the histogram, by default 100
    output : str, optional
        Output directory, by default 'output_dir'
    subprefix : str, optional
        Prefix of the output files, by default 'mp_dist'
    """
    fig, ax = plt.subplots()
    ax.hist(evl,
            bins=n_bins,
            density=True,
            label='empirical',
            alpha=0.5,
            color='b')
    ax.plot(x, y, 'r-', label='MP distribution')
    ax.set_xlabel('eigenvalues')
    ax.set_ylabel(r'$\mu$')

    ax_inset_1 = fig.add_axes([0.5, 0.24, 0.35, 0.25])
    ax_inset_1.set_xlim([2.5, 1.05 * evl.max()])
    ax_inset_1.set_ylim([0.0, 0.2])
    ax_inset_1.hist(evl,
                    bins=n_bins,
                    density=True,
                    alpha=0.5,
                    label='empirical',
                    color='b')
    ax_inset_1.plot(x, y, 'r-', label='theoretical')

    ax_inset_2 = fig.add_axes([0.25, 0.55, 0.35, 0.25])
    ax_inset_2.set_xlim([-0.1, 1.5])
    ax_inset_2.set_ylim([0.2, 0.9])
    ax_inset_2.hist(evl,
                    bins=n_bins,
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


def plot_potential(x_inf: float = 0.0,
                   x_sup: float = 0.0,
                   n_values: int = 1000,
                   mu1: float = 0.0,
                   mu2: float = 1.0,
                   mu3: float = 0.0,
                   output: str = 'output_dir',
                   prefix: str = 'initial_potential'):
    """
    Plot the initial potential.

    Parameters
    ----------
    x_inf : float, optional
        Lower bound of the domain, by default 0.0
    x_sup : float, optional
        Upper bound of the domain, by default 0.0
    n_values : int, optional
        Number of grid points, by default 1000
    mu1 : float, optional
        Mass parameter (quadratic term), by default 0.0
    mu2 : float, optional
        Quartic term, by default 1.0
    mu3 : float, optional
        6th power term, by default 0.0
    output : str, optional
        Output directory, by default 'output_dir'
    prefix : str, optional
        Prefix of the output files, by default 'initial_potential'
    """
    fig, ax = plt.subplots()
    x = np.linspace(x_inf, x_sup, n_values)
    y = mu1*x + mu2 * x**2 + mu3 * x**3

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
