# -*- coding: utf-8 -*-
"""
SSD - Stochastic Signal Detection

Simulation of the stochastic signal detection mechanism.
"""
import argparse
import sys

import matplotlib as mpl
from matplotlib import pyplot as plt

from ssd import __version__
from ssd.utils.utils import (collect_sqlite,
                             collect_values,
                             compute_eigenvalues,
                             create_data,
                             energy_scale,
                             get_configuration,
                             get_initial_expression,
                             get_output_directory,
                             inverse_spectrum,
                             log_expr,
                             mp_distribution,
                             plot_distributions,
                             plot_results,
                             simulation,
                             simulation_distribution)

mpl.use('agg')
mpl.rcParams['figure.figsize'] = (8, 6)
plt.style.use('ggplot')

__author__ = 'Riccardo Finotello'
__email__ = 'riccardo.finotello@cea.fr'
__description__ = 'Simulation of the stochastic signal detection mechanism.'
__epilog__ = 'For bug reports and info: ' + __author__ + ' <' + __email__ + '>'


def main(a: argparse.Namespace) -> int | str:

    # Organize the arguments
    cfg, log = get_configuration(a.config, a.arguments, a.print_config, a.log)

    # Create the output directory
    output_dir = get_output_directory(cfg, log)

    # Define the initial conditions
    expr = get_initial_expression(cfg, log)

    # Define the bulk distribution
    (Z, S, X, C), (matrix, signal) = create_data(cfg, log)

    # Plotting the distributions
    plot_distributions(cfg, log, output_dir, Z, S, X, signal)

    # Compute the eigenvalues of the covariance matrix
    E = compute_eigenvalues(log, C)

    # Define the MP distribution associated to the background
    ratio, mp = mp_distribution(cfg, log, output_dir, matrix, E)

    # Compute the inverse of the eigenvalues
    E_inv = inverse_spectrum(log, E)

    # Define the energy scale
    E_inv, (m2_bot, m2_top), e_scale, force_origin = energy_scale(cfg, log, mp, E_inv)

    # Define the distribution of the simulation
    (m2_bot, m2_top), dist = simulation_distribution(cfg, log, output_dir, ratio, e_scale,m2_bot, m2_top, force_origin, E_inv)

    # Define the simulation
    eq = simulation(cfg, log, expr, m2_bot, m2_top, dist)

    # Collect important quantities
    (dU_start,
     dU_end), (dim_kappa_bar, dim_mu_4_bar, dim_mu_6_bar,
               dim_mu_8_bar) = collect_values(cfg, log, eq)

    # Add information to the sqlite database
    collect_sqlite(cfg,
                   log,
                   m2_bot,
                   m2_top,
                   eq,
                   dU_start,
                   dU_end,
                   dim_kappa_bar,
                   dim_mu_4_bar,
                   dim_mu_6_bar,
                   dim_mu_8_bar)

    # Visualize the results
    plot_results(cfg,
                 log,
                 output_dir,
                 ratio,
                 m2_top,
                 eq,
                 dU_start,
                 dU_end,
                 dim_kappa_bar,
                 dim_mu_4_bar,
                 dim_mu_6_bar,
                 dim_mu_8_bar)

    # Record the last expression
    log_expr(log, eq)

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

    exit_code = main(args)

    sys.exit(exit_code)
