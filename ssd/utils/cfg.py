# -*- coding: utf-8 -*-
"""
Cfg nodes

Deal with CfgNode objects and import the parameters.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from yacs.config import CfgNode as CN

from ssd import __author__, __email__, __version__
from ssd.config import get_cfg_defaults


def logger(filename: Optional[Union[str, Path]]) -> logging.Logger:
    """
    Logger of the simulation.

    Parameters
    ----------
    filename : str or pathlib.Path
        Filename of the log file.

    Returns
    -------
    logger : logging.Logger
        Logger of the simulation.
    """
    log = logging.getLogger('SSD')
    log.setLevel(logging.DEBUG)
    form = '[%(asctime)s] %(levelname)s: %(message)s'

    # Create handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(form))
    log.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(str(filename))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(form))
        log.addHandler(file_handler)

    return log


def header(config: str, cfg: CN) -> str:
    """
    Header of the simulation.

    Parameters
    ----------
    config : str
        Configuration file.
    cfg : yacs.config.CfgNode
        Configuration parameters.

    Returns
    -------
    header : str
        Header of the simulation.
    """
    today = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    head = f"""\n
#########################################
#                                       #
#   SSD - Stochastic Signal Detection   #
#                                       #
#########################################

Authors:       H. Erbin, R. Finotello, B. W. Kpera, V. Lahoche, D. Ousmane Samary
Institution:   CEA List, Palaiseau, France
Code:          R. Finotello <{__email__}>
Version:       v{__version__}
Date:          {today}
Configuration: {config}

-----------------------------------------

{cfg.dump()}

"""

    return head


def get_params(config: str = None, arguments: List[str] = None) -> CN:
    """
    Get the parameters from the configuration file and the arguments.

    Parameters
    ----------
    config : str
        Configuration file.
    arguments : list
        List of arguments.

    Returns
    -------
    cfg : yacs.config.CfgNode
        Configuration parameters.
    """
    # Get the default parameters
    cfg = get_cfg_defaults()

    # Merge the configuration file
    if config is not None:
        cfg.merge_from_file(config)

    # Merge the arguments
    if arguments is not None:
        cfg.merge_from_list(arguments)

    # Freeze the configuration
    cfg.freeze()

    # Print the header
    log = logging.getLogger('SSD')
    log.info(header(config, cfg))

    return cfg


def print_config(cfg: CN) -> None:
    """
    Print the configuration parameters.

    Parameters
    ----------
    cfg : yacs.config.CfgNode
        Configuration parameters.
    """
    print(cfg.dump())
