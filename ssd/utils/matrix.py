# -*- coding: utf-8 -*-
"""
SSD - Utilities

Utility functions for the Stochastic Signal Detection project.
"""
from typing import Optional, Union

import numpy as np


def create_bulk(rows: int,
                ratio: float,
                random_state: Optional[Union[int, np.random.Generator]] = None,
                **kwargs) -> np.ndarray:
    """
    Create a bulk distribution matrix.

    Parameters
    ----------
    rows : int
        Number of rows (data vectors) of the matrix.
    ratio : float
        Ratio between the number of columns (variables) and the number of rows (data vectors).
    random_state : Optional[Union[int, np.random.Generator]], optional
        Random state, by default None. It can be an integer, in which case it is used as the seed for the default random number generator, or a ``Generator`` object (see `numpy <https://numpy.org/doc/stable/reference/random/generator.html>`_).
    kwargs : dict
        Additional keyword arguments to pass to the random distribution (``np.random.normal``).

    Returns
    -------
    np.ndarray
        Bulk distribution matrix.

    Raises
    ------
    TypeError
        If the random state is not an integer or a ``Generator`` object.
    """
    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    elif not isinstance(random_state, np.random.Generator):
        raise TypeError(
            f'The random state must be an integer or a numpy.random.Generator object, not {type(random_state)}!'
        )

    # Create the bulk distribution matrix
    cols = int(rows * ratio)
    bulk = random_state.normal(size=(rows, cols), **kwargs)

    return bulk


def create_signal(rows: int,
                  ratio: float,
                  rank: int,
                  random_state: Optional[Union[int,
                                               np.random.Generator]] = None,
                  **kwargs) -> np.ndarray:
    """
    Create a signal matrix.

    Parameters
    ----------
    rows : int
        Number of rows (data vectors) of the matrix.
    ratio : float
        Ratio between the number of columns (variables) and the number of rows (data vectors).
    rank : int
        Rank of the signal matrix.
    random_state : Optional[Union[int, np.random.Generator]], optional
        Random state, by default None. It can be an integer, in which case it is used as the seed for the default random number generator, or a ``Generator`` object (see `numpy <https://numpy.org/doc/stable/reference/random/generator.html>`_).
    kwargs : dict
        Additional keyword arguments to pass to the random distribution (``np.random.normal``).

    Returns
    -------
    np.ndarray
        Signal matrix.

    Raises
    ------
    TypeError
        If the random state is not an integer or a ``Generator`` object.
    """
    if random_state is None:
        random_state = np.random.default_rng()
    elif isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    elif not isinstance(random_state, np.random.Generator):
        raise TypeError(
            f'The random state must be an integer or a numpy.random.Generator object, not {type(random_state)}!'
        )

    # Create the signal matrix
    cols = int(rows * ratio)
    U = random_state.normal(size=(rows, rank), **kwargs)
    V = random_state.normal(size=(rank, cols), **kwargs)
    signal = U @ V

    return signal / np.sqrt(rank)
