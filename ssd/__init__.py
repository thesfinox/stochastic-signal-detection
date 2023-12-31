# -*- coding: utf-8 -*-
"""
SSD: Stochastic Signal Detection

A Python package for the detection of signals using stochastic processes
"""

# Import modules
from .distributions import (EmpiricalDistribution, HistogramDistribution,
                            InterpolateDistribution, InverseMarchenkoPastur,
                            InverseWignerSemicircle, MarchenkoPastur,
                            TranslatedInverseMarchenkoPastur,
                            TranslatedMarchenkoPastur, WignerSemicircle)
from .ssd import SSD

# Set the version number
__version__ = '1.0.0'

# Set the author information
__author__ = ', '.join([
    'Harold Erbin',
    'Riccardo Finotello',
    'Bio Wahabou Kpera',
    'Vincent Lahoche',
    'Dine Ousmane Samary',
])
__email__ = 'riccardo.finotello@cea.fr'

# Set the license and package information
__license__ = ''
__url__ = ''

# Package imports
__all__ = [
    'EmpiricalDistribution',
    'InverseMarchenkoPastur',
    'MarchenkoPastur',
    'TranslatedInverseMarchenkoPastur',
    'TranslatedMarchenkoPastur',
    'WignerSemicircle',
    'InverseWignerSemicircle',
    'HistogramDistribution',
    'InterpolateDistribution',
    'SSD',
]
