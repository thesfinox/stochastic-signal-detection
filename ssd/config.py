# -*- coding: utf-8 -*-
"""
Configuration

Create a list of default parameters for the SSD simulation.
"""
from yacs.config import CfgNode as CN


def get_cfg_defaults():

    # Initialize config
    _C = CN()

    # Output files and directories
    _C.OUTPUT = CN()  # output parameters

    _C.OUTPUT.OUTPUT_DIR = 'simulation'

    _C.OUTPUT.DB = CN()  # database parameters

    _C.OUTPUT.DB.OUTPUT_FILE = 'simulation.sqlite'
    _C.OUTPUT.DB.TABLE = 'simulation'

    # Input parameters
    _C.INPUT = CN()  # input parameters

    _C.INPUT.MATRIX = CN()  # matrix parameters
    _C.INPUT.MATRIX.ROWS = 1000
    _C.INPUT.MATRIX.COLUMNS = 800
    _C.INPUT.MATRIX.SEED = 42

    _C.INPUT.SIGNAL = CN()  # signal parameters
    _C.INPUT.SIGNAL.RATIO = 0.5
    _C.INPUT.SIGNAL.BY_DET = CN()  # signal by deterministic matrix
    _C.INPUT.SIGNAL.BY_DET.ENABLED = True
    _C.INPUT.SIGNAL.BY_DET.RANK = 500
    _C.INPUT.SIGNAL.BY_IMG = CN()  # signal by image
    _C.INPUT.SIGNAL.BY_IMG.ENABLED = False
    _C.INPUT.SIGNAL.BY_IMG.FILE = 'image.png'

    _C.INPUT.BINNING = CN()  # binning parameters
    _C.INPUT.BINNING.BINS = 100
    _C.INPUT.BINNING.SMOOTHING = 0.3

    _C.INPUT.E_SCALE = CN()  # energy scale of the simulation
    _C.INPUT.E_SCALE.BY_VALUE = CN()  # fixed energy scale
    _C.INPUT.E_SCALE.BY_VALUE.ENABLED = False
    _C.INPUT.E_SCALE.BY_VALUE.MIN = 0.0
    _C.INPUT.E_SCALE.BY_VALUE.MAX = 1.0
    _C.INPUT.E_SCALE.BY_MASS_SCALE = CN()  # energy scale at mass scale
    _C.INPUT.E_SCALE.BY_MASS_SCALE.ENABLED = True
    _C.INPUT.E_SCALE.BY_MASS_SCALE.WIDTH = 0.25

    _C.INPUT.INIT = CN()  # initial conditions of the simulation
    _C.INPUT.INIT.BY_TEMP = CN()  # choice by temperature
    _C.INPUT.INIT.BY_TEMP.ENABLED = False
    _C.INPUT.INIT.BY_TEMP.T = 1.0
    _C.INPUT.INIT.BY_TEMP.TC = 0.5
    _C.INPUT.INIT.BY_PARAMS = CN()  # choice by parameters
    _C.INPUT.INIT.BY_PARAMS.ENABLED = False
    _C.INPUT.INIT.BY_PARAMS.MU_0 = 1.0
    _C.INPUT.INIT.BY_PARAMS.MU_1 = 1.0
    _C.INPUT.INIT.BY_PARAMS.MU_2 = 0.0
    _C.INPUT.INIT.BY_INIT = CN()  # choice by initial conditions
    _C.INPUT.INIT.BY_INIT.ENABLED = True
    _C.INPUT.INIT.BY_INIT.KAPPA_0 = 0.5
    _C.INPUT.INIT.BY_INIT.MU_0 = 1.0
    _C.INPUT.INIT.BY_INIT.MU_1 = 1.0

    # Simulation parameters
    _C.SIM = CN()  # simulation parameters

    _C.SIM.INF = 0.0
    _C.SIM.SUP = 1.0
    _C.SIM.N_VALUES = 1000
    _C.SIM.N_STEPS = 1000
    _C.SIM.PERIODIC = False
    _C.SIM.IR_TO_UV = False

    return _C.clone()
