#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing DDT transform."""

# Basic import(s)
import gzip

# Scientific import(s)
import ROOT
import root_numpy
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.utils import loadclf
from adversarial.profile import profile

# Common definition(s)
BINS = np.linspace(-1, 6, 7 * 4 + 1, endpoint=True)  # Binning in rhoDDT
FIT_RANGE_TAU21 = (1.5, 4.0) # Range in rhoDDT to be fitted
FIT_RANGE_N2 = (2.1, 4.0)
FIT_RANGE_DECDEEP = (0.2, 2.5)
FIT_RANGE_DEEP = (2.1, 2.8)
VAR_TAU21  = 'tau21'
VAR_N2 = 'N2_B1'
VAR_DECDEEP = 'decDeepWvsQCD'
VAR_DEEP = 'DeepWvsQCD'
VAR_RHODDT = 'rhoDDT'    # 'rhoDDT'
VAR_WEIGHT = 'weight_test'

@profile
def add_ddt (data, feat=VAR_TAU21, newfeat=None, path='models/ddt/ddt_{}.pkl.gz'.format(VAR_TAU21)):
    """
    Add DDT-transformed `feat` to `data`. Modifies `data` in-place.

    Arguments:
        data: Pandas DataFrame to which to add the DDT-transformed variable.
        feat: Substructure variable to be decorrelated.
        newfeat: Name of output feature. By default, `{feat}DDT`.
        path: Path to trained DDT transform model.
    """

    # Check(s)
    if newfeat is None:
        newfeat = feat + 'DDT'
        pass

    # Load model
    ddt = loadclf(path)

    # Add new classifier to data array
    data[newfeat] = pd.Series(data[feat] - ddt.predict(data[[VAR_RHODDT]].values), index=data.index)
    return


@profile
def fill_profile (data, var):
    """
    Fill ROOT.TProfile with the average `var` as a function of rhoDDT.
    """

    profile = ROOT.TProfile('profile_{}'.format(var), "", len(BINS) - 1, BINS)
    root_numpy.fill_profile(profile, data[[VAR_RHODDT, var]].values, weights=data[VAR_WEIGHT].values)
    return profile
