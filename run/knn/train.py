#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf  #, initialise_backend
from adversarial.profile import profile, Profile
from adversarial.constants import *
#from run.adversarial.common import initialise_config

# Local import(s)
from .common import *

WORKING_POINTS = [5,10,15,20,25,30,35,40,45,50]

# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    #data, _, _ = load_data(args.input + 'data.h5', train=True)
    data, _, _ = load_data(args.input + 'data.h5', train_full_signal=True)

    #variable = VARTAU21
    #bg_eff = TAU21EFF
    variable = VARN2
    #bg_eff = N2EFF

    # training on a list of working points:
    for bg_eff in WORKING_POINTS:
	train(data, variable, bg_eff)
    return 0

    # -------------------------------------------------------------------------
    ####
    #### # Initialise Keras backend
    #### initialise_backend(args)
    ####
    #### # Neural network-specific initialisation of the configuration dict
    #### initialise_config(args, cfg)
    ####
    #### # Keras import(s)
    #### from keras.models import load_model
    ####
    #### # NN
    #### from run.adversarial.common import add_nn
    #### with Profile("NN"):
    ####     classifier = load_model('models/adversarial/classifier/full/classifier.h5')
    ####     add_nn(data, classifier, 'NN')
    ####     pass
    # -------------------------------------------------------------------------

    ## Compute background efficiency at sig. eff. = 50%
    #eff_sig = 0.5
    #fpr, tpr, thresholds = roc_curve(data['signal'], data[variable], sample_weight=data['weight_test'])
    #idx = np.argmin(np.abs(tpr - eff_sig))
    #print "Background acceptance @ {:.2f}% sig. eff.: {:.2f}% ({} < {:.2f})".format(eff_sig * 100., (1 - fpr[idx]) * 100., variable, thresholds[idx])
    #print "Chosen target efficiency: {:.2f}%".format(bg_eff)

    ## Filling profile
    #data = data[data['signal'] == 0]
    #profile_meas, (x,y,z) = fill_profile(data, variable, bg_eff)

    ## Format arrays
    #X = np.vstack((x.flatten(), y.flatten())).T
    #Y = z.flatten()

    ## Fit KNN regressor
    #knn = KNeighborsRegressor(weights='distance')
    #knn.fit(X, Y)

    ## Save KNN classifier
    #saveclf(knn, 'models/knn/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))
    #print "reached end of main()"
    #return 0

def train(data, variable, bg_eff):
    # Filling profile
    data = data[data['signal'] == 0]
    profile_meas, (x,y,z) = fill_profile(data, variable, bg_eff)

    # Format arrays
    X = np.vstack((x.flatten(), y.flatten())).T
    Y = z.flatten()

    # Fit KNN regressor
    knn = KNeighborsRegressor(weights='distance')
    knn.fit(X, Y)

    # Save KNN classifier
    saveclf(knn, 'models/knn2/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()  # (adversarial=True)

    # Call main function
    main(args)
    pass
