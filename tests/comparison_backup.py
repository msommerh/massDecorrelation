#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing comparison studies."""

# Basic import(s)
import re
import gc
import gzip
import itertools

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import numpy as np
import pandas as pd
import pickle
import root_numpy
from array import array
from scipy.stats import entropy
from sklearn.metrics import roc_curve, roc_auc_score

# Project import(s)
from adversarial.utils import initialise, initialise_backend, parse_args, load_data, mkdir, wpercentile, latex
from adversarial.profile import profile, Profile
from adversarial.constants import *
#from run.adversarial.common import initialise_config
from .studies.common import *
import studies

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Initialise Keras backend
    #initialise_backend(args)

    # Neural network-specific initialisation of the configuration dict
    #initialise_config(args, cfg)

    # Keras import(s)
    #import keras.backend as K
    #from keras.models import load_model

    # Project import(s)
    #from adversarial.models import classifier_model, adversary_model, combined_model, decorrelation_model

    # Load data
    #data, features, _ = load_data(args.input + 'data.h5', test=True)
    data, features, _ = load_data(args.input + 'data.h5', test_full_signal=True)


    # Common definitions
    # --------------------------------------------------------------------------
    # -- k-nearest neighbour
    kNN_var_N2 = 'N_{2}-k#minusNN'
    kNN_var_tau21 = 'tau_{21}-k#minusNN'

    def meaningful_digits (number):
        digits = 0
        if number > 0:
            digits = int(np.ceil(max(-np.log10(number), 0)))
            pass
        return '{l:.{d:d}f}'.format(d=digits,l=number)

    # -- Adversarial neural network (ANN) scan
    #lambda_reg  = 10.
    #lambda_regs = sorted([1., 3., 10.])
    #ann_vars    = list()
    #lambda_strs = list()
    #for lambda_reg_ in lambda_regs:
    #    lambda_str = meaningful_digits(lambda_reg_).replace('.', 'p')
    #    lambda_strs.append(lambda_str)

    #    ann_var_ = "ANN(#lambda={:s})".format(lambda_str.replace('p', '.'))
    #    ann_vars.append(ann_var_)
    #    pass

    #ann_var = ann_vars[lambda_regs.index(lambda_reg)]

    # -- uBoost scan
    #uboost_eff = 92
    #uboost_ur  = 0.3
    #uboost_urs = sorted([0., 0.01, 0.1, 0.3, 1.0])
    #uboost_var  =  'uBoost(#alpha={:s})'.format(meaningful_digits(uboost_ur))
    #uboost_vars = ['uBoost(#alpha={:s})'.format(meaningful_digits(ur)) for ur in uboost_urs]
    #uboost_pattern = 'uboost_ur_{{:4.2f}}_te_{:.0f}_rel21_fixed'.format(uboost_eff)

    # Tagger feature collection
    #tagger_features = ['Tau21','Tau21DDT', 'D2', kNN_var, 'D2', 'D2CSS', 'NN', ann_var, 'Adaboost', uboost_var]
    #tagger_features = ['tau21', 'tau21DDT', 'tau21', 'tau21kNN', 'tau21', 'tau21CSS', 'N2_B1', 'N2_B1DDT', 'N2_B1', 'N2_B1kNN', 'N2_B1', 'N2_B1CSS']; title="tau21_vs_N2_B1"
    #tagger_features = ['N2_B1', 'N2_B1DDT', 'N2_B1', 'N2_B1kNN', 'N2_B1', 'N2_B1CSS']; title="N2_B1"
    #tagger_features = ['tau21', 'tau21DDT', 'N2_B1', 'N2_B1kNN', 'N2_B1', 'N2_B1CSS']; title="ATLAS"
    tagger_features = ['decDeepWvsQCD', 'decDeepWvsQCDDDT', 'decDeepWvsQCD', 'decDeepWvsQCDkNN', 'decDeepWvsQCD', 'decDeepWvsQCDCSS']; title="decDeep"
    tagger_features = ['DeepWvsQCD', 'DeepWvsQCDDDT', 'DeepWvsQCD', 'DeepWvsQCDkNN', 'DeepWvsQCD', 'DeepWvsQCDCSS']; title="Deep"

    # Add variables
    # --------------------------------------------------------------------------
    with Profile("Add variables"):

        ## Tau21DDT
        #from run.ddt.common import add_ddt
        #add_ddt(data, feat='tau21', path='models/ddt/ddt_tau21.pkl.gz') 

        ## N2DDT
        #from run.ddt.common import add_ddt
        #add_ddt(data, feat='N2_B1', path='models/ddt/ddt_N2_B1.pkl.gz') 

	## decDeepQvsQCDDDT
        #from run.ddt.common import add_ddt
        #add_ddt(data, feat='decDeepWvsQCD', path='models/ddt/ddt_decDeepWvsQCD.pkl.gz') 

	# DeepQvsQCDDDT
        from run.ddt.common import add_ddt
        add_ddt(data, feat='DeepWvsQCD', path='models/ddt/ddt_DeepWvsQCD.pkl.gz') 

        ## Tau21-kNN
        #from run.knn.common import add_knn, VAR_TAU21 as kNN_basevar, TAU21_EFF as kNN_eff
        #print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var_tau21)
        #add_knn(data, feat=kNN_basevar, path='models/knn/knn_{}_{}.pkl.gz'.format(kNN_basevar, kNN_eff))

        ## N2-kNN
        #from run.knn.common import add_knn, VAR_N2 as kNN_basevar, N2_EFF as kNN_eff
        #print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var_N2)
        #add_knn(data, feat=kNN_basevar, path='models/knn/knn_{}_{}.pkl.gz'.format(kNN_basevar, kNN_eff))

        ## decDeepWvsQCD-kNN
        #from run.knn.common import add_knn, VAR_DECDEEP as kNN_basevar, DECDEEP_EFF as kNN_eff
        #print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var_N2)
        #add_knn(data, feat=kNN_basevar, path='models/knn/knn_{}_{}.pkl.gz'.format(kNN_basevar, kNN_eff))

        # DeepWvsQCD-kNN
        from run.knn.common import add_knn, VAR_DEEP as kNN_basevar, DEEP_EFF as kNN_eff
        print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var_N2)
        add_knn(data, feat=kNN_basevar, path='models/knn/knn_{}_{}.pkl.gz'.format(kNN_basevar, kNN_eff))

        ## Tau21-CSS
        #from run.css.common import add_css
        #add_css("tau21", data)

        ## N2-CSS
        #from run.css.common import add_css
        #add_css("N2_B1", data)

        ## decDeepWvsQCD-CSS
        #from run.css.common import add_css
        #add_css("decDeepWvsQCD", data)

        # DeepWvsQCD-CSS
        from run.css.common import add_css
        add_css("DeepWvsQCD", data)

        pass

    # Remove unused variables
    #used_variables = set(tagger_features + ann_vars + uboost_vars + ['m', 'pt', 'npv', 'weight_test'])
    used_variables = set(tagger_features + ['m', 'pt', 'weight_test', 'npv']) ## need to put 'npv' back in for robustness study
    unused_variables = [var for var in list(data) if var not in used_variables]
    data.drop(columns=unused_variables)
    gc.collect()

    # Perform performance studies
    #perform_studies (data, args, tagger_features, ann_vars, uboost_vars)
    perform_studies (data, args, tagger_features, title=title)

    return 0


#def perform_studies (data, args, tagger_features, ann_vars, uboost_vars):
def perform_studies (data, args, tagger_features, title=None):
    """
    Method delegating performance studies.
    """
    #masscuts  = [True, False]
    masscuts = [False]
    pt_ranges = [None, (200, 500), (500, 1000), (1000, 2000)]


    ## Perform combined robustness study    
    #with Profile("Study: Robustness"):
    #    for masscut in masscuts:
    #        studies.robustness_full(data, args, tagger_features, masscut=masscut, title=title)
    #        pass
    #    pass

    ## Perform jet mass distribution comparison study
    #with Profile("Study: Jet mass comparison"):
    #    for pt_range in pt_ranges:
    #        print "pt_range =", pt_range 
    #        studies.jetmasscomparison(data, args, tagger_features, pt_range, title=title)
    #    pass

    # Perform summary plot study  
    with Profile("Study: Summary plot"):
        #regex_nn = re.compile('\#lambda=[\d\.]+')
        #regex_ub = re.compile('\#alpha=[\d\.]+')

        #scan_features = {'NN':       map(lambda feat: (feat, regex_nn.search(feat).group(0)), ann_vars),
        #                 'Adaboost': map(lambda feat: (feat, regex_ub.search(feat).group(0)), uboost_vars)
        #                 }
        scan_features = dict()

        for masscut, pt_range in itertools.product(masscuts, pt_ranges):
            studies.summary(data, args, tagger_features, scan_features, masscut=masscut, pt_range=pt_range, title=title)
            pass
        pass

    ## Perform distributions study
    #with Profile("Study: Substructure tagger distributions"):
    #    mass_ranges = np.linspace(50, 300, 5 + 1, endpoint=True)
    #    mass_ranges = [None] + zip(mass_ranges[:-1], mass_ranges[1:])
    #    for feat, pt_range, mass_range in itertools.product(tagger_features, pt_ranges, mass_ranges):  # tagger_features
    #        studies.distribution(data, args, feat, pt_range, mass_range, title=title)
    #        pass
    #    pass

    # Perform ROC study
    with Profile("Study: ROC"):
        for masscut, pt_range in itertools.product(masscuts, pt_ranges):
            studies.roc(data, args, tagger_features, masscut=masscut, pt_range=pt_range, title=title)
            pass
        pass

    ## Perform JSD study
    #with Profile("Study: JSD"):
    #    for pt_range in pt_ranges:
    #        studies.jsd(data, args, tagger_features, pt_range, title=title)
    #        pass
    #    pass

    ## Perform efficiency study
    #with Profile("Study: Efficiency"):
    #    for feat in tagger_features:
    #        studies.efficiency(data, args, feat, title=title)
    #        pass
    #    pass

    #return

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
