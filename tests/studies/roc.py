#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, signal_low
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp

#import matplotlib.pyplot as plt
#from .ownROCS import Make_Binned_ROC_histograms

# Global variable definition(s)
ROOT.gStyle.SetTitleOffset(2.0, 'y')


#@showsave
def roc (data_, args, feature_dict, masscut=False, pt_range=(200, 2000), title=None):
    """
    Perform study of ...

    Saves plot `figures/roc.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for ...
        masscut: ...
    """

    # Extract features and count appearance of each base variable
    features = []
    appearances = []
    for basevar in feature_dict.keys():
	for suffix in feature_dict[basevar]:
	    features.append(basevar+suffix)
	appearances.append(len(feature_dict[basevar]))

    print "features =", features

    # Select pT-range
    if pt_range is not None:
	if pt_range[0] > pt_range[1]:
	    data = data_.loc[((data_['pt'] > 200) & (data_['pt'] < pt_range[1])) | ((data_['pt'] < 2000) & (data_['pt'] > pt_range[0]))]	
	else:
            data = data_.loc[(data_['pt'] > pt_range[0]) & (data_['pt'] < pt_range[1])]
    else:
        data = data_
        pass
	

    # (Opt.) masscut | @NOTE: Duplication with adversarial/utils/metrics.py
    #msk = (data['m'] > 60.) & (data['m'] < 100.) if masscut else np.ones_like(data['signal']).astype(bool)
    msk = np.ones_like(data['signal']).astype(bool) if masscut==False else (data['m'] > masscut[0]) & (data['m'] < masscut[1]) 

    # Computing ROC curves
    ROCs = dict()
    for feat in features:

        sign = -1. if signal_low(feat) else 1


        eff_bkg, eff_sig, thresholds = roc_curve(data.loc[msk, 'signal'].values,
                                                 data.loc[msk, feat]    .values * sign,
                                                 sample_weight=data.loc[msk, 'weight_test'].values)

        if masscut:
            eff_sig_mass = np.mean(msk[data['signal'] == 1])
            eff_bkg_mass = np.mean(msk[data['signal'] == 0])

            eff_sig *= eff_sig_mass
            eff_bkg *= eff_bkg_mass
            pass

        # Filter, to advoid background rejection blowing up
        indices = np.where((eff_bkg > 0) & (eff_sig > 0))
        eff_sig = eff_sig[indices]
        eff_bkg = eff_bkg[indices]

        # Subsample to 1% steps
        targets = np.linspace(0, 1, 100 + 1, endpoint=True)
        indices = np.array([np.argmin(np.abs(eff_sig - t)) for t in targets])
        eff_sig = eff_sig[indices]
        eff_bkg = eff_bkg[indices]

        # Store
        ROCs[feat] = (eff_sig, eff_bkg)
        pass

    # Computing ROC AUCs
    AUCs = dict()
    for feat in features:
        sign = -1. if signal_low(feat) else 1.
        AUCs[feat] = roc_auc_score(data['signal'].values,
                                   data[feat]    .values * sign,
                                   sample_weight=data['weight_test'].values)
        pass

    # Report scores
    print "\n== pT range: {:s}".format('inclusive' if pt_range is None else "[{:.0f}, {:.0f}] Gev".format(*pt_range))
    print "\n== {} masscut".format("With" if masscut else "Without")
    for feat in features:
        effsig = ROCs[feat][0]
        idx = np.argmin(np.abs(effsig - 0.5))
        print "\nFeature {}:".format(feat)
        print "  Background rejection at effsig = {:.0f}%: {:6.3f}".format(ROCs[feat][0][idx] * 100., 1. / ROCs[feat][1][idx])
        print "  AUC: {:5.4f}".format(AUCs[feat])
        pass


    # Perform plotting
    c = plot(args, data, features, ROCs, AUCs, masscut, pt_range, appearances)

    # Output
    if title is None:
	#path = 'figures/roc{}{:s}.pdf'.format('__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[1]) if pt_range is not None else '', '__masscut' if masscut else '')
        path = 'figures/roc{}{:s}.pdf'.format('__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[1]) if pt_range is not None else '', '' if masscut==False else '__masscut_'+str(masscut[0])+'to'+str(masscut[1]))
    else:
	#path = 'figures/'+title+'_roc{}{:s}.pdf'.format('__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[1]) if pt_range is not None else '', '__masscut' if masscut else '')
        path = 'figures/'+title+'_roc{}{:s}.pdf'.format('__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[1]) if pt_range is not None else '', '' if masscut==False else '__masscut_'+str(masscut[0])+'to'+str(masscut[1]))

    c.save(path = path) 

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, features, ROCs, AUCs, masscut, pt_range, appearances = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    # -- Random guessing
    bins = np.linspace(0.2, 1., 100 + 1, endpoint=True)  # original representation
    #bins = np.linspace(0., 1., 100 + 1, endpoint=True) #comparison with JME-18-002
    bins = np.array([bins[0], bins[0] + 0.01 * np.diff(bins[:2])[0]] + list(bins[1:]))
    #bins = np.array([0.2] + list(bins[1:]))
    #edges = bins[1:-1]
    edges = bins
    centres = edges[:-1] + 0.5 * np.diff(edges)
    c.hist(np.power(centres, -1.), bins=edges, linecolor=ROOT.kGray + 2, fillcolor=ROOT.kBlack, alpha=0.05, linewidth=1, option='HISTC')
    linestyles = [1, 3, 5, 7]

    # -- ROCs
    if len(appearances) != 2:
        for is_simple in [True, False]:

            # Split the legend into simple- and MVA taggers
            indices = np.array([0]+appearances).cumsum()
            for i in range(len(indices)-1):
                for ifeat, feat in filter(lambda t: is_simple == signal_low(t[1]), enumerate(features[indices[i]:indices[i+1]])):
                    eff_sig, eff_bkg = ROCs[feat]
                    c.graph(np.power(eff_bkg, -1.), bins=eff_sig, linestyle=linestyles[ifeat], linecolor=rp.colours[i % len(rp.colours)], linewidth=2, label=latex(feat, ROOT=True), option='L')  # original representation
            	#c.graph(eff_bkg, bins=eff_sig, linestyle=1 + ifeat, linecolor=rp.colours[i % len(rp.colours)], linewidth=2, label=latex(feat, ROOT=True), option='L')  #comparison with JME-18-002
                    pass

            # Draw class-specific legend
            width = 0.17 #moved from 0.17 to 0.25 and back to 0.17
            c.legend(header=("Analytical:" if is_simple else "MVA:"),
              width=width, xmin=0.45 + (width + 0.06) * (is_simple), ymax=0.888)   # xmin moved from 0.58 to 0.45, inserted width translation of 0.06

    else:
	for first_var in [True, False]:

	    indices = np.array([0]+appearances).cumsum()
            for i in [0,1]:
		if i == 0 and not first_var: continue
		if i == 1 and first_var: continue
		for ifeat, feat in enumerate(features[indices[i]:indices[i+1]]):
                    eff_sig, eff_bkg = ROCs[feat]
                    c.graph(np.power(eff_bkg, -1.), bins=eff_sig, linestyle=linestyles[ifeat], linecolor=rp.colours[i % len(rp.colours)], linewidth=2, label=latex(feat, ROOT=True), option='L')  # original representation
            	    #c.graph(eff_bkg, bins=eff_sig, linestyle=1 + ifeat, linecolor=rp.colours[i % len(rp.colours)], linewidth=2, label=latex(feat, ROOT=True), option='L')  #comparison with JME-18-002
                    pass

            # Draw class-specific legend
            width = 0.15 #moved from 0.17 to 0.25 and back to 0.15
	    c.legend(header=(latex(features[0], ROOT=True)+"-based:" if first_var else latex(features[appearances[1]], ROOT=True)+"-based:"), 
              width=width, xmin=0.55 + (width + 0.06) * (first_var), ymax=0.888)   # xmin moved from 0.58 to 0.45, inserted width translation of 0.06



    # Decorations
    c.xlabel("Signal efficiency #varepsilon_{sig}^{rel}")
    c.ylabel("Background rejection 1/#varepsilon_{bkg}^{rel}")
    c.text([], xmin=0.15, ymax=0.96, qualifier=QUALIFIER, ATLAS=False)
    c.text(["#sqrt{s} = 13 TeV",
            "#it{W} jet tagging"] + (
                ["p_{{T}} #in  [{:.0f}, {:.0f}] GeV".format(pt_range[0], pt_range[1])] if pt_range is not None else []
            ) + (
                #["Cut: m #in  [60, 100] GeV"] if masscut else []
		[] if masscut==False else ["Cut: m #in  [{:.0f}, {:.0f}] GeV".format(masscut[0], masscut[1])]
            ),
           ATLAS=False)

    if masscut!=False: masscut = True
    ranges = int(pt_range is not None) + int(masscut)
    mult = 10. if ranges == 2 else (2. if ranges == 1 else 1.)

    c.latex("Random guessing", 0.4, 1./0.4 * 0.9, align=23, angle=-12 + 2 * ranges, textsize=13, textcolor=ROOT.kGray + 2)
    c.xlim(0.2, 1.)
    #c.ylim(1E+00, 5E+02 * mult) # original representation
    c.ylim(1E+00, 2E+02 * mult)
    #c.xlim(0., 1.)  #comparison with JME-18-002
    #c.ylim(1E-04, 1.)
    c.logy()
    c.legend()

    return c
