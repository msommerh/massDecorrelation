#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle
import argparse #newly added
import itertools

# Scientific import(s)
import numpy as np
import ROOT
import root_numpy
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf, wpercentile, garbage_collect, loadclf
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
#from .common import *

# Project import(s)
from adversarial.utils import latex, parse_args, initialise, load_data, mkdir, loadclf  #, initialise_backend
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp

# Global definitions
VARX = 'rho'
VARY = 'pt'
VARS = [VARX, VARY]
AXIS = {      # Dict holding (num_bins, axis_min, axis_max) for axis variables
    'rho': (20, -7.0, -1.5),
    'pt':  (20, 200., 1200.),
}

BOUNDS = [
    ROOT.TF1('bounds_0', "TMath::Sqrt( TMath::Power( 50, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2]),
    ROOT.TF1('bounds_1', "TMath::Sqrt( TMath::Power(300, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2])
    ]

NB_CONTOUR = 13 * 16

BOUNDS[0].SetLineColor(ROOT.kGray + 3)
BOUNDS[1].SetLineColor(ROOT.kGray + 3)
for bound in BOUNDS:
    bound.SetLineWidth(1)
    bound.SetLineStyle(2)
    pass

#ZRANGE = (0., 0.8)  # for tau21
#ZRANGE = (0., 0.5) # for N2
#ZRANGE = (0.12,0.3) # for N2 in comparison with CMS-EXO-17-001
#ZRANGE = (0., 1.) # for decDeepWvsQCD
#ZRANGE = (0., 1.) # for DeepWvsQCD

#WORKING_POINTS = [5,10,15,20,25,30,35,40,45,50]
#WORKING_POINTS = [5,20]

parser = argparse.ArgumentParser(description="kNN fitter")
parser.add_argument('--input', default='input/data.h5',
                    help="Name of input HDF5 file.")
parser.add_argument('--variable', default='DeepWvsQCD',
                    help="Name of the variable to be decorrelated")
parser.add_argument('--signal_low', type=bool, default=False,
		    help="Set true if the signal peaks below the background.")
parser.add_argument('--working_points', nargs='+', type=float, default=[15],
                    help="List of background efficiencies in percent on which to perform the fit.")
parser.add_argument('--zrange', nargs=2, type=float, default=(0.,1.),
                    help="Tuple of the desired z-axis range for the percentile plots.")
parser.add_argument('--output', default='knn_fitter',
                    help="Directory in which to store the fit models and figures.")


# Main function definition
@profile
def main (args):

    # Initialise
    #args, cfg = initialise(args)
    WORKING_POINTS = args.working_points
    variable = args.variable
    signal_above = not args.signal_low

    # Load data
    #data, _, _ = load_data(args.input + 'data.h5', train_full_signal=True)
    data, _, _ = load_data(args.input, train_full_signal=True)

    #variable = VAR_TAU21; signal_above=False
    #bg_eff = TAU21_EFF
    #variable = VAR_N2; signal_above=False
    #bg_eff = N2_EFF
    #variable = VAR_DECDEEP; signal_above=True
    #bg_eff = DECDEEP_EFF
    #variable = VAR_DEEP; signal_above=True
    #bg_eff = DEEP_EFF

    # training on a list of working points:
    sign = 1. if signal_above else -1.
    fpr, tpr, thresholds = roc_curve(data['signal'], data[variable] * sign, sample_weight=data['weight_test'])

    for bg_eff in WORKING_POINTS:
	idx = np.argmin(np.abs(fpr - 0.01*bg_eff))
	print "Evaluating on a background efficiency of {:.2f} ({} {} {:.2f}), corresponding to a signal efficiency of {:.2f}%.".format(bg_eff, variable, '>' if signal_above else '<', thresholds[idx], 100.*tpr[idx])
        train(data, variable, bg_eff, signal_above=signal_above)
	test(data, variable, bg_eff, signal_above=signal_above)

    print "reached end of main()"
    return 0

def train(data, variable, bg_eff, signal_above=False):
    # Filling profile
    data_ = data[data['signal'] == 0]
    profile_meas, (x,y,z) = fill_profile(data_, variable, bg_eff, signal_above=signal_above)

    # Format arrays
    X = np.vstack((x.flatten(), y.flatten())).T
    Y = z.flatten()

    # Fit KNN regressor
    knn = KNeighborsRegressor(weights='distance')
    knn.fit(X, Y)

    # Save KNN classifier
    saveclf(knn, args.output+'/models/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))
    #saveclf(knn, 'knn_fitter/models/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))

def test(data, variable, bg_eff, signal_above=False):
    # Shout out to Cynthia Brewer and Mark Harrower
    # [http://colorbrewer2.org]. Palette is colorblind-safe.
    rgbs = [
        (247/255., 251/255., 255/255.),
        (222/255., 235/255., 247/255.),
        (198/255., 219/255., 239/255.),
        (158/255., 202/255., 225/255.),
        (107/255., 174/255., 214/255.),
        ( 66/255., 146/255., 198/255.),
        ( 33/255., 113/255., 181/255.),
        (  8/255.,  81/255., 156/255.),
        (  8/255.,  48/255., 107/255.)
        ]
    
    red, green, blue = map(np.array, zip(*rgbs))
    nb_cols = len(rgbs)
    stops = np.linspace(0, 1, nb_cols, endpoint=True)
    ROOT.TColor.CreateGradientColorTable(nb_cols, stops, red, green, blue, NB_CONTOUR)

    msk_sig = data['signal'] == 1
    msk_bkg = ~msk_sig

    # Fill measured profile
    with Profile("filling profile"):
        profile_meas, _ = fill_profile(data[msk_bkg], variable, bg_eff, signal_above=signal_above)

    # Add k-NN variable
    with Profile("adding variable"):
        knnfeat = 'knn'
        #add_knn(data, feat=variable, newfeat=knnfeat, path='knn_fitter/models/knn_{}_{}.pkl.gz'.format(variable, bg_eff))
        add_knn(data, feat=variable, newfeat=knnfeat, path=args.output+'/models/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))

    # Loading KNN classifier
    with Profile("loading model"):
        #knn = loadclf('knn_fitter/models/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))
        knn = loadclf(args.output+'/models/knn_{:s}_{:.0f}.pkl.gz'.format(variable, bg_eff))

    # Filling fitted profile
    with Profile("Filling fitted profile"):
        rebin = 8
        edges, centres = dict(), dict()
        for ax, var in zip(['x', 'y'], [VARX, VARY]):

            # Short-hands
            vbins, vmin, vmax = AXIS[var]

            # Re-binned bin edges
            edges[ax] = np.interp(np.linspace(0,    vbins, vbins * rebin + 1, endpoint=True),
                                  range(vbins + 1),
                                  np.linspace(vmin, vmax,  vbins + 1,         endpoint=True))

            # Re-binned bin centres
            centres[ax] = edges[ax][:-1] + 0.5 * np.diff(edges[ax])
            pass


        # Get predictions evaluated at re-binned bin centres
        g = dict()
        g['x'], g['y'] = np.meshgrid(centres['x'], centres['y'])
        g['x'], g['y'] = standardise(g['x'], g['y'])

        X = np.vstack((g['x'].flatten(), g['y'].flatten())).T
        fit = knn.predict(X).reshape(g['x'].shape).T

        # Fill ROOT "profile"
        profile_fit = ROOT.TH2F('profile_fit', "", len(edges['x']) - 1, edges['x'].flatten('C'),
                                                   len(edges['y']) - 1, edges['y'].flatten('C'))
        root_numpy.array2hist(fit, profile_fit)
        pass


    # Plotting
    for fit in [False, True]:

        # Select correct profile
        profile = profile_fit if fit else profile_meas

        # Plot
        plot(profile, fit, variable, bg_eff)
        pass
    pass

    # Plotting local selection efficiencies for D2-kNN < 0
    # -- Compute signal efficiency
    for sig, msk in zip([True, False], [msk_sig, msk_bkg]):
        if sig:
                print "working on signal"
        else:
                print "working on bg"

        if sig:
            rgbs = [
                (247/255., 251/255., 255/255.),
                (222/255., 235/255., 247/255.),
                (198/255., 219/255., 239/255.),
                (158/255., 202/255., 225/255.),
                (107/255., 174/255., 214/255.),
                ( 66/255., 146/255., 198/255.),
                ( 33/255., 113/255., 181/255.),
                (  8/255.,  81/255., 156/255.),
                (  8/255.,  48/255., 107/255.)
                ]

            red, green, blue = map(np.array, zip(*rgbs))
            nb_cols = len(rgbs)
            stops = np.linspace(0, 1, nb_cols, endpoint=True)
        else:
            rgbs = [
                (255/255.,  51/255.,   4/255.),
                (247/255., 251/255., 255/255.),
                (222/255., 235/255., 247/255.),
                (198/255., 219/255., 239/255.),
                (158/255., 202/255., 225/255.),
                (107/255., 174/255., 214/255.),
                ( 66/255., 146/255., 198/255.),
                ( 33/255., 113/255., 181/255.),
                (  8/255.,  81/255., 156/255.),
                (  8/255.,  48/255., 107/255.)
                ]

            red, green, blue = map(np.array, zip(*rgbs))
            nb_cols = len(rgbs)
            stops = np.array([0] + list(np.linspace(0, 1, nb_cols - 1, endpoint=True) * (1. - bg_eff / 100.) + bg_eff / 100.))
            pass

            ROOT.TColor.CreateGradientColorTable(nb_cols, stops, red, green, blue, NB_CONTOUR)

        # Define arrays
        shape   = (AXIS[VARX][0], AXIS[VARY][0])
        bins    = [np.linspace(AXIS[var][1], AXIS[var][2], AXIS[var][0] + 1, endpoint=True) for var in VARS]
        x, y, z = (np.zeros(shape) for _ in range(3))

        # Create `profile` histogram
        profile = ROOT.TH2F('profile', "", len(bins[0]) - 1, bins[0].flatten('C'), len(bins[1]) - 1, bins[1].flatten('C'))

        # Compute inclusive efficiency in bins of `VARY`
        effs = list()
        for edges in zip(bins[1][:-1], bins[1][1:]):
            msk_bin  = (data[VARY] > edges[0]) & (data[VARY] < edges[1])
            if signal_above:
                msk_pass = data[knnfeat] > 0  # ensure correct cut direction
            else:
                msk_pass =  data[knnfeat] < 0
	    num_msk = msk * msk_bin * msk_pass
            num = data.loc[num_msk, 'weight_test'].values.sum()
            den = data.loc[msk & msk_bin,            'weight_test'].values.sum()
            effs.append(num/den)
            pass

        # Fill profile
        with Profile("Fill profile"):
            for i,j in itertools.product(*map(range, shape)):
                #print "Fill profile - (i, j) = ({}, {})".format(i,j)
                # Bin edges in x and y
                edges = [bin[idx:idx+2] for idx, bin in zip([i,j],bins)]

                # Masks
                msks = [(data[var] > edges[dim][0]) & (data[var] <= edges[dim][1]) for dim, var in enumerate(VARS)]
                msk_bin = reduce(lambda x,y: x & y, msks)

                # Set non-zero bin content
                if np.sum(msk & msk_bin):
                    if signal_above:
                        msk_pass = data[knnfeat] > 0 # ensure correct cut direction
                    else:
                        msk_pass = data[knnfeat] < 0
                    num_msk = msk * msk_bin * msk_pass
                    num = data.loc[num_msk, 'weight_test'].values.sum()
                    den = data.loc[msk & msk_bin,            'weight_test'].values.sum()
                    eff = num/den
                    profile.SetBinContent(i + 1, j + 1, eff)
                    pass

        c = rp.canvas(batch=True)
        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetRightMargin(0.20)
        pad.SetLeftMargin(0.15)
        pad.SetTopMargin(0.10)

        # Styling
        profile.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True) + " = log(m^{2}/p_{T}^{2})")
        profile.GetYaxis().SetTitle("Large-#it{R} jet " + latex(VARY, ROOT=True) + " [GeV]")
        profile.GetZaxis().SetTitle("Selection efficiency for %s^{(%s%%)}" % (latex(variable, ROOT=True), bg_eff))

        profile.GetYaxis().SetNdivisions(505)
        profile.GetZaxis().SetNdivisions(505)
        profile.GetXaxis().SetTitleOffset(1.4)
        profile.GetYaxis().SetTitleOffset(1.8)
        profile.GetZaxis().SetTitleOffset(1.3)
        zrange = (0., 1.)
        if zrange:
            profile.GetZaxis().SetRangeUser(*zrange)
            pass
        profile.SetContour(NB_CONTOUR)

        # Draw
        profile.Draw('COLZ')

        # Decorations
        c.text(qualifier=QUALIFIER, ymax=0.92, xmin=0.15, ATLAS=False)
        c.text(["#sqrt{s} = 13 TeV", "#it{W} jets" if sig else "Multijets"], ATLAS=False)

        # -- Efficiencies
        xaxis = profile.GetXaxis()
        yaxis = profile.GetYaxis()
        tlatex = ROOT.TLatex()
        tlatex.SetTextColor(ROOT.kGray + 2)
        tlatex.SetTextSize(0.023)
        tlatex.SetTextFont(42)
        tlatex.SetTextAlign(32)
        xt = xaxis.GetBinLowEdge(xaxis.GetNbins())
        for eff, ibin in zip(effs,range(1, yaxis.GetNbins() + 1)):
            yt = yaxis.GetBinCenter(ibin)
            tlatex.DrawLatex(xt, yt, "%s%.1f%%" % ("#bar{#varepsilon}^{rel}_{%s} = " % ('sig' if sig else 'bkg') if ibin == 1 else '', eff * 100.))
            pass

        # -- Bounds
        BOUNDS[0].DrawCopy("SAME")
        BOUNDS[1].DrawCopy("SAME")
        c.latex("m > 50 GeV",  -4.5, BOUNDS[0].Eval(-4.5) + 30, align=21, angle=-37, textsize=13, textcolor=ROOT.kGray + 3)
        c.latex("m < 300 GeV", -2.5, BOUNDS[1].Eval(-2.5) - 30, align=23, angle=-57, textsize=13, textcolor=ROOT.kGray + 3)

        # Save
        mkdir('knn_fitter/figures/')
        c.save('knn_fitter/figures/knn_eff_{}_{:s}_{:.0f}.pdf'.format('sig' if sig else 'bkg', variable, bg_eff))
        mkdir(args.output+'/figures/')
        c.save(args.output+'/figures/knn_eff_{}_{:s}_{:.0f}.pdf'.format('sig' if sig else 'bkg', variable, bg_eff))
        pass

    return

def plot (profile, fit, variable, bg_eff):
    """
    Method for delegating plotting.
    """
    ZRANGE = args.zrange

    # rootplotting
    c = rp.canvas(batch=True)
    pad = c.pads()[0]._bare()
    pad.cd()
    pad.SetRightMargin(0.20)
    pad.SetLeftMargin(0.15)
    pad.SetTopMargin(0.10)

    # Styling
    profile.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True) + " = log(m^{2}/p_{T}^{2})")
    profile.GetYaxis().SetTitle("Large-#it{R} jet " + latex(VARY, ROOT=True) + " [GeV]")
    profile.GetZaxis().SetTitle("%s %s^{(%s%%)}" % ("#it{k}-NN fitted" if fit else "Measured", latex(variable, ROOT=True), bg_eff))

    profile.GetYaxis().SetNdivisions(505)
    profile.GetZaxis().SetNdivisions(505)
    profile.GetXaxis().SetTitleOffset(1.4)
    profile.GetYaxis().SetTitleOffset(1.8)
    profile.GetZaxis().SetTitleOffset(1.3)
    if ZRANGE:
        profile.GetZaxis().SetRangeUser(*ZRANGE)
        pass
    profile.SetContour(NB_CONTOUR)

    # Draw
    profile.Draw('COLZ')
    BOUNDS[0].DrawCopy("SAME")
    BOUNDS[1].DrawCopy("SAME")
    c.latex("m > 50 GeV",  -4.5, BOUNDS[0].Eval(-4.5) + 30, align=21, angle=-37, textsize=13, textcolor=ROOT.kGray + 3)
    c.latex("m < 300 GeV", -2.5, BOUNDS[1].Eval(-2.5) - 30, align=23, angle=-57, textsize=13, textcolor=ROOT.kGray + 3)

    # Decorations
    c.text(qualifier=QUALIFIER, ymax=0.92, xmin=0.15, ATLAS=False)
    c.text(["#sqrt{s} = 13 TeV", "Multijets"], ATLAS=False, textcolor=ROOT.kWhite)

    # Save
    mkdir(args.output+'/figures/')
    c.save(args.output+'/figures/knn_{}_{:s}_{:.0f}.pdf'.format('fit' if fit else 'profile', variable, bg_eff))
    #mkdir('knn_fitter/figures/')
    #c.save('knn_fitter/figures/knn_{}_{:s}_{:.0f}.pdf'.format('fit' if fit else 'profile', variable, bg_eff))
    pass

@garbage_collect
def standardise (array, y=None):
    """
    Standardise axis-variables for kNN regression.

    Arguments:
        array: (N,2) numpy array or Pandas DataFrame containing axis variables.

    Returns:
        (N,2) numpy array containing standardised axis variables.
    """

    # If DataFrame, extract relevant columns and call method again.
    if isinstance(array, pd.DataFrame):
        X = array[[VARX, VARY]].values.astype(np.float)
        return standardise(X)

    # If receiving separate arrays
    if y is not None:
        x = array
        assert x.shape == y.shape
        shape = x.shape
        X = np.vstack((x.flatten(), y.flatten())).T
        X = standardise(X)
        x,y = list(X.T)
        x = x.reshape(shape)
        y = y.reshape(shape)
        return x,y

    # Check(s)
    assert array.shape[1] == 2

    # Standardise
    X = np.array(array, dtype=np.float)
    for dim, var in zip([0,1], [VARX, VARY]):
        X[:,dim] -= float(AXIS[var][1])
        X[:,dim] /= float(AXIS[var][2] - AXIS[var][1])
        pass

    return X

@profile
def add_knn (data, feat='N2_B1', newfeat=None, path=None):
    """
    Add kNN-transformed `feat` to `data`. Modifies `data` in-place.

    Arguments:
        data: Pandas DataFrame to which to add the kNN-transformed variable.
        feat: Substructure variable to be decorrelated.
        newfeat: Name of output feature. By default, `{feat}kNN`.
        path: Path to trained kNN transform model.
    """

    # Check(s)
    assert path is not None, "add_knn: Please specify a model path."
    if newfeat is None:
        newfeat = '{}kNN'.format(feat)
        pass

    # Prepare data array
    X = standardise(data)

    # Load model
    knn = loadclf(path)

    # Add new classifier to data array
    data[newfeat] = pd.Series(data[feat] - knn.predict(X).flatten(), index=data.index)
    return

@profile
def fill_profile (data, variable, bg_eff, signal_above=False):
    """Fill ROOT.TH2F with the measured, weighted values of the bg_eff-percentile
    of the background `VAR`. """

    if signal_above: bg_eff = 100. - bg_eff  # ensures that region above cut is counted as signal, not below

    # Define arrays
    shape   = (AXIS[VARX][0], AXIS[VARY][0])
    bins    = [np.linspace(AXIS[var][1], AXIS[var][2], AXIS[var][0] + 1, endpoint=True) for var in VARS]
    x, y, z = (np.zeros(shape) for _ in range(3))

    # Create `profile` histogram
    profile = ROOT.TH2F('profile', "", len(bins[0]) - 1, bins[0].flatten('C'), len(bins[1]) - 1, bins[1].flatten('C')
)

    # Fill profile
    for i,j in itertools.product(*map(range, shape)):

        # Bin edges in x and y
        edges = [bin[idx:idx+2] for idx, bin in zip([i,j],bins)]

        # Masks
        msks = [(data[var] > edges[dim][0]) & (data[var] <= edges[dim][1]) for dim, var in enumerate(VARS)]
        msk = reduce(lambda x,y: x & y, msks)

        # Percentile
        perc = np.nan
        if np.sum(msk) > 20:  # Ensure sufficient statistics for meaningful percentile
            perc = wpercentile(data=   data.loc[msk, variable]          .values, percents=bg_eff,
                               weights=data.loc[msk, 'weight_test'].values)
            pass

        x[i,j] = np.mean(edges[0])
        y[i,j] = np.mean(edges[1])
        z[i,j] = perc

        # Set non-zero bin content
        if perc != np.nan:
            profile.SetBinContent(i + 1, j + 1, perc)
            pass
        pass

    # Normalise arrays
    x,y = standardise(x,y)

    # Filter out NaNs
    msk = ~np.isnan(z)
    x, y, z = x[msk], y[msk], z[msk]

    return profile, (x,y,z)

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parser.parse_args() 

    # Call main function
    main(args)
    pass
