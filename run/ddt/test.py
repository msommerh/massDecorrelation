#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing DDT transform."""

# Basic import(s)
import math
from array import array

# Scientific import(s)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf, latex, garbage_collect
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
from .common import *
from tests.studies.common import TemporaryStyle

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data(args.input + 'data.h5', test_full_signal=True)

    #variable = VAR_TAU21
    #variable = VAR_N2
    variable = VAR_DECDEEP
    #variable = VAR_DEEP

    if variable == VAR_N2:
        fit_range = FIT_RANGE_N2
    elif variable == VAR_TAU21:
        fit_range = FIT_RANGE_TAU21
    elif variable == VAR_DECDEEP:
	fit_range = FIT_RANGE_DECDEEP
    elif variable == VAR_DEEP:
	fit_range = FIT_RANGE_DEEP
    else:
	print "invalid variable"
	return 0

    # Add DDT variable
    add_ddt(data, feat=variable, path='models/ddt/ddt_{}.pkl.gz'.format(variable))

    # Load transform
    ddt = loadclf('models/ddt/ddt_{}.pkl.gz'.format(variable))

    # --------------------------------------------------------------------------
    # 1D plot

    # Define variable(s)
    msk = data['signal'] == 0

    # Fill profiles
    profiles = dict()
    for var in [variable, variable + 'DDT']:
        profiles[var] = fill_profile(data[msk], var)
        pass

    # Convert to graphs
    graphs = dict()
    for key, profile in profiles.iteritems():
        # Create arrays from profile
        arr_x, arr_y, arr_ex, arr_ey = array('d'), array('d'), array('d'), array('d')
        for ibin in range(1, profile.GetXaxis().GetNbins() + 1):
            if profile.GetBinContent(ibin) != 0. or profile.GetBinError(ibin) != 0.:
                arr_x .append(profile.GetBinCenter (ibin))
                arr_y .append(profile.GetBinContent(ibin))
                arr_ex.append(profile.GetBinWidth  (ibin) / 2.)
                arr_ey.append(profile.GetBinError  (ibin))
                pass
            pass

        # Create graph
        graphs[key] = ROOT.TGraphErrors(len(arr_x), arr_x, arr_y, arr_ex, arr_ey)
        pass

    # Plot 1D transform
    plot1D(graphs, ddt, arr_x, variable, fit_range)


    # --------------------------------------------------------------------------
    # 2D plot

    # Create contours
    binsx = np.linspace(1.5, 5.0, 40 + 1, endpoint=True)
    if variable == VAR_N2:
    	binsy = np.linspace(0.0, 0.8, 40 + 1, endpoint=True)
    else:
	binsy = np.linspace(0.0, 1.4, 40 + 1, endpoint=True)

    contours = dict()
    for sig in [0,1]:

        # Get signal/background mask
        msk = data['signal'] == sig

        # Normalise jet weights
        w  = data.loc[msk, VAR_WEIGHT].values
        w /= math.fsum(w)

        # Prepare inputs
        X = data.loc[msk, [VAR_RHODDT, variable]].values

        # Fill, store contour
        contour = ROOT.TH2F('2d_{}'.format(sig), "", len(binsx) - 1, binsx, len(binsy) - 1, binsy)
        root_numpy.fill_hist(contour, X, weights=w)
        contours[sig] = contour
        pass

    # Linear discriminant analysis (LDA)
    lda = LinearDiscriminantAnalysis()
    X = data[[VAR_RHODDT, variable]].values
    y = data['signal'].values
    w = data[VAR_WEIGHT].values
    p = w / math.fsum(w)
    indices = np.random.choice(y.shape[0], size=int(1E+06), p=p, replace=True)
    lda.fit(X[indices], y[indices])  # Fit weighted sample

    # -- Linear fit to decision boundary
    xx, yy = np.meshgrid(binsx, binsy)
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    yboundary = binsy[np.argmin(np.abs(Z - 0.5), axis=0)]
    xboundary = binsx
    lda = LinearRegression()
    lda.fit(xboundary.reshape(-1,1), yboundary)

    # Plot 2D scatter
    plot2D(data, ddt, lda, contours, binsx, binsy, variable)
    return


def plot1D (*argv):
    """
    Method for delegating 1D plotting.
    """

    # Unpack arguments
    graphs, ddt, arr_x, variable, fit_range = argv

    # Style
    ROOT.gStyle.SetTitleOffset(1.4, 'x')

    # Canvas
    c = rp.canvas(batch=True)

    # Setup
    pad = c.pads()[0]._bare()
    pad.cd()
    pad.SetTopMargin(0.10)
    pad.SetTopMargin(0.10)

    # Profiles
    if variable == VAR_TAU21:
    	c.graph(graphs[variable],         label="Original, #tau_{21}",          linecolor=rp.colours[4], markercolor=rp.colours[4], markerstyle=24, legend_option='PE')
    	c.graph(graphs[variable + 'DDT'], label="Transformed, #tau_{21}^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=20, legend_option='PE')
    elif variable == VAR_N2:
    	c.graph(graphs[variable],         label="Original, N_{2}",          linecolor=rp.colours[4], markercolor=rp.colours[4], markerstyle=24, legend_option='PE')
    	c.graph(graphs[variable + 'DDT'], label="Transformed, N_{2}^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=20, legend_option='PE')
    elif variable == VAR_DECDEEP:
    	c.graph(graphs[variable],         label="Original, dec_deepWvsQCD",          linecolor=rp.colours[4], markercolor=rp.colours[4], markerstyle=24, legend_option='PE')
    	c.graph(graphs[variable + 'DDT'], label="Transformed, dec_deepWvsQCD^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=20, legend_option='PE')
    elif variable == VAR_DEEP:
    	c.graph(graphs[variable],         label="Original, deepWvsQCD",          linecolor=rp.colours[4], markercolor=rp.colours[4], markerstyle=24, legend_option='PE')
    	c.graph(graphs[variable + 'DDT'], label="Transformed, deepWvsQCD^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=20, legend_option='PE')


    # Fit
    x1, x2 = min(arr_x), max(arr_x)
    intercept, coef = ddt.intercept_ + ddt.offset_, ddt.coef_
    y1 = intercept + x1 * coef
    y2 = intercept + x2 * coef
    c.plot([y1,y2], bins=[x1,x2], color=rp.colours[-1], label='Linear fit', linewidth=1, linestyle=1, option='L')

    # Decorations
    c.xlabel("Large-#it{R} jet #rho^{DDT} = log[m^{2} / (p_{T} #times 1 GeV)]")
    if variable == VAR_TAU21:
        c.ylabel("#LT#tau_{21}#GT, #LT#tau_{21}^{DDT}#GT")
    elif variable == VAR_N2:
	c.ylabel("#LTN_{2}#GT, #LTN_{2}^{DDT}#GT")
    elif variable == VAR_DECDEEP:
	c.ylabel("#LTdec_deepWvsQCD#GT, #LTdec_deepWvsQCD^{DDT}#GT")
    elif variable == VAR_DEEP:
	c.ylabel("#LTdeepWvsQCD#GT, #LTdeepWvsQCD^{DDT}#GT")

    c.text(["#sqrt{s} = 13 TeV,  Multijets"], qualifier=QUALIFIER, ATLAS=False)
    c.legend(width=0.25, xmin=0.57, ymax=None if "Internal" in QUALIFIER else 0.85)

    c.xlim(0, 6.0)
    if variable == VAR_N2:
	ymax = 0.8
    else:
	ymax = 1.4
    c.ylim(0, ymax)
    c.latex("Fit range", sum(fit_range) / 2., 0.08, textsize=13, textcolor=ROOT.kGray + 2)
    c.latex("Fit parameters:", 0.3, 0.7*ymax, align=11, textsize=14, textcolor=ROOT.kBlack)
    c.latex("   intercept = {:7.4f}".format(intercept[0]), 0.3, 0.65*ymax, align=11, textsize=14, textcolor=ROOT.kBlack)
    c.latex("   coef = {:7.4f}".format(coef[0]), 0.3, 0.6*ymax, align=11, textsize=14, textcolor=ROOT.kBlack)
    c.xline(fit_range[0], ymax=0.82, text_align='BR', linecolor=ROOT.kGray + 2)
    c.xline(fit_range[1], ymax=0.82, text_align='BL', linecolor=ROOT.kGray + 2)

    # Save
    mkdir('figures/ddt/')
    c.save('figures/ddt/ddt_{}.pdf'.format(variable))
    return


def plot2D (*argv):
    """
    Method for delegating 2D plotting.
    """

    # Unpack arguments
    data, ddt, lda, contours, binsx, binsy, variable = argv

    with TemporaryStyle() as style:

        # Style
        style.SetNumberContours(10)

        # Canvas
        c = rp.canvas(batch=True)

        # Axes
        c.hist([binsy[0]], bins=[binsx[0], binsx[-1]], linestyle=0, linewidth=0)

        # Plotting contours
        for sig in [0,1]:
            c.hist2d(contours[sig], linecolor=rp.colours[1 + 3 * sig], label="Signal" if sig else "Background", option='CONT3', legend_option='L')
            pass

        # Linear fit
        x1, x2 = 1.5, 5.0
        intercept, coef = ddt.intercept_ + ddt.offset_, ddt.coef_
        y1 = intercept + x1 * coef
        y2 = intercept + x2 * coef
        c.plot([y1,y2], bins=[x1,x2], color=rp.colours[-1], label='DDT transform fit', linewidth=1, linestyle=1, option='L')

        # LDA decision boundary
        y1 = lda.intercept_ + x1 * lda.coef_
        y2 = lda.intercept_ + x2 * lda.coef_
        c.plot([y1,y2], bins=[x1,x2],  label='LDA boundary', linewidth=1, linestyle=2, option='L')

        # Decorations
        c.text(["#sqrt{s} = 13 TeV"], qualifier=QUALIFIER, ATLAS=False)
        c.legend()
        c.ylim(binsy[0], binsy[-1])
        c.xlabel("Large-#it{R} jet " + latex('rhoDDT', ROOT=True))
	if variable == VAR_TAU21:
        	c.ylabel("Large-#it{R} jet " + latex('#tau_{21}',  ROOT=True)) #changed these to latex formatting
	elif variable == VAR_N2:
		c.ylabel("Large-#it{R} jet " + latex('N_{2}',  ROOT=True))
	elif variable == VAR_DECDEEP:
		c.ylabel("Large-#it{R} jet " + latex('dec_deepWvsQCD',  ROOT=True))
	elif variable == VAR_DEEP:
		c.ylabel("Large-#it{R} jet " + latex('deepWvsQCD',  ROOT=True))

        # Save
        mkdir('figures/ddt')
        c.save('figures/ddt/ddt_{}_2d.pdf'.format(variable))
        pass
    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
