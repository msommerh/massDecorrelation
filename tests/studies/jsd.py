#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import numpy as np

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, signal_low, JSD, MASSBINS, garbage_collect
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@garbage_collect
#@showsave
def jsd (data_, args, feature_dict, pt_range, title=None):
    """
    Perform study of ...

    Saves plot `figures/jsd.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for ...
    """

    # Extract features and count appearance of each base variable
    features = []
    appearances = []
    for basevar in feature_dict.keys():
        for suffix in feature_dict[basevar]:
            features.append(basevar+suffix)
        appearances.append(len(feature_dict[basevar]))

    # Select data
    if pt_range is not None:
        data = data_[(data_['pt'] > pt_range[0]) & (data_['pt'] < pt_range[1])]
    else:
        data = data_
        pass

    # Create local histogram style dict
    histstyle = dict(**HISTSTYLE)
    histstyle[True] ['label'] = "Pass"
    histstyle[False]['label'] = "Fail"

    # Define common variables
    msk  = data['signal'] == 0
    effs = np.linspace(0, 100, 10 * 2, endpoint=False)[1:].astype(int)

    # Loop tagger features
    jsd = {feat: [] for feat in features}
    for ifeat, feat in enumerate(features):

        if len(jsd[feat]): continue  # Duplicate feature.

        # Define cuts
        cuts = list()
        for eff in effs:
            cut = wpercentile(data.loc[msk, feat].values, eff if signal_low(feat) else 100 - eff, weights=data.loc[msk, 'weight_test'].values)
            cuts.append(cut)
            pass

        # Compute KL divergence for successive cuts
        for cut, eff in zip(cuts, effs):

            # Create ROOT histograms
            msk_pass = data[feat] > cut
            if signal_low(feat):
                msk_pass = ~msk_pass
                pass

            # Get histograms / plot
            c = rp.canvas(batch=not args.show)
            h_pass = c.hist(data.loc[ msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk, 'weight_test'].values, normalise=True, **histstyle[True])   #, display=False)
            h_fail = c.hist(data.loc[~msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk, 'weight_test'].values, normalise=True, **histstyle[False])  #, display=False)

            # Convert to numpy arrays
            p = root_numpy.hist2array(h_pass)
            f = root_numpy.hist2array(h_fail)

            # Compute Jensen-Shannon divergence
            jsd[feat].append(JSD(p, f, base=2))

            # -- Decorations
            #c.xlabel("Large-#it{R} jet mass [GeV]")
            #c.ylabel("Fraction of jets")
            #c.legend()
            #c.logy()
            #c.text(TEXT + [
            #    "{:s} {} {:.3f}".format(latex(feat, ROOT=True), '<' if signal_low(feat) else '>', cut),
            #    "JSD = {:.4f}".format(jsd[feat][-1])] + \
            #    (["p_{{T}} #in  [{:.0f}, {:.0f}] GeV".format(*pt_range)] if pt_range else []),
            #    qualifier=QUALIFIER, ATLAS=False)

            # -- Save
	    #if title is None:
            #    c.save('figures/temp_jsd_{:s}_{:.0f}{}.pdf'.format(feat, eff, '' if pt_range is None else '__pT{:.0f}_{:.0f}'.format(*pt_range)))
	    #else:
            #    c.save('figures/'+title+'_temp_jsd_{:s}_{:.0f}{}.pdf'.format(feat, eff, '' if pt_range is None else '__pT{:.0f}_{:.0f}'.format(*pt_range)))

            pass
        pass

    # Compute meaningful limit on JSD
    jsd_limits = list()
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    for eff in sigmoid(np.linspace(-5, 5, 20 + 1, endpoint=True)):
        limits = jsd_limit(data[msk], eff, num_bootstrap=5)
        jsd_limits.append((eff, np.mean(limits), np.std(limits)))
        pass

    # Perform plotting
    c = plot(args, data, effs, jsd, jsd_limits, features, pt_range, appearances)

    # Output
    if title is None:
        path = 'figures/jsd{}.pdf'.format('' if pt_range is None else '__pT{:.0f}_{:.0f}'.format(*pt_range))
    else:
	path = 'figures/'+title+'_jsd{}.pdf'.format('' if pt_range is None else '__pT{:.0f}_{:.0f}'.format(*pt_range))
    c.save(path = path)
    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, effs, jsd, jsd_limits, features, pt_range, appearances = argv

    with TemporaryStyle() as style:

        # Style
        style.SetTitleOffset(1.5, 'x')
        style.SetTitleOffset(2.0, 'y')

        # Canvas
        c = rp.canvas(batch=not args.show)

        # Plots
        ref = ROOT.TH1F('ref', "", 10, 0., 1.)
        for i in range(ref.GetXaxis().GetNbins()):
            ref.SetBinContent(i + 1, 1)
            pass
        c.hist(ref, linecolor=ROOT.kGray + 2, linewidth=1)
	linestyles = [1, 3, 5, 7]

        width = 0.15
	if len(appearances) != 2:
            for is_simple in [True, False]:
    
                indices = np.array([0]+appearances).cumsum()
                for i in range(len(indices)-1):
                    for ifeat, feat in enumerate(features[indices[i]:indices[i+1]]):
                        if is_simple != signal_low(feat): continue
                        colour = rp.colours[i % len(rp.colours)]
                        linestyle   =  1 + ifeat
                        if ifeat == 0:
                            markerstyle = 20
                        else:
                            markerstyle = 23 + ifeat
                        c.plot(jsd[feat], bins=np.array(effs) / 100., linecolor=colour, markercolor=colour, linestyle=linestyle, markerstyle=markerstyle, label=latex(feat, ROOT=True), option='PL')
                        pass
    
                c.legend(header=("Analytical:" if is_simple else "MVA:"),
                         width=width * (1 + 0.8 * int(is_simple)), xmin=0.42 + (width + 0.05) * (is_simple), ymax=0.888,
                         columns=2 if is_simple else 1,
                         margin=0.35)   # moved one intendation to the left
	else:
            for first_var in [True, False]:
    
                indices = np.array([0]+appearances).cumsum()
                for i in [0,1]:
                    if i==0 and not first_var: continue
                    if i==1 and first_var: continue
                    for ifeat, feat in enumerate(features[indices[i]:indices[i+1]]):
                        colour = rp.colours[i % len(rp.colours)]
                        linestyle   =  linestyles[ifeat]
                        if ifeat == 0:
                            markerstyle = 20
                        else:
                            markerstyle = 23 + ifeat
                        c.plot(jsd[feat], bins=np.array(effs) / 100., linecolor=colour, markercolor=colour, linestyle=linestyle, markerstyle=markerstyle, label=latex(feat, ROOT=True), option='PL')
                        pass
    
                c.legend(header=(latex(features[0], ROOT=True)+"-based:" if first_var else latex(features[appearances[1]], ROOT=True)+"-based:"),
                         width=width, xmin=0.45 + (width + 0.06) * (first_var), ymax=0.888)

            pass

####  c.legend(header=(features[0]+":" if first_var else features[appearances[1]]+":"), #work in progress!!!!!!!!!!!!!!!!!!!!!
####                  width=width, xmin=0.45 + (width + 0.06) * (first_var), ymax=0.888)

        # Meaningful limits on JSD
        x,y,ey = map(np.array, zip(*jsd_limits))
        ex = np.zeros_like(ey)
        gr = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        smooth_tgrapherrors(gr, ntimes=2)
        c.graph(gr, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

        # Redraw axes
        c.pads()[0]._primitives[0].Draw('AXIS SAME')

        # Decorations
        c.xlabel("Background efficiency #varepsilon_{bkg}^{rel}")
        c.ylabel("Mass correlation, JSD")
        c.text([], xmin=0.15, ymax = 0.96, qualifier=QUALIFIER, ATLAS=False)
        c.text(["#sqrt{s} = 13 TeV",  "Multijets"] + \
              (["p_{T} [GeV] #in", "    [{:.0f}, {:.0f}]".format(*pt_range)] if pt_range else []),
               ymax=0.85, ATLAS=None)

        c.latex("Maximal sculpting", 0.065, 1.2, align=11, textsize=11, textcolor=ROOT.kGray + 2)
        c.xlim(0, 1)
        c.ymin(5E-05)
        c.padding(0.45)
        c.logy()

        for leg in c.pad()._legends:
            leg.SetMargin(0.5)
            pass

        x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
        idx = gr.GetN() - 7
        gr.GetPoint(idx, x_,  y_)
        ey_ = gr.GetErrorY(idx)
        x_, y_ = map(float, (x_, y_))
        c.latex("Statistical limit", x_, y_ - ey_ / 2., align=23, textsize=11, angle=12, textcolor=ROOT.kGray + 2)
        pass

    return c
