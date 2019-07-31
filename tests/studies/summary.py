#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
from sklearn.metrics import roc_curve

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, metrics, signal_low, JSD, MASSBINS
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


#@showsave
def summary (data_, args, feature_dict, scan_features, target_tpr=0.5, num_bootstrap=5, masscut=False, pt_range=(200, 2000), title=None):
    """
    Perform study of combined classification- and decorrelation performance.

    Saves plot `figures/summary.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Python list of named features in `data` to study.
        scan_features: Python dict of parameter scan-features. The dict key is
            the base features to which the parameter scan belongs. The dict
            values are a list of tuples of `(name, label)`, where `name` is the
            name of the appropriate feature in `data` and `label` is drawn on
            the plot next to each scan point. For instance:
                scan_features = {'NN': [('ANN(#lambda=3)', '#lambda=3'),
                                        ('ANN(#lambda=10)', '#lambda=10'),
                                        ('ANN(#lambda=30)', '#lambda=30')],
                                 'Adaboost': ...,
                                 }
        target_tpr: ...
        num_bootstrap: ...
    """

    # Extract features and count appearance of each base variable
    features = []
    appearances = []
    for basevar in feature_dict.keys():
        for suffix in feature_dict[basevar]:
            features.append(basevar+suffix)
        appearances.append(len(feature_dict[basevar]))

    # Check(s)
    assert isinstance(features, list)
    assert isinstance(scan_features, dict)

    # Select pT-range
    if pt_range is not None:
        data = data_.loc[(data_['pt'] > pt_range[0]) & (data_['pt'] < pt_range[1])]
    else:
        data = data_
        pass

    # For reproducibility of bootstrap sampling
    np.random.seed(7)

    # Compute metrics for all features
    points = list()
    #for feat in features + map(lambda t: t[0], [it for gr in scan_features.itervalues() for it in gr]):
    for feat in features:
        print  "-- {}".format(feat)

        # Check for duplicates
        #if feat in map(lambda t: t[2], points):
        #    print "    Skipping (already encounted)"
        #    continue

        # Compute metrics
        _, rej, jsd = metrics(data, feat, masscut=masscut)

        # Add point to array
        points.append((rej, jsd, feat))
        pass

    # Compute meaningful limit for 1/JSD based on bootstrapping
    num_bootstrap = 10
    jsd_limits = list()
    for bkg_rej in np.logspace(np.log10(2.), np.log10(100), 2 * 10 + 1, endpoint=True):
        frac = 1. / float(bkg_rej)

        limits = 1./np.array(jsd_limit(data, frac, num_bootstrap=5))
        jsd_limits.append((bkg_rej, np.mean(limits), np.std(limits)))
        pass

    # Perform plotting
    c = plot(data, args, features, scan_features, points, jsd_limits, masscut, pt_range, appearances)

    # Output
    if title is None:
    	path = 'figures/summary{}{}.pdf'.format('__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[1]) if pt_range is not None else '', '__masscut' if masscut else '')
    else:
    	path = 'figures/'+title+'_summary{}{}.pdf'.format('__pT{:.0f}_{:.0f}'.format(pt_range[0], pt_range[1]) if pt_range is not None else '', '__masscut' if masscut else '')

    c.save(path = path)
    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, scan_features, points, jsd_limits, masscut, pt_range, appearances = argv

    with TemporaryStyle() as style:

        # Compute yaxis range
        ranges = int(pt_range is not None) + int(masscut)
        mult  = 10. if ranges == 2 else (5. if ranges == 1 else 1.)
        
        # Define variable(s)
        #axisrangex = (1.4,     100.)
	#axisrangey = (0.3, 100000. * mult)
	#axisrangex = (1.4,     40.)
        #axisrangey = (0.3, 300000. * mult)
        axisrangex = (1.4,     100.)
        axisrangey = (0.3, 500000.)
	aminx, amaxx = axisrangex
        aminy, amaxy = axisrangey

        # Styling
        scale = 0.95
        style.SetTitleOffset(1.8, 'x')
        style.SetTitleOffset(1.6, 'y')
        style.SetTextSize      ( style.GetTextSize()       * scale)
        style.SetLegendTextSize( style.GetLegendTextSize() * scale)

        # Canvas
        c = rp.canvas(batch=not args.show, size=(600,600))

        # Reference lines
        nullopts = dict(linecolor=0, linewidth=0, linestyle=0, markerstyle=0, markersize=0, fillstyle=0)
        lineopts = dict(linecolor=ROOT.kGray + 2, linewidth=1, option='L')
        boxopts  = dict(fillcolor=ROOT.kBlack, alpha=0.05, linewidth=0, option='HIST')
        c.hist([aminy], bins=list(axisrangex), **nullopts)
        c.plot([1, amaxy], bins=[2, 2],     **lineopts)
        c.plot([1, 1],     bins=[2, amaxx], **lineopts)
        c.hist([amaxy],    bins=[aminx, 2], **boxopts)
        c.hist([1],        bins=[2, amaxx], **boxopts)

        # Meaningful limits on 1/JSD
        x,y,ey = map(np.array, zip(*jsd_limits))
        ex = np.zeros_like(ey)
        gr = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        smooth_tgrapherrors(gr, ntimes=3)
        c.graph(gr, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

        x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
        idx = 3
        gr.GetPoint(idx, x_,  y_)
        ey_ = gr.GetErrorY(idx)
        x_, y_ = map(float, (x_, y_))
        c.latex("Statistical limit", x_, y_ + ey_, align=21, textsize=11, angle=-5, textcolor=ROOT.kGray + 2)


        # Markers
	if len(appearances) != 2:
            for is_simple in [True, False]:

                # Split the legend into simple- and MVA taggers
	        indices = np.array([0]+appearances).cumsum()
	        for i in range(len(indices)-1):
                    for ifeat, feat in filter(lambda t: is_simple == signal_low(t[1]), enumerate(features[indices[i]:indices[i+1]])):

                        # Coordinates, label
                        idx = map(lambda t: t[2], points).index(feat)
                        x, y, label = points[idx]

                        # Overwrite default name of parameter-scan classifier
                        label = 'ANN'    if label.startswith('ANN') else label
                        label = 'uBoost' if label.startswith('uBoost') else label

                        # Style
                        colour      = rp.colours[i % len(rp.colours)]
		        if ifeat == 0:
			    markerstyle = 20
		        else:
	                    markerstyle = 23 + ifeat

                        # Draw
                        c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, label='#scale[%.1f]{%s}' % (scale, latex(label, ROOT=True)), option='P')
                        pass

                # Draw class-specific legend
                width = 0.2 # chagned from 0.15 to 0.2
                c.legend(header=("Analytical:" if is_simple else "MVA:"),
                     width=width, xmin=0.50 + (width + 0.06) * (is_simple), ymax=0.888)  #, ymax=0.827) #changed xmin from 0.60 to 0.50, with translation from 0.02 to 0.06
            pass

	else:
            for first_var in [True, False]:

                # Split the legend into simple- and MVA taggers
	        indices = np.array([0]+appearances).cumsum()
	        for i in [0,1]:
		    if i == 0 and not first_var: continue
		    if i == 1 and first_var: continue
                    for ifeat, feat in enumerate(features[indices[i]:indices[i+1]]):

                        # Coordinates, label
                        idx = map(lambda t: t[2], points).index(feat)
                        x, y, label = points[idx]

                        # Style
                        colour      = rp.colours[i % len(rp.colours)]
		        if ifeat == 0:
			    markerstyle = 20
		        else:
	                    markerstyle = 23 + ifeat

                        # Draw
                        c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, label='#scale[%.1f]{%s}' % (scale, latex(label, ROOT=True)), option='P')
                        pass

                # Draw class-specific legend
                width = 0.15
                c.legend(header=(latex(features[0], ROOT=True)+"-based:" if first_var else latex(features[appearances[1]], ROOT=True)+"-based:"),
                     width=width, xmin=0.55 + (width + 0.06) * (first_var), ymax=0.9)

        # Make legends transparent
        for leg in c.pads()[0]._legends:
            leg.SetFillStyle(0)
            pass

        # Connecting lines (simple)
	indices = np.array([0]+appearances).cumsum()
	for i in range(len(indices)-1):
	    base_x, base_y, _ = points[indices[i]]
	    for j in range(appearances[i])[1:]:
		x1, y1, _ = points[indices[i]+j]
		color = rp.colours[i % len(rp.colours)]
		c.graph([base_y, y1], bins=[base_x, x1], linecolor=color, linestyle=2, option='L')
                pass

        # Decorations
        c.xlabel("Background rejection, 1 / #varepsilon_{bkg}^{rel} @ #varepsilon_{sig}^{rel} = 50%")
        c.ylabel("Mass-decorrelation, 1 / JSD @ #varepsilon_{sig}^{rel} = 50%")
        c.xlim(*axisrangex)
        c.ylim(*axisrangey)
        c.logx()
        c.logy()

        opts_text = dict(textsize=11, textcolor=ROOT.kGray + 2)
        midpointx = np.power(10, 0.5 * np.log10(amaxx))
        midpointy = np.power(10, 0.5 * np.log10(amaxy))
        c.latex("No separation",                       1.91, midpointy, angle=90, align=21, **opts_text)
        c.latex("Maximal sculpting",                   midpointx, 0.89, angle= 0, align=23, **opts_text)
        c.latex("    Less sculpting #rightarrow",      2.1, midpointy,  angle=90, align=23, **opts_text)
        c.latex("     Greater separation #rightarrow", midpointx, 1.1,  angle= 0, align=21, **opts_text)

        #c.text(TEXT + ["#it{W} jet tagging"], xmin=0.24, qualifier=QUALIFIER)
        c.text([], xmin=0.15, ymax=0.96, qualifier=QUALIFIER, ATLAS=False)
        c.text(TEXT + \
               ["#it{W} jet tagging"] + (
                    ["p_{{T}} #in  [{:.0f}, {:.0f}] GeV".format(pt_range[0], pt_range[1])] if pt_range is not None else []
                ) + (
                    ['Cut: m #in  [60, 100] GeV'] if masscut else []
                ),
               xmin=0.26, ATLAS=None)
        pass

    return c
