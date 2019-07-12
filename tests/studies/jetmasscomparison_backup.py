#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, signal_low, MASSBINS
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


#@showsave
def jetmasscomparison (data_, args, features, pt_range, eff_sig=50, title=None):
    """
    Perform study of jet mass distributions before and after subtructure cut for
    different substructure taggers.

    Saves plot `figures/jetmasscomparison__eff_sig_[eff_sig].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for which to plot signal- and background distributions.
        eff_sig: Signal efficiency at which to impose cut.
	pt_range: pT selection of the data.
    """

    # Define masks and direction-dependent cut value

    # Select pT-range
    if pt_range is not None:
        data = data_[(data_['pt'] > pt_range[0]) & (data_['pt'] < pt_range[1])]
    else:
        data = data_
        pass

    msk_sig = data['signal'] == 1
    cuts, msks_pass = dict(), dict()
    for feat in features:
        eff_cut = eff_sig if signal_low(feat) else 100 - eff_sig
        cut = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight_test'].values)
        msks_pass[feat] = data[feat] > cut

        # Ensure correct cut direction
        if signal_low(feat):
            msks_pass[feat] = ~msks_pass[feat]
            pass
        pass

    # Perform plotting
    c = plot(data, args, features, msks_pass, eff_sig, pt_range)

    # Perform plotting on individual figures
    plot_individual(data, args, features, msks_pass, eff_sig, pt_range, title)

    # Output
    #path = 'figures/jetmasscomparison__eff_sig_{:d}.pdf'.format(int(eff_sig))
    if title is None:
    	if pt_range is not None:
    	    path = 'figures/jetmasscomparison_pT{}to{}__eff_sig_{:d}.pdf'.format(pt_range[0], pt_range[1], int(eff_sig))
    	else:
    	    path = 'figures/jetmasscomparison__eff_sig_{:d}.pdf'.format(int(eff_sig))
    else:
    	if pt_range is not None:
    	    path = 'figures/'+title+'_jetmasscomparison_pT{}to{}__eff_sig_{:d}.pdf'.format(pt_range[0], pt_range[1], int(eff_sig))
    	else:
    	    path = 'figures/'+title+'_jetmasscomparison__eff_sig_{:d}.pdf'.format(int(eff_sig))

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, msks_pass, eff_sig, pt_range = argv

    with TemporaryStyle() as style:

        # Style
        ymin, ymax = 5E-05, 5E+00
        scale = 0.8
        for coord in ['x', 'y', 'z']:
            style.SetLabelSize(style.GetLabelSize(coord) * scale, coord)
            style.SetTitleSize(style.GetTitleSize(coord) * scale, coord)
            pass
        style.SetTextSize      (style.GetTextSize()       * scale)
        style.SetLegendTextSize(style.GetLegendTextSize() * scale)
        style.SetTickLength(0.07,                     'x')
        style.SetTickLength(0.07 * (5./6.) * (2./3.), 'y')

        # Global variable override(s)
        histstyle = dict(**HISTSTYLE)
        histstyle[True]['fillstyle'] = 3554
        histstyle[True] ['label'] = None
        histstyle[False]['label'] = None
        for v in ['linecolor', 'fillcolor']:
            histstyle[True] [v] = 16
            histstyle[False][v] = ROOT.kBlack
            pass
        style.SetHatchesLineWidth(1)

        # Canvas
        c = rp.canvas(batch=not args.show, num_pads=(2,3))

        # Plots
        # -- Dummy, for proper axes
        for ipad, pad in enumerate(c.pads()[1:], 1):
            pad.hist([ymin], bins=[50, 300], linestyle=0, fillstyle=0, option=('Y+' if ipad % 2 else ''))
            pass

        # -- Inclusive
        base = dict(bins=MASSBINS, normalise=True, linewidth=2)
        for signal, name in zip([False, True], ['bkg', 'sig']):
            msk = data['signal'] == signal
            histstyle[signal].update(base)
            for ipad, pad in enumerate(c.pads()[1:], 1):
                histstyle[signal]['option'] = 'HIST'
                pad.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, **histstyle[signal])
                pass
            pass

        for sig in [True, False]:
            histstyle[sig]['option'] = 'FL'
            pass

        c.pads()[0].legend(header='Inclusive selection:', categories=[
            ("Multijets",   histstyle[False]),
            ("#it{W} jets", histstyle[True])
            ], xmin=0.18, width= 0.60, ymax=0.28 + 0.07, ymin=0.001 + 0.07, columns=2)
        c.pads()[0]._legends[-1].SetTextSize(style.GetLegendTextSize())
        c.pads()[0]._legends[-1].SetMargin(0.35)

        # -- Tagged
        base['linewidth'] = 2
        for ifeat, feat in enumerate(features):
            opts = dict(
                linecolor = rp.colours[(ifeat // 2)],
                linestyle = 1 + (ifeat % 2),
                linewidth = 2,
                )
            cfg = dict(**base)
            cfg.update(opts)
            msk = (data['signal'] == 0) & msks_pass[feat]
            pad = c.pads()[1 + ifeat//2]
            pad.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, label=" " + latex(feat, ROOT=True), **cfg)
            pass

        # -- Legend(s)
        for ipad, pad in enumerate(c.pads()[1:], 1):
            offsetx = (0.20 if ipad % 2 else 0.05)
            offsety =  0.20 * ((2 - (ipad // 2)) / float(2.))
            #pad.legend(width=0.25, xmin=0.68 - offsetx, ymax=0.80 - offsety)
	    pad.legend(width=0.25, xmin=0.7 - offsetx, ymax=0.95 - offsety)
            #pad.latex("Tagged multijets:", NDC=True, x=0.93 - offsetx, y=0.84 - offsety, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.8, align=31)
	    if pt_range is not None:
		pad.latex("Tagged multijets with pT #in [{:.0f}, {:.0f}] GeV:".format(pt_range[0], pt_range[1]), NDC=True, x=0.83 - offsetx, y=0.84 - offsety, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.6, align=31)            
	    else:
		pad.latex("Tagged multijets:", NDC=True, x=0.93 - offsetx, y=0.84 - offsety, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.8, align=31)


	    #pad._legends[-1].SetMargin(0.35)   			##probably need to replace these to make the legend look better....
            #pad._legends[-1].SetTextSize(style.GetLegendTextSize())
            pass

        # Formatting pads
        margin = 0.2
        for ipad, pad in enumerate(c.pads()):
            tpad = pad._bare()  # ROOT.TPad
            right = ipad % 2
            f = (ipad // 2) / float(len(c.pads()) // 2 - 1)
            tpad.SetLeftMargin (0.05 + 0.15 * (1 - right))
            tpad.SetRightMargin(0.05 + 0.15 * right)
            tpad.SetBottomMargin(f * margin)
            tpad.SetTopMargin((1 - f) * margin)
            if ipad == 0: continue
            pad._xaxis().SetNdivisions(505)
            pad._yaxis().SetNdivisions(505)
            if ipad // 2 < len(c.pads()) // 2 - 1:  # Not bottom pad(s)
                pad._xaxis().SetLabelOffset(9999.)
                pad._xaxis().SetTitleOffset(9999.)
            else:
                pad._xaxis().SetTitleOffset(2.7)
                pass
            pass

        # Re-draw axes
        for pad in c.pads()[1:]:
            pad._bare().RedrawAxis()
            pad._bare().Update()
            pad._xaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
            pad._yaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
            pass

        # Decorations
        c.pads()[-1].xlabel("Large-#it{R} jet mass [GeV]")
        c.pads()[-2].xlabel("Large-#it{R} jet mass [GeV]")
        c.pads()[1].ylabel("#splitline{#splitline{#splitline{#splitline{}{}}{#splitline{}{}}}{#splitline{}{}}}{#splitline{}{#splitline{}{#splitline{}{Fraction of jets}}}}")
        c.pads()[2].ylabel("#splitline{#splitline{#splitline{#splitline{Fraction of jets}{}}{}}{}}{#splitline{#splitline{}{}}{#splitline{#splitline{}{}}{#splitline{}{}}}}")
        # I have written a _lot_ of ugly code, but this ^ is probably the worst.

        c.pads()[0].text(["#sqrt{s} = 13 TeV,  #it{W} jet tagging",
                    "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                    ], xmin=0.2, ymax=0.72, qualifier=QUALIFIER, ATLAS=False)

        for pad in c.pads()[1:]:
            pad.ylim(ymin, ymax)
            pad.logy()
            pass

        pass  # end temprorary style

    return c


def plot_individual (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, msks_pass, eff_sig, pt_range, title = argv

    with TemporaryStyle() as style:

        # Style @TEMP?
        ymin, ymax = 5E-05, 5E+00
        scale = 0.6
        for coord in ['x', 'y', 'z']:
            style.SetLabelSize(style.GetLabelSize(coord) * scale, coord)
            style.SetTitleSize(style.GetTitleSize(coord) * scale, coord)
            pass
        #style.SetTextSize      (style.GetTextSize()       * scale)
        #style.SetLegendTextSize(style.GetLegendTextSize() * (scale + 0.03))
        style.SetTickLength(0.07,                     'x')
        style.SetTickLength(0.07 * (5./6.) * (2./3.), 'y')

        # Global variable override(s)
        histstyle = dict(**HISTSTYLE)
        histstyle[True]['fillstyle'] = 3554
        histstyle[True] ['linewidth'] = 4
        histstyle[False]['linewidth'] = 4
        histstyle[True] ['label'] = None
        histstyle[False]['label'] = None
        for v in ['linecolor', 'fillcolor']:
            histstyle[True] [v] = 16
            histstyle[False][v] = ROOT.kBlack
            pass
        style.SetHatchesLineWidth(6)

        # Loop features
        ts  = style.GetTextSize()
        lts = style.GetLegendTextSize()
        for ifeat, feats in enumerate([None] + list(zip(features[::2], features[1::2])), start=-1):
            first = ifeat == -1

            # Style
            style.SetTitleOffset(1.25 if first else 1.2, 'x')
            style.SetTitleOffset(1.7  if first else 1.6, 'y')
            style.SetTextSize(ts * (0.8 if first else scale))
            style.SetLegendTextSize(lts * (0.8 + 0.03 if first else scale + 0.03))

            # Canvas
            c = rp.canvas(batch=not args.show, size=(300, 200))#int(200 * (1.45 if first else 1.))))

            if first:
                opts = dict(xmin=0.185, width=0.60, columns=2)
                c.legend(header=' ', categories=[
                            ("Multijets",   histstyle[False]),
                            ("#it{W} jets", histstyle[True])
                        ], ymax=0.45, **opts)
                c.legend(header='Inclusive selection:',
                         ymax=0.40, **opts)
                #c.pad()._legends[-2].SetTextSize(style.GetLegendTextSize())
                #c.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())
                c.pad()._legends[-2].SetMargin(0.35)
                c.pad()._legends[-1].SetMargin(0.35)

                c.text(["#sqrt{s} = 13 TeV,  #it{W} jet tagging",
                        "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                        ], xmin=0.2, ymax=0.80, qualifier=QUALIFIER, ATLAS=False)


            else:


                # Plots
                # -- Dummy, for proper axes
                c.hist([ymin], bins=[50, 300], linestyle=0, fillstyle=0)

                # -- Inclusive
                base = dict(bins=MASSBINS, normalise=True)
                for signal, name in zip([False, True], ['bkg', 'sig']):
                    msk = data['signal'] == signal
                    histstyle[signal].update(base)
                    histstyle[signal]['option'] = 'HIST'
                    c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, **histstyle[signal])
                    pass

                for sig in [True, False]:
                    histstyle[sig]['option'] = 'FL'
                    pass

                # -- Tagged
                for jfeat, feat in enumerate(feats):
                    opts = dict(
                        linecolor = rp.colours[((2 * ifeat + jfeat) // 2)],
                        linestyle = 1 + 6 * (jfeat % 2),
                        linewidth = 4,
                        )
                    cfg = dict(**base)
                    cfg.update(opts)
                    msk = (data['signal'] == 0) & msks_pass[feat]
                    c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, label=" " + latex(feat, ROOT=True), **cfg)
                    pass

                # -- Legend(s)
                y =  0.46  if first else 0.68
                dy = 0.025 if first else 0.04
                c.legend(width=0.25, xmin=0.63, ymax=y)
                #c.latex("Tagged multijets:", NDC=True, x=0.87, y=y + dy, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.9, align=31)
		if pt_range is not None:
		    c.latex("Tagged multijets #in [{:.0f}, {:.0f}] GeV:".format(pt_range[0], pt_range[1]), NDC=True, x=0.77, y=y + dy, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.7, align=31)
		else:
		    c.latex("Tagged multijets:", NDC=True, x=0.87, y=y + dy, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.9, align=31)
                c.pad()._legends[-1].SetMargin(0.35)
                c.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())

                # Formatting pads
                tpad = c.pad()._bare()
                tpad.SetLeftMargin  (0.20)
                tpad.SetBottomMargin(0.12 if first else 0.20)
                tpad.SetTopMargin   (0.39 if first else 0.05)

                # Re-draw axes
                tpad.RedrawAxis()
                tpad.Update()
                c.pad()._xaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
                c.pad()._yaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"

                # Decorations
                c.xlabel("Large-#it{R} jet mass [GeV]")
                c.ylabel("Fraction of jets")

                c.text(qualifier=QUALIFIER, xmin=0.25, ymax=0.82, ATLAS=False)

                c.ylim(ymin, ymax)
                c.logy()
                pass

            # Save

	    if title is None:
	    	if pt_range is not None:
	    	    c.save(path = 'figures/jetmasscomparison_pT'+str(pt_range[0])+'to'+str(pt_range[1])+'__eff_sig_{:d}__{}.pdf'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats)))
	    	else:
	    	    c.save(path = 'figures/jetmasscomparison__eff_sig_{:d}__{}.pdf'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats)))
	    else:
	    	if pt_range is not None:
	    	    c.save(path = 'figures/'+title+'_jetmasscomparison_pT'+str(pt_range[0])+'to'+str(pt_range[1])+'__eff_sig_{:d}__{}.pdf'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats)))
	    	else:
	    	    c.save(path = 'figures/'+title+'_jetmasscomparison__eff_sig_{:d}__{}.pdf'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats)))
            pass
        pass  # end temprorary style

    return
