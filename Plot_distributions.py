import ROOT
import numpy as np
import root_numpy 

from adversarial.utils import load_data

def Plot_variable(file_title, var_title, var, data, var_range, Normalized=False, uniform_signal_weights=False, scale_factor=1., bg_first=False):
    f1 = ROOT.TFile(file_title, "RECREATE")
    signal_dist = ROOT.TH1D("signal_"+var_title, "signal_"+var_title, 100, var_range[0], var_range[1])
    bg_dist = ROOT.TH1D("bg_"+var_title, "bg_"+var_title, 100, var_range[0], var_range[1])
    signal_data = data[data['signal']==1]
    bg_data = data[data['signal']==0]

    if uniform_signal_weights:
    	root_numpy.fill_hist(signal_dist, signal_data[var], weights=np.full(len(data[data['signal'] == 1]), 0.01))
    else:
 	root_numpy.fill_hist(signal_dist, signal_data[var], weights=signal_data["weight_test"]*scale_factor)
    root_numpy.fill_hist(bg_dist, bg_data[var], weights=bg_data["weight_test"])

    # normal scale
    canv = ROOT.TCanvas(var_title, var_title, 600, 600)
    signal_dist.SetLineColor(4)
    bg_dist.SetLineColor(2)
    leg = ROOT.TLegend(0.5,0.8,0.9,0.9)
    leg.AddEntry(signal_dist, "signal")
    leg.AddEntry(bg_dist, "bg")
    signal_dist.GetXaxis().SetTitle(var_title)
    if Normalized:
	if bg_first:
	        bg_dist.DrawNormalized()
        	signal_dist.DrawNormalized("SAME")
	else:
		signal_dist.DrawNormalized()
		bg_dist.DrawNormalized("SAME")
    else:
	if bg_first:
		bg_dist.Draw()
		signal_dist.Draw("SAME")
	else:
        	signal_dist.Draw()
        	bg_dist.Draw("SAME")
    leg.Draw()
    canv.Write()

    # log scale
    canv_log = ROOT.TCanvas(var_title+"_log", var_title+"_log", 600, 600)
    if Normalized:
        bg_dist.DrawNormalized()
        signal_dist.DrawNormalized("SAME")
    else:
        signal_dist.Draw()
        bg_dist.Draw("SAME")
    leg.Draw()
    canv_log.SetLogy()
    canv_log.Write()

    signal_int = signal_dist.Integral()
    bg_int = bg_dist.Integral()
    f1.Close()
    return signal_int, bg_int

def Plot_variable_from_data_2D(out_file, data_path, x_title, x_var, x_bins, x_range, y_title, y_var, y_bins, y_range, test=False, weights=None):
	if test:
		data, features, _ = load_data(data_path, test_full_signal=True)
	else:
		data, features, _ = load_data(data_path, train_full_signal=True)
	if weights is not None:
		weights = data[weights]
	f1 = ROOT.TFile(out_file, "RECREATE")
	hist = ROOT.TH2D('hist', 'hist', x_bins, x_range[0], x_range[1], y_bins, y_range[0], y_range[1])
	X = data[x_var]; Y = data[y_var]
	root_numpy.fill_hist(hist, np.vstack((X,Y)).T, weights=weights)
	canv = ROOT.TCanvas('canv', 'canv', 600, 600)
	hist.SetContour(256)
	hist.GetXaxis().SetTitle(x_title)
	hist.GetYaxis().SetTitle(y_title)
	hist.Draw("COLZ")
	canv.Write()
	f1.Close()

if __name__ == "__main__":

	#print "Background training jets: {}, test jets: {}".format(sum((data['train']==1)&(data['signal']==0)), sum((data['train']==0)&(data['signal']==0)))

	#data, features, _ = load_data('data.h5', test=True)
	data, features, _ = load_data('input/data.h5', test_full_signal=True)
	#nbg = len(data[data['signal']==0])
	#nbg2 = data[data['signal']==0].shape[0]
	#nsig = len(data[data['signal']==1])
	#nsig2 = data[data['signal']==1].shape[0]
	#print "There are {} background jets and {} signal jets.".format(nbg, nsig)
	#signal_int, bg_int = Plot_variable("jet_pT.root", "jet_pT", "pt", data, (0,2500), Normalized=False, scale_factor=1.)
	#print "The background integrates to {} and the signal to {}.".format(bg_int, signal_int)

	#Plot_variable_from_data_2D("2D_hist.root", "input/data.h5", "#rho", "rho", 20, (-7, -1), "p_{T}", "pt", 20, (200,2000), test=False, weights='weight_test')
	#Plot_variable_from_data_2D("2D_hist.root", "input/data.h5", "#rho", "rho", 20, (-7, -1), "p_{T}", "pt", 20, (200,2000), test=False, weights=None)		

	#signal_int, bg_int = Plot_variable("jet_N2.root", "N2", "N2_beta1", data, (0,3), Normalized=False, scale_factor=1.)
	#signal_int, bg_int = Plot_variable("nPV.root", "nPV", "npv", data, (0,100), Normalized=False, scale_factor=1.)
	signal_int, bg_int = Plot_variable("decDeep.root", "decDeepWvsQCD", "decDeepWvsQCD", data, (0.,1.), Normalized=False, scale_factor=1., bg_first=True)
	signal_int, bg_int = Plot_variable("Deep.root", "DeepWvsQCD", "DeepWvsQCD", data, (0.,1.), Normalized=False, scale_factor=1.)

