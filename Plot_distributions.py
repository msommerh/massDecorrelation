import ROOT
import numpy as np
import root_numpy 

import pandas as pd
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
    root_numpy.fill_hist(bg_dist, bg_data[var], weights=bg_data["weight_test"])  ##use this one for real data sets
    #root_numpy.fill_hist(bg_dist, bg_data[var], weights=np.full(len(data[data['signal'] == 0]), 0.01)) #this one only for nanoAOD input without correct weights

    # normal scale
    canv = ROOT.TCanvas(var_title, var_title, 600, 600)
    signal_dist.SetLineColor(4)
    bg_dist.SetLineColor(2)
    leg = ROOT.TLegend(0.65,0.8,0.9,0.9)
    leg.AddEntry(signal_dist, "Signal")
    leg.AddEntry(bg_dist, "QCD")
    signal_dist.GetXaxis().SetTitle(var_title)
    if Normalized:
	if bg_first:
		bg_dist.GetYaxis().SetTitle("(a.u.)")
	        bg_dist.DrawNormalized()
        	signal_dist.DrawNormalized("SAME")
	else:
		signal_dist.GetYaxis().SetTitle("(a.u.)")
		signal_dist.DrawNormalized()
		bg_dist.DrawNormalized("SAME")
    else:
	if bg_first:
		bg_dist.GetYaxis().SetTitle("(a.u.)")
		bg_dist.Draw()
		signal_dist.Draw("SAME")
	else:
		signal_dist.GetYaxis().SetTitle("(a.u.)")
        	signal_dist.Draw()
        	bg_dist.Draw("SAME")
    leg.Draw()
    canv.Write()
    signal_dist.Write()
    bg_dist.Write()

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

def loadclf (path, zip=True):
    """
    Load pickled classifier from file.
    """
    import gzip
    import pickle

    # Check file suffix
    if path.endswith('.gz'):
        zip = True
        pass

    # Determine operation
    op = gzip.open if zip else open

    # Load model
    with op(path, 'r') as f:
        clf = pickle.load(f)
        pass

    return clf

def add_ddt (data, feat='tau21', newfeat=None, path='models/ddt/ddt_{}.pkl.gz'.format('tau21')):
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
    data[newfeat] = pd.Series(data[feat] - ddt.predict(data[['rhoDDT']].values), index=data.index)
    return

def standardise (array, y=None):
    """
    Standardise axis-variables for kNN regression.

    Arguments:
        array: (N,2) numpy array or Pandas DataFrame containing axis variables.

    Returns:
        (N,2) numpy array containing standardised axis variables.
    """

    VARX = 'rho'
    VARY = 'pt'
    AXIS = {      # Dict holding (num_bins, axis_min, axis_max) for axis variables
    'rho': (20, -7.0, -1.0),
    'pt':  (20, 200., 2000.),
    }
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

def add_knn (data, feat='tau21', newfeat=None, path=None):
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


if __name__ == "__main__":

	#from knn_fitter import add_knn

	#print "Background training jets: {}, test jets: {}".format(sum((data['train']==1)&(data['signal']==0)), sum((data['train']==0)&(data['signal']==0)))

	#data, features, _ = load_data('input/data.h5', test_full_signal=True)
	#data, features, _ = load_data('/afs/cern.ch/work/m/msommerh/private/kNN_fitter/converter/larger_nanoAOD.h5', test_full_signal=True)

	import sys
	sys.path.insert(0, 'tests/studies')
	from ownROCS_general import GetData, GetSimpleData, GetData_corrected, GetSignal
	#data = GetData()
	#data = GetSimpleData()
	#data = GetData_corrected()

	#add_ddt(data, feat='tau21', path='models/ddt/ddt_tau21.pkl.gz')
        #add_knn(data, feat='N2_B1', path='models/knn/knn_{}_{}.pkl.gz'.format('N2_B1', 15))

	#nbg = len(data[data['signal']==0])
	#nbg2 = data[data['signal']==0].shape[0]
	#nsig = len(data[data['signal']==1])
	#nsig2 = data[data['signal']==1].shape[0]
	#print "There are {} background jets and {} signal jets.".format(nbg, nsig)
	#signal_int, bg_int = Plot_variable("jet_pT_nomasscut.root", "jet_pT", "pt", data, (0,2500), Normalized=False, scale_factor=1.)
	#print "The background integrates to {} and the signal to {}.".format(bg_int, signal_int)

	#decDeepWvsQCD = data['decDeepWvsQCD']
	#print "min(decDeepWvsQCD) =", min(decDeepWvsQCD)
	#print "max(decDeepWvsQCD) =", max(decDeepWvsQCD)
	#
	#signal_decDeep = decDeepWvsQCD[data['signal']==1]
	#bg_decDeep = decDeepWvsQCD[data['signal']==0]
	#
	#print "number of signal decDeep at -10 =", sum(signal_decDeep == -10.), "/", sum(signal_decDeep)
	#print "number of bg decDeep at -10 =", sum(bg_decDeep == -10.), "/", sum(bg_decDeep)

	#Plot_variable_from_data_2D("2D_hist.root", "input/data.h5", "#rho", "rho", 20, (-7, -1), "p_{T}", "pt", 20, (200,2000), test=False, weights='weight_test')
	#Plot_variable_from_data_2D("2D_hist.root", "input/data.h5", "#rho", "rho", 20, (-7, -1), "p_{T}", "pt", 20, (200,2000), test=False, weights=None)		
	#Plot_variable_from_data_2D("N2_vs_m.root", "input/data.h5", "N_{2}", "N2_B1", 100, (0., 0.5), "jet m", "m", 20, (30,300), test=False, weights='weight_test')
	#Plot_variable_from_data_2D("tau21_vs_m.root", "input/data.h5", "#tau_{21}", "tau21", 100, (0., 1.), "jet m", "m", 20, (30,300), test=False, weights='weight_test')

	#signal_int, bg_int = Plot_variable("jet_N2_kNN.root", "N2_kNN", "N2_B1kNN", data, (0.,0.5), Normalized=False, scale_factor=1., bg_first=True,uniform_signal_weights=False)
	#signal_int, bg_int = Plot_variable("jet_tau21_DDT.root", "tau21_DDT", "tau21DDT", data, (0,1), Normalized=False, scale_factor=1., bg_first=True, uniform_signal_weights=False)


	#signal_int, bg_int = Plot_variable("rew_corr_jet_N2.root", "N2", "N2_B1", data, (0.,0.5), Normalized=False, scale_factor=1., bg_first=True,uniform_signal_weights=False)
	#signal_int, bg_int = Plot_variable("rew_corr_jet_tau21.root", "tau21", "tau21", data, (0,1), Normalized=False, scale_factor=1., bg_first=False, uniform_signal_weights=False)
	##signal_int, bg_int = Plot_variable("nPV.root", "nPV", "npv", data, (0,100), Normalized=False, scale_factor=1.)
	#signal_int, bg_int = Plot_variable("nano_decDeep.root", "decDeepWvsQCD", "decDeepWvsQCD", data, (-10.1,1.), Normalized=False, scale_factor=1., bg_first=True,uniform_signal_weights=True)
	#signal_int, bg_int = Plot_variable("nano_Deep.root", "DeepWvsQCD", "DeepWvsQCD", data, (-1.1,1.), Normalized=False, scale_factor=1.,uniform_signal_weights=True)
	#signal_int, bg_int = Plot_variable("corr_jet_pT.root", "pt", "pt", data, (0,2000), Normalized=False, scale_factor=1., bg_first=True, uniform_signal_weights=False)
	#signal_int, bg_int = Plot_variable("nano_m.root", "m", "m", data, (0,500), Normalized=False, scale_factor=1.,uniform_signal_weights=True)
	#signal_int, bg_int = Plot_variable("larger_nano_jet_pT.root", "pt", "pt", data, (0,2000), Normalized=False, scale_factor=1., bg_first=True, uniform_signal_weights=True)

	#sys.path.insert(0, '/afs/cern.ch/work/m/msommerh/private/kNN_fitter/converter')
	#from Reweighting import find_weight_function
	#pT = data['pt']
	#weights = data['weight_test']
	#find_weight_function('matched_weights', pT[data['signal']==0], pT[data['signal']==1], 25, Numerator_weights=weights[data['signal']==0], pT_given_by_hist=False, scale_correction=1.)
	

	#f1 = ROOT.TFile.Open("corr_jet_pT.root")
	#f2 = ROOT.TFile.Open("matched_corr_jet_pT.root")
	#unmatched_pt = f1.Get("signal_pt")
	#unmatched_pt.SetNameTitle("unmatched", "unmatched")
	#matched_pt = f2.Get("signal_pt")
	#matched_pt.SetNameTitle("matched", "matched")
	# 
	#unmatched_bin_0 = unmatched_pt.GetXaxis().FindBin(0)
	#unmatched_bin_500 = unmatched_pt.GetXaxis().FindBin(500)
	#unmatched_bin_1000 = unmatched_pt.GetXaxis().FindBin(1000)
	#unmatched_bin_2000 = unmatched_pt.GetXaxis().FindBin(2000)
	#	
	#matched_bin_0 = matched_pt.GetXaxis().FindBin(0)
        #matched_bin_500 = matched_pt.GetXaxis().FindBin(500)
        #matched_bin_1000 = matched_pt.GetXaxis().FindBin(1000)
	#matched_bin_2000 = matched_pt.GetXaxis().FindBin(2000)

	#ratio_0_500 = float(matched_pt.Integral(matched_bin_0, matched_bin_500))/unmatched_pt.Integral(unmatched_bin_0, unmatched_bin_500)
	#ratio_500_1000 = float(matched_pt.Integral(matched_bin_500, matched_bin_1000))/unmatched_pt.Integral(unmatched_bin_500, unmatched_bin_1000)
	#ratio_1000_2000 = float(matched_pt.Integral(matched_bin_1000, matched_bin_2000))/unmatched_pt.Integral(unmatched_bin_1000, unmatched_bin_2000)

	#f1.Close()
	#f2.Close()	

	#matched_pt = GetSignal(selection=False, matching=True)['pt']
	#unmatched_pt = GetSignal(selection=False, matching=False)['pt']
	
	#matched_pt_200_500 = (matched_pt>200)&(matched_pt<500)
	#matched_pt_500_1000 = (matched_pt>500)&(matched_pt<1000)
	#matched_pt_1000_2000 = (matched_pt>1000)&(matched_pt<2000)
	#unmatched_pt_200_500 = (unmatched_pt>200)&(unmatched_pt<500)
        #unmatched_pt_500_1000 = (unmatched_pt>500)&(unmatched_pt<1000)
        #unmatched_pt_1000_2000 = (unmatched_pt>1000)&(unmatched_pt<2000)

	#ratio_0_500 = float(sum(matched_pt_200_500))/sum(unmatched_pt_200_500)
	#ratio_500_1000 = float(sum(matched_pt_500_1000))/sum(unmatched_pt_500_1000)
	#ratio_1000_2000 = float(sum(matched_pt_1000_2000))/sum(unmatched_pt_1000_2000)

	#print "ratio_0_500", ratio_0_500
        #print "ratio_500_1000", ratio_500_1000 
        #print "ratio_1000_2000", ratio_1000_2000
	#print "perc_0_500", (1.-ratio_0_500)*100
	#print "perc_500_1000", (1.-ratio_500_1000)*100 
	#print "perc_1000_2000", (1.-ratio_1000_2000)*100

	#signal_string = '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_{}.root'
	#f1 = ROOT.TFile.Open(signal_string.format(500))
	#f2 = ROOT.TFile.Open(signal_string.format(2000))
	#t1 = f1.Get("AnalysisTree")
	#t2 = f2.Get("AnalysisTree")
	#N1 = t1.GetEntries()
	#N2 = t1.GetEntries()
	#print "entries at {}GeV:".format(500), N1
	#print "entries at {}GeV:".format(2000), N2
	
	unmatched_data = GetSignal(selection=False, matching=False)
	unmatched_pt = unmatched_data['pt']
	unmatched_dR = unmatched_data['truth_dR']

	hist = ROOT.TH2D('hist', 'hist', 100, 200, 2000 , 100, 0, 1.)
	hist.GetXaxis().SetTitle("p_{T}")
	hist.GetYaxis().SetTitle("#DeltaR")
	root_numpy.fill_hist(hist, np.vstack((unmatched_pt, unmatched_dR)).T)	
	#hist.Draw("COLZ")
	hist.ProfileX().Draw()
	
	
