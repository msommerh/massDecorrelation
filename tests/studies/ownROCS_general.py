import ROOT as rt
import numpy as np
import matplotlib.pyplot as plt
import root_numpy
import numpy.lib.recfunctions as rfn
import sys
sys.path.insert(0, 'prepro')
from Reweighting import reweight

def bin_selection(pT,bins):
        """numpy array version of the bin_selection() function: takes an array of all pT values and the pT-bins and returns an array of same length as pT with the corresponing bins index at each entry. pT values outside the bins are labeled with -100"""
        bin_numbers = np.zeros(len(pT))
        for n in range(len(bins)-1):
                bin_numbers += (n+100)*(pT>bins[n])*(pT<bins[n+1])
        bin_numbers -=100
        return bin_numbers.astype(int)

def Get_ROC_Efficiencies(histogram,ran,nCuts,print_cut=False, cut_below=False):
        """Helper function used in Make_ROC_Curves(). Given a discriminant histogram, it finds the cut corresponding most closely to a 10% mistag rate"""
        Cuts = np.linspace(ran[0],ran[1],nCuts+1)
        bin_ran = (histogram.GetXaxis().FindBin(ran[0]),histogram.GetXaxis().FindBin(ran[1]))
        Efficiencies = np.zeros(nCuts+1)
        FullIntegral = histogram.Integral(bin_ran[0],bin_ran[1])
	if cut_below:
	        for n,cut in enumerate(Cuts):
	                bin_cut = histogram.GetXaxis().FindBin(cut)
	                Efficiencies[n] = histogram.Integral(bin_ran[0],bin_cut)/FullIntegral
	else:
	        for n,cut in enumerate(Cuts):
	                bin_cut = histogram.GetXaxis().FindBin(cut)
	                Efficiencies[n] = histogram.Integral(bin_cut,bin_ran[1])/FullIntegral

        diff = 1
        closest = 0
        for n,eff in enumerate(Efficiencies):
                if abs(eff - 0.1) < diff:
                        closest = n
                        diff = abs(eff - 0.1)
        if print_cut:
                print "Mistag rate:",Efficiencies[closest], "corresponding to a cut at", Cuts[closest]
        return Efficiencies, Cuts[closest]

def ROC_Curve(variable, labels, boundary, signal_low=False, sample_weights=None):
	rt.gROOT.SetBatch(True)
	N = len(variable)
	if sample_weights is not None: assert len(sample_weights) == N
	nbins = 200

	bg_hist = rt.TH1D("bg_hist","bg_hist", nbins, boundary[0], boundary[1])
	root_numpy.fill_hist(bg_hist, variable[labels==0], weights=sample_weights[labels==0])
	signal_hist = rt.TH1D("signal_hist", "signal_hist", nbins, boundary[0], boundary[1])
	root_numpy.fill_hist(signal_hist, variable[labels==1], weights=sample_weights[labels==1])

	fpr, thr = Get_ROC_Efficiencies(bg_hist, boundary, nbins, print_cut=False, cut_below=signal_low)
	tpr, thr = Get_ROC_Efficiencies(signal_hist, boundary, nbins, print_cut=False, cut_below=signal_low)

	return fpr, tpr, thr

def GetData():
	sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1200.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1400.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1600.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1800.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_600.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2500.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_800.root']
	bkg = [] #will be filled below
	treename = 'AnalysisTree'
	bkg_weights = [] #will be filled below
	bg_bins = ['170to300', '300to470', '470to600', '600to800', '800to1000', '1000to1400', '1400to1800', '1800to2400', '2400to3200', '3200toInf']
	bg_cross_sections = {'170to300':117276 , '300to470':7823 , '470to600':648.2, '600to800':186.9, '800to1000':32.293, '1000to1400':9.4183, '1400to1800':0.84265, '1800to2400':0.114943, '2400to3200':0.00682981, '3200toInf':0.000165445}
	bg_genEvents = {'170to300': 14796774.0 ,'300to470': 22470404.0 ,'470to600': 3959992.1 ,'600to800': 13119540.0 ,'800to1000': 19504239.0 ,'1000to1400': 9846615.0 ,'1400to1800': 2849545.0 ,'1800to2400': 1982038.0 ,'2400to3200': 996130.0 ,'3200toInf': 391735.0}
	for bin_ in bg_bins:
        	bkg.append("/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/QCD_Pt_{}.root".format(bin_))
        	bkg_weights.append(bg_cross_sections[bin_]/bg_genEvents[bin_])

	branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt','jj_l1_tau2/jj_l1_tau1','jj_l1_ecfN2_beta1','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	branches_updated = ['m', 'pt', 'tau21', 'N2_B1', 'DeepWvsQCD', 'decDeepWvsQCD', 'npv']

	weight_var = 'weight_test'

    	for n in range(len(sig)):
        	print "sig data loop, n =",n
        	data_sig_l1 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l1, selection=selections_l1)
        	data_sig_l2 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l2, selection=selections_l2)
        	data_sig_l1.dtype.names = branches_updated
        	data_sig_l2.dtype.names = branches_updated

        	sig_weights_l1 = reweight('prepro/weights/rescale.json', data_sig_l1['pt'], np.ones(len(data_sig_l1)), 25, scale_correction=1.3)
        	sig_weights_l2 = reweight('prepro/weights/rescale.json', data_sig_l2['pt'], np.ones(len(data_sig_l2)), 25, scale_correction=1.3)

        	data_sig_l1 = rfn.append_fields(data_sig_l1, weight_var, sig_weights_l1, usemask=False) #weights to look like signal
        	data_sig_l2 = rfn.append_fields(data_sig_l2, weight_var, sig_weights_l2, usemask=False)

        	if n == 0:
        	    	data_sig = np.concatenate((data_sig_l1, data_sig_l2))
        	else:
        	    	data_sig = np.concatenate((data_sig, np.concatenate((data_sig_l1, data_sig_l2))))

    	for n in range(len(bkg)):
        	print "bkg data loop, n =",n
        	data_bkg_l1 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l1, selection=selections_l1)
        	data_bkg_l2 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l2, selection=selections_l2)
        	data_bkg_l1.dtype.names = branches_updated
        	data_bkg_l2.dtype.names = branches_updated
        	data_bkg_l1 = rfn.append_fields(data_bkg_l1, weight_var, np.full(data_bkg_l1.shape[0], bkg_weights[n]),   usemask=False	)
        	data_bkg_l2 = rfn.append_fields(data_bkg_l2, weight_var, np.full(data_bkg_l2.shape[0], bkg_weights[n]),   usemask=False)
        	if n == 0:
            		data_bkg = np.concatenate((data_bkg_l1, data_bkg_l2))
        	else:
            		data_bkg = np.concatenate((data_bkg, np.concatenate((data_bkg_l1, data_bkg_l2))))

	data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
	data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

	data = np.concatenate((data_sig, data_bkg))
	return data


if __name__ == "__main__":
	
	variables = [("decDeepWvsQCD", True), ("DeepWvsQCD", True), ("tau21", False), ("N2_B1", False)]
	masscut = (65,105)
	pt_cut = (300,500)
	title = "Deep_Check"

	data = GetData()

	#import pandas as pd
	#print "loading data"
	#data = pd.read_hdf('input/data.h5', 'dataset')
	#print "skimming to test data"
	#data = data[(data['signal'] == 1) | ((data['train'] == 0) & (data['signal'] == 0))]
	
	print data[:20]

	print "define mass condition"
	mass_condition = (data['m']>masscut[0]) & (data['m']<masscut[1])
	print "define pt condition"
	pt_condition = (data['pt']>pt_cut[0]) & (data['pt']<pt_cut[1])
	print "apply cuts"
	data = data[mass_condition & pt_condition]

	#print data[:20]

	print "compute roc curves"
	from sklearn.metrics import roc_curve
	fpr, tpr = [], []
	for var in variables:
		#if var[1]:
		#	fpr_, tpr_, _ = roc_curve(data['signal'], data[var[0]], sample_weight=data['weight_test'])
		#else:
		#	fpr_, tpr_, _ = roc_curve(data['signal'], -1*data[var[0]], sample_weight=data['weight_test'])
		#	
		fpr_, tpr_, _ = ROC_Curve(data[var[0]], data['signal'], (0,1), signal_low= not var[1], sample_weights=data['weight_test'])
		fpr.append(fpr_)
		tpr.append(tpr_)

	#import matplotlib.pyplot as plt
	print "draw roc curves"
	plt.figure()
	plt.clf()
	for n, var in enumerate(variables):
		plt.semilogy(tpr[n], fpr[n], label=var[0])
	plt.xlabel(r"$\epsilon$_signal")
	plt.ylabel(r"$\epsilon$_background")
	plt.xlim(0., 1.)
	plt.ylim(1e-4, 1.)
	plt.legend(loc=4)
	plt.grid()
	plt.savefig("ROC_doublechecks/{}_ROC_Curves2.png".format(title))
	print "saved as ROC_doublechecks/{}_ROC_Curves2.png".format(title)
