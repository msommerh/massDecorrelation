import ROOT as rt
import numpy as np
import matplotlib.pyplot as plt
import root_numpy
import numpy.lib.recfunctions as rfn
import sys
import pickle
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
	signal_hist = rt.TH1D("signal_hist", "signal_hist", nbins, boundary[0], boundary[1])
	if sample_weights is None:
		root_numpy.fill_hist(bg_hist, variable[labels==0], weights=None)
		root_numpy.fill_hist(signal_hist, variable[labels==1], weights=None)
	else:
		root_numpy.fill_hist(bg_hist, variable[labels==0], weights=sample_weights[labels==0])
		root_numpy.fill_hist(signal_hist, variable[labels==1], weights=sample_weights[labels==1])

	fpr, thr = Get_ROC_Efficiencies(bg_hist, boundary, nbins, print_cut=False, cut_below=signal_low)
	tpr, thr = Get_ROC_Efficiencies(signal_hist, boundary, nbins, print_cut=False, cut_below=signal_low)

	return fpr, tpr, thr

def Get_ROC_Efficiencies_Semicut(num_histogram, FullIntegral, ran, nCuts, cut_below=False):
        Cuts = np.linspace(ran[0],ran[1],nCuts+1)
        bin_ran = (num_histogram.GetXaxis().FindBin(ran[0]), num_histogram.GetXaxis().FindBin(ran[1]))
        Efficiencies = np.zeros(nCuts+1)
	FullIntegral_num = num_histogram.Integral(bin_ran[0],bin_ran[1])
	print "num: {} \t denom: {}".format(FullIntegral_num, FullIntegral)
	if cut_below:
	        for n,cut in enumerate(Cuts):
	                bin_cut = num_histogram.GetXaxis().FindBin(cut)
	                Efficiencies[n] = num_histogram.Integral(bin_ran[0],bin_cut)/FullIntegral
	else:
	        for n,cut in enumerate(Cuts):
	                bin_cut = num_histogram.GetXaxis().FindBin(cut)
	                Efficiencies[n] = num_histogram.Integral(bin_cut,bin_ran[1])/FullIntegral

        return Efficiencies, Cuts


def ROC_Curve_Semicut(num_variable, denom_variable, num_labels, denom_labels, boundary, signal_low=False, num_weights=None, denom_weights=None):
	rt.gROOT.SetBatch(True)
	N_num = len(num_variable)
	if num_weights is not None: assert len(num_weights) == N_num
	N_denom = len(denom_variable)
	if denom_weights is not None: assert len(denom_weights) == N_denom

	print "there are {} jets in total and {} pass the mass cut".format(N_denom, N_num)

	nbins = 200

	bg_hist_num = rt.TH1D("bg_hist_num","bg_hist_num", nbins, boundary[0], boundary[1])
	signal_hist_num = rt.TH1D("signal_hist_num", "signal_hist_num", nbins, boundary[0], boundary[1])

	if num_weights is None:
		root_numpy.fill_hist(bg_hist_num, num_variable[num_labels==0], weights=None)
		root_numpy.fill_hist(signal_hist_num, num_variable[num_labels==1], weights=None)
	else:
		root_numpy.fill_hist(bg_hist_num, num_variable[num_labels==0], weights=num_weights[num_labels==0])
		root_numpy.fill_hist(signal_hist_num, num_variable[num_labels==1], weights=num_weights[num_labels==1])

	if denom_weights is not None:
		bg_FullIntegral = sum(denom_weights[denom_labels==0])
		signal_FullIntegral = sum(denom_weights[denom_labels==1])
	else:
		bg_FullIntegral = denom_variable[denom_labels==0].shape[0]
		signal_FullIntegral = denom_variable[denom_labels==1].shape[0]

	fpr, thr = Get_ROC_Efficiencies_Semicut(bg_hist_num, bg_FullIntegral, boundary, nbins, cut_below=signal_low)
	tpr, thr = Get_ROC_Efficiencies_Semicut(signal_hist_num, signal_FullIntegral, boundary, nbins, cut_below=signal_low)

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
	#selections_l1 = 'jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	#selections_l2 = 'jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
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
        	data_bkg_l1 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l1, selection=selections_l1, stop=2000000)
        	data_bkg_l2 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l2, selection=selections_l2, stop=2000000)
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

def GetData_corrected():
	sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1200.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1400.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1600.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1800.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_600.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2500.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_800.root']
	bkg = [] #will be filled below
	treename = 'AnalysisTree'
	bg_bins = ['170to300', '300to470', '470to600', '600to800', '800to1000', '1000to1400', '1400to1800', '1800to2400', '2400to3200', '3200toInf']
	#bg_genEvents = {'170to300': 14796774.0 ,'300to470': 22470404.0 ,'470to600': 3959992.1 ,'600to800': 13119540.0 ,'800to1000': 19504239.0 ,'1000to1400': 9846615.0 ,'1400to1800': 2849545.0 ,'1800to2400': 1982038.0 ,'2400to3200': 996130.0 ,'3200toInf': 391735.0}
	bg_genEvents_list = []
	for bin_ in bg_bins:
		bg_pickle = pickle.load(open( "/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/QCD_Pt_{}.pck".format(bin_), "rb" ))
        	bkg.append("/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/QCD_Pt_{}.root".format(bin_))
        	bg_genEvents_list.append(bg_pickle['events'])
	
	weight_var = 'weight_test'

	branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt','jj_l1_tau2/jj_l1_tau1','jj_l1_ecfN2_beta1','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert', 'genWeight*puWeight*xsec']
	selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert', 'genWeight*puWeight*xsec']
	selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	branches_updated = ['m', 'pt', 'tau21', 'N2_B1', 'DeepWvsQCD', 'decDeepWvsQCD', 'npv', weight_var]

    	for n in range(len(sig)):
        	print "sig data loop, n =",n
        	data_sig_l1 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l1, selection=selections_l1)#+' && TMath::Sqrt(TMath::Sq(jj_l1_eta-jj_l1_gen_eta)+TMath::Sq(jj_l1_phi-jj_l1_gen_phi))<0.6')
        	data_sig_l2 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l2, selection=selections_l2)#+' && TMath::Sqrt(TMath::Sq(jj_l2_eta-jj_l2_gen_eta)+TMath::Sq(jj_l2_phi-jj_l2_gen_phi))<0.6')
        	data_sig_l1.dtype.names = branches_updated
        	data_sig_l2.dtype.names = branches_updated

        	sig_weights_l1 = reweight('prepro/weights/correct_weights.json', data_sig_l1['pt'], np.ones(len(data_sig_l1)), 25, scale_correction=1.)
        	sig_weights_l2 = reweight('prepro/weights/correct_weights.json', data_sig_l2['pt'], np.ones(len(data_sig_l2)), 25, scale_correction=1.)

		data_sig_l1[weight_var] = sig_weights_l1
		data_sig_l2[weight_var] = sig_weights_l2

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
		data_bkg_l1[weight_var] = data_bkg_l1[weight_var]/float(bg_genEvents_list[n])
		data_bkg_l2[weight_var] = data_bkg_l2[weight_var]/float(bg_genEvents_list[n])
        	if n == 0:
            		data_bkg = np.concatenate((data_bkg_l1, data_bkg_l2))
        	else:
            		data_bkg = np.concatenate((data_bkg, np.concatenate((data_bkg_l1, data_bkg_l2))))

	data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
	data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

	data = np.concatenate((data_sig, data_bkg))
	return data

def GetSignal(selection=True, matching=True):
	sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1200.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1400.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1600.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1800.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_600.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2500.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_800.root']
	treename = 'AnalysisTree'
	weight_var = 'weight_test'

	branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt','jj_l1_tau2/jj_l1_tau1','jj_l1_ecfN2_beta1','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert', 'TMath::Sqrt(TMath::Sq(jj_l1_eta-jj_l1_gen_eta)+TMath::Sq(jj_l1_phi-jj_l1_gen_phi))', 'nVert/nVert']
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert', 'TMath::Sqrt(TMath::Sq(jj_l2_eta-jj_l2_gen_eta)+TMath::Sq(jj_l2_phi-jj_l2_gen_phi))', 'nVert/nVert']
	if selection:
		selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
		selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	else:
		selections_l1 = 'jj_l1_softDrop_pt>200' 
	        selections_l2 = 'jj_l2_softDrop_pt>200'
	branches_updated = ['m', 'pt', 'tau21', 'N2_B1', 'DeepWvsQCD', 'decDeepWvsQCD', 'npv', 'truth_dR', weight_var]

    	for n in range(len(sig)):
        	print "sig data loop, n =",n
		if matching:
			selections_l1 += ' && TMath::Sqrt(TMath::Sq(jj_l1_eta-jj_l1_gen_eta)+TMath::Sq(jj_l1_phi-jj_l1_gen_phi))<0.6'
			selections_l2 += ' && TMath::Sqrt(TMath::Sq(jj_l2_eta-jj_l2_gen_eta)+TMath::Sq(jj_l2_phi-jj_l2_gen_phi))<0.6'
        	data_sig_l1 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l1, selection=selections_l1)
        	data_sig_l2 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l2, selection=selections_l2)
        	data_sig_l1.dtype.names = branches_updated
        	data_sig_l2.dtype.names = branches_updated

        	if n == 0:
        	    	data_sig = np.concatenate((data_sig_l1, data_sig_l2))
        	else:
        	    	data_sig = np.concatenate((data_sig, np.concatenate((data_sig_l1, data_sig_l2))))

	data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
	return data_sig


def GetSimpleData():
	sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1200.root']
	bkg = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/QCD_Pt-15to7000.root']
	treename = 'AnalysisTree'
	branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt','jj_l1_tau2/jj_l1_tau1','jj_l1_ecfN2_beta1','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	#selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	selections_l1 = 'jj_l1_softDrop_mass>30 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	#selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	selections_l2 = 'jj_l2_softDrop_mass>30 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
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

        	data_sig_l1 = rfn.append_fields(data_sig_l1, weight_var, np.full(data_sig_l1.shape[0], 1.), usemask=False) #weights to look like signal
        	data_sig_l2 = rfn.append_fields(data_sig_l2, weight_var, np.full(data_sig_l2.shape[0], 1.), usemask=False)

        	if n == 0:
        	    	data_sig = np.concatenate((data_sig_l1, data_sig_l2))
        	else:
        	    	data_sig = np.concatenate((data_sig, np.concatenate((data_sig_l1, data_sig_l2))))

    	for n in range(len(bkg)):
        	print "bkg data loop, n =",n
        	data_bkg_l1 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l1, selection=selections_l1, stop=2000000)
        	data_bkg_l2 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l2, selection=selections_l2, stop=2000000)
        	data_bkg_l1.dtype.names = branches_updated
        	data_bkg_l2.dtype.names = branches_updated
        	data_bkg_l1 = rfn.append_fields(data_bkg_l1, weight_var, np.full(data_bkg_l1.shape[0], 1.),   usemask=False	)
        	data_bkg_l2 = rfn.append_fields(data_bkg_l2, weight_var, np.full(data_bkg_l2.shape[0], 1.),   usemask=False)
        	if n == 0:
            		data_bkg = np.concatenate((data_bkg_l1, data_bkg_l2))
        	else:
            		data_bkg = np.concatenate((data_bkg, np.concatenate((data_bkg_l1, data_bkg_l2))))

	data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
	data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

	data = np.concatenate((data_sig, data_bkg))
	return data

def GetWZData():
	sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_1000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_1200.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_1400.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_1600.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_1800.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_2000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_2500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_3000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_3500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_4000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_4500.root',	'/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_600.root' , '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/WprimeToWZToWhadZhad_narrow_800.root'] 

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
	#selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	selections_l1 = 'jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	#selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	selections_l2 = 'jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
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
        	data_bkg_l1 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l1, selection=selections_l1, stop=2000000)
        	data_bkg_l2 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l2, selection=selections_l2, stop=2000000)
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

def GetZZData():
	sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_1000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_1200.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_1400.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_1600.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_1800.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_2000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_2500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_3000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_3500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_4000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_500.root ', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_600.root ', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToZZToZhadZhad_narrow_800.root ']

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
	#selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	selections_l1 = 'jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
	branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt','jj_l2_tau2/jj_l2_tau1','jj_l2_ecfN2_beta1','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
	#selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
	selections_l2 = 'jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
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
        	data_bkg_l1 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l1, selection=selections_l1, stop=2000000)
        	data_bkg_l2 = root_numpy.root2array(bkg[n], treename=treename, branches=branches_l2, selection=selections_l2, stop=2000000)
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
	
	variables = [("decDeepWvsQCD", True), ("N2_B1", False), ("DeepWvsQCD", True), ("tau21", False)]
	masscut = (65,105)
	global_masscut = (55,215)
	pt_cut = (300,500)
	#pt_cut = (200,2000)
	title = "Deep_Check_matching"

	#data = GetData()
	#data = GetSimpleData()
	#data = GetWZData()
	#data = GetZZData()
	data = GetData_corrected()

	import pandas as pd
	#print "loading data"
	#data = pd.read_hdf('input/data.h5', 'dataset')
	#data = pd.read_hdf('/afs/cern.ch/work/m/msommerh/private/kNN_fitter/converter/larger_nanoAOD.h5', 'dataset')
	#print "skimming to test data"
	#data = data[(data['signal'] == 1) | ((data['train'] == 0) & (data['signal'] == 0))]
	
	#print data[:20]

	#print "define mass condition"
	#mass_condition = (data['m']>masscut[0]) & (data['m']<masscut[1])
	print "define pt condition"
	pt_condition = (data['pt']>pt_cut[0]) & (data['pt']<pt_cut[1])
	global_mass_condition = (data['m']>global_masscut[0]) & (data['m']<global_masscut[1])
	print "apply cuts"
	data_denom = data[pt_condition & global_mass_condition]
	mass_condition = (data_denom['m']>masscut[0]) & (data_denom['m']<masscut[1])
	data_num = data_denom[mass_condition]
	#data = data[mass_condition & pt_condition]

	#print data[:20]

	print "compute roc curves"
	from sklearn.metrics import roc_curve
	fpr, tpr = [], []
	for var in variables:
		print "var[0] =", var[0]
		print "len(data[var[0]]) =", len(data[var[0]])
		#if var[1]:
		#	fpr_, tpr_, _ = roc_curve(data['signal'], data[var[0]], sample_weight=data['weight_test'])
		#else:
		#	fpr_, tpr_, _ = roc_curve(data['signal'], -1*data[var[0]], sample_weight=data['weight_test'])
		#	
		#fpr_, tpr_, _ = ROC_Curve(data[var[0]], data['signal'], (0,1), signal_low= not var[1], sample_weights=data['weight_test'])
		#fpr_, tpr_, _ = ROC_Curve(data[var[0]], data['signal'], (0,1), signal_low= not var[1], sample_weights=None)
		
		#denom_weights=data_denom['weight_test']
		#print 'sum(denom_weights) =', sum(denom_weights)
		fpr_, tpr_, _ = ROC_Curve_Semicut(data_num[var[0]], data_denom[var[0]], data_num['signal'], data_denom['signal'], (-3,1), signal_low= not var[1], num_weights=data_num['weight_test'], denom_weights=data_denom['weight_test'])
	
		#fpr_, tpr_, _ = ROC_Curve_Semicut(data_num[var[0]], data_denom[var[0]], data_num['signal'], data_denom['signal'], (-3,1), signal_low= not var[1], num_weights=None, denom_weights=None)

		fpr.append(fpr_)
		tpr.append(tpr_)

	#import matplotlib.pyplot as plt
	print "draw roc curves"
	plt.figure()
	plt.clf()
	for n, var in enumerate(variables):
		plt.semilogy(tpr[n], fpr[n], label=var[0])
		#plt.plot(tpr[n], fpr[n], label=var[0])
	plt.xlabel(r"$\epsilon$_signal")
	plt.ylabel(r"$\epsilon$_background")
	plt.xlim(0., 1.)
	plt.ylim(1e-4, 1.)
	#plt.ylim(0., 1.)
	plt.legend(loc=4)
	#plt.legend(loc=2)
	plt.grid()
	plt.text(0.05, 0.44, r'$p_{T} \in$ ['+str(pt_cut[0])+', '+str(pt_cut[1])+'] GeV')
	plt.text(0.05, 0.3, r'$m >$ 30 GeV')
	plt.text(0.05, 0.2, r'tagging: $m \in$ ['+str(masscut[0])+', '+str(masscut[1])+'] GeV')
	plt.savefig("ROC_doublechecks/{}_ROC_Curves_Semicut.png".format(title))
	print "saved as ROC_doublechecks/{}_ROC_Curves_Semicut.png".format(title)
