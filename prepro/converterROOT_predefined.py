#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import argparse
import itertools
from glob import glob

# Scientific import(s)
import numpy as np
import numpy.lib.recfunctions as rfn
import ROOT
import root_numpy

# Project import(s)
from adversarial.profile import profile
from .Reweighting import reweight

# Command-line argument parser
parser = argparse.ArgumentParser(description="Convert generally non-flat ROOT file(s) to single HDF5 file")
parser.add_argument('--output', default='data.h5',
                    help="Name of output HDF5 file.")
parser.add_argument('--dataset', default='dataset',
                    help="Name of dataset tin output HDF5 file.")
parser.add_argument('--treename', default='jetTree/nominal',
                    help="Name of ROOT TTree to be used.")
parser.add_argument('--no-shuffle', action='store_false',
                    help="Don't shuffle data before (optionally) subsampling.")
parser.add_argument('--sample', type=float, default=0,
                    help="Fraction of combined data to subsample.")
parser.add_argument('--replace', action='store_true',
                    help="Whether to subsample with replacement.")
parser.add_argument('--frac-train', type=float, default=0.2, #changed from 0.8 to 0.5
                    help="Fraction of comined data to use for training.")
parser.add_argument('--seed', type=int, default=21,
                    help="Random-number generator seed, for reproducibility.")

#other global variables:
#sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2000.root']

sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1200.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_3500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1400.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4000.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1600.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_4500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_1800.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_500.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2000.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_600.root','/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2500.root', '/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_800.root']

bkg = [] #will be filled below
collection = 'fjet'
treename = 'AnalysisTree'
bkg_weights = [] #will be filled below
nleading = 1

bg_bins = ['170to300', '300to470', '470to600', '600to800', '800to1000', '1000to1400', '1400to1800', '1800to2400', '2400to3200', '3200toInf']
bg_cross_sections = {'170to300':117276 , '300to470':7823 , '470to600':648.2, '600to800':186.9, '800to1000':32.293, '1000to1400':9.4183, '1400to1800':0.84265, '1800to2400':0.114943, '2400to3200':0.00682981, '3200toInf':0.000165445}
bg_genEvents = {'170to300': 14796774.0 ,'300to470': 22470404.0 ,'470to600': 3959992.1 ,'600to800': 13119540.0 ,'800to1000': 19504239.0 ,'1000to1400': 9846615.0 ,'1400to1800': 2849545.0 ,'1800to2400': 1982038.0 ,'2400to3200': 996130.0 ,'3200toInf': 391735.0}
for bin_ in bg_bins:
	bkg.append("/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/QCD_Pt_{}.root".format(bin_))
	bkg_weights.append(bg_cross_sections[bin_]/bg_genEvents[bin_])


# Utility function(s)
glob_sort_list = lambda paths: sorted(list(itertools.chain.from_iterable(map(glob, paths))))

branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt', 'jj_l1_softDrop_eta', 'jj_l1_softDrop_phi', 'jj_l1_tau1', 'jj_l1_tau2', 'jj_l1_tau2/jj_l1_tau1','jj_l1_ecfN2_beta1', 'jj_l1_ecfN2_beta2','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt', 'jj_l2_softDrop_eta', 'jj_l2_softDrop_phi', 'jj_l2_tau1', 'jj_l2_tau2', 'jj_l2_tau2/jj_l2_tau1', 'jj_l2_ecfN2_beta1', 'jj_l2_ecfN2_beta2','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_MassDecorrelatedDeepBoosted_WvsQCD', 'nVert']
selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
branches_updated = ['m', 'pt', 'eta', 'phi', 'tau1', 'tau2', 'tau21', 'N2_B1', 'N2_B2','DeepWvsQCD', 'decDeepWvsQCD', 'npv']

def unravel (data, nleading):
    """
    ...
    """

    if not data.dtype.hasobject:
        return data

    if nleading == 0:
        nleading = 99999
        pass
    # Identify variable-length (i.e. per-jet) and scalar (i.e. per-event)
    # fields
    jet_fields = list()
    for field, (kind, _) in data.dtype.fields.iteritems():
        if kind.hasobject:
            jet_fields.append(field)
            pass
        pass
    jet_fields   = sorted(jet_fields)
    event_fields = sorted([field for field in data.dtype.names if field not in jet_fields])
    # Loop events, take up to `nleading` jets from each
    jets = list()
    data_events = data[event_fields]
    data_jets   = data[jet_fields]
    
    rows = list()
    for jets, event in zip(data_jets, data_events):
        for jet in np.array(jets.tolist()).T[:nleading]:
            row = event.copy()
            row = rfn.append_fields(row, jet_fields, jet.tolist(), usemask=False)
            rows.append(row)
            pass
        pass
    return np.concatenate(rows)



# Main function definition.
@profile
def main ():
    """
    ...
    """

    # Parse command-line argument
    args = parser.parse_args()

    # Check(s)
    assert not args.output.startswith('/')
    assert args.output.endswith('.h5')
    assert args.sample <= 1.0
    assert args.sample >= 0.0
    assert args.frac_train <= 1.0
    assert args.frac_train >= 0.0

    # Convenience
    shuffle = not args.no_shuffle

    # Renaming method
    def rename (name):
        name = name.replace(collection, 'fjet')
        return name

    # For reproducibility
    rng = np.random.RandomState(seed=args.seed)

    # Get glob'ed list of files for each category
    #sig = glob_sort_list(args.sig)
    #bkg = glob_sort_list(args.bkg)

    print "Found {} signal and {} background files.".format(len(sig), len(bkg))

    # Read in data

    weight_var = 'weight_test'
    #weight_var = 'mcEventWeight'

    for n in range(len(sig)):
	print "sig data loop, n =",n
	data_sig_l1 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l1, selection=selections_l1)
	data_sig_l2 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l2, selection=selections_l2)
	data_sig_l1.dtype.names = branches_updated
	data_sig_l2.dtype.names = branches_updated
	#data_sig_l1 = rfn.append_fields(data_sig_l1, weight_var, np.full(data_sig_l1.shape[0], sig_weights[n]),   usemask=False) # uniform weights
	#data_sig_l2 = rfn.append_fields(data_sig_l2, weight_var, np.full(data_sig_l2.shape[0], sig_weights[n]),   usemask=False)

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
	data_bkg_l1 = rfn.append_fields(data_bkg_l1, weight_var, np.full(data_bkg_l1.shape[0], bkg_weights[n]),   usemask=False)
	data_bkg_l2 = rfn.append_fields(data_bkg_l2, weight_var, np.full(data_bkg_l2.shape[0], bkg_weights[n]),   usemask=False)
	if n == 0:
	    data_bkg = np.concatenate((data_bkg_l1, data_bkg_l2))
	else:
	    data_bkg = np.concatenate((data_bkg, np.concatenate((data_bkg_l1, data_bkg_l2))))

    # Read in data
    #data_sig_l1 = root_numpy.root2array(sig, treename=treename, branches=branches_l1)
    #data_sig_l1 = rfn.append_fields(data_sig_l1, 'mcEventWeight', np.full(data_sig_l1.shape[0], weight),   usemask=False)

    #data_sig_l2 = root_numpy.root2array(sig, treename=treename, branches=branches_l2)
    #data_sig_l2.dtype.names = branches_updated
    #data_bkg_l1 = root_numpy.root2array(bkg, treename=treename, branches=branches_l1)
    #data_bkg_l1.dtype.names = branches_updated
    #data_bkg_l2 = root_numpy.root2array(bkg, treename=treename, branches=branches_l2)
    #data_bkg_l2.dtype.names = branches_updated
    #data_sig = np.concatenate((data_sig_l1, data_sig_l2))
    #data_bkg = np.concatenate((data_bkg_l1, data_bkg_l2))
    #data_sig.dtype.names = branches_updated
    #data_bkg.dtype.names = branches_updated

    # (Opt.) Unravel non-flat data
    data_sig = unravel(data_sig, nleading)
    data_bkg = unravel(data_bkg, nleading)

    # Append signal fields
    data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate arrays
    data = np.concatenate((data_sig, data_bkg))

    # Rename columns
    #data.dtype.names = map(rename, data.dtype.names)

    # Variable names
    var_m      = 'm'
    var_pt     = 'pt'
    var_rho    = 'rho'    # New variable
    var_rhoDDT = 'rhoDDT' # New variable
    #var_weight = 'mcEventWeight' # New variable

    # Object selection
    msk = (data[var_pt] > 10.) & (data[var_m] > 10.) # @TODO: Generalise?
    data = data[msk]

    # Append rhoDDT field
    data = rfn.append_fields(data, var_rho,    np.log(data[var_m]**2 / data[var_pt]**2),   usemask=False)
    data = rfn.append_fields(data, var_rhoDDT, np.log(data[var_m]**2 / data[var_pt] / 1.), usemask=False)

    # Append train field
    data = rfn.append_fields(data, "train", rng.rand(data.shape[0]) < args.frac_train, usemask=False)

    # (Opt.) Shuffle
    if shuffle:
        rng.shuffle(data)
        pass

    # (Opt.) Subsample
    if args.sample:
        data = rng.choice(data, args.sample, replace=args.replace)
        pass

    print "There are {} background jets and {} signal jets.".format(len(data_bkg), len(data_sig))
    print "There are in total {} jets.".format(len(data))

    # Save to HDF5 file
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset(args.dataset,  data=data, compression='gzip')
        pass

    print "Background training jets: {}, test jets: {}".format(sum((data['train']==1)&(data['signal']==0)), sum((data['train']==0)&(data['signal']==0)))

    return 

# Main function call.
if __name__ == '__main__':
    main()
    pass
