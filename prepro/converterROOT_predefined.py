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
parser.add_argument('--frac-train', type=float, default=0.8,
                    help="Fraction of comined data to use for training.")
parser.add_argument('--seed', type=int, default=21,
                    help="Random-number generator seed, for reproducibility.")

#other global variables:
sig = ['/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/BulkGravToWW_narrow_2000.root']
bkg = [] #will be filled below
collection = 'fjet'
treename = 'AnalysisTree'
sig_weights = [1.] #might need to adjust
bkg_weights = [] #will be filled below
nleading = 1

bg_bins = ['100to200', '200to300', '300to500', '500to700', '700to1000', '1000to1500', '1500to2000', '2000toInf']
bg_cross_sections =  {'100to200': 2.785*10**7,  '200to300': 1717000, '300to500': 351300, '500to700': 31630, '700to1000': 6802, '1000to1500': 1206, '1500to2000': 120.4, '2000toInf': 25.25}
bg_genEvents = {'100to200': 82003456.,  '200to300': 57580393., '300to500': 53096524., '500to700': 56533448., '700to1000': 36741540., '1000to1500': 15210939., '1500to2000': 11839357., '2000toInf': 6019541.}

for bin_ in bg_bins:
        bkg.append("/eos/cms/store/cmst3/group/exovv/VVtuple/FullRun2VVVHNtuple/2016_new/QCD_HT{}.root".format(bin_))
        bkg_weights.append(bg_cross_sections[bin_]/bg_genEvents[bin_])


# Utility function(s)
glob_sort_list = lambda paths: sorted(list(itertools.chain.from_iterable(map(glob, paths))))

branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt', 'jj_l1_softDrop_eta', 'jj_l1_softDrop_phi', 'jj_l1_tau1', 'jj_l1_tau2', 'jj_l1_tau2/jj_l1_tau1', '1-(jj_l1_tau2/jj_l1_tau1)', 'jj_l1_ecfN2_beta1', 'jj_l1_ecfN2_beta2','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_DeepBoosted_probWqq', 'nVert']
selections_l1 = 'jj_l1_softDrop_mass>50 && jj_l1_softDrop_mass<300 && jj_l1_softDrop_pt>200 && jj_l1_softDrop_pt<2000 && TMath::Abs(jj_l1_softDrop_eta)<2.4'
branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt', 'jj_l2_softDrop_eta', 'jj_l2_softDrop_phi', 'jj_l2_tau1', 'jj_l2_tau2', 'jj_l2_tau2/jj_l2_tau1', '1-(jj_l2_tau2/jj_l2_tau1)', 'jj_l2_ecfN2_beta1', 'jj_l2_ecfN2_beta2','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_DeepBoosted_probWqq', 'nVert']
selections_l2 = 'jj_l2_softDrop_mass>50 && jj_l2_softDrop_mass<300 && jj_l2_softDrop_pt>200 && jj_l2_softDrop_pt<2000 && TMath::Abs(jj_l2_softDrop_eta)<2.4'
branches_updated = ['fjet_mass', 'fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_tau1', 'fjet_tau2', 'fjet_tau21', 'fjet_tau21_r', 'fjet_N2_beta1', 'fjet_N2_beta2','fjet_DeepBoosted_WvsQCD', 'fjet_DeepBoosted_probWqq', 'npv']

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
	data_sig_l1 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l1, selection=selections_l1)
	data_sig_l2 = root_numpy.root2array(sig[n], treename=treename, branches=branches_l2, selection=selections_l2)
	data_sig_l1.dtype.names = branches_updated
	data_sig_l2.dtype.names = branches_updated
	data_sig_l1 = rfn.append_fields(data_sig_l1, weight_var, np.full(data_sig_l1.shape[0], sig_weights[n]),   usemask=False)
	data_sig_l2 = rfn.append_fields(data_sig_l2, weight_var, np.full(data_sig_l2.shape[0], sig_weights[n]),   usemask=False)
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
    data.dtype.names = map(rename, data.dtype.names)

    # Variable names
    var_m      = 'fjet_mass'
    var_pt     = 'fjet_pt'
    var_rho    = 'fjet_rho'    # New variable
    var_rhoDDT = 'fjet_rhoDDT' # New variable
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

    # Save to HDF5 file
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset(args.dataset,  data=data, compression='gzip')
        pass

    return 

# Main function call.
if __name__ == '__main__':
    main()
    pass
