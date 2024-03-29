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
parser.add_argument('--sig', nargs='+', required=True,
                    help="List of signal file(s), supporting wild-cards.")
parser.add_argument('--bkg', nargs='+', required=True,
                    help="List of background file(s), supporting wild-cards.")
parser.add_argument('--output', default='data.h5',
                    help="Name of output HDF5 file.")
parser.add_argument('--dataset', default='dataset',
                    help="Name of dataset tin output HDF5 file.")
parser.add_argument('--collection', default='AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsCalibSelect',
                    help="Name of jet collection, which will be renamed to `fjet`.")
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
parser.add_argument('--nleading', type=int, default=2,
                    help="Take only up to `nleading` jets in each event, given sorting.")
parser.add_argument('--weight', type=float, default=1.,
                    help="Background weight. Currently only single weight possible.")

# Utility function(s)
glob_sort_list = lambda paths: sorted(list(itertools.chain.from_iterable(map(glob, paths))))

branches_l1 = ['jj_l1_softDrop_mass', 'jj_l1_softDrop_pt', 'jj_l1_softDrop_eta', 'jj_l1_softDrop_phi', 'jj_l1_tau1', 'jj_l1_tau2', 'jj_l1_tau2/jj_l1_tau1', 'jj_l1_ecfN2_beta1', 'jj_l1_ecfN2_beta2','jj_l1_DeepBoosted_WvsQCD', 'jj_l1_DeepBoosted_probWqq']
branches_l2 = ['jj_l2_softDrop_mass', 'jj_l2_softDrop_pt', 'jj_l2_softDrop_eta', 'jj_l2_softDrop_phi', 'jj_l2_tau1', 'jj_l2_tau2', 'jj_l2_tau2/jj_l2_tau1', 'jj_l2_ecfN2_beta1', 'jj_l2_ecfN2_beta2','jj_l2_DeepBoosted_WvsQCD', 'jj_l2_DeepBoosted_probWqq']
branches_updated = ['fjet_mass', 'fjet_pt', 'fjet_eta', 'fjet_phi', 'fjet_tau1', 'fjet_tau2', 'fjet_tau21', 'fjet_N2_beta1', 'fjet_N2_beta2','fjet_DeepBoosted_WvsQCD', 'fjet_DeepBoosted_probWqq']

def unravel (data, nleading):
    """
    ...
    """

    print "unravel has been entered"    
    if not data.dtype.hasobject:
        return data

    if nleading == 0:
        nleading = 99999
        pass
    print "unravel point A"
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
    print "unravel point B"
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
    print "unravel point C"
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
    assert args.nleading >= 0

    # Convenience
    shuffle = not args.no_shuffle

    # Renaming method
    def rename (name):
        name = name.replace(args.collection, 'fjet')
        return name

    # For reproducibility
    rng = np.random.RandomState(seed=args.seed)

    # Get glob'ed list of files for each category
    sig = glob_sort_list(args.sig)
    bkg = glob_sort_list(args.bkg)

    print "Found {} signal and {} background files.".format(len(sig), len(bkg))

    # Read in data
    data_sig_l1 = root_numpy.root2array(sig, treename=args.treename, branches=branches_l1)
    data_sig_l1.dtype.names = branches_updated
    data_sig_l2 = root_numpy.root2array(sig, treename=args.treename, branches=branches_l2)
    data_sig_l2.dtype.names = branches_updated
    data_bkg_l1 = root_numpy.root2array(bkg, treename=args.treename, branches=branches_l1)
    data_bkg_l1.dtype.names = branches_updated
    data_bkg_l2 = root_numpy.root2array(bkg, treename=args.treename, branches=branches_l2)
    data_bkg_l2.dtype.names = branches_updated
    #data_sig = root_numpy.root2array(sig, treename=args.treename)
    #data_bkg = root_numpy.root2array(bkg, treename=args.treename)
    print "data_sig_l1.shape =", data_sig_l1.shape
    print "data_sig_l2.shape =", data_sig_l2.shape
    print "data_sig_l1[:20] =", data_sig_l1[:20]
    data_sig = np.concatenate((data_sig_l1, data_sig_l2))
    data_bkg = np.concatenate((data_bkg_l1, data_bkg_l2))
    data_sig.dtype.names = branches_updated
    data_bkg.dtype.names = branches_updated

    # (Opt.) Unravel non-flat data
    data_sig = unravel(data_sig, args.nleading)
    data_bkg = unravel(data_bkg, args.nleading)

    # Append signal fields
    data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate arrays
    data = np.concatenate((data_sig, data_bkg))

    # Rename columns
    data.dtype.names = map(rename, data.dtype.names)

    # Variable names
    var_m      = 'fjet_mass'
    #var_m      = 'fjet_JetConstitScaleMomentum_m'
    var_pt     = 'fjet_pt'
    var_rho    = 'fjet_rho'    # New variable
    var_rhoDDT = 'fjet_rhoDDT' # New variable
    var_weight = 'mcEventWeight' # New variable

    # Object selection
    msk = (data[var_pt] > 10.) & (data[var_m] > 10.) # @TODO: Generalise?
    data = data[msk]

    # Append rhoDDT field
    data = rfn.append_fields(data, var_rho,    np.log(data[var_m]**2 / data[var_pt]**2),   usemask=False)
    data = rfn.append_fields(data, var_rhoDDT, np.log(data[var_m]**2 / data[var_pt] / 1.), usemask=False)
    data = rfn.append_fields(data, var_weight, np.full(data.shape[0],args.weight),   usemask=False)

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
