import ROOT as rt
import numpy as np
from root_numpy import fill_hist, root2array

def find_weight_function(title, Numerator_pT, Denominator_pT, binsize, Numerator_weights=None, pT_given_by_hist=False, scale_correction=1.):

        maximum = 2500
        bins = range(0,maximum+1,binsize)
        nbins = len(bins)-1
        import array
        bins_ = array.array('d',bins)
        Numerator_hist = rt.TH1D("Numerator_hist","Numerator_hist",nbins,bins_)
        Denominator_hist = rt.TH1D("Denominator_hist","Denominator_hist",nbins,bins_)

	if pT_given_by_hist:
		Numerator_hist = Numerator_pT.Clone()	
		Denominator_hist = Denominator_pT.Clone()
	else:
		if Numerator_weights:
			fill_hist(Numerator_hist, Numerator_pT, weights=Numerator_weights)
		else:
			fill_hist(Numerator_hist, Numerator_pT)
		fill_hist(Denominator_hist, Denominator_pT)

        Ratio_hist = Numerator_hist.Clone()
        Ratio_hist.SetName("Ratio_hist")
        Ratio_hist.SetTitle("Ratio_hist")
        Ratio_hist.Divide(Denominator_hist)

        ratio_dict = {}
        for bin_nr in range(nbins):
                ratio_dict[bins[bin_nr]] = scale_correction*Ratio_hist.GetBinContent(bin_nr+1)

        import json
        with open("weights/{}.json".format(title), "w") as fp:
                json.dump(ratio_dict,fp)

def reweight(weight_file, train_pt, train_y, bin_size, scale_correction=1.):
        maximum = 2500

        import json
        with open(weight_file, "r") as fp:
                weight_function = json.load(fp)

        weights = np.zeros(len(train_pt))
        for n,entry in enumerate(train_y):
                if entry == 0:
                        weights[n] = 1
                else:
                        if train_pt[n] > maximum:
                                weights[n] = 0
                        else:
                                weights[n] = scale_correction*weight_function[str((int(train_pt[n])/bin_size)*bin_size)]
        return weights

#if __name__ == "__main__":

	
