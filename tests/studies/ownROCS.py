import ROOT as rt
import numpy as np
import matplotlib.pyplot as plt
import root_numpy

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

def Make_Binned_ROC_histograms(title, DDT, kNN, pT, bins, sample_weights=None):
        rt.gROOT.SetBatch(True)
        N = len(DDT)
        assert len(pT) == N and len(kNN) == N
	if sample_weights is not None: assert len(sample_weights) == N

        nbins = 100
        DDT_hist_list = []
        kNN_hist_list = []
        for bin_ in range(len(bins)-1):
                DDT_hist_list.append(rt.TH1D("DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]),"DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]),nbins,0,1))
                kNN_hist_list.append(rt.TH1D("kNN_"+str(bins[bin_])+"_"+str(bins[bin_+1]),"kNN_"+str(bins[bin_])+"_"+str(bins[bin_+1]),nbins,0,1))

		condition = np.logical_and((pT>bins[bin_]), (pT<bins[bin_+1]))
		root_numpy.fill_hist(DDT_hist_list[bin_], DDT[condition], weights=sample_weights)
		root_numpy.fill_hist(kNN_hist_list[bin_], kNN[condition], weights=sample_weights)

        tfile = rt.TFile("ROC_doublechecks/{}_ROC_histograms.root".format(title),"recreate")
        for hist in DDT_hist_list:
                hist.Write()
        for hist in kNN_hist_list:
                hist.Write()
        print "saved histograms in ROC_doublechecks/{}_ROC_histograms.root".format(title)

def Make_Binned_ROC_Curves(title,Signal_title,Background_title,bins,log=False, cut_below=False, bg_eff_representation=False):
        """accesses the files made by ANN_Make_Binned_ROC_histograms() directly to produce a ROC curve for ANN and CSV in the desired pT-bins. A log representation can be turned on"""
        if len(bins)<=6:
                color = ['red','green','blue','orange','brown']
        else:
                color = ['deepskyblue','rosybrown','olivedrab','royalblue','firebrick','chartreuse','navy','red','darkorchid','lightseagreen','mediumvioletred','blue']
        nbins = 100

        Signal_file = rt.TFile("ROC_doublechecks/{}_ROC_histograms.root".format(Signal_title),"READ")
        Background_file =   rt.TFile("ROC_doublechecks/{}_ROC_histograms.root".format(Background_title),"READ")

        plt.figure("ROC")
        plt.clf()

        for bin_ in range(len(bins)-1):
                DDT_Signal_Eff, _C = Get_ROC_Efficiencies(Signal_file.Get("DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1])),(0,1),nbins,0, cut_below=cut_below)
                DDT_BG_Eff, _C = Get_ROC_Efficiencies(Background_file.Get("DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1])),(0,1),nbins,0, cut_below=cut_below)
                kNN_Signal_Eff, _C = Get_ROC_Efficiencies(Signal_file.Get("kNN_"+str(bins[bin_])+"_"+str(bins[bin_+1])),(0,1),nbins,0, cut_below=cut_below)
                kNN_BG_Eff, _C = Get_ROC_Efficiencies(Background_file.Get("kNN_"+str(bins[bin_])+"_"+str(bins[bin_+1])),(0,1),nbins,0, cut_below=cut_below)

		#print "DDT_Signal_Eff[:20] =", DDT_Signal_Eff[:20]
		#print "DDT_BG_Eff[:20] =", DDT_BG_Eff[:20]
		#print "kNN_Signal_Eff[:20] =", kNN_Signal_Eff[:20]
		#print "kNN_BG_Eff[:20] =", kNN_BG_Eff[:20]

	        indices = np.where((DDT_BG_Eff > 0) & (DDT_Signal_Eff > 0))
        	DDT_Signal_Eff = DDT_Signal_Eff[indices]
        	DDT_BG_Eff = DDT_BG_Eff[indices]
	        indices = np.where((kNN_BG_Eff > 0) & (kNN_Signal_Eff > 0))
        	kNN_Signal_Eff = kNN_Signal_Eff[indices]
        	kNN_BG_Eff = kNN_BG_Eff[indices]


		if bg_eff_representation:
                	if log:
                	        plt.semilogy(DDT_Signal_Eff,DDT_BG_Eff, color = color[bin_], linestyle = '-',label="DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]))
                	        plt.semilogy(kNN_Signal_Eff,kNN_BG_Eff, color = color[bin_+1],linestyle = '-',label="kNN_"+str(bins[bin_])+"_"+str(bins[bin_+1]))

                	else:
                	        plt.plot(DDT_Signal_Eff,DDT_BG_Eff, color = color[bin_], linestyle = '-',label="DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]))
                	        plt.plot(kNN_Signal_Eff,kNN_BG_Eff, color = color[bin_+1],linestyle = '-',label="DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]))
			plt.ylabel(r"$\epsilon$_background")
			plt.ylim(0., 1.)

		else:
                	if log:
                	        plt.semilogy(DDT_Signal_Eff,np.power(DDT_BG_Eff,-1), color = color[bin_], linestyle = '-',label="DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]))
                	        plt.semilogy(kNN_Signal_Eff,np.power(kNN_BG_Eff,-1), color = color[bin_+1],linestyle = '-',label="kNN_"+str(bins[bin_])+"_"+str(bins[bin_+1]))

                	else:
                	        plt.plot(DDT_Signal_Eff,np.power(DDT_BG_Eff,-1), color = color[bin_], linestyle = '-',label="DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]))
                	        plt.plot(kNN_Signal_Eff,np.power(kNN_BG_Eff,-1), color = color[bin_+1],linestyle = '-',label="DDT_"+str(bins[bin_])+"_"+str(bins[bin_+1]))
			plt.ylabel(r"1/$\epsilon$_background")
			plt.ylim(1, 1e3)
	
	plt.xlabel(r"$\epsilon$_signal")
	plt.xlim(0.2, 1.)
	plt.legend(loc=1)
	plt.savefig("ROC_doublechecks/{}_ROC_Curves.png".format(title))
        print "saved as ROC_doublechecks/{}_ROC_Curves.png".format(title)

if __name__ == "__main__":
	Make_Binned_ROC_Curves("full_spectrum", "full_signal", "full_bg", [200,2000],log=True, cut_below=True, bg_eff_representation)
