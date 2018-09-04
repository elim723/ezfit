#!/usr/bin/env python
import os, sys, pickle, numpy, matplotlib

from scipy import interpolate
numpy.set_printoptions(threshold=numpy.nan)
matplotlib.use('Agg')
from matplotlib import pyplot
pyplot.rcParams['text.usetex'] = True
from matplotlib import gridspec

def Plots_1D(xvals, xbins, logx, logy, xlabel, ylim,
             weights_standard, weights_p1s, weights_m1s,
             weights_p1ns, weights_m1ns, xname):
    pyplot.figure()
    hist, bins = numpy.histogram(xvals,
                                 weights = weights_standard,
                                 bins = xbins)

    # Get the errorbars on this too
    err, bins = numpy.histogram(xvals,
                                weights = weights_standard**2,
                                bins = xbins)

    # Plot them together
    pyplot.bar(left = bins[:-1], 
               height = 2*numpy.sqrt(err), 
               width = numpy.diff(bins),
               bottom = hist - numpy.sqrt(err),
               color='k',
               alpha=0.4)
    pyplot.hist(bins[:-1],
                weights = hist,
                bins = bins,
                color = 'k',
                linewidth = 2,
                histtype='step')

    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")

    pyplot.grid()
    pyplot.xlabel(xlabel)
    pyplot.ylabel("Rate (mHz)")
    pyplot.ylim(ylim)
    pyplot.xlim([bins[0],bins[-1]])
    pyplot.title("Corsika 10282+10309+11905+12268+12332 (L4)")
    pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332%s.pdf"%xname)

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1ns, bins = numpy.histogram(xvals,
                                 weights = weights_p1ns,
                                 bins = xbins)
    m1ns, bins = numpy.histogram(xvals,
                                 weights = weights_m1ns,
                                 bins = xbins)

    gs = gridspec.GridSpec(2,1,height_ratios=[3,1])

    pyplot.subplot(gs[0])

    pyplot.bar(left = bins[:-1], 
               height = p1ns-hist, 
               width = numpy.diff(bins),
               bottom = hist,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = hist-m1ns, 
               width = numpy.diff(bins),
               bottom = m1ns,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')
    pyplot.hist(bins[:-1],
                weights = hist,
                bins = bins,
                color = 'k',
                linewidth = 1,
                histtype='step',
                label = 'Baseline')

    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")

    pyplot.grid()
    pyplot.ylabel("Rate (mHz)")
    pyplot.ylim(ylim)
    pyplot.xlim([bins[0],bins[-1]])
    pyplot.title("Corsika 10282+10309+11905+12268+12332 (L4) \n Including Primary Uncertainties (Normalised Data)")
    pyplot.legend(loc='best')

    pyplot.subplot(gs[1])

    pyplot.bar(left = bins[:-1], 
               height = numpy.nan_to_num((p1ns-hist)/hist), 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = numpy.nan_to_num((m1ns-hist)/hist), 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")

    pyplot.grid()
    pyplot.xlabel(xlabel)
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332%s_prim_unc_err_norm.pdf"%xname)

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1s, bins = numpy.histogram(xvals,
                                weights = weights_p1s,
                                bins = xbins)
    m1s, bins = numpy.histogram(xvals,
                                weights = weights_m1s,
                                bins = xbins)

    pyplot.subplot(gs[0])

    pyplot.bar(left = bins[:-1], 
               height = p1s-hist, 
               width = numpy.diff(bins),
               bottom = hist,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = hist-m1s, 
               width = numpy.diff(bins),
               bottom = m1s,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')
    pyplot.hist(bins[:-1],
                weights = hist,
                bins = bins,
                color = 'k',
                linewidth = 1,
                histtype='step',
                label = 'Baseline')

    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")

    pyplot.grid()
    pyplot.ylabel("Rate (mHz)")
    pyplot.ylim(ylim)
    pyplot.xlim([bins[0],bins[-1]])
    pyplot.title("Corsika 10282+10309+11905+12268+12332 (L4) \n Including Primary Uncertainties (Unmodified Data)")
    pyplot.legend(loc='best')

    pyplot.subplot(gs[1])

    pyplot.bar(left = bins[:-1], 
               height = numpy.nan_to_num((p1s-hist)/hist), 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = numpy.nan_to_num((m1s-hist)/hist), 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")

    pyplot.grid()
    pyplot.xlabel(xlabel)
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332%s_prim_unc_err.pdf"%xname)

    pyplot.close()

    return bins, numpy.nan_to_num((p1s-hist)/hist), numpy.nan_to_num((p1ns-hist)/hist)

def Save_1D(outfile, xvals, yvals):

    f = open('Uncertainties/%s'%outfile, 'w')

    for xval, yval in zip(xvals, yvals):
        f.write('%.4f %.4f\n'%(xval,yval))

    f.close()
    
#######################################################
# Load the results dictionary with this structure:
#
# results = {"weights":weights,
#            "energy_of_primary": energy_of_primary,
#            "energy_at_depth": energy_at_depth,
#            "coszens": coszens,
#            }
#######################################################

results = pickle.load(open("Data/Level5/10282+10309+11905+12268+12332corsika.pckl"))

all_data_sorted_by_code = {}
for code in set(results['code_of_primary']):
    all_data_sorted_by_code[code] = {}
    for val_key in results.keys():
        all_data_sorted_by_code[code][val_key] = []

for i in range(0,len(results['energy_at_depth'])-1):
    
    code = results['code_of_primary'][i]
    for val_key in results.keys():
        all_data_sorted_by_code[code][val_key].append(results[val_key][i])

pdg_labels = {}
pdg_colours = {}
all_codes = []
singular_weights = []

if 1000160320 in all_data_sorted_by_code.keys():
    pdg_labels[1000160320] = r'Silicon 32 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000160320]['weights'])* 1e6)
    pdg_colours[1000160320] = 'coral'
    all_codes.append(1000160320)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000160320]['weights'])* 1e6))

if 1000040090 in all_data_sorted_by_code.keys():
    pdg_labels[1000040090] = r'Beryllium 9 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000040090]['weights']) * 1e6)
    pdg_colours[1000040090] = 'indigo'
    all_codes.append(1000040090)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000040090]['weights'])* 1e6))

if 1000100200 in all_data_sorted_by_code.keys():
    pdg_labels[1000100200] = r'Neon 20 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000100200]['weights']) * 1e6)
    pdg_colours[1000100200] = 'lime'
    all_codes.append(1000100200)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000100200]['weights'])* 1e6))
        
if 1000080160 in all_data_sorted_by_code.keys():
    pdg_labels[1000080160] = r'Oxygen 16 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000080160]['weights']) * 1e6)
    pdg_colours[1000080160] = 'deepskyblue'
    all_codes.append(1000080160)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000080160]['weights'])* 1e6))

if 1000060120 in all_data_sorted_by_code.keys():
    pdg_labels[1000060120] = r'Carbon 12 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000060120]['weights']) * 1e6)
    pdg_colours[1000060120] = 'mediumorchid'
    all_codes.append(1000060120)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000060120]['weights'])* 1e6))

if 1000260560 in all_data_sorted_by_code.keys():
    pdg_labels[1000260560] = r'Iron 56 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000260560]['weights']) * 1e6)
    pdg_colours[1000260560] = 'saddlebrown'
    all_codes.append(1000260560)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000260560]['weights'])* 1e6))

if 1000130270 in all_data_sorted_by_code.keys():
    pdg_labels[1000130270] = r'Aluminium 27 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000130270]['weights']) * 1e6)
    pdg_colours[1000130270] = 'dimgray'
    all_codes.append(1000130270)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000130270]['weights'])* 1e6))

if 1000070140 in all_data_sorted_by_code.keys():
    pdg_labels[1000070140] = r'Nitrogen 14 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000070140]['weights']) * 1e6)
    pdg_colours[1000070140] = 'blue'
    all_codes.append(1000070140)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000070140]['weights'])* 1e6))

if 1000020040 in all_data_sorted_by_code.keys():
    pdg_labels[1000020040] = r'Helium 4 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000020040]['weights']) * 1e6)
    pdg_colours[1000020040] = 'green'
    all_codes.append(1000020040)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000020040]['weights'])* 1e6))
        
if 2212 in all_data_sorted_by_code.keys():
    pdg_labels[2212] = r'Protons (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[2212]['weights']) * 1e6)
    pdg_colours[2212] = 'red'
    all_codes.append(2212)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[2212]['weights'])* 1e6))

if 1000250550 in all_data_sorted_by_code.keys():
    pdg_labels[1000250550] = r'Manganese 55 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000250550]['weights'])* 1e6)
    pdg_colours[1000250550] = 'red'
    all_codes.append(1000250550)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000250550]['weights'])* 1e6))
        
if 1000050110 in all_data_sorted_by_code.keys():
    pdg_labels[1000050110] = r'Boron 11 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000050110]['weights'])* 1e6)
    pdg_colours[1000050110] = 'red'
    all_codes.append(1000050110)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000050110]['weights'])* 1e6))
        
if 1000110230 in all_data_sorted_by_code.keys():
    pdg_labels[1000110230] = r'Sodium 23 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000110230]['weights'])* 1e6)
    pdg_colours[1000110230] = 'red'
    all_codes.append(1000110230)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000110230]['weights'])* 1e6))
        
if 1000200400 in all_data_sorted_by_code.keys():
    pdg_labels[1000200400] = r'Calcium 40 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000200400]['weights'])* 1e6)
    pdg_colours[1000200400] = 'red'
    all_codes.append(1000200400)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000200400]['weights'])* 1e6))
        
if 1000090190 in all_data_sorted_by_code.keys():
    pdg_labels[1000090190] = r'Flourine 19 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000090190]['weights'])* 1e6)
    pdg_colours[1000090190] = 'red'
    all_codes.append(1000090190)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000090190]['weights'])* 1e6))
        
if 1000240520 in all_data_sorted_by_code.keys():
    pdg_labels[1000240520] = r'Chromium 52 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000240520]['weights'])* 1e6)
    pdg_colours[1000240520] = 'red'
    all_codes.append(1000240520)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000240520]['weights'])* 1e6))
        
if 1000180400 in all_data_sorted_by_code.keys():
    pdg_labels[1000180400] = r'Argon 40 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000180400]['weights'])* 1e6)
    pdg_colours[1000180400] = 'red'
    all_codes.append(1000180400)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000180400]['weights'])* 1e6))
        
if 1000150310 in all_data_sorted_by_code.keys():
    pdg_labels[1000150310] = r'Phosphorus 31 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000150310]['weights'])* 1e6)
    pdg_colours[1000150310] = 'red'
    all_codes.append(1000150310)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000150310]['weights'])* 1e6))
        
if 1000230510 in all_data_sorted_by_code.keys():
    pdg_labels[1000230510] = r'Vanadium 51 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000230510]['weights'])* 1e6)
    pdg_colours[1000230510] = 'red'
    all_codes.append(1000230510)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000230510]['weights'])* 1e6))

if 1000220480 in all_data_sorted_by_code.keys():
    pdg_labels[1000220480] = r'Titanium 48 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000220480]['weights'])* 1e6)
    pdg_colours[1000220480] = 'red'
    all_codes.append(1000220480)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000220480]['weights'])* 1e6))

if 1000120240 in all_data_sorted_by_code.keys():
    pdg_labels[1000120240] = r'Magnesium 24 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000120240]['weights'])* 1e6)
    pdg_colours[1000120240] = 'red'
    all_codes.append(1000120240)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000120240]['weights'])* 1e6))

if 1000030070 in all_data_sorted_by_code.keys():
    pdg_labels[1000030070] = r'Lithium 7 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000030070]['weights'])* 1e6)
    pdg_colours[1000030070] = 'red'
    all_codes.append(1000030070)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000030070]['weights'])* 1e6))
        
if 1000190390 in all_data_sorted_by_code.keys():
    pdg_labels[1000190390] = r'Potassium 39 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000190390]['weights'])* 1e6)
    pdg_colours[1000190390] = 'red'
    all_codes.append(1000190390)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000190390]['weights'])* 1e6))
        
if 1000140280 in all_data_sorted_by_code.keys():
    pdg_labels[1000140280] = r'Silicon 28 (%.2f $\mu$Hz)'%sum(numpy.array(all_data_sorted_by_code[1000140280]['weights'])* 1e6)
    pdg_colours[1000140280] = 'red'
    all_codes.append(1000140280)
    singular_weights.append(sum(numpy.array(all_data_sorted_by_code[1000140280]['weights'])* 1e6))

ordered_codes = [x for y,x in sorted(zip(singular_weights,all_codes))]

ProtonNormUncs = open('Uncertainties/SimpleProtonNorm1SigmaValues.txt','r')
HeliumNormUncs = open('Uncertainties/SimpleHeliumNorm1SigmaValues.txt','r')
ProtonUncs = open('Uncertainties/SimpleProton1SigmaValues.txt','r')
HeliumUncs = open('Uncertainties/SimpleHelium1SigmaValues.txt','r')

ProtonNormUncLogEPoints = []
ProtonNormUncUPoints = []
ProtonUncLogEPoints = []
ProtonUncUPoints = []
HeliumNormUncLogEPoints = []
HeliumNormUncUPoints = []
HeliumUncLogEPoints = []
HeliumUncUPoints = []

for line in ProtonUncs:
    ProtonUncLogEPoints.append(numpy.log10(float(line.rstrip().split(' ')[0])))
    ProtonUncUPoints.append(float(line.rstrip().split(' ')[1]))

for line in HeliumUncs:
    HeliumUncLogEPoints.append(numpy.log10(float(line.rstrip().split(' ')[0])))
    HeliumUncUPoints.append(float(line.rstrip().split(' ')[1]))

for line in ProtonNormUncs:
    ProtonNormUncLogEPoints.append(numpy.log10(float(line.rstrip().split(' ')[0])))
    ProtonNormUncUPoints.append(float(line.rstrip().split(' ')[1]))

for line in HeliumNormUncs:
    HeliumNormUncLogEPoints.append(numpy.log10(float(line.rstrip().split(' ')[0])))
    HeliumNormUncUPoints.append(float(line.rstrip().split(' ')[1]))

ProtonUncF = interpolate.interp1d(ProtonUncLogEPoints,ProtonUncUPoints)
HeliumUncF = interpolate.interp1d(HeliumUncLogEPoints,HeliumUncUPoints)
ProtonNormUncF = interpolate.interp1d(ProtonNormUncLogEPoints,ProtonNormUncUPoints)
HeliumNormUncF = interpolate.interp1d(HeliumNormUncLogEPoints,HeliumNormUncUPoints)

results["weights_p1ns"] = []
results["weights_m1ns"] = []
results["weights_p1s"] = []
results["weights_m1s"] = []

for weight, code, energy in zip(results["weights"],results["code_of_primary"],results["energy_of_primary"]):

    if code == 2212:
        # Apply proton re-weighting
        try:
            nfactor = ProtonNormUncF(numpy.log10(energy))/100.0
            factor = ProtonUncF(numpy.log10(energy))/100.0
        except:
            nfactor = ProtonNormUncF(ProtonNormUncLogEPoints[-1])/100.0
            factor = ProtonUncF(ProtonUncLogEPoints[-1])/100.0
        results["weights_p1ns"].append(weight*(1+nfactor))
        results["weights_m1ns"].append(weight*(1-nfactor))
        results["weights_p1s"].append(weight*(1+factor))
        results["weights_m1s"].append(weight*(1-factor))
    else:
        # Apply Helium re-weighting
        try:
            nfactor = HeliumNormUncF(numpy.log10(energy))/100.0
            factor = HeliumUncF(numpy.log10(energy))/100.0
        except:
            nfactor = HeliumNormUncF(HeliumNormUncLogEPoints[-1])/100
            factor = HeliumUncF(HeliumUncLogEPoints[-1])/100
        results["weights_p1ns"].append(weight*(1+nfactor))
        results["weights_m1ns"].append(weight*(1-nfactor))
        results["weights_p1s"].append(weight*(1+factor))
        results["weights_m1s"].append(weight*(1-factor))

results["weights_p1s"] = numpy.array(results["weights_p1s"])
results["weights_m1s"] = numpy.array(results["weights_m1s"])
results["weights_p1ns"] = numpy.array(results["weights_p1ns"])
results["weights_m1ns"] = numpy.array(results["weights_m1ns"])

#######################################################
# Plot the energy at the generator surface just for 
# a check to make sure i'm okay
#######################################################

ead_bins,ead_p1sunc,ead_p1nsunc = Plots_1D(
    xvals = results["energy_at_depth"],
    xbins = numpy.linspace(100, 600, 26),
    logx = False,
    logy = False,
    xlabel = "Energy at MuonGun Surface (GeV)",
    ylim = [0,6],
    weights_standard = results["weights"]*1e3,
    weights_p1s = results["weights_p1s"]*1e3,
    weights_m1s = results["weights_m1s"]*1e3,
    weights_p1ns = results["weights_p1ns"]*1e3,
    weights_m1ns = results["weights_m1ns"]*1e3,
    xname = "energy_at_depth")

ead_bin_cens = []
for i in range(0,len(ead_bins)-1):
    ead_bin_cens.append((ead_bins[i]+ead_bins[i+1])/2)

Save_1D(outfile = 'Muon1SigmaUncertaintiesEnergyAtDepth.txt',
        xvals = ead_bin_cens,
        yvals = ead_p1sunc)

#######################################################
# Plot the depth distribution at the generator surface
#######################################################

d_bins,d_p1sunc,d_p1nsunc = Plots_1D(
    xvals = results["depth"],
    xbins = numpy.linspace(-700, 900, 33),
    logx = False,
    logy = False,
    xlabel = "Depth of Muon on MuonGun Surface (m)",
    ylim = [0,8],
    weights_standard = results["weights"]*1e3,
    weights_p1s = results["weights_p1s"]*1e3,
    weights_m1s = results["weights_m1s"]*1e3,
    weights_p1ns = results["weights_p1ns"]*1e3,
    weights_m1ns = results["weights_m1ns"]*1e3,
    xname = "depth")

d_bin_cens = []
for i in range(0,len(d_bins)-1):
    d_bin_cens.append((d_bins[i]+d_bins[i+1])/2)

Save_1D(outfile = 'Muon1SigmaUncertaintiesDepth.txt',
        xvals = d_bin_cens,
        yvals = d_p1sunc)

#######################################################
# Plot the muon zenith
#######################################################

cz_bins,cz_p1sunc,cz_p1nsunc = Plots_1D(
    xvals = results["coszens"],
    xbins = numpy.linspace(0.0, 1.0, 21),
    logx = False,
    logy = False,
    xlabel = r"Muon $\cos\theta_Z$",
    ylim = [0,7],
    weights_standard = results["weights"]*1e3,
    weights_p1s = results["weights_p1s"]*1e3,
    weights_m1s = results["weights_m1s"]*1e3,
    weights_p1ns = results["weights_p1ns"]*1e3,
    weights_m1ns = results["weights_m1ns"]*1e3,
    xname = "muon_zenith")

cz_bin_cens = []
for i in range(0,len(cz_bins)-1):
    cz_bin_cens.append((cz_bins[i]+cz_bins[i+1])/2)

Save_1D(outfile = 'Muon1SigmaUncertaintiesCosZenith.txt',
        xvals = cz_bin_cens,
        yvals = cz_p1sunc)

#######################################################
# Plot the energy of the primary CR that produced this event
#######################################################

eop_bins,eop_p1sunc,eop_p1nsunc = Plots_1D(
    xvals = results["energy_of_primary"],
    xbins = numpy.logspace(2, 7, 26),
    logx = True,
    logy = False,
    xlabel = "Energy of Primary (GeV)",
    ylim = [0,8],
    weights_standard = results["weights"]*1e3,
    weights_p1s = results["weights_p1s"]*1e3,
    weights_m1s = results["weights_m1s"]*1e3,
    weights_p1ns = results["weights_p1ns"]*1e3,
    weights_m1ns = results["weights_m1ns"]*1e3,
    xname = "energy_of_primary")

eop_bin_cens = []
for i in range(0,len(eop_bins)-1):
    eop_bin_cens.append((eop_bins[i]+eop_bins[i+1])/2)

Save_1D(outfile = 'Muon1SigmaUncertaintiesEnergyofPrimary.txt',
        xvals = eop_bin_cens,
        yvals = eop_p1sunc)
