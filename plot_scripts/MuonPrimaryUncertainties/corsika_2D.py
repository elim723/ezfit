#!/usr/bin/env python
import os, sys, pickle, numpy, matplotlib

from scipy import interpolate
numpy.set_printoptions(threshold=numpy.nan)
matplotlib.use('Agg')
from matplotlib import pyplot
import matplotlib.patheffects as PathEffects
pyplot.rcParams['text.usetex'] = True
from matplotlib import gridspec

def Plots_2D(xvals, yvals, xbins, ybins, logx, logy, xlabel, ylabel, vmax,
             weights_standard, weights_p1s, weights_p1ns, xname, yname):
    
    hist, xedges, yedges = numpy.histogram2d(xvals, yvals,
                                             weights = weights_standard,
                                             bins = [xbins, ybins])

    # Get the errorbars on this too
    err, xedges, yedges = numpy.histogram2d(xvals, yvals,
                                            weights = (weights_standard)**2,
                                            bins = [xbins, ybins])
    
    p1s, xedges, yedges = numpy.histogram2d(xvals, yvals,
                                            weights = weights_p1s,
                                            bins = [xbins, ybins])

    

    p1ns, xedges, yedges = numpy.histogram2d(xvals, yvals,
                                             weights = weights_p1ns,
                                             bins = [xbins, ybins])
    
    err_diff = numpy.sqrt(err)/hist*100.0
    err_diff[numpy.isnan(err_diff)] = 0.0
    p1s_diff = (p1s-hist)/hist*100.0
    p1s_diff[numpy.isnan(p1s_diff)] = 0.0
    p1ns_diff = (p1ns-hist)/hist*100.0
    p1ns_diff[numpy.isnan(p1ns_diff)] = 0.0

    gs = gridspec.GridSpec(1,3)
    
    pyplot.figure(figsize=(30,10))
    pyplot.subplot(gs[0])

    # Plot them together
    pyplot.pcolormesh(xedges, yedges, hist.T, cmap='Oranges', vmin=0)
    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")  
    pyplot.colorbar().set_label(label=r'Rate ($\mu$Hz)',size=36)

    for i, x in enumerate(xedges[:-1]):
        for j, y in enumerate(yedges[:-1]):
            if not numpy.isnan(hist[i,j]):
                pyplot.text(x+(xedges[i+1]-x)/2, y+(yedges[j+1]-y)/2, 
                            r'%.1f' % hist[i,j],
                            verticalalignment='center',
                            horizontalalignment='center',
                            color='w',
                            path_effects=[PathEffects.withStroke(linewidth=3,
                                                                 foreground='k')])
        
    pyplot.ylabel(ylabel, fontsize=36)

    pyplot.subplot(gs[1])

    pyplot.pcolormesh(xedges, yedges, err_diff.T, cmap='Greens', vmin=0, vmax=vmax)
    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")
    pyplot.colorbar().set_label(label=r'MC Statistical Uncertainty (\%)',size=36)

    for i, x in enumerate(xedges[:-1]):
        for j, y in enumerate(yedges[:-1]):
            if not err_diff[i,j] == 0.0:
                pyplot.text(x+(xedges[i+1]-x)/2, y+(yedges[j+1]-y)/2, 
                            r'%.1f' % err_diff[i,j],
                            verticalalignment='center',
                            horizontalalignment='center',
                            color='w',
                            path_effects=[PathEffects.withStroke(linewidth=3,
                                                                 foreground='k')])
        
    pyplot.xlabel(xlabel, fontsize=36)

    pyplot.subplot(gs[2])

    pyplot.pcolormesh(xedges, yedges, p1s_diff.T, cmap='Blues', vmin=0, vmax=vmax)
    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")
    pyplot.colorbar().set_label(label='Primary Cosmic Ray Uncertainty (\%)\n Unmodified Data',size=36)

    for i, x in enumerate(xedges[:-1]):
        for j, y in enumerate(yedges[:-1]):
            if not p1s_diff[i,j] == 0.0:
                pyplot.text(x+(xedges[i+1]-x)/2, y+(yedges[j+1]-y)/2, 
                            r'%.1f' % p1s_diff[i,j],
                            verticalalignment='center',
                            horizontalalignment='center',
                            color='w',
                            path_effects=[PathEffects.withStroke(linewidth=3,
                                                                 foreground='k')])


    pyplot.suptitle("Corsika 10282+10309+11905+12268+12332 (L4)", fontsize=64)
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=0.85)
    pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332%s_vs_%s_unc_err.pdf"%(xname,yname))

    pyplot.cla()
    pyplot.pcolormesh(xedges, yedges, p1ns_diff.T, cmap='Blues', vmin=0, vmax=vmax)
    if logx:
        pyplot.xscale("log")
    if logy:
        pyplot.yscale("log")
    pyplot.colorbar().set_label(label='Primary Cosmic Ray Uncertainty (\%) \n Normalised Data',size=36)

    for i, x in enumerate(xedges[:-1]):
        for j, y in enumerate(yedges[:-1]):
            if not p1ns_diff[i,j] == 0.0:
                pyplot.text(x+(xedges[i+1]-x)/2, y+(yedges[j+1]-y)/2, 
                            r'%.1f' % p1ns_diff[i,j],
                            verticalalignment='center',
                            horizontalalignment='center',
                            color='w',
                            path_effects=[PathEffects.withStroke(linewidth=3,
                                                                 foreground='k')])

    pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332%s_vs_%s_unc_err_norm.pdf"%(xname,yname))
    pyplot.close()

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
# 2D Muon Energy and Primary Energy
#######################################################

Plots_2D(xvals = results["energy_at_depth"],
         yvals = results["energy_of_primary"],
         xbins = numpy.linspace(100, 600, 16),
         ybins = numpy.logspace(3, 6, 21),
         logx = False,
         logy = True,
         xlabel = "Energy at MuonGun Surface (GeV)",
         ylabel = "Energy of Primary (GeV)",
         vmax = 21.0,
         weights_standard = results["weights"]*1e6,
         weights_p1s = results["weights_p1s"]*1e6,
         weights_p1ns = results["weights_p1ns"]*1e6,
         xname = "energy_at_depth",
         yname = "primary_energy")

#######################################################
# 2D Muon Zenith and Primary Energy
#######################################################

Plots_2D(xvals = results["coszens"],
         yvals = results["energy_of_primary"],
         xbins = numpy.linspace(0, 1, 11),
         ybins = numpy.logspace(3, 6, 21),
         logx = False,
         logy = True,
         xlabel = r"Muon $\cos\theta_Z$",
         ylabel = "Energy of Primary (GeV)",
         vmax = 21,
         weights_standard = results["weights"]*1e6,
         weights_p1s = results["weights_p1s"]*1e6,
         weights_p1ns = results["weights_p1ns"]*1e6,
         xname = "muon_coszen",
         yname = "primary_energy")

#######################################################
# 2D Muon Energy and Zenith
#######################################################

Plots_2D(xvals = results["energy_at_depth"],
         yvals = results["coszens"],
         xbins = numpy.linspace(100, 600, 16),
         ybins = numpy.linspace(0, 1, 11),
         logx = False,
         logy = False,
         xlabel = "Energy at MuonGun Surface (GeV)",
         ylabel = r"Muon $\cos\theta_Z$",
         vmax = 10,
         weights_standard = results["weights"]*1e6,
         weights_p1s = results["weights_p1s"]*1e6,
         weights_p1ns = results["weights_p1ns"]*1e6,
         xname = "energy_at_depth",
         yname = "muon_coszen")

#######################################################
# 2D Muon Energy and depth
#######################################################

Plots_2D(xvals = results["energy_at_depth"],
         yvals = results["depth"],
         xbins = numpy.linspace(100, 600, 16),
         ybins = numpy.linspace(-700, 900, 16),
         logx = False,
         logy = False,
         xlabel = "Energy at MuonGun Surface (GeV)",
         ylabel = "Depth of Muon on MuonGun Surface (m)",
         vmax = 10,
         weights_standard = results["weights"]*1e6,
         weights_p1s = results["weights_p1s"]*1e6,
         weights_p1ns = results["weights_p1ns"]*1e6,
         xname = "energy_at_depth",
         yname = "muon_depth")

#######################################################
# 2D Muon zenith and depth
#######################################################

Plots_2D(xvals = results["depth"],
         yvals = results["coszens"],
         xbins = numpy.linspace(-700, 900, 16),
         ybins = numpy.linspace(0, 1, 11),
         logx = False,
         logy = False,
         xlabel = "Depth of Muon on MuonGun Surface (m)",
         ylabel = r"Muon $\cos\theta_Z$",
         vmax = 10,
         weights_standard = results["weights"]*1e6,
         weights_p1s = results["weights_p1s"]*1e6,
         weights_p1ns = results["weights_p1ns"]*1e6,
         xname = "muon_depth",
         yname = "muon_coszen")
