#!/usr/bin/env python
import os, sys, pickle, numpy, matplotlib

from scipy import interpolate
numpy.set_printoptions(threshold=numpy.nan)
matplotlib.use('Agg')
from matplotlib import pyplot
pyplot.rcParams['text.usetex'] = True
from matplotlib import gridspec

#######################################################
# Load the results dictionary with this structure:
#
# results = {"weights":weights,
#            "energy_of_primary": energy_of_primary,
#            "energy_at_depth": energy_at_depth,
#            "coszens": coszens,
#            }
#######################################################

infiles = ["9036corsika.pckl",
           "9255corsika.pckl",
           "10282corsika.pckl",
           "10309corsika.pckl",
           "10369corsika.pckl",
           "11905corsika.pckl",
           "12268corsika.pckl",
           "12332corsika.pckl"]

for filename in infiles:

    if "9036" in filename:
        name = "9036"
    elif "9255" in filename:
        name = "9255"
    elif "10282" in filename:
        name = "10282"
    elif "10309" in filename:
        name = "10309"
    elif "10369" in filename:
        name = "10369"
    elif "11905" in filename:
        name = "11905"
    elif "12268" in filename:
        name = "12268"
    elif "12332" in filename:
        name = "12332"

    print "Working on %s"%name

    results = pickle.load(open("Data/Level5/%s"%filename))

    all_data_sorted_by_code = {}
    for code in set(results['code_of_primary']):
        all_data_sorted_by_code[code] = {}
        for val_key in results.keys():
            all_data_sorted_by_code[code][val_key] = []

    for i in range(0,len(results['energy_at_depth'])-1):
    
        code = results['code_of_primary'][i]
        for val_key in results.keys():
            all_data_sorted_by_code[code][val_key].append(results[val_key][i])

    print all_data_sorted_by_code.keys()

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
    pyplot.figure()
    hist, bins = numpy.histogram(results["energy_at_depth"],
                                 weights = results["weights"] * 1e3,
                                 bins = numpy.linspace(100, 600, 26),
                             )

    # Get the errorbars on this too
    err, bins = numpy.histogram(results["energy_at_depth"],
                                weights = (results["weights"]*1e3)**2,
                                bins = numpy.linspace(100, 600, 26))

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

    pyplot.grid()
    pyplot.xlabel("Energy at MuonGun Surface (GeV)")
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5)"%name)
    pyplot.savefig("Plots/%s_energy_at_depth.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1ns, bins = numpy.histogram(results["energy_at_depth"],
                                 weights = (results["weights_p1ns"]*1e3),
                                 bins = numpy.linspace(100, 600, 26))
    m1ns, bins = numpy.histogram(results["energy_at_depth"],
                                 weights = (results["weights_m1ns"]*1e3),
                                 bins = numpy.linspace(100, 600, 26))

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

    pyplot.grid()
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5) Including Primary Uncertainties (Normalised Data)"%name)
    pyplot.legend()

    pyplot.subplot(gs[1])

    pyplot.bar(left = bins[:-1], 
               height = (p1ns-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = (m1ns-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    pyplot.grid()
    pyplot.xlabel("Energy at MuonGun Surface (GeV)")
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/%s_energy_at_depth_prim_unc_err_norm.pdf"%name)
    pyplot.close()
    
    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1s, bins = numpy.histogram(results["energy_at_depth"],
                                weights = (results["weights_p1s"]*1e3),
                                bins = numpy.linspace(100, 600, 26))
    m1s, bins = numpy.histogram(results["energy_at_depth"],
                                weights = (results["weights_m1s"]*1e3),
                                bins = numpy.linspace(100, 600, 26))

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
    
    pyplot.grid()
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5) Including Primary Uncertainties (Unmodified Data)"%name)
    pyplot.legend()

    pyplot.subplot(gs[1])

    pyplot.bar(left = bins[:-1], 
               height = (p1s-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = (m1s-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    pyplot.grid()
    pyplot.xlabel("Energy at MuonGun Surface (GeV)")
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/%s_energy_at_depth_prim_unc_err.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do it as a stack
    wanted_data = []
    wanted_weights = []
    wanted_labels = []
    wanted_colours = []
    for code in ordered_codes:
        wanted_data.append(all_data_sorted_by_code[code]["energy_at_depth"])
        wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
        wanted_labels.append(pdg_labels[code])
        wanted_colours.append(pdg_colours[code])

    pyplot.hist(wanted_data,
                weights = numpy.array(wanted_weights),
                bins = numpy.linspace(100, 600, 26),
                label = wanted_labels,
                color = wanted_colours,
                stacked = True,
                histtype='step')
                         
    pyplot.grid()
    pyplot.xlabel("Energy at MuonGun Surface (GeV)")
    pyplot.ylabel("Rate (mHz)")
    pyplot.subplots_adjust(bottom=0.12,top=0.8)
    pyplot.title(r'Corsika %s (L5)'%name,size='x-large',x=0.5,y=1.20)
    pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
    pyplot.savefig("Plots/%s_energy_at_depth_stack.pdf"%name)

    #######################################################
    # Plot the muon zenith
    #######################################################
    pyplot.figure()
    hist, bins = numpy.histogram(results["coszens"],
                                 weights = results["weights"] * 1e3,
                                 bins = numpy.linspace(0.0, 1.0, 21))

    # Get the errorbars on this too
    err, bins = numpy.histogram(results["coszens"],
                                weights = (results["weights"]*1e3)**2,
                                bins = numpy.linspace(0.0, 1.0, 21))

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

    pyplot.grid()
    pyplot.xlabel(r"Muon $\cos\theta_Z$")
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5)"%name)
    pyplot.savefig("Plots/%s_muon_zenith.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1ns, bins = numpy.histogram(results["coszens"],
                                 weights = (results["weights_p1ns"]*1e3),
                                 bins = numpy.linspace(0.0, 1.0, 21))
    m1ns, bins = numpy.histogram(results["coszens"],
                                 weights = (results["weights_m1ns"]*1e3),
                                 bins = numpy.linspace(0.0, 1.0, 21))

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

    pyplot.grid()
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5) Including Primary Uncertainties (Normalised Data)"%name)
    pyplot.legend(loc='upper left')

    pyplot.subplot(gs[1])
    
    pyplot.bar(left = bins[:-1], 
               height = (p1ns-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = (m1ns-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    pyplot.grid()
    pyplot.xlabel(r"Muon $\cos\theta_Z$")
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/%s_muon_zenith_prim_unc_err_norm.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1s, bins = numpy.histogram(results["coszens"],
                                weights = (results["weights_p1s"]*1e3),
                                bins = numpy.linspace(0.0, 1.0, 21))
    m1s, bins = numpy.histogram(results["coszens"],
                                weights = (results["weights_m1s"]*1e3),
                                bins = numpy.linspace(0.0, 1.0, 21))

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

    pyplot.grid()
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5) Including Primary Uncertainties (Unmodified Data)"%name)
    pyplot.legend(loc='upper left')

    pyplot.subplot(gs[1])

    pyplot.bar(left = bins[:-1], 
               height = (p1s-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = (m1s-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    pyplot.grid()
    pyplot.xlabel(r"Muon $\cos\theta_Z$")
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/%s_muon_zenith_prim_unc_err.pdf"%name)
    pyplot.close()
    
    pyplot.figure()

    # Now do it as a stack
    wanted_data = []
    wanted_weights = []
    wanted_labels = []
    wanted_colours = []
    for code in ordered_codes:
        wanted_data.append(all_data_sorted_by_code[code]["coszens"])
        wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
        wanted_labels.append(pdg_labels[code])
        wanted_colours.append(pdg_colours[code])

    pyplot.hist(wanted_data,
                weights = numpy.array(wanted_weights),
                bins = numpy.linspace(0.0, 1.0, 21),
                label = wanted_labels,
                color = wanted_colours,
                stacked = True,
                histtype='step')
                         
    pyplot.grid()
    pyplot.xlabel(r"Muon $\cos\theta_Z$")
    pyplot.ylabel("Rate (mHz)")
    pyplot.subplots_adjust(bottom=0.12,top=0.8)
    pyplot.title(r'Corsika %s (L5)'%name,size='x-large',x=0.5,y=1.20)
    pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
    pyplot.savefig("Plots/%s_muon_zenith_stack.pdf"%name)

    #######################################################
    # Plot the energy of the primary CR that produced this event
    #######################################################
    pyplot.figure()
    hist, bins = numpy.histogram(results["energy_of_primary"],
                                 weights = results["weights"] * 1e3,
                                 bins = numpy.logspace(2, 7, 26))

    # Get the errorbars on this too
    err, bins = numpy.histogram(results["energy_of_primary"],
                                weights = (results["weights"] * 1e3)**2,
                                bins = numpy.logspace(2, 7, 26))

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

    pyplot.grid()
    pyplot.xlabel("Energy of Primary (GeV)")
    pyplot.xscale("log")
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5)"%name)
    pyplot.savefig("Plots/%s_energy_of_primary.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1ns, bins = numpy.histogram(results["energy_of_primary"],
                                 weights = (results["weights_p1ns"]*1e3),
                                 bins = numpy.logspace(2, 7, 26))
    m1ns, bins = numpy.histogram(results["energy_of_primary"],
                                 weights = (results["weights_m1ns"]*1e3),
                                 bins = numpy.logspace(2, 7, 26))

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

    pyplot.grid()
    pyplot.xscale("log")
    pyplot.ylim(0,1.0)
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5) Including Primary Uncertainties (Normalised Data)"%name)
    pyplot.legend()

    pyplot.subplot(gs[1])
    
    pyplot.bar(left = bins[:-1], 
               height = (p1ns-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = (m1ns-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    pyplot.grid()
    pyplot.xlabel("Energy of Primary (GeV)")
    pyplot.xscale("log")
    pyplot.ylabel("Difference (\%)")

    pyplot.savefig("Plots/%s_energy_of_primary_prim_unc_err_norm.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do +/- 1 sigma uncertainty from primaries
    p1s, bins = numpy.histogram(results["energy_of_primary"],
                                weights = (results["weights_p1s"]*1e3),
                                bins = numpy.logspace(2, 7, 26))
    m1s, bins = numpy.histogram(results["energy_of_primary"],
                                weights = (results["weights_m1s"]*1e3),
                                bins = numpy.logspace(2, 7, 26))

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

    pyplot.grid()
    pyplot.xscale("log")
    pyplot.ylim(0,1.0)
    pyplot.ylabel("Rate (mHz)")
    pyplot.title("Corsika %s (L5) Including Primary Uncertainties (Unmodified Data)"%name)
    pyplot.legend()

    pyplot.subplot(gs[1])
    
    pyplot.bar(left = bins[:-1], 
               height = (p1s-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='green',
               alpha=0.4,
               label = 'Plus 1 Sigma')
    pyplot.bar(left = bins[:-1], 
               height = (m1s-hist)/hist, 
               width = numpy.diff(bins),
               bottom = 0,
               color='red',
               alpha=0.4,
               label = 'Minus 1 Sigma')

    pyplot.grid()
    pyplot.xlabel("Energy of Primary (GeV)")
    pyplot.xscale("log")
    pyplot.ylabel("Difference (\%)")
    pyplot.savefig("Plots/%s_energy_of_primary_prim_unc_err.pdf"%name)
    pyplot.close()

    pyplot.figure()

    # Now do it as a stack
    wanted_data = []
    wanted_weights = []
    wanted_labels = []
    wanted_colours = []
    for code in ordered_codes:
        wanted_data.append(all_data_sorted_by_code[code]["energy_of_primary"])
        wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
        wanted_labels.append(pdg_labels[code])
        wanted_colours.append(pdg_colours[code])

    pyplot.hist(wanted_data,
                weights = numpy.array(wanted_weights),
                bins = numpy.logspace(2, 7, 26),
                label = wanted_labels,
                color = wanted_colours,
                stacked = True,
                histtype='step')

    pyplot.grid()
    pyplot.xlabel("Energy of Primary (GeV)")
    pyplot.xscale("log")
    pyplot.ylabel("Rate (mHz)")
    pyplot.subplots_adjust(bottom=0.12,top=0.8)
    pyplot.title(r'Corsika %s (L5)'%name,size='x-large',x=0.5,y=1.20)
    pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                  ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
    pyplot.savefig("Plots/%s_energy_of_primary_stack.pdf"%name)
    pyplot.close()


    #######################################################
    # Now plot the 2D histogram with text giving rates and errors
    #######################################################
    fig = pyplot.figure()
    hist, xedges, yedges = numpy.histogram2d(results["coszens"],
                                             results["energy_of_primary"],
                                             weights = (results["weights"]*1e6),
                                             bins = [numpy.linspace(0, 1, 11), 
                                                     numpy.logspace(3, 6, 21)])

    # Get the errorbars on this too
    err, xedges, yedges = numpy.histogram2d(results["coszens"],
                                            results["energy_of_primary"],
                                            weights = (results["weights"]*1e6)**2,
                                            bins = [numpy.linspace(0, 1, 11), 
                                                    numpy.logspace(3, 6, 21)])

    # Plot them together
    pyplot.pcolormesh(xedges[:-1], yedges[:-1], hist.T, cmap='YlOrBr', vmin=0)
    pyplot.colorbar(label=r'Rate ($\mu$Hz)')

    for i, x in enumerate(xedges[:-2]):
        for j, y in enumerate(yedges[:-1]):
            value = hist[i,j]
            if value > 0.5 * numpy.max(hist): color = 'w'
            else: color='k'

            pyplot.text( x+(xedges[i+1]-x)/2, y+(yedges[j+1]-y)/2, 
                         r'%3.1f$\pm$%3.1f' % (hist[i,j], numpy.sqrt(err[i,j])),
                         verticalalignment='center',
                         horizontalalignment='center',
                         color=color,
                         fontsize=4)#transform=fig.transFigure)
        
    pyplot.ylabel("Energy of Primary (GeV)")
    pyplot.yscale("log")
    pyplot.xlabel("Cos(Muon Zenith)")
    pyplot.title("Corsika %s (L5)"%name)
    pyplot.savefig("Plots/%s_primary_energy_vs_coszen.pdf"%name)
    pyplot.close()
