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

# Now do it as a stack
pyplot.figure()
wanted_data = []
wanted_weights = []
wanted_labels = []
wanted_colours = []
last_hist = None
for code in ordered_codes:
    wanted_data.append(all_data_sorted_by_code[code]["energy_at_depth"])
    wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
    wanted_labels.append(pdg_labels[code])
    wanted_colours.append(pdg_colours[code])
    hist, bins = numpy.histogram(all_data_sorted_by_code[code]["energy_at_depth"],
                                 weights = numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3,
                                 bins = numpy.linspace(100, 600, 26))
    err, bins = numpy.histogram(all_data_sorted_by_code[code]["energy_at_depth"],
                                weights = (numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3)**2,
                                bins = numpy.linspace(100, 600, 26))
    if last_hist is not None:
        hist += last_hist
    pyplot.bar(left = bins[:-1],
               height = 2*numpy.sqrt(err),
               width = numpy.diff(bins),
               bottom = hist - numpy.sqrt(err),
               color = pdg_colours[code],
               alpha = 0.4,
               edgecolor = pdg_colours[code])
    last_hist = hist

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
pyplot.ylim([0,6])
pyplot.subplots_adjust(bottom=0.12,top=0.8)
pyplot.title(r'Corsika 10282+10309+11905+12268+12332 (L4)',size='x-large',x=0.5,y=1.20)
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332energy_at_depth_stack.pdf")


pyplot.yscale("log")
pyplot.ylim([1e-3,1e1])
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332energy_at_depth_stack_log.pdf")

#######################################################
# Plot the depth distribution at the generator surface
#######################################################

# Now do it as a stack
pyplot.figure()
wanted_data = []
wanted_weights = []
wanted_labels = []
wanted_colours = []
last_hist = None
for code in ordered_codes:
    wanted_data.append(all_data_sorted_by_code[code]["depth"])
    wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
    wanted_labels.append(pdg_labels[code])
    wanted_colours.append(pdg_colours[code])
    hist, bins = numpy.histogram(all_data_sorted_by_code[code]["depth"],
                                 weights = numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3,
                                 bins = numpy.linspace(-700, 900, 33))
    err, bins = numpy.histogram(all_data_sorted_by_code[code]["depth"],
                                weights = (numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3)**2,
                                bins = numpy.linspace(-700, 900, 33))
    if last_hist is not None:
        hist += last_hist
    pyplot.bar(left = bins[:-1],
               height = 2*numpy.sqrt(err),
               width = numpy.diff(bins),
               bottom = hist - numpy.sqrt(err),
               color = pdg_colours[code],
               alpha = 0.4,
               edgecolor = pdg_colours[code])
    last_hist = hist

pyplot.hist(wanted_data,
            weights = numpy.array(wanted_weights),
            bins = numpy.linspace(-700, 900, 33),
            label = wanted_labels,
            color = wanted_colours,
            stacked = True,
            histtype='step')
                         
pyplot.grid()
pyplot.xlabel("Depth of Muon on MuonGun Surface (m)")
pyplot.ylabel("Rate (mHz)")
pyplot.ylim([0,6])
pyplot.subplots_adjust(bottom=0.12,top=0.8)
pyplot.title(r'Corsika 10282+10309+11905+12268+12332 (L4)',size='x-large',x=0.5,y=1.20)
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332depth_stack.pdf")

pyplot.yscale("log")
pyplot.ylim([1e-3,1e1])
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332depth_stack_log.pdf")

#######################################################
# Plot the muon zenith
#######################################################

# Now do it as a stack
pyplot.figure()
wanted_data = []
wanted_weights = []
wanted_labels = []
wanted_colours = []
last_hist = None
for code in ordered_codes:
    wanted_data.append(all_data_sorted_by_code[code]["coszens"])
    wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
    wanted_labels.append(pdg_labels[code])
    wanted_colours.append(pdg_colours[code])
    hist, bins = numpy.histogram(all_data_sorted_by_code[code]["coszens"],
                                 weights = numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3,
                                 bins = numpy.linspace(0.0, 1.0, 21))
    err, bins = numpy.histogram(all_data_sorted_by_code[code]["coszens"],
                                weights = (numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3)**2,
                                bins = numpy.linspace(0.0, 1.0, 21))
    if last_hist is not None:
        hist += last_hist
    pyplot.bar(left = bins[:-1],
               height = 2*numpy.sqrt(err),
               width = numpy.diff(bins),
               bottom = hist - numpy.sqrt(err),
               color = pdg_colours[code],
               alpha = 0.4,
               edgecolor = pdg_colours[code])
    last_hist = hist

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
pyplot.ylim([0,7])
pyplot.subplots_adjust(bottom=0.12,top=0.8)
pyplot.title(r'Corsika 10282+10309+11905+12268+12332 (L4)',size='x-large',x=0.5,y=1.20)
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332muon_zenith_stack.pdf")

pyplot.yscale("log")
pyplot.ylim([1e-3,1e1])
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332muon_zenith_stack_log.pdf")

#######################################################
# Plot the energy of the primary CR that produced this event
#######################################################

# Now do it as a stack
pyplot.figure()
wanted_data = []
wanted_weights = []
wanted_labels = []
wanted_colours = []
last_hist = None
for code in ordered_codes:
    wanted_data.append(all_data_sorted_by_code[code]["energy_of_primary"])
    wanted_weights.append(numpy.array(all_data_sorted_by_code[code]["weights"])*1e3)
    wanted_labels.append(pdg_labels[code])
    wanted_colours.append(pdg_colours[code])
    hist, bins = numpy.histogram(all_data_sorted_by_code[code]["energy_of_primary"],
                                 weights = numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3,
                                 bins = numpy.logspace(2, 7, 26))
    err, bins = numpy.histogram(all_data_sorted_by_code[code]["energy_of_primary"],
                                weights = (numpy.array(all_data_sorted_by_code[code]["weights"]) * 1e3)**2,
                                bins = numpy.logspace(2, 7, 26))
    if last_hist is not None:
        hist += last_hist
    pyplot.bar(left = bins[:-1],
               height = 2*numpy.sqrt(err),
               width = numpy.diff(bins),
               bottom = hist - numpy.sqrt(err),
               color = pdg_colours[code],
               alpha = 0.4,
               edgecolor = pdg_colours[code])
    last_hist = hist

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
pyplot.ylim([0,8])
pyplot.subplots_adjust(bottom=0.12,top=0.8)
pyplot.title(r'Corsika 10282+10309+11905+12268+12332 (L4)',size='x-large',x=0.5,y=1.20)
pyplot.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=3, mode="expand", borderaxespad=0.,fontsize='x-small')
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332energy_of_primary_stack.pdf")

pyplot.yscale("log")
pyplot.ylim([1e-3,1e1])
pyplot.savefig("Plots/10282+10309+11905+12268+12332/10282+10309+11905+12268+12332energy_of_primary_stack_log.pdf")
