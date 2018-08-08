from __future__ import print_function

import os, numpy

from scipy import interpolate

def Muon_Primary_Spline(kind='linear'):
    ''' 
    Constructs a muon primary uncertainty spline.

    It is expected that the .txt file is in a subdirectory called Uncertainties 
    wherever this python script is located with the name 
    Muon1SigmaUncertaintiesCosZenith.txt. This is how it was provided on SVN.
    '''

    dirname = os.path.dirname(os.path.realpath(__file__))
    MuonUncs = open(os.path.join(dirname,'Uncertainties/Muon1SigmaUncertaintiesCosZenith.txt'))

    MuonZenithPoints = []
    MuonUncPoints = []

    for line in MuonUncs:
        MuonZenithPoints.append(float(line.rstrip().split(' ')[0]))
        MuonUncPoints.append(float(line.rstrip().split(' ')[1]))

    MuonZenithPoints = numpy.array(MuonZenithPoints)
    MuonUncPoints = numpy.array(MuonUncPoints)

    # Need to deal with zeroes. This may be a problem if this work ever
    # changes but is fine for the zenith spline here.
    while 0.0 in MuonUncPoints:
        ZeroIndices = numpy.where(MuonUncPoints == 0)[0]
        for ZeroIndex in ZeroIndices:
            MuonUncPoints[ZeroIndex] = MuonUncPoints[ZeroIndex+1]

    # Add a dummy point for coszenith = 0
    MuonZenithPoints = numpy.insert(MuonZenithPoints,0,0.0)
    MuonUncPoints = numpy.insert(MuonUncPoints,0,MuonUncPoints[0])
    # Add a dummy poiny for coszenith = 1
    MuonZenithPoints = numpy.insert(MuonZenithPoints,-1,1.0)
    MuonUncPoints = numpy.insert(MuonUncPoints,-1,MuonUncPoints[-1])

    indicies = numpy.argsort(MuonZenithPoints)
    MuonZenithPoints = MuonZenithPoints[indicies]
    MuonUncPoints = MuonUncPoints[indicies]

    #print ('MuonZenithPoints: {0}'.format(MuonZenithPoints))
    MuonUncF = interpolate.interp1d(MuonZenithPoints, MuonUncPoints, kind=kind)

    return MuonUncF

def Muon_Primary_Unc(cosZenith, sigma=0, MuonUncF=None, kind='linear'):
    ''' 
    Takes the truth cosZenith of the muon and returns a percentage uncertainty 
    corresponding to that due to the uncertainty on the primary cosmic rays. 
    The construction of the spline is a different function so you only have to
    call it once. If you don't provide one then it will be made. It is expected
    that the .txt file is in a subdirectory called Uncertainties wherever this 
    python script is located with the name Muon1SigmaUncertaintiesCosZenith.txt.
    This is how it was provided on SVN.

    For details on the calculation please see 

    https://wiki.icecube.wisc.edu/index.php/DeepCore_Muon_Background_Systematics
    
    i.e. the section on Uncertainties from Primary Cosmic Rays
    '''

    if MuonUncF is None:
        MuonUncF = Muon_Primary_Spline(kind=kind)

    print ("Modifying CR by {0}".format(sigma))
    weight_mod = 1 + sigma*MuonUncF(cosZenith)

    return weight_mod

if __name__ == '__main__':
    '''
    This is a debug for the splines.
    Make a plot see if it looks sensible.
    '''
    from matplotlib import pyplot
    pyplot.rcParams['text.usetex'] = True

    dirname = os.path.dirname(os.path.realpath(__file__))
    MuonUncs = open(os.path.join(dirname,'Uncertainties/Muon1SigmaUncertaintiesCosZenith.txt'))

    MuonZenithPoints = []
    MuonUncPoints = []

    for line in MuonUncs:
        MuonZenithPoints.append(float(line.rstrip().split(' ')[0]))
        MuonUncPoints.append(float(line.rstrip().split(' ')[1]))

    MuonZenithPoints = numpy.array(MuonZenithPoints)
    MuonUncPoints = numpy.array(MuonUncPoints)

    MuonUncFL = Muon_Primary_Spline()
    MuonUncFC = Muon_Primary_Spline(kind='cubic')

    MuonZenithPointsFine = numpy.linspace(0,1.,101)

    pyplot.plot(MuonZenithPoints,MuonUncPoints*100.0,'o',
                MuonZenithPointsFine,MuonUncFL(MuonZenithPointsFine)*100.0,'-',
                MuonZenithPointsFine,MuonUncFC(MuonZenithPointsFine)*100.0,'--')
    pyplot.legend(['data', 'linear', 'cubic'], loc='best')
    pyplot.xlabel(r'Muon $\cos\theta_Z$')
    pyplot.ylabel('Percentage Uncertainty')
    pyplot.savefig("splinesvisualised.png")
