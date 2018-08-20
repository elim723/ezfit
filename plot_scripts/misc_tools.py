#!/usr/bin/env python
from __future__ import print_function

###########################################################################
#### It is a tool box for any misc functions, such as 
####    -- define Map class for python dictionary <-> attributes
####    -- print event rates 
####    -- read nuisance parameter textfile
####    -- load each member from pFiles
####    -- name output histo dictionary file
####    -- calculate qTotal / nChannel / nString
###########################################################################

import numpy as np
import os, sys, cPickle

from I3Tray import *
from icecube import dataio, dataclasses, icetray
from icecube.dataclasses import I3RecoPulse

seconds_per_year = 3600.*24.*365.24

neutrino_holeices = ['30', '33', '36', '45', '55', '67', '80', '100'] ## ignoring 0cm
neutrino_domeffs  = ['0.85', '0.90', '0.95', '1.05', '1.10', '1.15']
neutrino_dima_holeices = ['15', '20', '30', '35']
#neutrino_dima_domeffs = ['0.88', '0.94', '0.97', '1.03', '1.06', '1.12']
neutrino_dima_domeffs = ['0.88', '0.94', '0.97', '1.03', '1.12']
neutrino_dima_forwards = ['n5', 'n3', 'n1', 'p1', 'p2']

msu_neutrino_holeices = ['30', '36', '45', '55', '80', '100'] ## ignoring 0cm
msu_neutrino_domeffs  = neutrino_domeffs
msu_neutrino_dima_holeices = neutrino_dima_holeices
msu_neutrino_dima_domeffs = ['0.88', '0.94', '0.97', '1.03', '1.06', '1.12']
msu_neutrino_dima_forwards = neutrino_dima_forwards

muongun_domeffs   = ['0.80', '0.90', '1.10', '1.20']
muongun_holeices  = ['30', '100'] ## only for muongun15
muongun_dima_holeices = ['15', '30']
muongun_dima_domeffs = ['0.70', '0.80']
muongun_dima_forwards = ['n2', 'n4']

###########################################################
### lists for new pegleg (still reco-ing)
neutrino_qIndPegleg_holeices = ['15', '20', '25', '30', '35']
neutrino_qIndPegleg_domeffs = ['0.88', '0.94', '0.97', '1', '1.03', '1.06', '1.12']
neutrino_qIndPegleg_forwards = ['n5', 'n3', 'n1', '0', 'p1', 'p2']
muongun_qIndPegleg_holeices = ['15', '25', '30']
muongun_qIndPegleg_domeffs = ['0.69', '0.79', '0.99']
muongun_qIndPegleg_forwards = ['n2', 'n4', '0']
muongun_qIndPegleg_absorption = ['0.8', '1', '1.1'] ### add 0.929/1.142 if ishyperplane and both absorption and scattering are turned on
muongun_qIndPegleg_scattering = ['0.8', '1', '1.1']
neutrino_qIndPegleg_coins = ['0', '1.0']
neutrino_qIndPegleg_absorption = ['1', '1.1'] ### add 0.929 if ishyperplane and both absorption and scattering are turned on
neutrino_qIndPegleg_scattering = ['1', '1.1']
###########################################################

def grab_discrete_sets (dtype, systematic,
                        qIndPegleg = False, dima_sets = False, ishyperplane=False, hasBothBulkice=False):
    if 'muongun' in dtype:
        if dima_sets:
            if systematic=='domeff':
                return muongun_dima_domeffs
            elif systematic=='holeice':
                return muongun_dima_holeices
            elif systematic=='forward':
                return muongun_dima_forwards
        elif qIndPegleg:
            if systematic=='domeff':
                return muongun_qIndPegleg_domeffs
            elif systematic=='holeice':
                return muongun_qIndPegleg_holeices
            elif systematic=='forward':
                return muongun_qIndPegleg_forwards
            elif systematic=='absorption':
                sets = ['0.929'] + muongun_qIndPegleg_absorption + ['1.142'] if ishyperplane and hasBothBulkice else muongun_qIndPegleg_absorption
                return sets
            elif systematic=='scattering':
                return muongun_qIndPegleg_scattering
        else: ## H2 sets
            if systematic=='domeff':
                return muongun_domeffs
            elif systematic=='holeice':
                return muongun_holeices
    else : ## neutrinos
        if dima_sets:
            if systematic=='domeff':
                return neutrino_dima_domeffs
            elif systematic=='holeice':
                return neutrino_dima_holeices
            elif systematic=='forward':
                return neutrino_dima_forwards
        elif qIndPegleg:
            if systematic=='domeff':
                return neutrino_qIndPegleg_domeffs
            elif systematic=='holeice':
                return neutrino_qIndPegleg_holeices
            elif systematic=='forward':
                return neutrino_qIndPegleg_forwards
            elif systematic=='absorption':
                sets = ['0.929'] + neutrino_qIndPegleg_absorption if ishyperplane and hasBothBulkice else neutrino_qIndPegleg_absorption
                return sets
            elif systematic=='scattering':
                return neutrino_qIndPegleg_scattering
            elif systematic=='coin':
                return neutrino_qIndPegleg_coins
        else: ## H2 sets
            if systematic=='domeff':
                return neutrino_domeffs
            elif systematic=='holeice':
                return neutrino_holeices
    return None

###########################################################################
#### Map Class : key <-> attributes
###########################################################################
class Map(dict):

    ''' A class to change dictionary key to attributes
        e.g. m = Map({'e':[1,2,3]}) can be accessed by m.e '''

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems(): self[k] = v
        if kwargs:
            for k, v in kwargs.iteritems(): self[k] = v

    def __getattr__(self, attr): 
        if attr.startswith('__') and attr.endswith('__'):
            return super(Map, self).__getattr__(attr)
        return self.__getitem__(attr)

    def __setattr__(self, key, value): 
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item): 
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self): 
        return self.__dict__

    def __setstate__(self, d): 
        self.update(d)
        self.__dict__.update(d)

###########################################################################
##### Define exceptions 
###########################################################################
def check_param_setting (method):
    if method not in ['linear', 'parabola', 'exponential', 'exp']:
        raise InvalidParamSettingError()
    return

class InvalidParamSettingError(Exception):
    def __str__(self):
        return 'Invalid settings for discrete parameterization. See documentation in nuisance_textfile.'

class DoubleRandomnessError(Exception):
    def __str__(self):
        return 'Either --poisson_fluctuation OR --random_sample. Not both.'

class InvalidOscProbError(Exception):
    def __str__(self):
        return 'OscProb is either vacuum OR pro3.'

class InvalidOutPathError(Exception):
    def __str__(self):
        return 'Invalid outPath.'

###########################################################################
##### Get nuisance_parameters from a text file
###########################################################################
def get_nuisance_params( nuisance_params_textfile='' ):

    ''' Obtain nuisance parameters from a given text file
        nuparams = misc_tools.get_nuisance_params (nuisance_params_textfile='~/nuparams.txt') '''
    
    params = Map({})
    textfile = open(nuisance_params_textfile, "r")
    f = lambda arg: None if (arg=='None' or not arg) else True if arg=='True' else False if arg=='False' else float(arg)

    for setting_line in ( raw.strip().split() for raw in textfile ):
        if not setting_line or setting_line[0][0]=='#':continue
        
        if len(setting_line)==2:
            ## barlow on/off
            setattr (params, setting_line[0], Map( {'included' :f(setting_line[1])} ))
        elif len(setting_line)==10:
            ## continuous systematics with penalty
            setattr (params, setting_line[0], Map( {'seeded_mc':f(setting_line[1])  , 'injected_data':f(setting_line[2]), 
                                                    'included' :f(setting_line[3])  , 'value'        :f(setting_line[4]), 
                                                    'limit'    :( f(setting_line[5]), f(setting_line[6]) ),
                                                    'error'    :f(setting_line[7])  , 'prior':f(setting_line[8]),
                                                    'penalty'  :f(setting_line[9]) } ))
        elif len(setting_line)==8:
            ## continuous systematics with no penalty
            setattr (params, setting_line[0], Map( {'seeded_mc':f(setting_line[1])  , 'injected_data':f(setting_line[2]), 
                                                    'included' :f(setting_line[3])  , 'value'        :f(setting_line[4]), 
                                                    'limit'    :( f(setting_line[5]), f(setting_line[6]) ),
                                                    'error'    :f(setting_line[7]) } ))
        elif len(setting_line)==12:
            ## discrete systematics with no penalty
            setattr (params, setting_line[0], Map( {'nu_method':setting_line[1]   , 'atmmu_method' :setting_line[2]   , 
                                                    'shifted'  :f(setting_line[3]), 'refit'        :f(setting_line[4]), 
                                                    'seeded_mc':f(setting_line[5]), 'injected_data':f(setting_line[6]), 
                                                    'included' :f(setting_line[7]), 'value'        :f(setting_line[8]), 
                                                    'limit'    :( f(setting_line[9]), f(setting_line[10]) ),
                                                    'error'    :f(setting_line[11]) } ))
        elif len(setting_line)==14:
            ## discrete systematics with prior mean and penalty
            setattr (params, setting_line[0], Map( {'nu_method':setting_line[1]   ,  'atmmu_method' :setting_line[2]   , 
                                                    'shifted'  :f(setting_line[3]),  'refit'        :f(setting_line[4]), 
                                                    'seeded_mc':f(setting_line[5]),  'injected_data':f(setting_line[6]), 
                                                    'included' :f(setting_line[7]),  'value'        :f(setting_line[8]), 
                                                    'limit'    :( f(setting_line[9]), f(setting_line[10]) ),
                                                    'error'    :f(setting_line[11]), 'prior'        :f(setting_line[12]),
                                                    'penalty'  :f(setting_line[13])} ))
        else:
            print ('Something is wrong in {0}'.format(setting_line))
            print ('Please comment it out, or follow the format [parameter_name, seeded_mc, injected_data, included, start_value, lower_lim, upp_lim, error, sigma]')
            sys.exit()

    textfile.close()

    #### check parameterization settings if discrete systematic is included
    for sysname in ['domeff', 'holeice', 'forward', 'coin', 'absorption', 'scattering']:
        if params[sysname].included:
            check_param_setting ( params[sysname].nu_method )
            check_param_setting ( params[sysname].atmmu_method )

    return params

###############################################################################################
#####                            Used in i3_pydict_converter.py                         #######
###############################################################################################

###################################################
##### Function to add particle info from i3 frame  
###################################################
def add_particle_info(particle, ddict,
                      pid=None, energy=None, zenith=None):

    """ Extra energy/zenith info from the given I3Particle to the given datadict
        If pid/energy/zenith argument is given, those values will be used instead

        Add true info if pid == None.
        Add reco info if pid not None. """

    e = energy if type(energy)==float else particle.energy
    z = zenith if type(zenith)==float else particle.dir.zenith
    ddict.e.append  ( e )
    ddict.z.append  ( z )
    ddict.cz.append ( np.cos(z) )
    if type(pid)==float:
        ddict.pid.append ( pid )
    else: ## if pdg, particle has to be MCTree[0]
        ddict.pdg.append( particle.pdg_encoding )
    return

###############################################################################################
#####                                  Used in fit_xxxx.py                              #######
###############################################################################################

###################################################
##### Function to print nuisance parameter settings  
###################################################

def print_nuisance_params_settings (fit_params):

    f = lambda bvalue: 'True' if bvalue else 'False'

    print ('#### your continuous nuisance parameters set up:')
    print ('####')
    print ('####   {0:16} | {1:16} | {2:16} | {3:16} | {4:16} | {5:16} | {6:16} | {7:16} | {8:16} | {9:16}'.format
           ('parameter'.center(16)  , 'seeded_mc'.center(16)  , 'injected_data'.center(16), 'included'.center(16), 'start_value'.center(16), 
            'lower_limit'.center(16), 'upper_limit'.center(16), 'error'.center(16)        , 'prior'.center(16), 'penalty'.center(16)   ) )
    print ('####   {0} | {0} | {0} | {0} | {0} | {0} | {0} | {0} | {0} | {0} '.format('='*16))

    for param in fit_params.keys():

        thisparam = fit_params[param]
        ### separate print outs for discrete systematics
        if param in ['domeff', 'holeice', 'forward', 'coin', 'absorption', 'scattering']: continue

        if param=='barlow':
            print ('####   {0:16} | {1:16} | {1:16} | {2:16} | {1:16} | {1:16} | {1:16} | {1:16} | '.format
                   ( param.center(16), '---', f(thisparam.included) ) )
        elif 'prior' not in thisparam.keys():
            print ('####   {0:16} | {1:16.7f} | {2:16.7f} | {3:16} | {4:16.7f} | {5:16.7f} | {6:16.7f} | {7:16.7f} | '.format
                   ( param.center(16), thisparam.seeded_mc, thisparam.injected_data, f(thisparam.included), 
                     thisparam.value , thisparam.limit[0] , thisparam.limit[1]     , thisparam.error ) )
        else:
            print ('####   {0:16} | {1:16.7f} | {2:16.7f} | {3:16} | {4:16.7f} | {5:16.7f} | {6:16.7f} | {7:16.7f} | {8:16.7f} | {9:16.7f}'.format
                   ( param.center(16), thisparam.seeded_mc, thisparam.injected_data, f(thisparam.included), 
                     thisparam.value , thisparam.limit[0] , thisparam.limit[1]     , thisparam.error      , thisparam.prior, thisparam.penalty ) )

    print ('#### ')
    print ('#### your discrete nuisance parameters set up:')
    print ('####')
    print ('####   {0:11} | {1:11} | {2:11} | {3:11} | {4:11} | {5:11} | {6:14} | {7:11} | {8:11} | {9:11} | {10:11} | {11:11} | {12:11} | {13:11} '.format
           ( 'parameter'.center(11)  , 'nu method'.center(11)    , 'mu method'.center(11),'shifted?'.center(11), 'redo fit?'.center(11), 
             'seeded_mc'.center(11)  , 'injected_data'.center(12), 'included'.center(11) , 'start_value'.center(11),
             'lower_limit'.center(11), 'upper_limit'.center(11)  , 'error'.center(11)    , 'prior'.center(11), 'penalty'.center(11)   ) )
    print ('####   {0} | {0} | {0} | {0} | {0} | {0} | {1} | {0} | {0} | {0} | {0} | {0} | {0} | {0} '.format('='*11, '='*14))
    for param in ['domeff', 'holeice', 'forward', 'coin', 'absorption', 'scattering']:
        thisparam = fit_params[param]
        if 'prior' not in thisparam.keys():
            print ('####   {0:11} | {1:11} | {2:11} | {3:11} | {4:11} | {5:11.5f} | {6:14.7f} | {7:11} | {8:11.5f} | {9:11.5f} | {10:11.5f} | {11:11.5f} '.format
                   ( param.center(11)   , thisparam.nu_method    , thisparam.atmmu_method, f(thisparam.shifted) , f(thisparam.refit), 
                     thisparam.seeded_mc, thisparam.injected_data, f(thisparam.included) , thisparam.value   ,
                     thisparam.limit[0] , thisparam.limit[1]     , thisparam.error ) )
        else:
            print ('####   {0:11} | {1:11} | {2:11} | {3:11} | {4:11} | {5:11.5f} | {6:14.7f} | {7:11} | {8:11.5f} | {9:11.5f} | {10:11.5f} | {11:11.5f} | {12:11.5f} | {13:11.5f} '.format
                   ( param.center(11)   , thisparam.nu_method    , thisparam.atmmu_method, f(thisparam.shifted) , f(thisparam.refit), 
                     thisparam.seeded_mc, thisparam.injected_data, f(thisparam.included) , thisparam.value   ,
                     thisparam.limit[0] , thisparam.limit[1]     , thisparam.error       , thisparam.prior   , thisparam.penalty  ) )

    print ('####')
    return

###############################################################################################
#####                                Used in masterdict.py                              #######
###############################################################################################

###########################################################################
##### Combine numunc and nuenc python dictionary into one nunc dictionary
##### Used in get_members (see next function)
#####
##### Since the cross section of an oscillated NC neutrinos is the same 
##### as the one for unoscillated neutrinos of the original flavor, NC 
##### neutrinos are not oscillated in this fitting algorithm (to save 
##### computational time). In addition, unoscillated nutau flux is 0. So,
##### only numunc and nuenc are in nunc.
###########################################################################
def select_events (ddict, erange, zrho='michael', recoZcut=None):

    recoX, recoY, recoZ = np.array (ddict.reco.X), np.array (ddict.reco.Y), np.array (ddict.reco.Z)
    rho = np.sqrt( (recoX-46.29)*(recoX-46.29) + (recoY+34.88)*(recoY+34.88) )
    if zrho=='martin':
        cut = recoZ < -230.
        cut *= rho < 140.
        vertex = cut*np.logical_or((recoZ+230.)/(rho-90)<-4.4, rho< 90.)
    else:
        cut = ddict.reco.Z < -200
        cut *= rho < 100
        vertex = cut*np.logical_or((recoZ+100)/(rho-50)<-2.5, rho< 50)
    boolean = np.logical_and (vertex, np.logical_and (ddict.hits.SRT_nCh<100., ddict.geo.charge_asym<0.85))
    if recoZcut: boolean = np.logical_and (recoZ>recoZcut, boolean)
    #if ddict.has_key ('mc') and ddict.mc.has_key ('ismuongun'):
    #    print ('this is merged muongun (before {0})'.format (np.sum (ddict.w[boolean])))
    #    goodmuons = np.logical_or (ddict.mc.ismuongun, np.logical_and (~ddict.mc.ismuongun, ~ddict.mc.inmuongun))
    #    boolean = np.logical_and (boolean, goodmuons)
    #    print ('this is merged muongun (after {0})'.format (np.sum (ddict.w[boolean])))
    e, pid = np.array (ddict.reco.e), np.array (ddict.reco.pid)
    cut = np.logical_and (pid<1000., np.logical_and (e>=erange[0], e<erange[1]))
    boolean = np.logical_and (cut, boolean)
    return apply_cut_to_dict (ddict, boolean)

def apply_cut_to_dict (ddict, cut):

    new_ddict = Map({})
    for key in ddict.keys():
        if isinstance(ddict[key], float): continue
        if isinstance(ddict[key], list) or isinstance(ddict[key],  np.ndarray):
            new_ddict[key] = ddict[key][cut] if len(ddict[key])>0 else []
            continue
        new_ddict[key] = Map({})
        for subkey in ddict[key].keys():
            if key=='reco' and 'llh' in subkey: continue
            new_ddict[key][subkey] = ddict[key][subkey][cut] if len(ddict[key][subkey])>0 else []
    return new_ddict

def concatenate_dicts (ddict1, ddict2, ddict3=None):

    combined_ddict = Map({})
    for key in ddict1.keys():
        if key not in ddict1 or key not in ddict2: continue
        if isinstance(ddict1[key], list) or isinstance(ddict1[key],  np.ndarray):
            #print ('len ddict1.{0}, ddict2.{0}: {1}'.format(key, len (ddict1[key]), len (ddict2[key])))
            combined_ddict[key] = np.array([]) if len(ddict1[key])==0 else \
                                  np.concatenate ((np.array(ddict1[key]), np.array(ddict2[key]))) if ddict3==None else \
                                  np.concatenate ((np.concatenate((ddict1[key], ddict2[key])), ddict3[key]))
            continue
        combined_ddict[key] = Map({})
        for subkey in ddict1[key].keys():
            if subkey not in ddict1[key].keys() or subkey not in ddict2[key].keys(): continue
            #print ('len ddict1.{0}.{1}, ddict2.{0}.{1}: {2}'.format(key, subkey, len (ddict1[key][subkey]), len (ddict2[key][subkey])))
            combined_ddict[key][subkey] = np.array([]) if len(ddict1[key][subkey])==0  else \
                                          np.concatenate ((np.array(ddict1[key][subkey]), np.array(ddict2[key][subkey]))) if ddict3==None else \
                                          np.concatenate ((np.concatenate ((ddict1[key][subkey], ddict2[key][subkey])), ddict3[key][subkey]))
    return combined_ddict

def get_nunc_member ( basepath      ='/data/user/elims/datasets/mlarson/pFiles/dllh/',
                      domeff        = '1'  , holeice    = '50' , forward      = '0',
                      scattering    = ''   , absorption = '' ,
                      noRDE_sets    = False, dima_sets  = False, SPiceHD_sets = False    , qIndPegleg = False,
                      include_nugen = False, isbaseline = False,
                      extra_cuts    = False, recoZcut   = None , zrho         = 'michael',
                      iscoin        = False, erange     = (10**0.75, 10**1.75)):

    isabs = absorption not in ['', '1']
    issca = scattering not in ['', '1']
    isoff = absorption == '0.929'
    if domeff=='0.99': domeff='1'
    ext = '_forward'+forward if (dima_sets or noRDE_sets or SPiceHD_sets) and (not iscoin and not isabs and not issca and not isoff) else ''
    if SPiceHD_sets:
        ext += '_SPiceHD'
    elif noRDE_sets and (not iscoin and not isabs and not issca and not isoff): 
        ext += '_noRDE'

    if iscoin: ext += '_coincident'
    if isabs and not isoff: ext += '_absorption'+absorption
    if issca and not isoff: ext += '_scattering'+scattering
    if isoff : ext += '_offaxis0.929'
    if qIndPegleg: ext += '_qIndPegleg'
    if include_nugen and isbaseline: ext += '_merged'
    
    tmp = {}
    for dtype in ['numunc', 'nuenc', 'nutaunc']:
        print ('####                          {0} pfile: {1}'.format (dtype, dtype+'_domeff'+domeff+'_holeice'+holeice+ext+'.p'))
        with open( basepath+dtype+'_domeff'+domeff+'_holeice'+holeice+ext+'.p', "rb") as f:
            ddict = cPickle.load (f)
        f.close()
        tmp[dtype] = select_events (ddict, erange, zrho=zrho, recoZcut=recoZcut) if extra_cuts else ddict
    return concatenate_dicts (tmp['numunc'], tmp['nuenc'], ddict3=tmp['nutaunc'])

###########################################################################
##### Get the dictionary of baseline members (and any systematic sets that
##### the user asked ) NOTE : updated to muongun15 only
###########################################################################
def get_members ( basepath      = '/data/user/elims/datasets/mlarson/pFiles/dllh/',
                  datatypes     = ['noise', 'muongun15', 'numucc', 'nuecc', 'nutaucc', 'nunc'], 
                  discrete_sys  = []   , erange          = (10**0.75, 10**1.75),
                  include_nugen = False, include_corsika = False    ,
                  extra_cuts    = False, recoZcut        = None     , zrho            = 'michael', ishyperplane = False,
                  noRDE_sets    = False, SPiceHD_sets    = False    , dima_sets       = False    , qIndPegleg   = False ):
                  

    ''' return a datadict that stores information from all events of all datatypes '''
    members = Map ({})

    #### get baseline set members
    print ('####           getting baseline sets ...')
    members['baseline'] = Map({})
    for dtype in datatypes:
        #### genie and nugen nc
        if dtype == 'nunc':
            holeice = '10' if SPiceHD_sets else '25' if dima_sets or noRDE_sets or qIndPegleg else '50'
            members['baseline']['nunc'] = get_nunc_member (basepath=basepath, holeice=holeice, erange=erange,
                                                           noRDE_sets=noRDE_sets, dima_sets=dima_sets, SPiceHD_sets=SPiceHD_sets, qIndPegleg=qIndPegleg,
                                                           include_nugen=include_nugen,  isbaseline=True,
                                                           extra_cuts=extra_cuts, recoZcut=recoZcut, zrho=zrho,
                                                           iscoin=False)
        else:
            #### muongun / noise / genie cc sets
            pfile = basepath + dtype if dtype in ['exp', 'noise', 'burnsample', 'muon_201X'] else \
                    basepath + dtype + '_domeff1_holeice10_forward0_SPiceHD' if 'nu' in dtype and SPiceHD_sets else \
                    basepath + dtype + '_domeff1_holeice25_forward0_noRDE' if 'nu' in dtype and noRDE_sets else \
                    basepath + dtype + '_domeff1_holeice25_forward0' if dima_sets or (noRDE_sets and 'muongun' in dtype)else \
                    basepath + dtype + '_domeff1_holeice50'
            if qIndPegleg: pfile += '_qIndPegleg'
            if include_nugen and 'nu' in dtype: pfile += '_merged'
            if include_corsika and 'muongun' in dtype: pfile += '_mergedcorsika'
            print ('####              {0} pfile: {1}.p'.format (dtype, os.path.split (pfile)[1]))
            with open( pfile+'.p', "rb") as f:
                ddict = cPickle.load(f)
            f.close()
            members['baseline'][dtype] = select_events (ddict, erange, zrho=zrho, recoZcut=recoZcut) if extra_cuts else ddict
    print ('####')
    
    #### get systematic set members if asked
    if discrete_sys:
        for systematic in discrete_sys:
            print ('####           getting {0} sets ... '.format(systematic))
            members[systematic] = Map({})
            for dtype in datatypes:
                ## noise and msu muon_201X doesn't have systematic sets
                if dtype in ['muon_201X', 'noise', 'exp', 'burnsample']: continue
                if systematic in ['coin'] and dtype=='muongun15': continue
                print ('####                    -- {0} '.format(dtype))
                members[systematic][dtype] = Map({})
                dsets = grab_discrete_sets (dtype, systematic, qIndPegleg=qIndPegleg, dima_sets=dima_sets, ishyperplane=ishyperplane, hasBothBulkice=('absorption' in discrete_sys and 'scattering' in discrete_sys))
                ## get systematic members of this systematic set
                for dset in dsets:
                    domeff  = dset if systematic=='domeff' else '1'
                    holeice = dset if systematic=='holeice' else \
                              '25' if 'muongun' in dtype and systematic in ['absorption', 'scattering'] else \
                              '30' if ('nu' in dtype and dima_sets and systematic=='forward') or ('muongun' in dtype and (qIndPegleg or dima_sets or noRDE_sets)) else \
                              '25' if dima_sets or noRDE_sets or SPiceHD_sets or systematic in ['coin', 'absorption', 'scattering'] else '50'
                    forward = dset if systematic=='forward' else '0'
                    absorption = dset if systematic=='absorption' else '1'
                    scattering = dset if systematic=='scattering' else \
                                 '0.929' if systematic=='absorption' and dset=='0.929' else \
                                 '1.142' if systematic=='absorption' and dset=='1.142' else '1'
                    dset_key = str(round(1./float(dset),10)) if systematic=='holeice' and not dima_sets and not qIndPegleg else \
                               forward.replace('n', '-') if systematic=='forward' and 'n' in forward else \
                               forward.replace('p', '+') if systematic=='forward' and 'p' in forward else \
                               absorption.replace('n', '-') if systematic=='absorption' and 'n' in absorption else \
                               '0.693' if dset=='0.69' else '0.792' if dset=='0.79' else \
                               dset 
                    if dtype == 'nunc':
                        members[systematic]['nunc'][dset_key] = get_nunc_member (basepath=basepath    , erange=erange      , 
                                                                                 domeff=domeff        , holeice=holeice    , forward=forward   , 
                                                                                 scattering=scattering, absorption=absorption,
                                                                                 noRDE_sets=noRDE_sets, dima_sets=dima_sets, SPiceHD_sets=False, qIndPegleg=qIndPegleg,
                                                                                 include_nugen=False  , isbaseline=False   ,
                                                                                 extra_cuts=extra_cuts, recoZcut=recoZcut  , zrho=zrho         ,
                                                                                 iscoin=(systematic=='coin' and dset_key=='1.0') )
                    else:
                        ext = ''
                        if 'nu' in dtype:
                            if   systematic=='coin'       and dset_key=='1.0'  : 
                                ext += '_coincident'
                            elif systematic in ['scattering', 'absorption'] and not dset_key=='1': 
                                if dset in ['0.929']:
                                    ext += '_offaxis' + dset
                                else:
                                    ext += '_'+systematic+dset
                            elif dima_sets or noRDE_sets:
                                ext += '_forward'+forward
                                if noRDE_sets: ext += '_noRDE'
                        if 'muongun' in dtype:
                            if dima_sets or qIndPegleg: 
                                ext += '_forward'+forward
                            if systematic in ['scattering', 'absorption'] and dset_key=='1': 
                                ext += '_base'
                            elif systematic in ['scattering', 'absorption']: 
                                if dset in ['0.929', '1.142']:
                                    ext += '_offaxis'+dset
                                else:
                                    ext += '_'+systematic+dset
                        if qIndPegleg: ext += '_qIndPegleg'
                        pfile = basepath + dtype + '_domeff'+domeff+'_holeice'+holeice+ext+'.p'
                        print ('####                          {0} pfile: {1}'.format (dtype, os.path.split (pfile)[1]))
                        with open( pfile, "rb") as f:
                            ddict = cPickle.load(f)
                        f.close()
                        members[systematic][dtype][dset_key] = select_events (ddict, erange, zrho=zrho, recoZcut=recoZcut) if extra_cuts else ddict
                print ('####')
    return members

###########################################################################
##### Function to decompose total 3D histogram into a track histo 
##### and a cascade histo (ONLY for sample with 2 PID bins)
###########################################################################
def get_track_cscd_counts (histo):

    trackh = []; cscdh = []
    for index in np.arange(len(histo)):
        cscdh.append(histo[index][:,0]); trackh.append(histo[index][:,1])
    cscdh = np.vstack(cscdh); trackh = np.vstack(trackh)
    return cscdh, trackh

###########################################################################
##### Print number of expected / raw events
###########################################################################
def print_nevents (mdict, discrete_sys=[], expected=False, n_years=1, use_msusample=False):

    muon_key = 'muon_201X' if use_msusample else 'muongun15'
    datatypes = ['nuecc', 'numucc', 'nutaucc', 'nunc', muon_key, 'noise']
    datatypesets = ['baseline'] + discrete_sys
    number_type = 'expected' if expected else 'raw'
    print ('#### {0:39} {1:8} number of events {0:39} '.format ('='*39, number_type.center(8)) )
    print ('#### == {0:27} | {1:9} | {2:9} | {3:9} | {4:9} | {5:9} | {6:9} =='.format 
           ( '', 'nuecc', 'numucc', 'nutaucc', 'nunc', muon_key, 'noise') )

    for datatypeset in datatypesets:
        members = mdict[datatypeset]; nevents = {}

        if datatypeset=='baseline':
            for dtype in datatypes:
                nevents[dtype] = int(np.sum(members[dtype].w)*seconds_per_year*n_years) if dtype in members.keys() and expected else \
                                 len(members[dtype].reco.e) if dtype in members.keys() else '---'
            print ('#### == {0:27} | {1:9} | {2:9} | {3:9} | {4:9} | {5:9} | {6:9} =='.format 
                   ( 'baseline'.center(27), nevents['nuecc'], nevents['numucc'], nevents['nutaucc'], nevents['nunc'], nevents[muon_key], nevents['noise']) )

        else: ### in case domeff/holeice/forward
            for dtype in members.keys():
                nevents[dtype] = {}
                for dset in members[dtype].keys():
                    nevents[dtype][dset] = int(np.sum(members[dtype][dset].w)*seconds_per_year*n_years) if expected else len(members[dtype][dset].reco.e)
            for dset in nevents[nevents.keys()[0]].keys():
                bg_events = ('-'*3).center(7)
                mg_events =  nevents['muongun15']['0.99'] if 'muongun15' in nevents.keys() and datatypeset=='domeff' and dset=='1' else \
                             nevents['muongun15'][dset] if 'muongun15' in nevents.keys() and dset in ['0.033', '0.01', '0.0333333333', '15', '25', '1', '1.1'] else bg_events
                print ('#### == {0:27} | {1:9} | {2:9} | {3:9} | {4:9} | {5:9} | {6:9} =='.format 
                       ( datatypeset+' '+dset, nevents['nuecc'][dset], nevents['numucc'][dset], nevents['nutaucc'][dset], 
                         nevents['nunc'][dset], mg_events, bg_events ) )
            if 'domeff' in discrete_sys and 'muongun15' in nevents:
                for dset in nevents['muongun15'].keys():
                    non_mg_events = ('-'*3).center(7)
                    print ('#### == {0:27} | {1:9} | {2:9} | {3:9} | {4:9} | {5:9} | {6:9} =='.format 
                           ( datatypeset+' '+dset, non_mg_events, non_mg_events, non_mg_events, non_mg_events, nevents['muongun15'][dset], non_mg_events ) )
    print ('#### {0:105}'.format('='*105))
    print ('####')
    return

###########################################################################
##### Print number of expected events that are in the total MC histograms
###########################################################################
def get_histo_nevent_counts (histos, factor, is2D=False, realdata=False): 

    ''' get counts / errors per PID bin '''

    nevents = {}
    if is2D:
        if realdata:
            nevents['counts'] = int (np.sum( histos.mch * factor ))
            nevents['errors'] = int (np.sqrt(np.sum(counts)))
        else:
            for dtype in histos.keys():
                nevents[dtype] = {}
                nevents[dtype]['counts'] = int(np.sum( histos[dtype].mch * factor ))
                nevents[dtype]['errors'] = int(np.sqrt(np.sum( histos[dtype].mch2 * factor**2 )))
    else: ## 3D
        nevents['track'] = {}; nevents['cascade'] = {}
        if realdata:
            cscdh, trackh = get_track_cscd_counts ( histos.mch * factor )
            nevents['cascade']['counts'], nevents['track']['counts'] = int (np.sum(cscdh)), int (np.sum(trackh))
            nevents['cascade']['errors'], nevents['track']['errors'] = round (np.sqrt(np.sum(cscdh)), 3), round (np.sqrt(np.sum(trackh)), 3)
        else:
            for dtype in histos.keys():
                nevents['track'][dtype] = {}; nevents['cascade'][dtype] = {}
                cscdh, trackh = get_track_cscd_counts ( histos[dtype].mch * factor )
                nevents['cascade'][dtype]['counts'], nevents['track'][dtype]['counts'] = int (np.sum(cscdh)), int (np.sum(trackh))
                cscdh2, trackh2 = get_track_cscd_counts ( histos[dtype].mch2 * factor**2 )
                nevents['cascade'][dtype]['errors'], nevents['track'][dtype]['errors'] = round (np.sqrt(np.sum(cscdh2)), 3), round (np.sqrt(np.sum(trackh2)), 3)

    return nevents

def print_2D_histo_nevents (nevents, muon_key):

    #### define counts/errors for datatypes that are not included in the table header
    for dtype in ['nuecc', 'numucc', 'nutaucc', 'nunc', muon_key, 'noise']:
        if dtype not in nevents.keys():
            nevents[dtype] = {'counts':'---', 'errors':'--'}

    #### print numbers 
    print ('#### == {0:7} | {1:5} ({2:3}) | {3:5} ({4:3}) | {5:5} ({6:3}) | {7:5} ({8:3}) | {9:5} ({10:4}) | {11:5} ({12:4}) =='.format
           ( ' '*7, nevents['nuecc']['counts'], nevents['nuecc']['errors'], nevents['numucc']['counts'], nevents['numucc']['errors'],
             nevents['nutaucc']['counts'], nevents['nutaucc']['errors'], nevents['nunc']['counts'] , nevents['nunc']['errors'],
             nevents[muon_key]['counts'], nevents[muon_key]['errors'], nevents['noise']['counts'], nevents['noise']['errors'] ) )
    print ('#### {0:99}'.format('='*99))
    print ('####')
    return

def print_3D_histo_nevents (nevents, muon_key):

    #### define counts/errors for datatypes that are not included in the table header
    for dtype in ['nuecc', 'numucc', 'nutaucc', 'nunc', muon_key, 'noise']:
        for pidtype in ['track', 'cascade']:
            if dtype not in nevents[pidtype]: 
                nevents[pidtype][dtype] = {'counts':'---', 'errors':'--'}

    for pidtype in ['track', 'cascade']:
        print ('#### == {0:7} | {1:5} ({2:3}) | {3:5} ({4:3}) | {5:5} ({6:3}) | {7:5} ({8:3}) | {9:5} ({10:4}) | {11:5} ({12:4}) =='.format 
               ( pidtype.center(7), nevents[pidtype]['nuecc']['counts'], nevents[pidtype]['nuecc']['errors'], nevents[pidtype]['numucc']['counts'], nevents[pidtype]['numucc']['errors'],
                 nevents[pidtype]['nutaucc']['counts'], nevents[pidtype]['nutaucc']['errors'], nevents[pidtype]['nunc']['counts'], nevents[pidtype]['nunc']['errors'],
                 nevents[pidtype][muon_key]['counts'], nevents[pidtype][muon_key]['errors'], nevents[pidtype]['noise']['counts'], nevents[pidtype]['noise']['errors'] ) )
    print ('#### {0:99}'.format('='*99))
    print ('####')
    return

def print_histo_nevents( histos, n_years=1, is2D=False, inHz=True, total=False, realdata=False, use_msusample=False ):

    htype = 'MC' if inHz else 'data'
    if total: htype += ' total'
    factor = seconds_per_year * n_years if inHz else 1.
    nevents = get_histo_nevent_counts (histos, factor, is2D=is2D, realdata=realdata)

    if realdata:
        print ('#### === expected number of events in final data histogram ')
        if is2D:
            print ('#### === {0} ({1}) '.format(nevents['counts'], nevents['errors']))
        else:
            print ('#### === track   | {0} ({1}) '.format(nevents['track']['counts'], nevents['track']['errors']))
            print ('#### === cascade | {0} ({1}) '.format(nevents['cascade']['counts'], nevents['cascade']['errors']))
        return

    muon_key = 'muon_201X' if use_msusample else 'muongun15'
    print ('#### {0:21} expected number of events in final {1:10} histogram {0:21} '.format ('='*21, htype.center(10)) )
    print ('#### == {0:11} | {1:15} | {2:15} | {3:15} | {4:15} | {5:15} | {6:15} =='.format 
           ( '', 'nuecc'.center(15), 'numucc'.center(15), 'nutaucc'.center(15), 'nunc'.center(15), muon_key.center(15), 'noise'.center(15)))
    
    if is2D:
        print_2D_histo_nevents (nevents, muon_key)
    else:
        print_3D_histo_nevents (nevents, muon_key)

    return

###########################################################################
##### Print all info about the master dictionary.
###########################################################################
def print_check_info (info):

    f = lambda (value): round(value,4) if type(value) in [float, np.float64] else value

    print ('#### master dictionary info:')
    print ('####')
    print ('####           neutrinos                          {0}'.format(info['neutrinos']))
    print ('####           other backgrounds                  {0}'.format(info['backgrounds']))
    print ('####           pickled data dict from             {0}'.format(info['pdictpath']))
    print ('####')
    print ('####           osc prob from                      {0} '.format(info['oscProb']))
    print ('####           oscillate NC neutrinos?            {0} '.format(info['oscNC']))
    print ('####           number of expected years           {0} year(s)'.format(info['mcparams']['nyears']))
    print ('#### ')
    print ('####           energy axis bin edges              {0}'.format( np.round(info['eedges'], decimals=2) ))
    print ('####           zenith axis bin edges              {0}'.format( np.round(info['zedges'], decimals=2) ))
    print ('####           pid axis bin edges                 {0}'.format( np.round(info['pedges'], decimals=2) ))
    print ('####')
    print ('####           poisson fluctuate?                 {0}'.format(info['poisson']))
    print ('####           random sample?                     {0}'.format(info['random']))
    print ('####           fakedata expected from             {0} year(s)'.format(info['dataparams']['nyears']))
    print ('####')
    print ('####           seeded/injected values of parameters (MC templates, pseudo data):')
    print ('####')
    for key in info['mcparams'].keys():
        if key=='barlow': continue
        if key in ['domeff', 'holeice', 'forward', 'coin', 'absorption']:
            printout = '' if key not in info['shifted'].keys() else '( shifted to nominal value )' if info['shifted'][key] else '( NOT shifted to nominal value )'
            print ('####           {0:34} {1:16}: {2:10}, {3:10} {4:30}'.format(' '*34, key, f(info['mcparams'][key]), f(info['dataparams'][key]), printout.center(30) ))
            continue
        print ('####           {0:34} {1:16}: {2:10}, {3:10}'.format(' '*34, key, f(info['mcparams'][key]), f(info['dataparams'][key]) ))
    print ('####')
    print ('####           dict written to                    {0}'.format(info['outPath']))
    print ('####')
    check_info (info)
    return

###########################################################################
##### Check all info about the master dictionary.
###########################################################################
def check_info (info):

    if info['poisson'] and info['random']:
       raise DoubleRandomnessError()

    if info['oscProb'] not in ['vacuum', 'prob3']: 
        raise InvalidOscProbError()

    if not os.path.exists(info['outPath']):
        raise InvalidOutPathError()

    return
    
###########################################################################
##### Get name for output master dictionary
###########################################################################
def get_supname (info):

    if info['datafrom']:
        return info['prefix']+info['datafrom']
    if len (info['outname'])>0:
        return info['prefix']+info['outname']

    injected = ('wbf' if (info['dataparams']['dm31'] == 2.43E-3 and info['dataparams']['theta23'] == 0.7365) else
                str(info['dataparams']['dm31']) + '_maxmix' if np.sin(info['dataparams']['theta23'])**2 == 0.5 else
                str(info['dataparams']['dm31']) + '_' + str(info['dataparams']['theta23']))
    subname = injected+'_'+info['oscProb']+'_'+str(float(info['dataparams']['nyears']))
    subsubname = subname+'_'+'2D' if info['is2D'] else subname+'_'+'3D'
    subsubsubname = subsubname+'_'+'oscnc' if info['oscNC'] else subsubname+'_'+'nullNC'
    name = ( info['prefix']+'poisson_'+subsubsubname if info['poisson'] else 
             info['prefix']+'random_'+subsubsubname  if info['random']  else
             info['prefix']+'norandomness_'+subsubsubname )
    return name

###########################################################################
#### Functions that count qTotal / nCh / nStrings
#### InIcePulses (OfflinePulses before 2011) stores all pulse info from HLC and SLC hits.                                                                                                                                         ####
####    --- For each hit (DOM launch) registering > 0.25PE ,
####    --- HLC hits = number of self/neignboring DOMs get hit with 1 microsecond
####    --- SLC hits = number of other DOMs on the same string get hit with 1 microsecond
###########################################################################
def get_qtotal ( pulse ):

    ''' return total charge in a given pulseseries '''

    qtot = 0
    for dom in pulse.keys():
        for hit in pulse[dom]:
            qtot += hit.charge
    return float (qtot)

def get_nch ( pulse, HLC_hits=False ):

    ''' return number of channels and number of strings in a given pulseseries '''

    nch = 0
    strings = []
    for dom in pulse.keys():
        #### loop through each hit in DOM to check HLC
        if HLC_hits:
            isHLC = False
            for hit in pulse[dom]:
                if hasattr(hit, 'lc_bit'):
                    if hit.lc_bit:
                        strings.append (dom[0])
                        isHLC=True
                        break
                if hasattr(hit, 'flags'):
                    if hit.flags & I3RecoPulse.PulseFlags.LC:
                        strings.append (dom[0])
                        isHLC=True
                        break
            nch += isHLC
        else:
            #### nCh = number of DOMs that get hits
            strings.append (dom[0])
            nch += 1
    return float (nch), float ( len(np.unique (strings)) )

def hits_info ( frame,
                cleanedpulses='SRTTWOfflinePulsesDC',
                uncleanedpulses='InIcePulses' ) :

    ''' return rtotal, hlc_nch, hlc_nstring, srt_nch, srt_nstring from 
        a cleaned and an uncleaned pulseseries'''

    qtot, hlc_nch, hlc_nstr, srt_nch, srt_nstr = 0., 0., 0., 0., 0.

    #### qtotal, nstrings, nch from cleanedpulses                                                                                                                                                          
    if frame.Has(cleanedpulses):
        pulses = frame[cleanedpulses].apply(frame) if type(frame[cleanedpulses]) in [dataclasses.I3RecoPulseSeriesMapMask, dataclasses.I3RecoPulseSeriesMapUnion] \
                 else frame[cleanedpulses]

        srt_nch, srt_nstr  = get_nch ( pulses, HLC_hits=False )
        qtot =  get_qtotal ( pulses )

    #### nstrings, nch from cleanedpulses                                                                                                                                                                  
    if frame.Has(uncleanedpulses):
        pulses = frame[uncleanedpulses].apply(frame) if type(frame[uncleanedpulses]) in [dataclasses.I3RecoPulseSeriesMapMask, dataclasses.I3RecoPulseSeriesMapUnion] \
                 else frame[uncleanedpulses]

        hlc_nch, hlc_nstr  = get_nch ( pulses, HLC_hits=True )

    return qtot, hlc_nch, hlc_nstr, srt_nch, srt_nstr
