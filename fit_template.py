#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script is Step 2 of the fitting algorithm.
#### It takes a template object and perform minimization.
####
#### Command line to run:
#### $ python fit_template.py --outfile <outfile> --test_statistics modchi2
####                          --nuparam_textfile <nuparam textfile>
####                          --template <template file>
####                         (--dm31_grid 2.526e-3 --theta23_grid 0.7252
####                          --norm_nutau_grid 1.0 --verbose 1)
####
###############################################################

from __future__ import print_function
from optparse import OptionParser
import cPickle, socket, time
from copy import deepcopy
import numpy as np

from misc import Map
from fitter import Fitter
from nuparams import Nuparams
from likelihood import Likelihood

## ignore RuntimeWarning
import warnings
warnings.filterwarnings("ignore")
###########################################
#### print machine name
###########################################
print ('####################################################################')
print ('#### This job is running on {0}.'.format (socket.gethostname ()))
print ('####################################################################')
print (' ')

###########################################
#### essential functions
###########################################
def get_fix_oparams (dm31, theta23, norm_nutau):

    ''' determine the parameters to be fixed at grid point

        :type  dm31: a float or None
        :param dm31: If not provided, it is a bestfit
                     performance. dm31 is allowed to float.

        :type  theta23: a float or None
        :param theta23: If not provided, it is a bestfit
                        performance. theta23 is allowed to float.

        :type  norm_nutau: a float or None
        :param norm_nutau: If not provided, it is a bestfit
                           performance. norm_nutau is allowed to float.

        :return fix_oparams: a dictionary
                fix_oparams: values at which osc params to be fixed
    '''
    
    fix_oparams = {}
    for param in ['dm31', 'theta23', 'norm_nutau']:
        if eval (param):
            fix_oparams [param] = eval (param)
    return fix_oparams

def get_params (nuparams, dm31, theta23, norm_nutau):

    ''' get parameters

        :type  nuparams: a Nuparams object
        :param nuparams: all user's input values

        :return fparams: a numpy array / list
                fparams: oscillation parameters to be fixed

        :return params: a dictionary
                params: minimizer start values
    '''
    
    ## get oscillation parameter that is fixed
    fixed_params = get_fix_oparams (dm31, theta23, norm_nutau)
    ## list of oscillation parameters to be fixed
    fparams = sorted ([ p for p in fixed_params ])
    ## get user's start values 
    params = nuparams.extract_params ('value')
    ## replace values of osc parameter that is
    ## fixed to grid points
    for p in fparams:
        params [p] = fixed_params [p]
    return fparams, params

def get_lib_histos (dtypes, params):

    ''' get library, hyperplaned histos,
        and data histogram

        :type  dtypes: a numpy array / list
        :param dtypes: data types / members included
    
        :type  params: a dictionary
        :param params: values of parameters from
                       nuparams' minimizer start values
                       and input osc parameters

        :return  lib: a Library object
                 lib: contains baseline events for all
                      members and their weighters

        :return  mhistos: a dictionary
                 mhistos: hyperplane-ly modified histograms
                          and their variances  from all data types

        :return  dhisto: a dictionary
                       : data histogram and variance
    '''

    lib, bhistos = temp.get_baseline_histograms (dtypes, params)
    mhistos, mc, var = temp.get_template (dtypes, params, lib, bhistos)
    dhisto = temp.dhisto
    if verbose > 0:
        temp._print_rates ('hplaned', mhistos)
        temp._print_rates ('template', {'H':mc, 'H2':var})
        temp._print_rates ('data', dhisto)
    return lib, mhistos, dhisto

def collect_events (lib, dtypes):

    ''' collect baseline events.

        :type  lib: a Library object
        :param lib: contains events for all members

        :type  dtypes: a list or numpy array
        :param dtypes: datatypes involved
    
        :return events: a dictionary / Map
                events: baseline events for all dtypes
    '''
    
    events = Map ({})
    for dtype in dtypes:
        events[dtype] = lib._baseline[dtype]._events
    return events

def get_likelihood (test_statistics, dhisto, dtypes, lib, params):

    ''' get likelihood object

        :type  test_statistics: a string
        :param test_statistics: test statistics method
                                chi2 / modchi2 / barlow / poisson

        :type  dhisto: a dictionary
        :param dhisto: data histogram {'H':H, 'H2':H2}

        :type  dtypes: a list / numpy array
        :param dtypes: list of dtypes involved

        :type  lib: a Library object
        :param lib: library object to get baseline events

        :type  params: a dictionary
        :param params: minimizer start values by user

        :return LH: a Likelihood class
                LH: likelihood object
    '''
    
    LH = Likelihood (dhisto, test_statistics, verbose=verbose)
    ## special info for barlow LLH
    if 'barlow' in test_statistics:
        events = collect_events (lib, dtypes)
        unhistos, norms = get_barlow_params (events, params)
        LH.set_barlow (unhistos, norms)
    return LH

def get_barlow_params (events, params):

    ''' get unweighted histograms and normalization
        factors for barlow LH

        :type  events: a dictionary
        :param events: baseline events from all dtypes

        :type  params: a dictionary
        :param params: minimizer start values by user

        :return unhistos: a dictionary
                unhistos: unweighted histograms for all dtypes
     
        :return norms: a dictionary
                norms: normalization factors
    '''

    unhistos, norms = Map ({}), Map ({})
    edges = info.get_edges ()
    for dtype in events:
        ## get unweighted histograms
        weights = np.ones (len (events[dtype].reco.e))
        H, H2 = member.get_histogram (edges, weights=weights)
        unhistos[dtype] = Map ({'H':H, 'H2':H2})
        ## get the normalization factor
        key = 'atmmu' if 'muon' in dtype else 'noise' if 'noise' in dtype else 'numu'
        norm = params['norm_nutau'] if 'nutau' in dtype else \
               params['norm_nc'] if 'nc' in dtype else 1
        norm *= params['norm_'+key]
        norms[dtype] = norm
        
    return unhistos, norms

def convert_to_dict (obj):

    ''' convert an iminuit output into regular
        dictionary for pickling 

        :type  obj: iminuit type
        :param obj: iminuit output

        :return dictionary: python dictionary
                dictionary: converted dictionary
    '''
    
    dictionary = {}
    for key in obj:
        dictionary [key] = obj [key]
    return dictionary

def print_result (output):

    print ('#### Minimization results: ')
    print ('####')
    line = '#### {0:20} | {1:20.10f}'
    for key in output ['params']:
        value = round (output ['params'][key], 10)
        print (line.format (key.center (10), value))
    print ('####')
    print ('#### TS value: {0}'.format (2*output['ts']))
    print ('####')
    
###########################################
#### parse options/params
###########################################
parser = OptionParser ()
parser.add_option ("--dm31_grid", type = "float", default = None,
                   help = "submitted delta m31 sq grid point")
parser.add_option ("--theta23_grid", type = "float", default = None,
                   help = "submitted theta23 grid point")
parser.add_option ("--norm_nutau_grid", type = "float", default = None,
                   help = "submitted norm_nutau grid point")
parser.add_option ("--outfile", type="string", default=None,
                   help = "address and name for out put file (with extension .p)")
parser.add_option ("--template", type = "string", default = None,
                   help = "address to template")
parser.add_option ("--test_statistics", type="string", default = 'modchi2',
                   help = "modchi2 / chi2 / poissonllh / barlowllh")
parser.add_option ("--nuparam_textfile", type="string", default = None,
                   help = "address to nuisance parameter textfile")
parser.add_option ("--verbose", type="int", default = 1,
                   help = "0: no print. 1: min print. 2: print iter, blinded. 3: print iter, unblinded.")
(options, args) = parser.parse_args()

###########################################
#### define more arguments
###########################################

### misc
verbose = options.verbose
outfile = options.outfile

### essential files
tempfile = options.template
with open (tempfile, 'rb') as f:
    temp = cPickle.load (f)
f.close ()
temp = cPickle.loads (temp)
info = temp.info

### set nuparams
nuparam_textfile= options.nuparam_textfile
# if not specified, get the textfile from template
if not nuparam_textfile:
    nuparam_textfile = temp.info ('nuparam_textfile')
nuparams = Nuparams (nuparam_textfile,
                     isinverted=info ('inverted'))

### set parameter dictionary
dm31 = options.dm31_grid
theta23 = options.theta23_grid
norm_nutau = options.norm_nutau_grid
fparams, params = get_params (nuparams, dm31, theta23, norm_nutau)

### get library, hplaned, data histograms
dtypes = info.get_datatypes ()
lib, mhistos, dhisto = get_lib_histos (dtypes, params)

## set likelihood
test_statistics = options.test_statistics
LH = get_likelihood (test_statistics, dhisto, dtypes, lib, params)
LH.set_histos (mhistos)
ts, bints, As = LH.get_ts ()
if verbose > 0:
    print ('#### {0}: {1}'.format (test_statistics, 2*ts))
    print ('####')

###########################################
#### start fitter
###########################################
start_time = time.time()

## perform fit
fit = Fitter (nuparams, params, lib, LH, temp,
              fparams=fparams,
              verbose=verbose)

## dump output
output = {'bints'    : fit.bestfit.ts,
          'ts'       : fit.results.fval,
          'params'   : convert_to_dict (fit.results.values),
          'errors'   : convert_to_dict (fit.results.errors),
          'histos'   : Map ({ 'H':fit.bestfit.H, 'H2':fit.bestfit.H2 }),
          'barlow_As': fit.barlow_As }
    
with open (outfile, 'wb') as f:
    cPickle.dump (output, f, protocol=2)
f.close ()

## print results if fitter doesn't
if verbose < 4:
    print_result (output)

print('  .... it takes {0} minuites.'.format ( (time.time() - start_time)/60. ))
