#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script is to change the nuparam_textfile
#### for data challenge for numu disappearance.
#### This will produce a set of N x N x 2 x 2
#### nuparam_textfiles, where N is set by arguments.
####
#### For oscillation parameters, N grid points are
#### included for each oscillation parameter.
#### That is, N x N.
####
#### For non oscillation parameters, two values are
#### set for each of them
####  -- parameters with priors: +/- #sigma sigmas
####  -- parameters without priors:
####              norms: +/- 20%
####              forward: +/- 3
#### That is, N x N x 2.
####
#### Finally, one minimization is performed for each
#### octant. That is, N x N x 2 x 2.
####
#### Command line to run:
#### $ python for_convergence.py
####              --outdir <path/to/out/folder>
####              --gridpoints 5
####
#### Note: user can also change the textfile
#### template itself directly without this script.
####
###############################################################

from __future__ import print_function
from optparse import OptionGroup, OptionParser
from copy import deepcopy
import numpy as np
import sys, os

###########################################
#### paths for essential tools
###########################################
sys.path.append ('../')
import nuparams

###########################################
#### functions 
###########################################
def get_injected_values (params):

    ''' get injected values for non-osc parameters

        :type  params: a list
        :param params: parameters included by users

        :return values: a dictionary
                      : two injected values for
                        each parameter
    '''
    
    values = {}
    for param in params:
        if param in ['dm31', 'theta23']: continue
        ## get lower and upper values
        if 'prior' in nuParams (param):
            prior = nuParams (param).prior
            penalty = nuParams (param).penalty
            values [param] = [prior - nsigmas*penalty, prior + nsigmas*penalty]
        elif 'norm' in param:
            injected = nuParams (param).injected
            values [param] = [0.8*injected, 1.2*injected]
        else:
            injected = nuParams (param).injected
            lower    = nuParams (param).lower_limit
            upper    = nuParams (param).upper_limit
            values [param] = [(injected-lower)/2., (upper-injected)/2.]
    return values

def get_injected_strings (dm31, sin2theta23, index):

    ''' get the injected values for all included parameters
    
        :type  dm31: a float
        :param dm31: this dm31 value in meV2

        :type  sin2theta23: a float
        :param sin2theta23: this sin2theta23 value

        :type  index: an int
        :param index: index - 0 = lower limit / 1 = upper limit

        :return injecteds: a dictionary
                injecteds: injected values in string
    '''
    
    injecteds = {}
    for param in params:
        if param in ['dm31', 'theta23']:
            ### injected osc params
            injecteds [param] = str (dm31) if param=='dm31' else \
                                str (round (np.arcsin (np.sqrt (sin2theta23)), 4))
        else:
            injecteds [param] = str (round (values [param][index], 4))
    return injecteds

def get_oline (nuParams, param, injecteds, line):

    ''' get out line

        :type  nuParams: a nuParams object
        :param nuParams: nuParams for this nufile

        :type  params: a list / array
        :param params: list of parameters included

        :type  injecteds: a dictionary
        :param injecteds: the injected values in string

        :type  line: string
        :param line: the initial line from textfile

        :return  oline: string
                 oline: replaced line
    '''
    
    # ====================================
    # temporary: exclude hplane / dparams
    if nuParams (param).isdiscrete:
        return line.replace ('True', 'False')
    # ====================================
    initial = '2.526' if 'dm31' in param else \
              str (nuParams (param).injected)
    updated = injecteds [param]
    return line.replace  (initial, updated, 2).replace (updated, initial, 1)

###########################################
#### parse user's options
###########################################
usage = "%prog [options]"
parser = OptionParser(usage=usage)
parser.add_option ("-o", "--outdir", type="string", default='~/',
                   help = "output nuisance textfile directory")
parser.add_option ("-n", "--gridpoints", type="int", default=5,
                   help = "number of grid points for oscillation parameters")
parser.add_option ("-v", "--verbose", type="int", default=0,
                   help = "print outs")
(options, args) = parser.parse_args()

outdir = options.outdir
verbose = options.verbose
gridpoints = options.gridpoints

indir = os.path.dirname (os.path.realpath(__file__))
nufiles = [ indir + '/nuparams_template_o1.txt',
            indir + '/nuparams_template_o2.txt' ]
nsigmas = 1.25 ## for non-oscillation parameters

###########################################
#### define more variables
###########################################
### define new injected values for non-osc parameters
nuParams = nuparams.Nuparams (nufiles[0])
## only perform convergence on parameters included
params = [ param for param in nuParams.get_all_params () 
           if nuParams (param).included ]

### define injected values for non-osc parameters
values = get_injected_values (params)

### define grid (injected) points
dm31s = np.round (np.linspace (2, 3, gridpoints), 4) ## in 1e-3
sin2theta23s = np.linspace (0.25, 0.75, gridpoints)
theta23s = np.round (np.arcsin (np.sqrt (sin2theta23s)), 4)

###########################################
#### spit out textfiles
###########################################
for nufile in nufiles:
    
    ### define variables for this nufile
    octant = 'o1' if 'o1' in os.path.split (nufile)[1] else 'o2'
    nuParams = nuparams.Nuparams (nufile)

    if verbose > 0: print ('############## {0} ##############'.format (octant))
    
    ### loop through osc parameters
    for dm31 in dm31s:
        for sin2theta23 in sin2theta23s:
            ## loop through upper and lower injected values
            for j, limit in enumerate ([ 'lower', 'upper' ]):

                if verbose > 0:
                    print ('#### dm31, sin2theta23, limit: {0}, {1}, {2}'.format (dm31, sin2theta23, limit))
                    print ('####')
                
                ### get input file
                intxt = open (nufile, 'r')
                
                # define output file
                outfile = outdir + 'injected_' + str (dm31) + '_' + \
                          str (sin2theta23) + '_' + octant + '_' + limit + '.txt'
                outtxt = open (outfile, 'w')

                # define injected values in strings
                injecteds = get_injected_strings (dm31, sin2theta23, j)
                if verbose > 0:
                    print ('#### injected strings')
                    for key in injecteds:
                        print ('####           {0}: {1}'.format (key, injecteds[key]))
                    print ('####')
                
                # write out file line by line
                for index, line in enumerate (intxt):
                    if line=='' or line[0] in ['#', ' ', '\n']:
                        outtxt.write(line)
                    elif line.strip ().split ()[0] in params:
                        param = line.strip ().split ()[0]
                        oline = get_oline (nuParams, param, injecteds, line)
                        if verbose > 0: print ('{0}'.format (oline))
                        outtxt.write(oline)
                    else:
                        outtxt.write (line)

                ### close all files
                intxt.close()
                outtxt.close()
                if verbose > 0: print ('####')
