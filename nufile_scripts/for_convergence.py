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
        ## exclude oscillation parameters to be measured
        if param in ['dm31', 'theta23']: continue
        ## get lower and upper values
        if param=='coin':
            lower, upper = '0.0', '0.1'
        elif param in ['forward']:
            lower, upper = '-2.0', '2.0'
        elif 'prior' in nuParams (param):
            prior = nuParams (param).prior
            penalty = nuParams (param).penalty
            lower = str (round (prior-nsigmas*penalty, 4))
            upper = str (round (prior+nsigmas*penalty, 4))
        elif 'norm' in param:
            injected = nuParams (param).injected
            lower = str (round (0.8*injected, 4))
            upper = str (round (1.2*injected, 4))
        else:
            injected = nuParams (param).injected
            lowlim  = nuParams (param).lower_limit
            upplim  = nuParams (param).upper_limit
            lower = str (round ((injected-lowlim)/2., 4))
            upper = str (round ((upplim-injected)/2., 4))
        values [param] = [lower, upper]
    return values

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
### get nuParams
nuParams = nuparams.Nuparams (nufiles[0])
## only perform convergence on parameters included
params = [ param for param in nuParams.get_all_params () 
           if nuParams (param).included ]
### define injected values for non-osc parameters
values = get_injected_values (params)

### define grid (injected) points
dm31s = np.round (np.linspace (2, 3, gridpoints), 4)  ## in 1e-3
sin2theta23s = np.round (np.linspace (0.25, 0.75, gridpoints), 4)

###########################################
#### spit out textfiles
###########################################
for nufile in nufiles:
    
    ### define octant and parameters of this nufile
    octant = 'o1' if 'o1' in os.path.split (nufile)[1] else 'o2'
    nuParams = nuparams.Nuparams (nufile)

    ### print verbose header 
    if verbose > 0:
        print ('############## {0} ##############'.format (octant))
    
    ### loop through osc parameters
    for dm31 in dm31s:
        for sin2theta23 in sin2theta23s:

            ## define oscillation injected values in strings
            oscvalues = {'dm31':str (dm31),
                         'theta23':str (np.round (np.arcsin (np.sqrt (sin2theta23)), 4))}
            
            ## loop through upper and lower injected values
            for j, limit in enumerate ([ 'lower', 'upper' ]):

                if verbose > 0:
                    print ('#### dm31, sin2theta23, limit: {0}, {1}, {2}'.format (dm31, sin2theta23, limit))
                
                # define output file
                outfile = outdir + 'injected_' + str (dm31) + '_' + \
                          str (sin2theta23) + '_' + octant + '_' + limit + '.txt'
                outtxt = open (outfile, 'w')

                ### get input file
                intxt = open (nufile, 'r')

                # write out file line by line
                for index, line in enumerate (intxt):
                    if line=='' or line[0] in ['#', ' ', '\n']:
                        outtxt.write(line)
                    elif line.strip ().split ()[0] in params:

                        param = line.strip ().split ()[0]
                        initial = '2.526' if 'dm31' in param else \
                                  str (int (nuParams (param).injected)) if 'holeice' in param else \
                                  str (nuParams (param).injected)
                        value = oscvalues[param] if param in oscvalues else \
                                values [param][j]
                        oline = line.replace  (initial, value, 2).replace (value, initial, 1)
                        if verbose > 0: print ('#### {0}'.format (oline))
                        outtxt.write(oline)
                    else:
                        outtxt.write (line)

                ### close all files
                intxt.close()
                outtxt.close()
                if verbose > 0: print ('####')
