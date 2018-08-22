#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script is to change the nuparam_textfile
#### for template effects for numu disappearance.
#### This will produce a set of N x 2 nuparam_textfiles,
#### where N is the number of parameters included.
####
#### For each parameters, two values are set 
####   -- for parameters with priors: +/- 1 sigma
####   -- for parameters - norms: +/- 10%
####   -- others: half way between limit and injected
####
#### Only octant 1 template is used for histo effect.
#### No minimization is done.
####
#### Command line to run:
#### $ python for_histoeffect.py
####              --outdir <path/to/out/folder>
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
                        each parameter in string
    '''
    
    values = {}
    for param in params:
        ## get lower and upper values
        if param=='coin':
            lower, upper = '0.0', '0.1'
        elif 'prior' in nuParams (param):
            prior = nuParams (param).prior
            penalty = nuParams (param).penalty
            lower = str (round (prior-penalty, 4))
            upper = str (round (prior+penalty, 4))
        elif 'norm' in param:
            injected = nuParams (param).injected
            lower = str (round (0.9*injected, 4))
            upper = str (round (1.1*injected, 4))
        elif param in ['dm31', 'theta23']:
            lower = '2.426' if 'dm31' in param else '0.6652'
            upper = '2.626' if 'dm31' in param else '0.7852'
        elif param in ['forward']:
            lower, upper = '-2.0', '2.0'
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
parser.add_option ("-v", "--verbose", type="int", default=0,
                   help = "print outs")
(options, args) = parser.parse_args()

outdir = options.outdir
verbose = options.verbose

indir = os.path.dirname (os.path.realpath(__file__))
nufile = indir + '/nuparams_template_o1.txt'

###########################################
#### define more variables
###########################################
### define new injected values for non-osc parameters
nuParams = nuparams.Nuparams (nufile)
## only perform convergence on parameters included
params = [ param for param in nuParams.get_all_params () 
           if nuParams (param).included ]

### define injected values for all included parameters
values = get_injected_values (params)

###########################################
#### spit out textfiles
###########################################
### change one parameter per file
for param in params:

    if verbose > 0:
        print ('############## {0} ##############'.format (param))
    
    ## change upper / lower value per param
    for index, value in enumerate (values[param]):
    
        ### define nufile
        nufile = indir + '/nuparams_template_o1.txt'
        intxt = open (nufile, 'r')
                
        # define output file
        ext = 'upper' if index==1 else 'lower'
        outfile = outdir + param + '_' + ext + '.txt'
        outtxt = open (outfile, 'w')
        
        ### spit out a file
        if verbose > 0:
            print ('#### value: {0}'.format (value))
            print ('####')
            
        ### write out file line by line
        for index, line in enumerate (intxt):
            if line=='' or line[0] in ['#', ' ', '\n']:
                outtxt.write(line)
            elif param in line.strip ().split ()[0]:
                initial = '2.526' if 'dm31' in param else \
                          str (int (nuParams (param).injected)) if 'holeice' in param else \
                          str (nuParams (param).injected)
                oline = line.replace  (initial, value, 2).replace (value, initial, 1)
                if verbose > 0: print ('{0}'.format (oline))
                outtxt.write(oline)
            else:
                outtxt.write (line)

        ### close all files
        intxt.close()
        outtxt.close()
        if verbose > 0: print ('####')
