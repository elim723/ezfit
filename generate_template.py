#!/usr/bin/env python

#################################################################################################
####
#### By Elim Cheung (07/24/2018)
####
#### This script is Step 1 of the fitting algorithm.
#### It creates the template object from template.py.
####
#### Command line to run:
#### $ python generate_templates.py
####            --outfile <address/to/outputfile.p>
####		--nuparam_textfile <nuparam file> --verbose 1
####            (optional: --oscnc --inverted --fit_data)
####
#### NOTE: This script is cleaned up to reproduce GRECO
####       numu disappearance result.
####       However, one can add more arguments to gain
####       access to the flexibility of the tool. 
####
#################################################################################################

from __future__ import print_function
from optparse import OptionParser
from template import template
import numpy as np
import time, os, cPickle, socket
import copy_reg, types, multiprocessing

from misc import eedges, zedges, pedges

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
#### function to pickle instances
###########################################
def pickle_method (m):

    ''' pickle self objects '''
    
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

###########################################
#### parse options/params
###########################################
parser = OptionParser()
parser.add_option ("--outfile", type="string", default='output.p',
                   help = "address and name for out put file (with extension .p)")
parser.add_option ("--nuparam_textfile", type="string", 
                   default='~/nuisance_parameters_settings.txt',
                   help = "a textfile with a table of nuisance parameter settings")
parser.add_option ("--verbose", type="int", default=1,
                   help = "0 for no print out; 2 for detailed print out")
parser.add_option ("--oscnc", action="store_true", default=False,
                   help = "oscillate NC")
parser.add_option ("--inverted", action="store_true", default=False,
                   help = "inverted hierarchy")
parser.add_option ("--fit_data", action="store_true", default=False,
                   help = "use data as data histogram")
(options, args) = parser.parse_args()

###########################################
#### define more arguments
###########################################
start_time = time.time()

fit_data         = options.fit_data
neutrinos        = ['numucc', 'nuecc', 'nutaucc', 'numunc', 'nuenc', 'nutaunc']
backgrounds      = ['noise', 'muon']

pdictpath        = os.path.dirname(os.path.abspath( __file__ )) + '/../pickled_files/'
nuparam_textfile = options.nuparam_textfile
outfile          = options.outfile

matter           = True
oscnc            = options.oscnc
inverted         = options.inverted

verbose          = options.verbose

###########################################
#### creat temp object
###########################################
temp = template (fit_data=fit_data,
                 neutrinos=neutrinos,
                 backgrounds=backgrounds,
                 eedges=eedges,
                 zedges=zedges,
                 pedges=pedges,
                 pdictpath=pdictpath,
                 nuparam_textfile=nuparam_textfile,
                 outfile=outfile,
                 matter=matter,
                 oscnc=oscnc,
                 inverted=inverted,
                 verbose=verbose)

###########################################
#### save temp object
###########################################
with open (temp.info ('outfile'), 'wb') as f:
    cPickle.dump (temp, f, protocol=2)
f.close ()

print (' ... it took {0} minuites'.format ((time.time() - start_time)/60.))
