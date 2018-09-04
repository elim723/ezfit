#!/usr/bin/env python

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
#### This script does the following procedures:
####
#### 0. Initialize Template instance 
####
#### TO OBTAIN MC TEMPLATE
#### 1. Set baseline histograms
#### 2. Set hyperplane objects for all data types
#### 3. Multiply baseline histogram by factors
####
#### TO OBTAIN DATA HISTO
#### 4. Get data histogram
####
#### TO COMPARE DATA AND MC BEFORE FITTING
#### 5. Evaluate chi2 out of the box
####
#### TO SAVE
#### 6. Pickle template instance
###############################################################

from __future__ import print_function
from optparse import OptionParser
import time, os, cPickle, socket
import numpy as np

## classes for template
from template import Template
from nuparams import Nuparams
from likelihood import Likelihood
from misc import eedges, zedges, pedges
from misc import Toolbox

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
fit_data         = options.fit_data
members          = ['numucc', 'nuecc', 'nutaucc', 'numunc',
                    'nuenc', 'nutaunc', 'noise', 'muon']
edges            = {'e':eedges,
                    'z':zedges,
                    'p':pedges }

pdictpath        = os.path.dirname(os.path.abspath( __file__ )) + \
                   '/../pickled_files/'
nuparam_textfile = options.nuparam_textfile
outfile          = options.outfile

matter           = True
oscnc            = options.oscnc
inverted         = options.inverted

verbose          = options.verbose

###########################################
#### check file/path before work
###########################################
toolbox = Toolbox ()
## does nuparam_textfile exist?
toolbox.check_file (nuparam_textfile)
## is pdictpath valid?
toolbox.check_path (pdictpath)
## is outdir valid?
toolbox.check_path (os.path.split (outfile)[0])

start_time = time.time()
###########################################
#### Step 0: initialize
###########################################
print ('#### Initializing template ...')
print ('####')

## initialize template and nuparams
temp = Template (verbose=verbose)

## set up nuparams
temp.nufile = nuparam_textfile
temp.inverted = inverted
nuparams = Nuparams (nuparam_textfile,
                     isinverted=inverted)
print ('{0}'.format (nuparams))
seeded   = nuparams.extract_params ('seeded')
injected = nuparams.extract_params ('injected')

###########################################
#### Step 1: get baseline histograms
###########################################
print ('#### Setting baseline histograms ...')
print ('####')

## set up properties needed 
temp.ppath   = pdictpath
temp.members = members
temp.edges   = edges
temp.oscnc   = oscnc
temp.matter  = matter

## set baseline histograms 
library, bhistos = temp.get_baseline_histograms (seeded)
temp.bhistos = bhistos

###########################################
#### Step 2: get hyperplanes
###########################################
print ('#### Setting hyperplane objects ...')
print ('####')

temp.hplanes = library.get_hplanes (nuparams,
                                    verbose=verbose)

###########################################
#### Step 3: get MC template
###########################################
print ('#### Getting MC template ...')
print ('####')

mhistos, mc = temp.get_template (seeded, library, bhistos)
temp.mhistos  = mhistos
temp.template = mc

###########################################
#### Step 4: get data histogram to be fit
###########################################
print ('#### Getting data histogram ...')
print ('####')

## set up information needed
data = temp.get_data (injected, fit_data,
                      nuparams.diff_injected_seeded ())

###########################################
#### Step 5: calculate chi2 before fit
###########################################
print ('#### Evaluating chi2 out of the box ...')
print ('####')

LH = Likelihood (data, 'modchi2',
                 verbose=verbose )
LH.set_histos (mhistos)
ts, bints, As = LH.get_ts ()
print ('#### modified chi2: {0}'.format (2*ts))

###########################################
#### Step 6: save temp object
###########################################
temp.outfile = outfile
pstring = cPickle.dumps (temp)
with open (outfile, 'wb') as f:
    cPickle.dump (pstring, f, protocol=2)
f.close ()

print ('#### ################################################')
print (' ... it took {0} minuites'.format ((time.time() - start_time)/60.))
