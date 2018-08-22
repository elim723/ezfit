#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script plot the change in histogram due to
#### systematic effect.
####
#### Command line:
#### $ python histoeffect.py --outdir plots/ --indir <indir>
####                    (optional --norm --chi)
####
####################################################################

from __future__ import print_function
from optparse import OptionGroup, OptionParser
from glob import glob
import numpy as np
import sys, cPickle, logging

logging.basicConfig (level=logging.INFO)
logger = logging.getLogger (__name__)
###########################################
#### import essential objects
###########################################
sys.path.append ('../')
from misc import Map, InvalidArguments
from template import Template

###########################################
#### import plot styling
###########################################
import plotter
from plotter import HistoEffect
labels = plotter.labels

########################################### 
#### parse options/params 
###########################################
usage = "%prog [options] <inputfiles>"
parser = OptionParser(usage=usage)
parser.add_option ('-o', '--outdir', type='string', default='~/',
                   help = "out directory for plotting")
parser.add_option ('-i', '--indir', type='string', default='',
                   help = "indir where all histo are located")
parser.add_option ('-n', '--norm', action='store_true', default=False,
                   help = "normalize histograms for shape comparison")
parser.add_option ('-c', '--chi', action='store_true', default=False,
                   help = "calculate sqrt (chi2) instead of percentage for comparison")
(options, args) = parser.parse_args()
indir   = options.indir
outdir  = options.outdir
norm    = options.norm
chi     = options.chi

###########################################
#### Functions to get templates
###########################################
def get_histos (params, filenames):

    ''' get a set of histograms with either lower
        or upper sigmas

        :type  params: list
        :param params: a list of parameter names

        :type  sigma: list
        :param sigma: a list of filenames with histograms

        :return results: dictionary
                results: cascade and track chi / percentage
                         {'param': {'cascade':[], 'track':[]} }
    '''
    
    results = Map ({})

    for param in params:
        logger.info ('####   -- {0}'.format (param))
        filename = [ filename for filename in filenames if param in filename ]
        ### check filename length; must be 1.
        if not len (filename) == 1:
            message = 'histoeffect:get_histos :: ' + param + \
                      ' has a file length of '+str (len (filename))
            raise InvalidArguments (message)
        ### get histograms
        with open (filename[0], 'rb') as f:
            histo = cPickle.load (f)
        f.close ()
        histo = cPickle.loads (histo)
        ### calculate comparisons in cascade/track
        results [param] = compare (histo.template, histo.dhisto)
    edges = {'x': np.log10 (histo.info ('eedges')),
             'y': np.cos (histo.info ('zedges'))[::-1]}
    logger.info ('#### ')
    return results, edges

def compare (template, data):

    ''' return the comparison betwee template and dhisto.
        The default comparison is in percentage w/o norm.

        :type  template: a dictionary
        :param template: MC histogram counts and variances

        :type  data: a dictionary
        :param data: data histogram counts and variances

        :return results: a dictionary
                results: either percentage or chi of cascade/track histograms
    '''

    ### get data info
    dcounts, dvars = data['H'], data['H2']
    ### get mc info
    factor = np.sum (dcounts) / np.sum (template['H']) if norm else 1.0
    mcounts = template['H'] * factor
    mvars = template['H2'] * factor**2

    ### get comparison in PIDs
    results = {}
    for index, pid in enumerate (['cascade', 'track']):
        dc, mc = dcounts[:,:,index], mcounts[:,:,index]
        dv, mv = dvars[:,:,index]  , mvars[:,:,index]
        ## compare
        result = (dc - mc) / np.sqrt (dv + mv) if chi else \
                 (dc - mc) / mc * 100.
        results [pid] = result.T[::-1].T
    return results
    
###########################################
#### Functions to get plot
###########################################
def plot_effects (param, lower, upper, **kwargs):

    ''' plot a systematic effect on the histogram for
        a given parameters
    
        :type  param: string
        :param param: name of the systematic to be plotted

        :type  lower: a dictionary
        :param lower: comparison between lower sigma and baseline
                      {'cascade':[], 'track':[]} }

        :type  upper: a dictionary
        :param upper: comparison between upper sigma and baseline
                      {'cascade':[], 'track':[]} }

        :type  kwargs: a dictionary
        :param kwargs: plot settings
    '''

    ### set up canvas and set style
    canvas = HistoEffect (**kwargs)
    canvas.set_style (**kwargs)

    ### plot 
    canvas.plot (param, lower, upper)

    ### save
    sname = 'chi' if chi else 'percent'
    name = outdir + '/histoeffect_'+param+'_'+sname+'.pdf'
    canvas.save (name)

    ### delete
    del canvas

###########################################
#### Everything starts here :)
###########################################
#### get lower / upper histograms
lowers = sorted (glob (indir + '/*lower*.p'))
uppers = sorted (glob (indir + '/*upper*.p'))
params = [ param for param in labels['sys'] for lower in lowers if param in lower ]

logger.info ('########################################')
logger.info ('########### histo effect ###############')
logger.info ('####')
logger.info ('#### You have included ...')
for param in params:
    logger.info ('####    -- {0}'.format (param))
logger.info ('####')
    
#### get comparisons
comparisons = {}
for sigma in ['lowers', 'uppers']:
    logger.info ('#### Working on getting {0} comparisons ...'.format (sigma))
    comparisons [sigma], edges = get_histos (params, eval (sigma))

logger.info ('#### Done collecting histos')
logger.info ('####')
    
#### set plot style kwargs
rounding = 2 if chi else 1
cbarlabel = r'(changed - baseline) / $\sigma_{\text{tot}}$' if chi else \
            r'(changed - baseline) / baseline ($\%$)' ## percent

kwargs = {'edges'      : edges,
          'labelsize'  : 10,
          'labels'     : {'x':labels['obs']['reco_e'], 'y':labels['obs']['reco_cz']},
          'ticksize'   : 8,
          'contentsize': 5,
          'rounding'   : rounding,
          'cbarlabel'  : cbarlabel,
          'cmap'       : plotter.plt.get_cmap ('RdYlBu'),
          'print2D'    : True }

#### plot for each param
logger.info ('#### Working on plots')
for param in params:
    logger.info ('####    -- {0}'.format (param))
    kwargs['fmt']   = '%10.2f' if param=='muon_flux' else '%10.1f'
    kwargs['title'] = labels['sys'][param]
    plot_effects (param, 
                  comparisons['lowers'][param],
                  comparisons['uppers'][param],
                  **kwargs)

#### done
logger.info ('####')
logger.info ('########################################')
