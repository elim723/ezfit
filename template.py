#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script generates MC template by the following procedures.
####
#### 0. Save info from input arguments
#### 1. Load all pickled files in Step 0
####
#### ================ TO OBTAIN MC TEMPLATE =================
#### 2. Get baseline histograms
####    base weighters and propmaps are saved to Template
#### 3. Get hyperplane objects for all data types
####    hyperplanes are saved to Template
#### 5. Multiply baseline histogram by factors
#### ====== The resultant histogram is the MC template ======
####
#### ================= TO OBTAIN DATA HISTO =================
#### 6. Get data histogram
#### ==== The resultant histogram is the data histogram =====
####
#### 7. Evaluate chi2 out of the box
####################################################################

from __future__ import print_function
import numpy as np
import cPickle

from misc import Map, Info
from member import Member
from library import Library
from nuparams import Nuparams
from likelihood import Likelihood

####################################################################
#### Template class
####################################################################
class Template (object):

    ''' A class to create an object with MC template and data
        histogram.
    '''
    
    def __init__(self, **kwargs):

        ''' initialize template class '''
        
        print ('#### ##################################################')
        print ('#### ############# Generate Template  #################')
        print ('####')

        ###################################################
        #### check / store / print user's inputs
        ###################################################
        print ('#### Checking your inputs ...')
        print ('####')
        self.info = Info (**kwargs)
        self.nuparams = Nuparams (self.info ('nuparam_textfile'),
                                  isinverted=self.info ('inverted'))
        print ('{0}'.format (self.info))
        print ('{0}'.format (self.nuparams))

        ###################################################
        #### collect constants
        ###################################################
        params    = self.nuparams.extract_params ('seeded')
        datatypes = self.info.get_datatypes ()
        outfile   = self.info ('outfile')
        
        ###################################################
        #### get baseline histograms (bhistos)
        ###################################################
        print ('#### Getting baseline histograms ...')
        print ('####')
        lib, bhistos = self.get_baseline_histograms (datatypes, params)
        self.bhistos = bhistos
        self.probmaps = lib.probmaps
        self._print_rates ('baseline', bhistos)

        ######################################################
        #### get hyperplanes and modify base_histos (mhistos)
        ######################################################
        print ('#### Getting hyperplane objects ...')
        print ('####')
        self.hplanes = lib.get_hplanes (self.nuparams, params,
                                        verbose=self.info ('verbose'))
        
        ###################################################
        #### get total mc template
        ###################################################
        print ('#### Getting MC template ...')
        print ('####')
        mhistos, template, variance = self.get_template (datatypes,
                                                         params, lib,
                                                         bhistos)
        self.mhistos = mhistos
        self._print_rates ('hplaned', mhistos)
        self.template = {'H':template, 'H2':variance}
        self._print_rates ('template', self.template)

        ###################################################
        #### get data histogram
        ###################################################
        print ('#### Getting data histogram ...')
        print ('####')
        template, variance = self.get_data (datatypes, params)
        self.dhisto = Map ({'H':template, 'H2':variance})
        self._print_rates ('data', self.dhisto)
        
        ###################################################
        #### calculate chi2 before any fit
        ###################################################
        print ('#### Evaluating chi2 out of the box ...')
        print ('####')
        LH = Likelihood (self.dhisto, 'modchi2',
                         verbose=self.info ('verbose') )
        LH.set_histos (self.mhistos)
        ts, bints, As = LH.get_ts ()
        print ('#### modified chi2: {0}'.format (2*ts))
        print ('#### ################################################')

    def __getstate__ (self):

        ''' get state for pickling '''
        
        return self.__dict__

    def __setstate__ (self, d):

        ''' set state for pickling '''

        self.__dict__ = d
    
    def get_baseline_histograms (self, dtypes, params):

        ''' obtain all baseline histograms for all data types

            :type  dtypes: a list / array
            :param dtypes: data types included in this template

            :type  params: dictionary
            :param params: values of floating parameters

            :retrun  lib: a Library class
                     lib: for manipulating members
        
            :retrun  bhistos: a dictionary
                     bhistos: baseline histograms from all members
        '''

        lib = Library (dtypes, self.info ('pdictpath'),
                       ranges=self.info.get_ranges (),
                       edges=self.info.get_edges (),
                       verbose=self.info ('verbose'))
        lib.set_weighters (params,
                           matter=self.info ('matter'),
                           oscnc=self.info ('oscnc'))
        bhistos = lib.collect_base_histograms (params)
        return lib, bhistos
    
    def get_template (self, dtypes, params, lib, histos):

        ''' obtain a template from all data types

            :type  dtypes: a list / array
            :param dtypes: data types included in this template

            :type  params: dictionary
            :param params: values of floating parameters

            :type  lib: a Library class
            :param lib: for manipulating members
        
            :type  histos: a dictionary
            :param histos: baseline histograms from all members

            :return mhistos: a dictionary
                    mhistos: modified histogram from hyperplane

            :retrun mc: a multi-dimensional array
                    mc: MC template histogram

            :return var: a multi-dimensional array
                    var: variance of MC template
        '''
        
        mhistos = lib.apply_hplanes (histos, self.hplanes, params)

        ### sum up histograms and variances
        mc = np.array (sum ([ mhistos[dtype]['H'] for dtype in dtypes ]))
        var = np.array (sum ([ mhistos[dtype]['H2'] for dtype in dtypes ]))
        return mhistos, mc, var
        
    def get_data (self, datatypes, params):

        ''' obtain data histogram

            :type  params: dictionary
            :param params: values of floating parameters

            :type  datatypes: a list / array 
            :param datatypes: name of data types involved

            :retrun H: a multi-dimensional array
                    H: data histogram

            :return H2: a multi-dimensional array
                    H2: variance of data histogram
        '''
        
        if self.info ('fit_data'):
            data = Member ('data', self.info ('pdictpath'),
                           ranges=self.info.get_ranges ())
            w = data.get_weights (params)
            return data.get_histogram (self.info.get_edges (), weights=w)
        elif self.nuparams.diff_injected_seeded ():
            ## get injdected parameters
            params    = self.nuparams.extract_params ('injected')
            ## library and baseline histograms from the injected param
            lib, bhistos = self.get_baseline_histograms (datatypes, params)
            ## same hyperplane
            ## get template with injected data
            mhisto, H, H2 = self.get_template (datatypes, params, lib, bhistos)
            return H, H2
        
        return self.template['H'], self.template['H2']

    def _print_rates (self, htype, histos):

        ''' print histogram rate '''

        ## print header
        print ('#### ##################################################')
        print ('#### ############## {0:8} histogram ################'.format (htype))
        print ('####')
        line = '####  {0:9} | {1:9} | {2:9} | {3:9} |'
        print (line.format ('dtypes'.center (9), 'nevents'.center (9),
                            'cascade'.center (9), 'track'.center (9) ))
        print ('#### {0}'.format ('-'*48))
        ## print line
        if htype in ['data', 'template']:
            self._print_line (htype, histos)
        else: ## per data type
            for dtype in self.info.get_datatypes ():
                self._print_line (dtype, histos[dtype])
        ## print end
        print ('#### {0}'.format ('-'*48))
        print ('####')
    
    def _print_line (self, dtype, histo):

        ''' print rate info of a given histogram '''
        
        line = '####  {0:9} | {1:5} {2:3} | {3:5} {4:3} | {5:5} {6:3} |'
        numbers = self._collect_numbers (histo)
        print (line.format (dtype.center (7),
                            int (numbers['nevents'][0]),
                            int (numbers['nevents'][1]),
                            int (numbers['cascade'][0]),
                            int (numbers['cascade'][1]),
                            int (numbers['track'][0]),
                            int (numbers['track'][1])  ))

    def _collect_numbers (self, histo):

        ''' collect rates (and uncertainty)

            :type   histo: a dictionary
            :param  histo: {'H':[], 'H2':[]}

            :return numbers: a dictionary
                    numbers: total rates / cascade / track of a given histo
        '''
        
        cascade  = np.sum (histo['H'][:,:,0])
        cascade2 = np.sqrt (np.sum (histo['H2'][:,:,0]))
        track  = np.sum (histo['H'][:,:,1])
        track2 = np.sqrt (np.sum (histo['H2'][:,:,1]))
        nevents  = np.sum (histo['H'])
        nevents2 = np.sqrt (np.sum (histo['H2']))
        return {'nevents': (nevents, nevents2),
                'cascade': (cascade, cascade2),
                'track'  : (track  , track2  )}
