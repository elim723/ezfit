#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contain Library class which
####
####    - manipulate all members (load / weights / histograms)
####    - get total template
####    - create hyperplane for a specific data type
####      (deal with discrete sets)
####    - print event rates and histograms
####
####################################################################

from __future__ import print_function
from copy import deepcopy
import numpy as np
import time

from misc import get_sets, Map, InvalidArguments
from weightcalculator import WeightCalculator
from hyperplane import HyperPlane
from member import Member

####################################################################
#### import constants needed
####################################################################
from misc import datatypes, seconds_per_year, default_edges, discrete_parameters
from misc import default_ranges, default_nu_sysvalues, default_mu_sysvalues

####################################################################
#### Library class
####################################################################
class Library (object):

        ''' A class to 
                  - load baseline events
                  - set weighters (given floating parameter values)
                  - create baseline histograms
                  - create hyperplane for a given data type
                  - sum up histograms for MC template
                  - build data histogram

            Example to create Library object
            --------------------------------
            In [0] : import library, nuparams
            In [1] : lib = library.Library (['numucc'], 'pickled_files/')

            Example to set weights and get baseline histograms
            --------------------------------------------------
            In [2] : nufile = 'nuisance_textfiles/nuparams_template.txt'
            In [3] : nuParams = nuparams.Nuparams (nufile, isinverted=False)
            In [4] : params = nuParams.extract_params ('seeded')
            In [5] : lib.set_weighters (params, oscnc=False)
            In [6] : basehistos = lib.collect_base_histograms (params)
            NOTE: weighters for baseline members are stored in Library.
            
            Example to get all systematic histograms for a specific data type
            -----------------------------------------------------------------
            In [7] : hparams = nuParams.get_hplaned_dparams () 
            In [8] : refvalues, histos = lib.collect_sys_histograms ('numucc',
                                                                     params, hparams)
            NOTE: everytime collect_sys_histograms () is called, it creates
                  a one-time use weighter for each member called.

            Example to get HyperPlane objects for all data types
            ----------------------------------------------------
            In [10]: hplanes = lib.get_hplanes (nuParams)

            Example to print all rates of samples and histograms
            ----------------------------------------------------
        '''

        def __init__ (self, dtypes, pdictpath,
                      ranges=default_ranges,
                      edges=default_edges,
                      verbose=1):

                ''' Initialize a Library object given a list of
                    members and discrete parameters included

                    :type   dtypes: a list or a numpy array 
                    :param  dtypes: list of all members included for templates

                    :type   pdictpath: a string
                    :param  pdictpath: path to all pickled dictionaries

                    :type   ranges: a dictionary of tuples
                    :param  ranges: limits of template ranges; events outside are removed

                    :type    edges: a dictionary of arrays
                    :param   edges: template bin edges 

                    :type   verbose: int
                    :param  verbose: If 0 or 1, no printout
                                     If 2, printout progress
                '''
                
                self._dtypes = dtypes
                self._ranges = ranges
                self._edges = edges
                self._pdictpath = pdictpath
                self._verbose = verbose
                
                ## a library must have baseline sets
                self._baseline = self._collect_base_members ()

        def __getstate__ (self):

                ''' get state for pickling '''

                return self.__dict__

        def __setstate__ (self, d):

                ''' set state for pickling '''

                self.__dict__ = d

        def __call__ (self, dtype):

                ''' return baseline event info (no weights) in this library '''
                
                return self._baseline [dtype]

        def _check_setid (self, setid, default):

                ''' check if setid already exists in a dictionary

                    :type    default: a dictionary
                    :param   default: keys/default values of discrete systematics

                    :type  setid: a string
                    :param setid: systematic values separated by '_' as the id of this set

                    :return  isref: a boolean
                    :        isref: If True, this is a reference set
                '''
                
                defid = self._get_setid (default)
                return defid == setid

        def _get_setid (self, sysvalues):

                ''' get the ID for the given sysvalues dictionary 

                    :return  sysvalues: a dictionary
                    :        sysvalues: keys/values of discrete systematics for the set

                    :return      setid: a string
                    :            setid: systematic values separated by '_' as the id of this set
                '''
                
                setid = ''
                for i, dp in enumerate (sorted (sysvalues)):
                        setid += str (float (sysvalues[dp]))
                        if not i == len (sysvalues)-1: setid += '_'
                return setid
        
        def _get_setvalues (self, set_value, dtype, hparam, default):

                ''' get the values dictionary for the specific discrete set

                    :type     set_value: a float
                    :param    set_value: value of one of the dparam sets

                    :type     hparam: a string
                    :param    hparam: name of the discrete systematics

                    :type      dtype: a string
                    :param     dtype: name of the data type

                    :type    default: a dictionary
                    :param   default: keys/default values of discrete systematics

                    :return  setvalues: a dictionary
                    :        setvalues: keys/values of discrete systematics for the set

                    :return      setid: a string
                    :            setid: systematic values separated by '_' as the id of this set
                '''

                setvalues = deepcopy (default)
                setvalues[hparam] = set_value
                ### modification for muons
                if 'muon' in dtype:
                        ## if absorption / scattering: oversizing = 3
                        if hparam in ['absorption', 'scattering']: setvalues['oversizing'] = 3
                        ## if domeff / forward: holeice = 30
                        if hparam in ['domeff', 'forward']: setvalues['holeice'] = 30
                ### modification for off axis bulkice sets
                if hparam == 'absorption' and set_value in [0.929, 1.142]:
                        setvalues['scattering'] = set_value

                ### define ID for this set
                setid = self._get_setid (setvalues)
                return setid, setvalues
        
        def _collect_base_members (self):

                ''' collect all baseline members

                    :return  members: a Map object
                             members: contains all members objects
                '''
                
                members = Map ({})
                for dtype in self._dtypes:
                        members[dtype] = Member (dtype, self._pdictpath,
                                                 ranges=self._ranges,
                                                 baseline=True )
                return members

        def _collect_sys_members (self, dtype, hparam, default, has_bulkice):

                ''' collect all set members for a specific data type
                    and a specific discrete parameter

                    :type     hparam: a string
                    :param    hparam: name of the discrete systematics

                    :type      dtype: a string
                    :param     dtype: name of the data type

                    :type    default: a dictionary
                    :param   default: keys/default values of discrete systematics

                    :type    has_bulkice: a boolean
                    :param   has_bulkice: if True, include bulk ice off axis points

                    :return  members: a Map object
                             members: 99contains all members objects
                '''
                
                set_members = Map ({})
                sets = get_sets (dtype, hparam, has_bulkice=has_bulkice)
                for s in sets:
                        setid, setvalues = self._get_setvalues (s, dtype, hparam, default)
                        isdef = self._check_setid (setid, default)
                        ## don't waste time if it is the default set and already stored
                        if setid in set_members:
                                if not defid == setid:
                                        message = 'Library:'+fname+' : WARNING : setid (' + \
                                                  setid + ') already exist in the dictionary !'
                                        print ('{0}'.format (message))
                                continue
                        set_members [setid] = Member (dtype, self._pdictpath,
                                                      ranges=self._ranges,
                                                      baseline=False,
                                                      sysvalues=setvalues)
                return set_members

        def _get_histogram (self, member, params,
                            isbaseline=False, matter=True, oscnc=False):

                ''' get one histogram from one member

                    :type    member: a Member object
                    :param   member: events to be histogrammed

                    :type    params: a dictionary
                    :param   params: values of floating parameters

                    :type   isbaseline: boolean
                    :param  isbaseline: if True, use self.weighters
                                        if False, redefine weighters for sys sets

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events

                    :return   hdict: a dictionary
                              hdict: 'H' = histogram weighted by weight
                                     'H2' = histogram weighted by weight**2
                '''

                dtype = member.get_dtype ()
                if isbaseline:
                        ## BASELINE: weighters in self
                        if not hasattr (self, 'weighters'):
                                self.set_weighters (params, matter=matter, oscnc=oscnc)
                        weighter = self.weighters [dtype]
                else:
                        ## NOT BASELINE: one-time weighter
                        pmap = self.probmaps[dtype[:-2]] if 'nu' in dtype else None
                        weighter = member.get_weighter (params, matter=matter,
                                                        oscnc=oscnc, pmap=pmap)
                        
                weights = member.get_weights (params, weighter=weighter)
                H, H2 = member.get_histogram (self._edges, weights=weights)

                return Map ({'H':H, 'H2':H2})

        def _get_sys_histograms (self, dtype, params, hparams, default,
                                 matter=True, oscnc=False):

                ''' get all systematic histograms of a data type.
                    
                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type     params: a dictionary
                    :param    params: values of floating parameters

                    :type    hparams: a list
                    :param   hparams: discrete parameters to be included in hplane

                    :type    default: a dictionary
                    :param   default: keys/default values of discrete systematics

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events
                
                    :return  histos: a dictionary
                             histos: all systematic histograms
                '''
                
                histos = Map ({})
                has_bulkice = 'scattering' in hparams and 'absorption' in hparams
                for hp in sorted (hparams):
                        if self._verbose > 1: print ('####    -- {0}'.format (hp))
                        ## get member of this discrete param
                        members = self._collect_sys_members (dtype, hp, default, has_bulkice)
                        ## get histogram from each set
                        for setid in members:
                                if self._verbose > 1: print ('####        -- {0}'.format (setid))
                                # check if it is default set
                                isdef = self._check_setid (setid, default)
                                # don't waste time on redoing default set
                                if isdef and setid in histos: continue
                                # get histogram of this setid
                                histos [setid] = self._get_histogram (members[setid], params,
                                                                      isbaseline=False, 
                                                                      matter=matter, oscnc=oscnc)
                                # define normalization factor
                                if isdef: norm = np.sum (histos[setid]['H'])
                                # apply normalization factor if coin set
                                if hp=='coin':
                                        hsum = np.sum (histos [setid]['H'])
                                        factor = norm/hsum if hsum else 1.0
                                        histos [setid]['H'] *= factor
                        if self._verbose > 1: print ('####')
                return histos
        
        def get_dtypes (self):

                ''' return data types included in this library '''
                
                return self._dtypes

        def get_ranges (self, obs):

                ''' return observable template ranges in this library

                    :type  obs: a string
                    :param obs: observable: e/z/p

                    :return  range: a tuple
                             range: range of the given observable
                '''
                        
                return self._ranges [obs]

        def get_edges (self, obs):
        
                ''' return observable template edges in this library

                    :type  obs: a string
                    :param obs: observable: e/z/p

                    :return  edge: a numpy array 
                             edge: bin edges of this observable in this library.
                '''
                
                return self._edges [obs]

        def set_weighters (self, params,
                           matter=True, oscnc=False):

                ''' set weighters for each member into self.
                    weighters per data type (for all systematic sets)
                    probmaps per numu / nue / nutau (for both CC and
                    NC and all systematic sets)

                    :type    params: a dictionary
                    :param   params: values of floating parameters

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events
                '''
                
                weighters = Map ({})
                pmaps = self.probmaps if hasattr (self, 'probmaps') else Map ({})
                for dtype in self._dtypes:
                        pmap = pmaps [dtype[:-2]] if 'nu' in dtype and \
                               dtype[:-2] in pmaps \
                               else None
                        weighters [dtype] = self._baseline[dtype].get_weighter (params,
                                                                                matter=matter,
                                                                                oscnc=oscnc,
                                                                                pmap=pmap)
                        if 'nu' in dtype: pmaps [dtype[:-2]] = weighters[dtype].probmap 
                self.weighters = weighters
                self.probmaps  = pmaps

        def collect_base_histograms (self, params,
                                     matter=True, oscnc=False):

                ''' get all baseline histograms from all members

                    Note: You might want to have weighters set defined
                          otherwise, it will do it here.

                    :type     params: a dictionary
                    :param    params: values of floating parameters

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events

                    :return  histos: a dictionary
                             histos: all baseline histograms
                '''
                start_time = time.time ()
                
                ## collect baseline histograms
                histos = Map ({})
                for dtype in self._dtypes:
                        if self._verbose > 1: print ('####    -- {0}'.format (dtype))
                        member = self._baseline [dtype]
                        histos [dtype] = self._get_histogram (member, params, isbaseline=True,
                                                              matter=matter, oscnc=oscnc)
                if self._verbose > 1: print ('####')

                dtime = (time.time () - start_time)/60.
                return histos

        def collect_sys_histograms (self, dtype, params, hparams,
                                    matter=True, oscnc=False):

                ''' collect all systematic histograms of a data type

                    Note: You might want to have weighters set defined
                          otherwise, it will do it here.

                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type     params: a dictionary
                    :param    params: values of floating parameters

                    :type    hparams: a list
                    :param   hparams: discrete parameters included in hyperplane

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events
                
                    :return  histos: a dictionary
                             histos: all baseline histograms
                '''

                ## collect sys histogram from all discrete sets of this data type
                default = default_mu_sysvalues if 'muon' in dtype else default_nu_sysvalues
                hparams = sorted ([ param for param in default if param in hparams ])
                histos = self._get_sys_histograms (dtype, params, hparams, default,
                                                   matter=matter, oscnc=oscnc)
                return histos

        def get_hplanes (self, nuparams, matter=True, oscnc=False, verbose=1):

                ''' collect hyperplane objects from all data types
                    Note: hyperplanes are based upon systematic histograms
                          weighted by the seeded MC values in nuparams

                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type   nuparams: a Nuparams object
                    :param  nuparams: user settings of floating parameters

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events
                
                    :type   verbose: an int
                    :param  verbose: If 0, no printout
                                     If 1, print out basic info
                                     If 2, print out info within chi2 fit

                    :return  hplanes: a dictionary
                             hplanes: a dictionary of hyperplane objects
                '''
                
                ## collect systematic histograms
                params = nuparams.extract_params ('seeded')
                ## only for neutrinos and muons
                dtypes = [ dtype for dtype in self._dtypes if dtype[:2] in ['nu', 'mu'] ]
                ## collect systematic information
                syshistos = Map ({})
                if self._verbose > 1: print ('####  collecting sys histograms')
                for dtype in dtypes:
                        if self._verbose > 1: print ('####  -- {0}'.format (dtype))
                        hparams = nuparams.get_hplaned_dparams (dtype)                        
                        syshistos[dtype] = self.collect_sys_histograms (dtype, params,
                                                                        hparams,
                                                                        matter=matter,
                                                                        oscnc=oscnc)
                if self._verbose > 1: print ('####')
                        
                ## special treatment for nunc
                nc = sorted ([ dtype for dtype in dtypes if 'nc' in dtype ])
                if len (nc) > 0:
                        dtypes = sorted ([ dtype for dtype in dtypes
                                           if not 'nc' in dtype ] + ['nunc'])

                ## collect hyperplane objects
                hplanes = Map ({})
                ## get hplanes
                if self._verbose > 1: print ('####  collecting hplanes')
                for dtype in dtypes:
                        if self._verbose > 1: print ('####    -- {0}'.format (dtype))
                        hparams = nuparams.get_hplaned_dparams (dtype)
                        ## hyperplane is built only if at least one discrete parameter
                        if len (hparams) == 0:
                                hplanes[dtype] = None; continue
                        expparams = nuparams.get_exp_dparams (dtype)
                        histos = self.merge_nunc (dtype, syshistos, nc)
                        hplanes[dtype] = HyperPlane (dtype, histos, expparams,
                                                     hparams, verbose=verbose)
                if self._verbose > 1: print ('####')
                return hplanes

        def merge_nunc (self, dtype, histos, ncdtypes):

                ''' merge any nc members into one histogram

                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type  histos: a dictionary
                    :param histos: histograms of all discrete sets from all dtypes

                    :type  ncdtypes: a list 
                    :param ncdtypes: nc data type

                    :return refvalues: a dictionary
                            refvalues: reference values of the discrete parameters

                    :return histos: a dictionary
                            histos: all systematic histograms from this data type
                '''
                
                ## return if not nunc
                if not 'nc' in dtype:
                        return histos[dtype]

                ## massage nc
                nchistos = Map ({})
                for setid in histos[ncdtypes[0]]:
                        nchistos[setid] = {'H': sum ([ histos[ncdtype][setid]['H']
                                                       for ncdtype in ncdtypes ]),
                                           'H2': sum ([ histos[ncdtype][setid]['H2']
                                                        for ncdtype in ncdtypes ]) }
                return nchistos

        def scale_histos (self, histos, params):

                ''' scale histograms by appropriate factors

                    :type  histos: a dictionary
                    :param histos: histogram to be scaled

                    :type  params: a dictionary
                    :param params: values of floating parameters
                '''

                shistos = Map ({})

                for dtype in histos:
                        ## get the normalization factor
                        key = 'atmmu' if 'muon' in dtype else 'noise' if 'noise' in dtype else 'numu'
                        norm = params['norm_nutau'] if 'nutau' in dtype else \
                               params['norm_nc'] if 'nc' in dtype else 1.
                        norm *= params['norm_'+key] * seconds_per_year * params['nyears']
                        ## scale this histogram
                        shistos[dtype] = {'H' : histos[dtype]['H'] * norm,
                                          'H2': histos[dtype]['H2'] * norm**2}
                return shistos
        
        def apply_hplanes (self, bhistos, hplanes, params):

                ''' multiply hyperplane factor to each member template

                    :type  bhistos: a dictionary
                    :param bhistos: all baseline histograms

                    :type  hplanes: a dictionary
                    :param hplanes: hyperplane objects for all data types

                    :type   params: a dictionary
                    :param  params: values of floating parameters

                    :return mhistos: a dictionary
                            mhistos: modified baseline histograms
                '''

                mhistos = Map ({})
                for dtype in self._dtypes:
                        hplane = hplanes['nunc'] if 'nc' in dtype else \
                                 hplanes[dtype] if dtype in hplanes else None
                        ## if no hyperplane, histo same base histo
                        if not hplane:
                                mhistos[dtype] = bhistos[dtype]; continue
                        factors = hplane.apply (params)
                        mhistos[dtype] = {'H' : bhistos[dtype]['H'] * factors,
                                          'H2': bhistos[dtype]['H2'] * factors**2}
                return mhistos
