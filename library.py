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

from misc import get_sets, Map, InvalidArguments, Toolbox
from weightcalculator import WeightCalculator
from hyperplane import HyperPlane
from member import Member

####################################################################
#### import constants needed
####################################################################
from misc import datatypes, seconds_per_year, default_edges, discrete_parameters
from misc import default_ranges, default_nu_sysvalues, default_mu_sysvalues
toolbox = Toolbox ()

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
            In [4] : params = nuParams.extract_params ('seeded_mc')
            In [5] : lib.set_weighters (params, oscnc=False)
            In [6] : basehistos = lib.collect_base_histograms (params)
            NOTE: weighters for baseline members are stored in Library.
            
            Example to get all systematic histograms for a specific data type
            -----------------------------------------------------------------
            In [7] : dparams = nuParams.get_active_dparams () 
            In [8] : refvalues, histos = lib.collect_sys_histograms ('numucc', params, dparams)
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
                      edges=default_edges ):

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
                '''
                
                self._dtypes = dtypes
                self._ranges = ranges
                self._edges = edges
                self._pdictpath = pdictpath
                self._check_args ()
                
                ## a library must have baseline sets
                self._baseline = self._collect_base_members ()

        def __call__ (self, dtype):

                ''' return baseline event info (no weights) in this library '''

                if dtype not in self._baseline:
                        message = 'Library:__call__ :: '+dtype+' is not registerd.'
                        raise InvalidArguments (message)
                
                return self._baseline [dtype]

        def _check_dtype (self, fname, dtypes):

                ''' check dtype type and valid

                    :type  fname: string
                    :param fname: function name that calls _check_dtype

                    :type  dtype: a list
                    :param dtype: name to be checked
                '''

                if not toolbox.is_array (dtypes):
                        message = 'Library:'+fname+' :: dtypes must be a list / array.'
                        raise InvalidArguments (message)
                
                for dtype in dtypes:
                        if not dtype in datatypes:
                                message = 'Library:'+fname+' :: '+dtype+' not registed as data types.'
                                raise InvalidArguments (message)
                
        def _check_args (self):

                ''' check argument inputs '''

                ## check data types
                self._check_dtype ('check_args', self._dtypes)
                ## check dictionaries
                for arg in ['ranges', 'edges']:
                        if not toolbox.is_dict (eval ('self._' + arg)):
                                message = 'Library:check_args :: '+arg+' must be a dictionary/'
                                raise InvalidArguments (message)
                return

        def _check_setid (self, fname, dictionary, setid, refvalues):

                ''' check if setid already exists in a dictionary

                    :type  fname: a string
                    :param fname: name of the function calling _check_setid

                    :type  dictionary: a Map / python dictionary
                    :param dictionary: dictionary to be checked

                    :type  setid: a string
                    :param setid: systematic values separated by '_' as the id of this set

                    :type  refvalues: a dictionary
                    :param refvalues: keys/values of reference set

                    :return  isref: a boolean
                    :        isref: If True, this is a reference set
                '''

                refid = self._get_setid (refvalues)
                
                if setid in dictionary and not refid == setid:
                        message = 'Library:'+fname+' : WARNING : setid (' + setid + ') already exist in the dictionary !'
                        print ('{0}'.format (message))
                        message = 'Library:'+fname+' : WARNING : going to over write for now ..'
                        print ('{0}'.format (message))
                return refid == setid

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
        
        def _get_sysvalues (self, set_value, dtype, dparam, sysvalues):

                ''' get the sysvalues dictionary for the specific discrete set

                    :type     set_value: a float
                    :param    set_value: value of one of the dparam sets

                    :type     dparam: a string
                    :param    dparam: name of the discrete systematics

                    :type      dtype: a string
                    :param     dtype: name of the data type

                    :type    sysvalues: a dictionary
                    :param   sysvalues: default nominal keys/values of discrete systematics

                    :return  sysvalues: a dictionary
                    :        sysvalues: keys/values of discrete systematics for the set

                    :return      setid: a string
                    :            setid: systematic values separated by '_' as the id of this set
                '''
                
                sysvalues = deepcopy (sysvalues)
                sysvalues[dparam] = set_value
                ### modification for muons
                if 'muon' in dtype:
                        ## if absorption / scattering: oversizing = 3
                        if dparam in ['absorption', 'scattering']: sysvalues['oversizing'] = 3
                        ## if domeff / forward: holeice = 30
                        if dparam in ['domeff', 'forward']: sysvalues['holeice'] = 30
                ### modification for off axis bulkice sets
                if dparam == 'absorption' and set_value in [0.929, 1.142]:
                        sysvalues['scattering'] = set_value

                ### define ID for this set
                setid = self._get_setid (sysvalues)
                return setid, sysvalues
        
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

        def _collect_sys_members (self, dtype, dparam, default_sysvalues, has_bulkice):

                ''' collect all set members for a specific data type
                    and a specific discrete parameter

                    :type     dparam: a string
                    :param    dparam: name of the discrete systematics

                    :type      dtype: a string
                    :param     dtype: name of the data type

                    :type    default_sysvalues: a dictionary
                    :param   default_sysvalues: nominal keys/values of discrete systematics

                    :type    has_bulkice: a boolean
                    :param   has_bulkice: if True, include bulk ice off axis points

                    :return  members: a Map object
                             members: contains all members objects
                '''
                
                set_members = Map ({})
                sets = get_sets (dtype, dparam, has_bulkice=has_bulkice)
                for s in sets:
                        setid, sysvalues = self._get_sysvalues (s, dtype, dparam, default_sysvalues)
                        isref = self._check_setid ('collect_sys_members', set_members, setid, default_sysvalues)
                        ## don't waste time if it is the ref set and already stored
                        if isref and setid in set_members: continue
                        set_members [setid] = Member (dtype, self._pdictpath,
                                                      ranges=self._ranges,
                                                      baseline=False,
                                                      sysvalues=sysvalues)
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
                        weighter = member.get_weighter (params, matter=matter, oscnc=oscnc, pmap=pmap)
                
                weights = member.get_weights (params, weighter=weighter)
                H, H2 = member.get_histogram (self._edges, weights=weights)
                return Map ({'H':H, 'H2':H2})

        def _get_sys_histograms (self, dtype, params, refvalues, has_bulkice,
                                 matter=True, oscnc=False):

                ''' get all systematic histograms of a data type.
                    
                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type     params: a dictionary
                    :param    params: values of floating parameters

                    :type    refvalues: a dictionary
                    :param   refvalurs: reference values of discrete parameters

                    :type   has_bulkice: a boolean
                    :param  has_bulkice: if True, include bulk ice off axis sets

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events
                
                    :return  histos: a dictionary
                             histos: all systematic histograms
                '''
                
                histos = Map ({})
                for dp in sorted (refvalues):
                        ## get member of this discrete param
                        members = self._collect_sys_members (dtype, dp, refvalues, has_bulkice)
                        ## get histogram from each set
                        for setid in members:
                                isref = self._check_setid ('get_sys_histograms', histos, setid, refvalues)
                                histos [setid] = self._get_histogram (members[setid], params,
                                                                      isbaseline=False, 
                                                                      matter=matter, oscnc=oscnc)
                                # define normalization factor if reference set
                                if isref: norm = np.sum (histos[setid]['H'])
                                # apply normalization factor if coin set
                                if dp=='coin': histos [setid]['H'] *= norm / np.sum (histos [setid]['H'])

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

                if not obs in self._ranges:
                        message = 'Library:get_ranges::'+obs+' is not an observable in this library.'
                        raise InvalidArgument (message)
                        
                return self._ranges [obs]

        def get_edges (self, obs):
        
                ''' return observable template edges in this library

                    :type  obs: a string
                    :param obs: observable: e/z/p

                    :return  edge: a numpy array 
                             edge: bin edges of this observable in this library.
                '''

                if not obs in self._edges:
                        message = 'Library:get_edges::'+obs+' is not an observable in this library.'
                        raise InvalidArgument (message)
                
                return self._edges [obs]

        def set_weighters (self, params,
                           matter=True, oscnc=False):

                ''' set weighters for each member into self.
                    weighters per data type (for all systematic sets)
                    probmaps per numu / nue / nutau (for both CC and NC and all systematic sets)

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
                        pmap = pmaps [dtype[:-2]] if 'nu' in dtype and dtype[:-2] in pmaps \
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

                ## check args
                if not toolbox.is_dict (params):
                        message = 'Library:collect_base_histograms :: params must be a dictionary.'
                        raise InvalidArguments (message)
                
                ## collect baseline histograms
                histos = Map ({})
                for dtype in self._dtypes:
                        member = self._baseline [dtype]
                        histos [dtype] = self._get_histogram (member, params, isbaseline=True,
                                                              matter=matter, oscnc=oscnc)
                return histos

        def collect_sys_histograms (self, dtype, params, dparams,
                                    matter=True, oscnc=False):

                ''' collect all systematic histograms of a data type

                    Note: You might want to have weighters set defined
                          otherwise, it will do it here.

                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type     params: a dictionary
                    :param    params: values of floating parameters

                    :type    dparams: a list
                    :param   dparams: discrete parameters included

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events
                
                    :return  refvalues: a dictionary
                             refvalues: reference values of discrete systematics

                    :return  histos: a dictionary
                             histos: all baseline histograms
                '''

                ## check dicts
                if not toolbox.is_dict (params):
                        message = 'Library:get_sys_histograms :: params must be a dictionary.'
                        raise InvalidArguments (message)
                ## check list
                if not toolbox.is_array (dparams):
                        message = 'Library:get_sys_histograms :: dparams must be a list / array.'
                        raise InvalidArguments (message)

                ## collect sys histogram from all discrete sets of this data type
                refvalues = default_mu_sysvalues if 'muon' in dtype else default_nu_sysvalues 
                has_bulkice = 'scattering' in dparams and 'absorption' in dparams
                histos = self._get_sys_histograms (dtype, params, refvalues, has_bulkice,
                                                   matter=matter, oscnc=oscnc)
                return refvalues, histos

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
                
                ## collect hyperplane objects
                hplanes = Map ({})
                dparams = nuparams.get_active_dparams ()
                ## hyperplane is built only if at least one discrete parameter
                if len (dparams) == 0: return hplanes
                
                ## floating parameter values from which weights
                ## are calculated for all systematic histograms
                params = nuparams.extract_params ('seeded_mc')
                ## only for neutrinos and muons
                dtypes = [ dtype for dtype in self._dtypes if dtype[:2] in ['nu', 'mu'] ]
                ## collect systematic information
                sysinfo = Map ({})
                for dtype in dtypes:
                        refvalues, histos = self.collect_sys_histograms (dtype, params, dparams,
                                                                         matter=matter, oscnc=oscnc)
                        sysinfo[dtype] = {'refvalues':refvalues, 'histos':histos}
                ## special treatment for nunc
                nc = sorted ([ dtype for dtype in dtypes if 'nc' in dtype ])
                if len (nc) > 0:
                        dtypes = sorted ([ dtype for dtype in dtypes if not 'nc' in dtype ] + ['nunc'])
                ## get hplanes
                for dtype in dtypes:
                        expparams = nuparams.get_exp_dparams (dtype)
                        refvalues, histos = self.merge_nunc (dtype, sysinfo, nc)
                        hplanes[dtype] = HyperPlane (dtype, histos, expparams, refvalues, verbose=verbose) 
                return hplanes

        def merge_nunc (self, dtype, sysinfo, ncdtypes):

                ''' merge any nc members into one histogram

                    :type      dtype: a string
                    :param     dtype: name of data type

                    :type  sysinfo: a dictionary
                    :param sysinfo: includes refvalues and systematic histograms

                    :type  ncdtypes: a list 
                    :param ncdtypes: nc data type

                    :return refvalues: a dictionary
                            refvalues: reference values of the discrete parameters

                    :return histos: a dictionary
                            histos: all systematic histograms from this data type
                '''

                ## return if not nunc
                if not 'nc' in dtype:
                        return sysinfo[dtype]['refvalues'], sysinfo[dtype]['histos']

                ## massage nc
                histos = Map ({})
                for setid in sysinfo[ncdtypes[0]]['histos']:
                        histos[setid] = {'H': sum ([ sysinfo[ncdtype]['histos'][setid]['H'] for ncdtype in ncdtypes ]),
                                         'H2': sum ([ sysinfo[ncdtype]['histos'][setid]['H2'] for ncdtype in ncdtypes ]) }
                return sysinfo[ncdtypes[0]]['refvalues'], histos
                        
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
                                 None if not dtype in hplanes else hplanes[dtype]
                        ## if no hyperplane, histo same base histo
                        if not hplane:
                                mhistos[dtype] = bhistos[dtype]; continue
                        factors = hplane.apply (params)
                        mhistos[dtype] = {'H' : bhistos[dtype]['H'] * factors,
                                          'H2': bhistos[dtype]['H2'] * factors**2}
                return mhistos
