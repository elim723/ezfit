#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contain Member class which
####
####    - loads / selects / stores events of a given memeber
####    - function to define weighter
####    - function to get weights
####    - function to create a histogram
####
####################################################################

from __future__ import print_function
import numpy as np
import os, cPickle
from misc import Map, Toolbox, InvalidArguments
from nuparams import discrete_parameters as dparams
import weightcalculator

####################################################################
#### import constants needed
####################################################################
from misc import datatypes, seconds_per_year, default_ranges, greco_nyears
toolbox = Toolbox ()

####################################################################
#### Member class
####################################################################
class Member (object):

        ''' A class to store all events for generating a MC template.

            Given whichever discrete parameters are included, this
            class create a library with all events needed.

            Example
            -------
            In [0]: from library import Member
            In [1]: pdictpath = '/data/condor_builds/users/elims/ezfits/clean_ezfit/pickled_files/'
            In [2]: numucc = Member ('numucc', pdictpath, baseline=True)
            In [3]: print numucc.events.keys ()
        '''

        def __init__ (self, dtype, pdictpath,
                      ranges=default_ranges,
                      baseline=False, sysvalues={}):

            ''' Initialize an Events object given data types
                and discrete parameters included

                :type   dtype: a string
                :param  dtype: data type name
                               nu*cc / nu*nc / noise / muon / data

                :type   pdictpath: a string
                :param  pdictpath: path to all pickled dictionaries

                :type   ranges: a dictionary of tuples
                :param  ranges: limits of template ranges; events outside are removed

                :type   baseline: boolean
                :param  baseline: if True, get baseline sets

                :type   sysvalues: a python dictionary
                :param  sysvalues: dictionary must contain systematic values:
                                    - domeff
                                    - holeice
                                    - forward
                                    - absorption
                                    - scattering
                                    - coin / oversizing
            '''

            self._dtype      = dtype
            self._isbaseline = baseline
            self._sysvalues  = sysvalues
            self._pdictpath  = pdictpath
            self._ranges     = ranges
            self._check_args ()
            
            self._pfile = self._get_pfile ()
            self._events = self._load_events ()

        def __getstate__ (self):

                ''' get state for pickling '''

                return self.__dict__

        def __setstate__ (self, d):

                ''' set state for pickling '''

                self.__dict__ = d
            
        def get_events (self):
                
                ''' return event dictionary of this member '''

                return self._events
            
        def get_dtype (self):
                
                ''' return data type of this member '''

                return self._dtype

        def get_ranges (self):

                ''' return template ranges for member's template '''

                return self._ranges

        def get_isbaseline (self):

                ''' return whether this member is baseline set '''

                return self._isbaseline

        def get_sysvalues (self):

                ''' return which systematic set this member is '''

                return self._sysvalues

            
        def _check_args (self):

                ''' check argument inputs '''
            
                ## check data type
                if not self._dtype in datatypes:
                        message = 'Member:check_args :: '+self._dtype+' not registed as data types.'
                        raise InvalidArguments (message)

                ## check data vs baseline sets vs sysvalues
                if 'data' in self._dtype:
                        # if data set, ignore baseline / sys values
                        if self._isbaseline:
                                message = 'Member:check_args :: WARNING: getting data; isbaseline ignored.'
                                print ('#### {0}'.format (message))
                        if len (self._sysvalues) > 0:
                                message = 'Member:check_args :: WARNING: getting data; systematic values ignored.'
                                print ('#### {0}'.format (message))
                elif self._isbaseline:
                        # if baseline set, ignore sys values
                        if len (self._sysvalues) > 0:
                                message = 'Member:check_args :: WARNING: getting baseline sets; systematic values ignored.'
                                print ('#### {0}'.format (message))
                else:
                        # if noise, must be baseline set
                        if 'noise' in self._dtype:
                                message = 'Member:check_args :: noise must be baseline set.'
                                raise InvalidArguments (message)
                        # if not baseline set, sysvalues keys within discrete_parameters
                        matches = len ([ sysname for sysname in self._sysvalues.keys () if not sysname in dparams ])
                        if matches > 0:
                                message = 'Member:check_args :: sys parameter name(s) does not match. Check spelling ?'
                                print ('your sysnames: {0}'.format (self._sysvalues.keys ()))
                                raise InvalidArguments (message)
                return

        def _get_pfile (self):

                ''' define and check pickled file
                
                    :return  pfile: a string
                             pfile: full file address to the pickled file
                '''
            
                pname = self._dtype
                if 'data' in self._dtype:
                        pname += '.p'
                elif self._isbaseline:
                        pname += '_baseline.p'
                else:
                        forward = str (self._sysvalues['forward']).replace ('-', 'n')
                        if self._sysvalues['forward'] > 0: forward = 'p' + forward
                        extra = '_coin'   + str (self._sysvalues['coin']) if 'nu' in self._dtype else \
                                '_oversizing' + str (self._sysvalues['oversizing']) 
                        pname +=  '_domeff'  + str (self._sysvalues['domeff'])  + \
                                  '_holeice' + str (self._sysvalues['holeice']) + \
                                  '_forward' + forward + \
                                  '_absorption' + str (self._sysvalues['absorption']) + \
                                  '_scattering' + str (self._sysvalues['scattering']) + \
                                  extra + '.p'
                        
                # check if pickled file exist
                pfile = self._pdictpath + '/' + pname
                if not os.path.exists (pfile):
                        message = 'Member:check_args :: pickled file (' + pname + \
                                  ') does not exist in ' + self._pdictpath
                        raise InvalidArguments (message)
                return pfile

        def _select_events (self, ddict):

                ''' define and apply final cuts

                    :type   ddict: a Map object
                    :param  ddict: a dictionary containing all events
            
                    :return sdict: a Map object
                            sdict: a dictionary containing all selected events
                '''
            
                ### define containment cuts
                recoX, recoY, recoZ = np.array (ddict.reco.X), np.array (ddict.reco.Y), np.array (ddict.reco.Z)
                rho = np.sqrt( (recoX-46.29)*(recoX-46.29) + (recoY+34.88)*(recoY+34.88) )
                cut = recoZ < -230.
                cut *= rho < 140.
                vertex = cut*np.logical_or((recoZ+230.)/(rho-90)<-4.4, rho< 90.)
                boolean = np.logical_and (recoZ>-500, vertex)
                ### define misc cuts
                boolean = np.logical_and (boolean, ddict.hits.SRT_nCh<100.)
                boolean = np.logical_and (boolean, ddict.geo.charge_asym<0.85)
                ### define template range cuts
                e, z, p = np.array (ddict.reco.e), np.array (ddict.reco.z), np.array (ddict.reco.pid)
                ecut = np.logical_and (e>=self._ranges['e'][0], e<self._ranges['e'][1])
                zcut = np.logical_and (z>=self._ranges['z'][0], z<self._ranges['z'][1])
                pcut = np.logical_and (p>=self._ranges['p'][0], p<self._ranges['p'][1])
                tempcut = np.logical_and (np.logical_and (ecut, zcut), pcut)
                boolean = np.logical_and (boolean, tempcut)
                return self.apply_cut (ddict, boolean)
        
        def _load_events (self):

                ''' load events and apply standard cuts
            
                    :return  sdict: a Map object
                             sdict: a dictionary containing all selected events
                '''

                with open (self._pfile, "rb") as f:
                        ddict = cPickle.load(f)
                f.close()
                return self._select_events (ddict)

        def apply_cut (self, ddict, cut):

                ''' apply cut to a dictionary

                    :type    ddict: a Map object
                    :param   ddict: a dictionary containing all events

                    :type      cut: an array of boolean
                    :param     cut: boolean to select events

                    :return  cdict: a Map object
                             cdict: a dictionary containing all selected events
                '''
            
                cdict = Map({})
                for key in ddict.keys():
                        ### deal with arrays
                        cdict[key] = toolbox.chop (ddict[key], cut)
                        ### deal with arrays in dictionary
                        if toolbox.is_dict (ddict[key]):
                                cdict[key] = Map ({})
                                for skey in ddict[key].keys():
                                        cdict[key][skey] = toolbox.chop (ddict[key][skey], cut)
                return cdict
        
        def get_weighter (self, params, matter=True, oscnc=False, pmap=None):

                ''' calculate weights for each events (simulation only)

                    :type    params: a dictionary
                    :param   params: values of floating parameters

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events

                    :type      pmap: ProbMap object
                    :param     pmap: use the provided spline function
                                     instead of creating new one

                    :return weighter: a weightcalculator object
                            weighter: a weight calculator 
                '''

                weighter = None if 'data' in self._dtype else \
                           weightcalculator.NoiseWeighter (params, self._events) if 'noise' in self._dtype else \
                           weightcalculator.MuonWeighter  (params, self._events) if 'muon'  in self._dtype else \
                           weightcalculator.NeutrinoWeighter (self._dtype, params, self._events,
                                                              matter=matter, oscnc=oscnc, pmap=pmap)
                return weighter

        def get_weights (self, params, weighter=None,
                         matter=True, oscnc=False, pmap=None):

                ''' get weights for each events

                    :type  params: a dictionary
                    :param params: values of floating parameters

                    :type  weighter: a WeightCalculator
                    :param weighter: if None, define one here

                    :type    matter: boolean
                    :param   matter: if True, include matter effect

                    :type     oscnc: boolean
                    :param    oscnc: if True, oscillate NC events

                    :type      pmap: ProbMap object
                    :param     pmap: if provided, use the provided spline function
                                     instead of creating new one

                    :return weights: a 1D numpy array
                            weights: weights for each event
                '''

                ## check params
                if not toolbox.is_dict (params):
                        message = 'Member:get_weights :: params must be a dictionary.'
                        raise InvalidArguments (message)

                ## normalization factors
                livetime = seconds_per_year * params ['nyears']

                ## data weights
                if 'data' in self._dtype:
                        ## return equal weights if data
                        greco_livetime = seconds_per_year * greco_nyears
                        return np.ones (len (self._events.reco.e)) / greco_livetime * livetime
                
                ## check weighter
                if weighter and not isinstance (weighter, weightcalculator.WeightCalculator):
                        message = 'Member:get_weights :: weighter must be a WeightCalculator.'
                        raise InvalidArguments (message)
                
                ## define weighter if not parsed
                if not weighter:
                        weighter = self.get_weighter (params, matter=matter, oscnc=oscnc, pmap=pmap)

                ## normalization factors
                key = 'atmmu' if 'muon' in self._dtype else 'noise' if 'noise' in self._dtype else 'numu'
                norm = params['norm_nutau'] if 'nutau' in self._dtype else \
                       params['norm_nc'] if 'nc' in self._dtype else 1
                norm *= params['norm_'+key] * livetime
                
                ## get weights
                return norm * weighter.reweight (params)

        def get_histogram (self, edges, weights=[], params=None):

                ''' get histogram based on template

                    :type   edges: a dictionary
                    :param  edges: {'e': 10**np.linspace (0.75, 1.75, 9),
                                   'z': np.arccos(np.linspace(-1.,1.,11))[::-1],
                                   'p': np.array ([0., 50., 1000.])     }
                
                    :type  weights: a 1D numpy array
                    :param weights: weight for each events
                                    If empty, params must not be None.

                    :type   params: a dictionary
                    :param  params: nuisance parameter values

                    :return H: a 3D numpy array
                            H: histogram based on given edges (weight)

                            H2: a 3D numpy array
                            H2: histogram based on given edges (weight**2)
                '''

                reco = self._events.reco
                e, z, pid = np.array (reco.e), np.array (reco.z), np.array (reco.pid)
                w = self.get_weights (params=params) if len (weights) == 0 else \
                                              np.array (weights)

                #### make sure event variables are finite
                finite = np.logical_and (np.logical_and (np.logical_and (np.isfinite (e), np.isfinite (z)),
                                                         np.isfinite (w)), np.isfinite (pid))

                #### define data/edges/ranges
                data = [ e[finite], z[finite], pid[finite] ]
                hedges = (edges['e'], edges['z'], edges['p'])
                ranges = [ [edges['e'][0], edges['e'][-1]],
                           [edges['z'][0], edges['z'][-1]],
                           [edges['p'][0], edges['p'][-1]] ]
                
                #### build histogram
                H, edge = np.histogramdd (np.array (data).T, hedges, range=ranges, weights=w[finite])
                H2, edge = np.histogramdd (np.array (data).T, hedges, range=ranges, weights=(w[finite])**2)
                
                return H, H2
