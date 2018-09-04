#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contain several weight related classes:
####
####    1. a WeightCalculator class for neutrino / muon / noise weights
####    2. a Muon
####
####################################################################

from __future__ import print_function
import os, time
import numpy as np
from misc import InvalidArguments, seconds_per_year
from copy import deepcopy
from nuparams import Nuparams
from probmap import ProbMap

#### all reweighting classes
from specorrector import SPEPeakCorrector

####################################################################
#### General Weight Calculator
####################################################################
class WeightCalculator (object):

            ''' A class to calculate weights for all data types

                Example
                -------
                In [0]: from weightcalculator import WeightCalculator
                In [1]: 
            '''

            def __init__ (self, dtype, params, events):

                        ''' Initialize a Weight Calculator object
                          
                            :type     dtype: a string
                            :param    dtype: data type name
                                             nu*cc / nu*nc / noise / muon / data
                         
                            :type    params: a dictionary
                            :param   params: a dictionary containing values of parameters

                            :type    events: a dictionary
                            :param   events: all event variables
                        '''
                        self._dtype    = dtype
                        self._params   = params
                        self._events   = events
                        
                        ## initialize info to store previous values
                        self._info     = {}

            def __getstate__ (self):

                        ''' get state for pickling '''

                        return self.__dict__

            def __setstate__ (self, d):
                                    
                        ''' set state for pickling '''

                        self.__dict__ = d

            def _update (self, pname, new_value):

                        ''' check if this param is updated;
                            this is where self._info update / values get modified
                        
                            :type  pname: a string
                            :param pname: name of the parameter

                            :type  new_value: a float
                            :param new_value: new value of the parameter
                        '''
                        
                        ## if self._info does not have this parameter, register and return True
                        if not self._info.has_key (pname):
                                    if not new_value == self._params [pname]:
                                                message = 'WeightCalc:update : WARNING : {0} different new and default values.'
                                                print (message.format (pname))
                                    self._info[pname] = {'old_value':None, 'new_value':new_value, 'update':True}
                                    return
                                    
                        ## exchange new and old values
                        self._info [pname]['old_value'] = deepcopy (self._info[pname]['new_value'])
                        self._info [pname]['new_value'] = new_value
                        ## if new and old values are different, turn update to True
                        self._info [pname]['update'] = self._info [pname]['old_value'] != new_value

            def _get_spe_factor (self, params, weights):

                        ''' determine SPE correction factors
                            this is where self._info spe-related factors get modified

                            :type    params: a dictionary
                            :param   params: values of floating parameters

                            :type   weights: a 1D numpy array
                            :param  weights: weights of all events
                        '''

                        param = 'spe_corr'
                        self._update (param, params[param])
                        if self._info [param]['update']:
                                    hits = self._events.hits
                                    q_nch = np.divide (hits.qTotal, hits.SRT_nCh)
                                    specorr = SPEPeakCorrector (q_nch, weights)
                                    self._info ['spe_corr']['factor'] = specorr (params ['spe_corr'])
                        
            def get_total_events (self, weights):

                        ''' get total number of expected events
                        
                            :type  weight: a 1D array
                            :param weight: weights

                            :return rate: a float
                                    rate: total number of expecter events from nyears 
                        '''
                        return np.sum (weights) * seconds_per_year * params ['nyears']
            
####################################################################
#### Muon Weight Calculator
####################################################################
class MuonWeighter (WeightCalculator):

            ''' A class to calculate weights for muon

                Example
                -------
                In [0]: from weightcalculator import MuonWeighter
                In [1]: from member import Member
                In [2]: from misc import Nuparams, seconds_per_year
                In [3]: nuparams = Nuparams ('nunuisance_textfiles/nuparams_template.txt')
                In [4]: params = nuparams.extract_params ('seeded')
                In [5]: pdictpath = '/data/condor_builds/users/elims/ezfits/clean_ezfit/pickled_files/'
                In [6]: muon = Member ('muon', pdictpath, baseline=True)
                In [7]: weighter = weightcalculator.MuonWeighter (params, muon._events)
                In [8]: scaling = params['norm_atmmu'] * params['nyears'] * seconds_per_year
                In [9]: muon_rate = scaling * np.sum (weighter.reweight (params))
            '''

            def __init__ (self, params, events):

                        ''' initialize a Weight Calculator object
                          
                            :type    params: a dictionary
                            :param   params: floating parameter values

                            :type    events: a dictionary
                            :param   events: all event variables
                        '''

                        WeightCalculator.__init__ (self, 'muon', params, events)
                        
                        #### muon flux uncertainties (work by ste)
                        import MuonPrimaryUncertainties
                        muflux_table = os.getcwd () + \
                                       'MuonPrimaryUncertainties/Uncertainties/Muon1SigmaUncertaintiesCosZenith.txt'
                        self._muflux = MuonPrimaryUncertainties.Muon_Primary_Spline (kind='linear')

                        ## classify muongun events
                        mc = self._events.mc
                        self._ismuongun = np.logical_or (mc.ismuongun, np.logical_and (~mc.ismuongun, mc.inmuongun)) \
                                          if mc.has_key ('ismuongun') else np.ones (len (mc.e)).astype (bool)
                        
            def _get_factors (self, params, weights):

                        ''' determine muon reweighting factors
                            this is where self._info factors get modified

                            :type  params: a dictionary
                            :param params: values of reweighting parameters
                        
                            :type  weights: a 1D array 
                            :param weights: initial event weights 
                        '''

                        ## get muon flux / norm corsika factors
                        for param in ['muon_flux', 'norm_corsika']:
                                    self._update (param, params[param])
                                    if self._info [param]['update']:
                                                factor = params['norm_corsika'] \
                                                         if param == 'norm_corsika' else \
                                                         1 + params['muon_flux'] * \
                                                         self._muflux (self._events.mc.cz)
                                                self._info [param]['factor'] = factor

            def _get_weights (self, weights):

                        ''' get muon weights reweighted

                            :type  weights: a 1D numpy array
                            :param weights: reweighted muon weights
                        '''

                        ## apply norm_corsika factor
                        weights [~self._ismuongun] *= self._info ['norm_corsika']['factor']
                        ## apply muon_flux factor (shape only)
                        total_w = deepcopy (np.sum (weights))
                        weights *= self._info ['muon_flux']['factor']
                        weights *= total_w / np.sum (weights)
                        return weights
                        
            def reweight (self, params):

                        ''' calculate muon weights given parameters

                            :type  params: a dictionary
                            :param params: values of reweighting parameters
                        '''

                        ## get muon reweighting factors
                        weights = deepcopy (np.array (self._events.w))
                        self._get_factors (params, weights)
                        weights = self._get_weights (weights)

                        ## apply SPE corr factor
                        self._get_spe_factor (params, weights)
                        weights *= self._info ['spe_corr']['factor']
                        return weights

####################################################################
#### Noise Weight Calculator
####################################################################
class NoiseWeighter (WeightCalculator):

            ''' A class to calculate weights for noise

                Example
                -------
                In [0]: from weightcalculator import NoiseWeighter
                In [1]: from member import Member
                In [2]: from misc import Nuparams, seconds_per_year
                In [3]: nuparams = Nuparams ('nunuisance_textfiles/nuparams_template.txt')
                In [4]: params = nuparams.extract_params ('seeded')
                In [5]: pdictpath = '/data/condor_builds/users/elims/ezfits/clean_ezfit/pickled_files/'
                In [6]: noise = Member ('noise', pdictpath, baseline=True)
                In [7]: weighter = weightcalculator.NoiseWeighter (params, noise._events)
                In [8]: scaling = params['norm_noise'] * params['nyears'] * seconds_per_year
                In [9]: noise_rate = scaling * np.sum (weighter.reweight (params))
            '''

            def __init__ (self, params, events):

                        ''' initialize a Weight Calculator object for noise
                          
                            :type    params: a dictionary
                            :param   params: values of floating parameters

                            :type    events: a dictionary
                            :param   events: all event variables
                        '''

                        WeightCalculator.__init__ (self, 'noise', params, events)
            
            def reweight (self, params):

                        ''' calculate noise weights given parameters

                            :type  params: a dictionary
                            :param params: values of reweighting parameters

                            :return weights: a 1D array
                                    weights: noise weights
                        '''

                        weights = deepcopy (np.array (self._events.w))

                        ## apply SPE corr factor
                        self._get_spe_factor (params, weights)
                        weights *= self._info ['spe_corr']['factor']

                        return weights

####################################################################
#### Neutrino Weight Calculator
####################################################################
class NeutrinoWeighter (WeightCalculator):

            ''' A class to calculate weights for noise

                Example
                -------
                In [0]: from weightcalculator import NeutrinoWeighter
                In [1]: from member import Member
                In [2]: from misc import Nuparams, seconds_per_year
                In [3]: nuparams = Nuparams ('nunuisance_textfiles/nuparams_template.txt')
                In [4]: params = nuparams.extract_params ('seeded')
                In [5]: pdictpath = '/data/condor_builds/users/elims/ezfits/clean_ezfit/pickled_files/'
                In [6]: numucc = Member ('noise', pdictpath, baseline=True)
                In [7]: weighter = weightcalculator.NeutrinoWeighter ('numucc', params,
                                                                      numucc._events,
                                                                      matter=True, oscnc=False)
                In [8]: scaling = params['norm_numu'] * params['nyears'] * seconds_per_year
                In [9]: numucc_rate = scaling * np.sum (weighter.reweight (params))
            '''

            def __init__ (self, dtype, params, events,
                          matter=True, oscnc=False, pmap=None):

                        ''' initialize a Weight Calculator object for neutrinos
                          
                            :type     dtype: a string
                            :param    dtype: a class containing all the nuisance parameters user inputs

                            :type    params: a dictionary
                            :param   params: values of floating parameters

                            :type    events: a dictionary
                            :param   events: all event variables

                            :type    matter: boolean
                            :param   matter: if True, matter effect is included

                            :type     oscnc: boolean
                            :param    oscnc: if True, oscillate NC events

                            :type      pmap: a PropMap object
                            :param     pmap: if provided, use the PropMap instead
                        '''

                        WeightCalculator.__init__ (self, dtype, params, events)
                        self._matter = matter
                        self._oscnc  = oscnc
                        ## Prob3 prep
                        self.probmap = pmap if pmap else \
                                       ProbMap (matter=matter, params=params)

                        ## identify event types
                        self._classify_events ()
                        ## Honda flux object
                        from hondamodifier import HondaModifier
                        self._flux_modifier = HondaModifier ()

            @staticmethod
            def axialMassVar (coeff=np.zeros(2), Ma=0.):

                        ''' A static method to modify weights based on axial mass

                            :type  coeff: an array of two elements
                            :param coeff: fitted parameters in reweighting factor vs axial mass sigma

                            :type     Ma: a float
                            :param    Ma: axial mass floating parameter

                            :return factor: a float
                                    factor: reweighted factor
                        '''
                        return 1 + coeff[:,0]*Ma**2 + coeff[:,1]*Ma

            @staticmethod
            def tkDISreweight (a=0., b=1., bjorken_x=np.zeros(1)):
                        
                        ''' A static method to modify weights based on DIS
                            See https://drive.google.com/file/d/0B8TQi1F3KxYmQkwwU0VHOTdnNEU/view

                            :type      a: a float
                            :param     a: a DIS floating parameter

                            :type      b: a float
                            :param     b: a a-dependent coefficient

                            :type  bjorken_x: a 1D numpy array
                            :param bjorken_x: genie x values

                            :return factor: a float
                                    factor: reweighted factor
                        '''
                        return b*bjorken_x**(-a)
                        
            def _classify_events (self):
                                    
                        ''' classify nugen / res / qe / dis nu / dis nubar events '''
                        
                        e, pdg = self._events.mc.e, self._events.mc.pdg
                        misc = self._events.misc
                        
                        self._isnugen = np.array (misc.isnugen) \
                                        if misc.has_key ('isnugen') else \
                                           np.zeros (len (e)).astype (bool)
                        self._isnugenHE = np.logical_and (e>=5000., np.array (misc.isnugen)) \
                                          if misc.has_key ('isnugen') else \
                                          np.zeros (len (e)).astype (bool)
                        notnugen = ~np.logical_or (self._isnugen, self._isnugenHE)
                        self._res = np.logical_and (notnugen, np.array (misc.scattering)==2)
                        self._qe  = np.logical_and (notnugen, np.array (misc.scattering)==3)
                        self._nu_dis    = np.logical_and (notnugen, np.logical_and
                                                          (np.array (misc.scattering)==1, pdg>0))
                        self._nubar_dis = np.logical_and (notnugen, np.logical_and
                                                          (np.array (misc.scattering)==1, pdg<0))
                        
            def _get_osc_factors (self, params, e, cz, pdg):

                        ''' determine neutrino oscillation reweighting factors
                            this is where self._info oscillation-related factors get modified

                            :type    params: a dictionary
                            :param   params: values of floating parameters

                            :type    energy: a 1D numpy array
                            :param   energy: MC truth energy (GeV)

                            :type    coszen: a 1D numpy array
                            :param   coszen: MC truth cos zenith angle

                            :type       pdg: a 1D numpy array
                            :param      pdg: particle encoding (+/- 12/4/6)
                        '''

                        ## oscillate NC if self._oscnc
                        if not self._oscnc and 'nc' in self._dtype:
                                    ## nuenc  : atm_nue = 1.0, atm_numu = 0.0
                                    ## numunc : atm_nue = 0.0, atm_numu = 1.0
                                    ## nutaunc: atm_nue = 0.0, atm_numu = 0.0
                                    atm_nue  = np.ones  (len (e)) if 'nue'  in self._dtype else \
                                               np.zeros (len (e))
                                    atm_numu = np.ones  (len (e)) if 'numu' in self._dtype else \
                                               np.zeros (len (e))
                                    self._info ['oscprob'] = {'atm_nue':atm_nue, 'atm_numu':atm_numu}
                        else:
                                    ## PropMap internally checks if new osc map needs to
                                    ## be regenerated
                                    prob = self.probmap.get_prob (e, cz, pdg, params)
                                    self._info ['oscprob'] = {'atm_nue':prob[:,0],
                                                              'atm_numu':prob[:,1]}
                                                
            def _get_flux_factors (self, params, e, cz, pdg):

                        ''' determine neutrino flux reweighting factors
                            this is where self._info flux-related factors get modified

                            :type    params: a dictionary
                            :param   params: values of reweighting parameters

                            :type    energy: a 1D numpy array
                            :param   energy: MC truth energy (GeV)

                            :type    coszen: a 1D numpy array
                            :param   coszen: MC truth cos zenith angle

                            :type       pdg: a 1D numpy array
                            :param      pdg: particle encoding (+/- 12/4/6)
                        '''

                        for param in ['nue_numu_ratio', 'gamma', 'barr_nubar_ratio', 'barr_uphor_ratio']:
                                    self._update (param, params[param])
                                    if self._info [param]['update']:
                                                if param == 'barr_nubar_ratio':
                                                            factor = self._flux_modifier.mod_ratio_NuBar (e, cz, pdg,
                                                                                                          params['barr_nu_nubar'],
                                                                                                          params['barr_nubar_ratio'])
                                                elif param == 'barr_uphor_ratio':
                                                            factor = self._flux_modifier.mod_ratio_UpHor (e, cz, pdg,
                                                                                                          params['barr_uphor_ratio'])
                                                elif param == 'nue_numu_ratio':
                                                            factor = params['nue_numu_ratio'] 
                                                else: ## gamma
                                                            factor = np.power (e, params['gamma'])
                                                ### define factor
                                                self._info [param]['factor'] = factor

            def _get_xsec_factors (self, params):
                                    
                        ''' determine neutrino cross section reweighting factors
                            this is where self._info xsection-related factors get modified

                            :type    params: a dictionary
                            :param   params: values of reweighting parameters
                        '''

                        misc = self._events.misc
                        genie_x = np.array (misc.genie_x).astype ('complex')
                        mparams = ['DISa_nu', 'DISa_nubar']
                        if 'cc' in self._dtype: mparams += ['axm_res', 'axm_qe']
                        for param in mparams:
                                    self._update (param, params[param])
                                    if self._info [param]['update']:
                                                if 'res' in param or 'qe' in param:
                                                            coeff = np.array (misc.ma_res)[self._res] \
                                                                    if 'res' in param else \
                                                                    np.array (misc.ma_qe)[self._qe]
                                                            factor = self.axialMassVar (coeff = coeff,
                                                                                        Ma = params[param])
                                                else: ## DIS
                                                            b = 1.-1.8073*params['DISa_nubar'] if 'nubar' in param else \
                                                                1.-1.65125*params['DISa_nu']
                                                            isdis = self._nubar_dis if 'nubar' in param else self._nu_dis
                                                            rew = self.tkDISreweight (a = params[param], b = b,
                                                                                      bjorken_x = genie_x[isdis])
                                                            factor = np.absolute (rew)
                                                self._info [param]['factor'] = factor

            def _get_nugen_factors (self, params):

                        ''' determine neutrino nugen reweighting factors
                            this is where self._info nugen-related factors get modified

                            :type    params: a dictionary
                            :param   params: values of reweighting parameters
                        '''

                        for param in ['norm_nugen', 'norm_nugenHE']:
                                    self._update (param, params[param])
                                    if self._info [param]['update']:
                                                self._info [param]['factor'] = params[param]
                                                
            def _get_nu_factors (self, params):

                        ''' determine neutrino weighting factors
                            this is where self._info factors (flux / oscprob) get modified

                            :type  params: a dictionary
                            :param params: values of reweighting parameters
                        '''

                        mc = self._events.mc
                        e, cz, pdg = np.array (mc.e), np.array (mc.cz), np.array (mc.pdg)

                        ## get flux factors
                        self._get_flux_factors (params, e, cz, pdg)
                        ## get oscprob factors
                        self._get_osc_factors  (params, e, cz, pdg)

            def _get_nu_weights (self):

                        ''' get neutrino weights up to oscillation prob

                            :return  weights: a 1D array 
                                     weights: neutrino weights per events
                        '''

                        misc = self._events.misc

                        ## apply honda correction
                        corr = self._flux_modifier.honda_correction (np.log10 (self._events.mc.e))
                        fluxes = {'atm_nue' : misc.fluxes[:,0] * np.array (misc.detector_w) * corr,
                                  'atm_numu': misc.fluxes[:,1] * np.array (misc.detector_w) * corr }

                        ## apply honda related factors
                        for param in ['nue_numu_ratio', 'gamma', 'barr_nubar_ratio', 'barr_uphor_ratio']:
                                    fluxes['atm_nue']  *= self._info [param]['factor']
                                    ## nue_numu_ratio is applied to atm_nue component
                                    if param=='nue_numu_ratio': continue
                                    fluxes['atm_numu'] *= self._info [param]['factor']
                                    
                        ## apply oscillation probability
                        fluxes['atm_nue']  *= self._info ['oscprob']['atm_nue']
                        fluxes['atm_numu'] *= self._info ['oscprob']['atm_numu']

                        ## weights from both atmospheric nue and numu
                        return fluxes['atm_nue'] + fluxes['atm_numu']

            def _get_factors (self, params, weights):

                        ''' determine neutrino reweighting factors
                            this is where self._info factors (xsec / spe / nugen) get modified

                            :type  params: a dictionary
                            :param params: values of reweighting parameters
                        
                            :type  weights: a 1D array 
                            :param weights: initial event weights 
                        '''

                        ## get xsection factors
                        self._get_xsec_factors (params)
                        ## get nugen factors
                        self._get_nugen_factors (params)

            def _get_weights (self, weights):

                        ''' get neutrino weights after xsec / spe / nugen '''

                        ## apply xsec factors
                        mparams = ['DISa_nu', 'DISa_nubar']
                        ## axm are only for cc types
                        if 'cc' in self._dtype: mparams += ['axm_res', 'axm_qe']
                        for param in mparams:
                                    if 'res' in param or 'qe' in param:
                                                key = eval ('self._'+param.split('_') [1])
                                                weights[key] *= self._info [param]['factor']
                                    else: ## DIS
                                                isdis = self._nubar_dis if 'nubar' in param else self._nu_dis
                                                weights[isdis] *= self._info [param]['factor']

                        ## apply nugen factors
                        weights[self._isnugen]   *= self._info ['norm_nugen']['factor']
                        weights[self._isnugenHE] *= self._info ['norm_nugenHE']['factor']
                        
                        return weights
                        
            def reweight (self, params):
                        
                        ''' calculate neutrino weights given parameters
                        
                            :type  params: a dictionary
                            :param params: values of reweighting parameters

                            :return weights: a 1D array
                                    weights: neutrino weights
                        '''

                        ## nu factors up to oscillation probability
                        self._get_nu_factors (params)
                        weights = self._get_nu_weights ()

                        ## nu factors related to xsection / spe / nugen
                        self._get_factors (params, weights)
                        weights = self._get_weights (weights)

                        ## apply SPE corr factor
                        self._get_spe_factor (params, weights)
                        weights *= self._info ['spe_corr']['factor']
                        return weights

