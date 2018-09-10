#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script is fitter class in which
#### minimization is performed.
####
#### Command line to run:
#### $ python fitter.py
####
####
###############################################################

from __future__ import print_function
from optparse import OptionParser
import numpy as np
import cPickle, iminuit

from misc import Map, Toolbox
from nuparams import Nuparams
from library import Library
from likelihood import Likelihood
from template import Template

###########################################
#### Fitter class
###########################################
class Fitter (object):

    ''' A class to perform minimization '''
    
    def __init__ (self, nuparams, params, lib, LH, temp,
                  fparams=None,
                  verbose=1):

        ''' initialize fitter class

            :type  nuparams: a Nuparams object
            :param nuparams: users setting of nuisance parameters

            :type  params: a dictionary
            :param params: values of parameters

            :type  lib: a Library object
            :param lib: info of all members / dtypes involved

            :type  LH: a Likelihood object
            :param LH: likelihood for minimization

            :type  temp: a Template object
            :param temp: template objects for printing histogram rates

            :type  fparams: a numpy array / list
            :param fparams: osc parmaeters to be fixed

            :type  verbose: int
            :param verbose: If 0: no printout.
                            If 1: minimum printout.
                            If 2: print iteration, oscparams blinded.
                            If 3: print iteration, unblinded.
                            If 4: print info per bin per iteration; oscparams unblineded
        '''
        
        print ('#### ##################################################')
        print ('#### ################### Fitter #######################')
        print ('####')


        self._nuparams = nuparams
        self._params = params
        self._lib, self._LH, self._temp = lib, LH, temp
        self._fparams = fparams
        self._verbose = verbose

        ## define holders
        self.bestfit = Map ({'H':[], 'H2':[], 'ts':[]})
        self.tsvalue = np.inf
        self.barlow_As = None

        ## print info
        if self._verbose > 1: self._print_header ()

        ## minimize !
        self.results = self._minimize ()
        print ('#### ##################################################')

    def _print_header (self):

        ''' print header

            :return params: a list
                    params: names of parameters for printing
        '''

        line = '#### '
        nparams = self._nuparams.get_all_params ()
        rparams = self._rename_params (nparams)
        for i in np.arange (len (rparams)):
            length = max (len (rparams [i]), 3)
            line +='{'+str(i)+':'+str(length+2)+'}|'
        print (line.format (rparams[0].center (len (rparams[0])+2) , rparams[1].center (len (rparams[1])+2) ,
                            rparams[2].center (len (rparams[2])+2) , rparams[3].center (len (rparams[3])+2) ,
                            rparams[4].center (len (rparams[4])+2) , rparams[5].center (len (rparams[5])+2) ,
                            rparams[6].center (len (rparams[6])+2) , rparams[7].center (len (rparams[7])+2) ,
                            rparams[8].center (len (rparams[8])+2) , rparams[9].center (len (rparams[9])+2) ,
                            rparams[10].center(len (rparams[10])+2), rparams[11].center(len (rparams[11])+2),
                            rparams[12].center(len (rparams[12])+2), rparams[13].center(len (rparams[13])+2),
                            rparams[14].center(len (rparams[14])+2), rparams[15].center(len (rparams[15])+2),
                            rparams[16].center(len (rparams[16])+2), rparams[17].center(len (rparams[17])+2),
                            rparams[18].center(len (rparams[18])+2), rparams[19].center(len (rparams[19])+2),
                            rparams[20].center(len (rparams[20])+2), rparams[21].center(len (rparams[21])+2),
                            rparams[22].center(len (rparams[22])+2), rparams[23].center(len (rparams[23])+2),
                            rparams[24].center(len (rparams[24])+2), rparams[25].center(len (rparams[25])+2),
                            rparams[26].center(len (rparams[26])+2), rparams[27].center(len (rparams[27])+2),
                            rparams[28].center(len (rparams[28])+2) ))
        print (line.format ('='*(len (rparams[0])+2) , '='*(len (rparams[1])+2) , '='*(len (rparams[3])+2) ,
                            '='*(len (rparams[3])+2) , '='*(len (rparams[4])+2) , '='*(len (rparams[5])+2) ,
                            '='*(len (rparams[6])+2) , '='*(len (rparams[7])+2) , '='*(len (rparams[8])+2) ,
                            '='*(len (rparams[9])+2) , '='*(len (rparams[10])+2), '='*(len (rparams[11])+2),
                            '='*(len (rparams[12])+2), '='*(len (rparams[13])+2), '='*(len (rparams[14])+2),
                            '='*(len (rparams[15])+2), '='*(len (rparams[16])+2), '='*(len (rparams[17])+2),
                            '='*(len (rparams[18])+2), '='*(len (rparams[19])+2), '='*(len (rparams[20])+2),
                            '='*(len (rparams[21])+2), '='*(len (rparams[22])+2), '='*(len (rparams[23])+2),
                            '='*(len (rparams[24])+2), '='*(len (rparams[25])+2), '='*(len (rparams[26])+2),
                            '='*(len (rparams[27])+2), '='*(len (rparams[28])+2) ))

    def _get_line (self, params, samebools):

        ''' get information about this iteration

            :type  params: a dictionary
            :param params: current values of all parameters

            :type  samebools: a dictionary
            :param samebools: whether the values are the same
                              as before

            :return line: a string
                    line: a line template

            :return rparams: list
                    rparams: names of the parameters

            :return values: list
                    values: values of parameters at
                            this iteraction

            :return sames: list
                    sames: boolean of parameters
                           If True, this parameter value
                           is the same as previous iteration
        '''
        nparams = self._nuparams.get_all_params ()
        rparams = self._rename_params (nparams)

        ## get line template and values and sames
        line, values, sames = '#### ', [], []
        for i, param in enumerate (nparams):
            ## get line template (space for value + space for boolean)
            length = max (len (rparams [i]), 3)
            ext = '{'+str(i*2)+':'+str(length)+'} ' if self._verbose < 3 and \
                  param in ['dm31', 'theta23', 'norm_nutau'] else \
                  '{'+str(i*2)+':'+str(length)+'.'+str(length/2)+'f} '
            line += ext + '{'+str(i*2+1)+':1}|'
            ## get same
            same = samebools [param]
            ## get value
            value = '--' if self._verbose < 3 and param in ['dm31', 'theta23', 'norm_nutau'] else \
                    params [param]
            if not type (value)==str:
                if param == 'dm31': value *= 1000.
                value = round (value, length/2)
            ## append
            values.append (value)
            sames.append (str (same)[0])
            
        return line, values, sames
        
    def _print_line (self, params, samebools):

        ''' print a line of parameter values

            :type  params: a dictionary
            :param params: current values of all parameters

            :type  samebools: a dictionary values are the same
                              as before
        '''

        ## get information about this iteration
        line, values, sames = self._get_line (params, samebools)
        
        ## print information
        print (line.format (values[0] , sames[0] , values[1] , sames[1] ,
                            values[2] , sames[2] , values[3] , sames[3] ,
                            values[4] , sames[4] , values[5] , sames[5] ,
                            values[6] , sames[6] , values[7] , sames[7] ,
                            values[8] , sames[8] , values[9] , sames[9] ,
                            values[10], sames[10], values[11], sames[11],
                            values[12], sames[12], values[13], sames[13],
                            values[14], sames[14], values[15], sames[15],
                            values[16], sames[16], values[17], sames[17],
                            values[18], sames[18], values[19], sames[19],
                            values[20], sames[20], values[21], sames[21],
                            values[22], sames[22], values[23], sames[23],
                            values[24], sames[24], values[25], sames[25],
                            values[26], sames[26], values[27], sames[27],
                            values[28], sames[28] ))

    def _get_penalties (self, penalties):

        ''' get penalty values

            :type  penalties: a dictionary
            :param penalties: penalties of the parameters

            :return rparams: a list
                    rparams: names of parameters

            :return values: a list
                    values: values of penalties
        '''

        nparams = self._nuparams.get_all_params ()
        rparams = self._rename_params (nparams)

        ## get penalty values
        line, values = '#### ', []
        for i in np.arange (len (rparams)):
            length = max (len (rparams[i]), 3)
            has_penalty = nparams[i] in penalties
            # line template
            ext = '{'+str(i)+':'+str(length+2)+'}|' if not has_penalty else \
                  '{'+str(i)+':'+str(length+2)+'.'+str((length+2)/2)+'f}|'
            line += ext
            # penalty value
            value = np.round (penalties [nparams[i]], (length+2)/2) \
                    if has_penalty else '--'
            values.append (value)

        return line, values
            
    def _print_penalties (self, ts, penalties):

        ''' print penalties

            :type  ts: a float
            :param ts: test statistics value of this iteration

            :type  penalties: a dictionary
            :param penalties: penalties of the parameters
        '''
        
        line, values = self._get_penalties (penalties)
        ## print penalty values
        print (line.format (values[0] , values[1] , values[2] , values[3] ,
                            values[4] , values[5] , values[6] , values[7] ,
                            values[8] , values[9] , values[10], values[11],
                            values[12], values[13], values[14], values[15],
                            values[16], values[17], values[18], values[19],
                            values[20], values[21], values[22], values[23],
                            values[24], values[25], values[26], values[27],
                            values[28] ))
        print ('#### test statistic value: {0}'.format (2*ts))
        print ('#### '+'-'*280)

    def _print_rates (self, uhistos, template, ts):
        
        ''' print histogra rate info for this iteration
        
            :type  uhistos: a dictionary
            :param uhistos: updated histograms from all dtypes
            
            :type  template: a dictionary
            :param template: MC histogram {'H':H, 'H2':H2}
            
            :type  ts: a float
            :param ts: raw test statistic value before penalty
        '''
            
        self._temp._print_rates ('hplaned', uhistos)
        self._temp._print_rates ('template', template)
        self._temp._print_rates ('data', self._temp.dhisto)
        print ('#### {0}: {1}'.format (self._LH._method, 2*ts))
        print ('#### {0}'.format ('-'*280))
        
    def _minimize (self):

        ''' minimize ts function via migrad()

            :return m: a dictionary
                    m: minimizer result
        '''

        kwargs = self._get_settings ()
        m = iminuit.Minuit (self.ts_func,
                            **kwargs)
        m.strategy = 1
        m.tol      = 1e-30
        m.errordef = 0.5
        m.migrad (ncall=10000)
        if self._verbose > 1: print ('####')
        return m

    def _get_settings (self):

        ''' minimzer settings from users

            :return settings: a dictionary
                    settings: user minimizer settings
        '''

        settings = {}
        params = sorted (self._nuparams.get_all_params ())
        ## define settings
        for i, param in enumerate (params):
            user = self._nuparams (param)
            settings [param] = user.value
            if param in self._fparams or not user.included:
                settings ['fix_'+param]   = True
            elif user.included:
                settings ['limit_'+param] = (user.lower_limit, user.upper_limit)
                settings ['error_'+param] = user.error
        settings ['print_level'] = 1 if self._verbose > 3 else -1
        settings ['errordef']    = 0.5
        return settings
        
    def _rename_params (self, params):

        ''' rename parameters for printing

            :type  params: list
            :param params: list of parameter names
        '''
        
        param_names = []
        for param in params:
            name = param[:3] if param in ['absorption', 'scattering'] else \
                   param.split ('_')[1] if 'axm' in param else \
                   param[:-6] if param=='nue_numu_ratio' else \
                   param[5:] if 'barr' in param else \
                   param.replace ('norm', 'N') 
            param_names.append (name)
        return param_names
        
    def _check_params (self, params):

        ''' check if current parameter values the same
            as values at previous minimization steps
        
            :type  params: a dictionary
            :param params: current parameter values
        '''

        same = {}
        for param in self._params:
            same [param] = params [param] == self._params [param]
        if self._verbose > 1: self._print_line (params, same)
        return 
    
    def _add_penalties (self, ts, params):

        ''' add penalties to test statistics

            :type  ts: a float
            :param ts: test statistic value of this iteration
                       before adding penalties

            :type  params: a dictionary
            :param params: values of parameters for this round

            :return ts: a float
                    ts: test statistics value of this iteration
                        after adding penalties
        '''
        
        penalties = {}
        priors = self._nuparams.extract_params ('prior')
        sigmas = self._nuparams.extract_params ('penalty')
        for param in priors:
            ## only for floating parameters
            if self._nuparams (param).included:
                ## Gaussian penalty assumed
                penalty = 0.5 * ((params[param] - priors[param])/sigmas[param])**2
                ts += penalty
                ## save info for printing
                penalties [param] = penalty
        if self._verbose > 1: self._print_penalties (ts, penalties)
        return ts

    def ts_func (self, dm31, theta23, theta13,
                 nyears, gamma, nue_numu_ratio, muon_flux,
                 barr_nu_nubar, barr_nubar_ratio, barr_uphor_ratio, 
                 norm_noise, norm_numu, norm_nc, norm_atmmu, 
                 norm_nugen, norm_nugenHE, norm_corsika, norm_nutau,
                 domeff, holeice, forward, coin, absorption, scattering,
                 axm_res, axm_qe, DISa_nu, DISa_nubar, spe_corr):

        ''' determine ts value. This function will be looped many many times. '''

        ### recognize, check, and print parameters
        local = locals ()
        params = {}
        for p in local:
            if p in self._nuparams.get_all_params ():
                params[p] = local[p]
        self._check_params (params)

        ### update histogram with these params
        uhistos = self._lib.collect_base_histograms (params)
        uhistos = self._lib.apply_hplanes (uhistos, self._temp.hplanes, params)
        uhistos = self._lib.scale_histos (uhistos, params)
        template = Map ({  'H':sum ([uhistos[dtype]['H']  for dtype in uhistos]),
                          'H2':sum ([uhistos[dtype]['H2'] for dtype in uhistos]) })

        ### calculate test statistics
        self._LH.set_histos (uhistos)
        rawts, bints, As = self._LH.get_ts ()

        ### add prior terms
        ts = self._add_penalties (rawts, params)

        ### print outs
        if self._verbose > 3:
            self._print_rates (uhistos, template, rawts)
        ### save results if needed
        if ts < self.tsvalue:
            self.tsvalue = ts
            self.bestfit = Map ({ 'H':template['H'], 'H2':template['H2'], 'ts':bints })
            self.barlow_As = As

        ### update internal parameter values
        self._params = params
        return ts
