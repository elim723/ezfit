#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contains a Nuparam class to
####    - read/store/check a nuisance text file
####    - provide access for user settings
####
#### Note: this class must be modified if new systematic
####       parameters are added/removed.
####
######################################################################

from __future__ import print_function
import numpy as np

from misc import Map, InvalidArguments

###########################################################################
#### define constants
###########################################################################
from misc import discrete_parameters
from probmap import default_params as default_oscparams

cont_values = [('name'       ,str  ), ('seeded_mc',float), ('injected_data',float),
               ('included'   ,bool ), ('value'    ,float), ('lower_limit'  ,float),
               ('upper_limit',float), ('error'    ,float), ('prior'        ,float),
               ('penalty'    ,float)]
disc_values = [('name'     ,str  ), ('nu_func'      ,str  ), ('mu_func'    ,str  ),
               ('seeded_mc',float), ('injected_data',float), ('included'   ,bool ),
               ('value'    ,float), ('lower_limit'  ,float), ('upper_limit',float),
               ('error'    ,float), ('prior'        ,float), ('penalty'    ,float)]

###########################################################################
#### define functions needed
###########################################################################
### a short function function to change arg type
f = lambda p: eval (p) if p in ['True', 'False'] else float(p)

def read_param (values, isinverted, *args):
        ''' A function to interprete a line defined for a given parameter

            Parameters
            ----------
            :value :list
                   :either cont_values or disc_values
        
            :isinverted :boolean
                        :if true, dm31 is inverted

            :*args      :an array
                        :a split line from nuparam_textfile

            For a continuous param
            ----------------------
            required: [0] : name       ; [1] : seeded_mc; [2] : injected_data;
                      [3] : included   ; [4] : value    ; [5] : lower_limit  ;
                      [6] : upper_limit; [7] : error        
            optional: [8] : prior   ; [9] : penalty

            For a discrete param
            --------------------
            required: [0] : name     ; [1] : nu_func      ; [2] : mu_func    ;
                      [3] : seeded_mc; [4] : injected_data; [5] : included   ;
                      [6] : value    ; [7] : lower_limit  ; [8] : upper_limit;
                      [9] : error        
            optional: [10]: prior    ; [11]: penalty
        
            Returns
            -------
            :pdict :a Map object
                   :a dictionary with user's setting for this parameter

            Note
            ----
            Ordering of parameters and columns in textfile are set to be the same.

            Example
            -------
            In[0]: from misc import read_param, disc_values, cont_values, discrete_parameters
            In[1]: textfile = open ('nuisance_textfiles/nuparams_template.txt', "r")
            In[2]: for line in (raw.strip().split() for raw in textfile):
            -----:      if not line or line[0][0]=='#': continue
            -----:      break
            In[3]: values = disc_values if line[0] in discrete_parameters else cont_values
            In[4]: this_param = read_param (values, True, line)
        '''
        param = Map ({})
        args  = args[0]
        
        ## check length of arguments
        if not (len(args) == len(values) or len(args) == len(values)-2):
                message = 'nuparams:read_param :: ' + args[0] + ' ('+str(len(values)) + \
                          '): unexpected number of arguements (' + str (len(args)) + ')'
                raise InvalidArguments (message)

        ## loop through each arguement
        for index, value in enumerate (values):
                ## skip first column (name)
                #if index==0: continue
                ## skip prior / penalty if not given
                if index in np.arange (len (values))[-2:] and len(args) == len(values)-2: continue
                vname, vtype = value
                setting = args[index] if index==0 or 'func' in vname else f(args[index])
                ## check if type is correct
                if not vtype == type (setting):
                        message = 'nuparams:read_param :: ' + args[0] + ' ' + vname + ' ('+ vtype + \
                                  '): unexpected arguement type (' + type(setting) + ')'
                        raise InvalidArguments (message)
                ## check if correct function name for nu/mu_func
                if 'func' in vname:
                        if not setting in ['linear', 'parabola', 'exp']:
                                message = 'nuparams:read_param :: ' + args[0] + ' ' + vname + \
                                          ': unexpected function name (' + type(setting) + ')'
                                raise InvalidArguments (message)
                ## modify values if inverted
                if isinverted and args[0]=='dm31':
                        if value in ['seeded_mc', 'injected_data', 'value', 'lower_limit', 'upper_limit']:
                                setting *= -1.
                ## put into param
                param[vname] = setting
        param['isdiscrete'] = values==disc_values
        ## if inverted, reverse dm31 limit
        if isinverted and args[0]=='dm31':
                limits = (param['lower_limit'], param['upper_limit'])
                param['lower_limit'], param['upper_limit'] = limits[1], limits[0]
        return param

###########################################################################
#### Nuparams class
###########################################################################
class Nuparams (object):

        ''' A class to store parameter info from a text file
        
            Given a nuparam textfile, this class check, save,
            print parameter information given by the user.
        
            Example
            -------
            In [0]: from misc import Nuparams
            In [1]: nuparams = Nuparams ('nunuisance_textfiles/nuparams_template.txt')
            In [2]: print nuparams ('dm31').value
            Out[2]: 0.002526
        '''
        
        def __init__ (self, nuparam_textfile, isinverted=False):
                '''Initialize a nuparam set given by the user via a text file
                
                :type   nuparam_textfile: string
                :param  nuparam_textfile: path to the nuisance parameter textfile.

                :type   isinverted: boolean
                :param  isinverted: if true, dm31 is converted
                '''
                
                self._textfile = nuparam_textfile
                self._isinverted = isinverted
                self._params = self.read_nuparams ()
                
        def __call__ (self, param):
                '''Return the user setting of a given param.

                :type   param: string
                :param  param: name of a given parameter.
                
                :return dictionary: a python dictionary.
                '''

                if not param in self._params:
                        message = 'nuparams:Nuparams :: your param ' + \
                                  param +' is not specified in Nuparams object.'
                        raise InvalidArguments (message)

                return self._params[param]

        def __str__ (self):

                '''Print out tables of nuisance parameter settings.'''

                cparams = np.array ([ p for p in self._params if not self._params[p]['isdiscrete'] ])
                dparams = np.array ([ p for p in self._params if self._params[p]['isdiscrete'] ])

                ## print out continuous params
                print ('#### your continuous nuisance parameters set up:')
                print ('####')
                ## cont params header
                header = '####   '
                for i in np.arange (len (cont_values)):
                        header += '| {'+str(i)+':16} '
                header += '|'
                print (header.format (cont_values[0][0].center(16),
                                      cont_values[1][0].center(16),
                                      cont_values[2][0].center(16),
                                      cont_values[3][0].center(16),
                                      cont_values[4][0].center(16),
                                      cont_values[5][0].center(16),
                                      cont_values[6][0].center(16),
                                      cont_values[7][0].center(16),
                                      cont_values[8][0].center(16),
                                      cont_values[9][0].center(16) ))
                divider = '####   '+'| {0} '*len (cont_values)+'|'
                print (divider.format ('='*16))
                ## cont params setting
                for param in sorted (cparams):
                        pdict = self._params[param]
                        pline = '####   '
                        for i in np.arange (len (pdict)-1):
                                line = '| {'+str(i)+':16} ' if i in [0, 3] else '| {'+str(i)+':16.7f} '
                                pline += line
                        pline += '|'
                        ## param with prior given
                        if len (pdict)-1 == len (cont_values):
                                print (pline.format (pdict[cont_values[0][0]],
                                                     pdict[cont_values[1][0]],
                                                     pdict[cont_values[2][0]],
                                                     str (pdict[cont_values[3][0]]),
                                                     pdict[cont_values[4][0]],
                                                     pdict[cont_values[5][0]],
                                                     pdict[cont_values[6][0]],
                                                     pdict[cont_values[7][0]],
                                                     pdict[cont_values[8][0]],
                                                     pdict[cont_values[9][0]] ))
                        else:
                                print (pline.format (pdict[cont_values[0][0]],
                                                     pdict[cont_values[1][0]],
                                                     pdict[cont_values[2][0]],
                                                     str (pdict[cont_values[3][0]]),
                                                     pdict[cont_values[4][0]],
                                                     pdict[cont_values[5][0]],
                                                     pdict[cont_values[6][0]],
                                                     pdict[cont_values[7][0]] ))
                print ('####') # end continous param print out
                                
                ## print out discrete params
                print ('#### your discrete nuisance parameters set up:')
                print ('####')
                ## discrete params header
                header = '####   '
                for i in np.arange (len (disc_values)):
                        header += '| {'+str(i)+':13} '
                header += '|'
                print (header.format (disc_values[0][0].center(13),
                                      disc_values[1][0].center(13),
                                      disc_values[2][0].center(13),
                                      disc_values[3][0].center(13),
                                      disc_values[4][0].center(13),
                                      disc_values[5][0].center(13),
                                      disc_values[6][0].center(13),
                                      disc_values[7][0].center(13),
                                      disc_values[8][0].center(13),
                                      disc_values[9][0].center(13),
                                      disc_values[10][0].center(13),
                                      disc_values[11][0].center(13) ))
                divider = '####   '+'| {0} '*len (disc_values)+'|'
                print (divider.format ('='*13))
                ## disc params setting
                for param in sorted (dparams):
                        pdict = self._params[param]
                        pline = '####   '
                        for i in np.arange (len (pdict)-1):
                                line = '| {'+str(i)+':13} ' if i in [0, 1, 2, 5] else \
                                       '| {'+str(i)+':13.5f} '
                                pline += line
                        pline += '|'
                        ## param with prior given
                        if len (pdict)-1 == len (disc_values):
                                print (pline.format (pdict[disc_values[0][0]],
                                                     pdict[disc_values[1][0]],
                                                     pdict[disc_values[2][0]],
                                                     pdict[disc_values[3][0]],
                                                     pdict[disc_values[4][0]],
                                                     str (pdict[disc_values[5][0]]),
                                                     pdict[disc_values[6][0]],
                                                     pdict[disc_values[7][0]],
                                                     pdict[disc_values[8][0]],
                                                     pdict[disc_values[9][0]],
                                                     pdict[disc_values[10][0]],
                                                     pdict[disc_values[11][0]] ))
                        else:
                                print (pline.format (pdict[disc_values[0][0]],
                                                     pdict[disc_values[1][0]],
                                                     pdict[disc_values[2][0]],
                                                     pdict[disc_values[3][0]],
                                                     pdict[disc_values[4][0]],
                                                     str (pdict[disc_values[5][0]]),
                                                     pdict[disc_values[6][0]],
                                                     pdict[disc_values[7][0]],
                                                     pdict[disc_values[8][0]],
                                                     pdict[disc_values[9][0]] ))
                        
                return ('####') # end discrete param print out
        
        def read_nuparams (self):
                ''' Read text file and store information
                
                    :return: a dictionary of user's parameter settings
                '''
                
                params = Map({})
                textfile = open (self._textfile, "r") # open text file

                # loop through each line in the text file
                for line in (raw.strip().split() for raw in textfile):
                        # skip this line if no character or first character is #
                        if not line or line[0][0]=='#': continue
                        pname = line[0]
                        # check cont or disc values to be used
                        values = disc_values if pname in discrete_parameters else cont_values
                        params[pname] = read_param (values, self._isinverted, line)
                
                textfile.close() # close text file
                return params

        def diff_injected_seeded (self):

                ''' Is any injected value different from seeded ?
                
                    :return: a boolean
                             If True, injected are the same as seeded
                '''

                return not self.extract_params ('seeded_mc') == self.extract_params ('injected_data')
                
        def get_active_dparams (self):

                ''' Obtain a list of discrete parameters that is turned on

                    :return: a list of names of active discrete parameters
                '''
                
                return sorted ([ p for p in self._params if self._params[p]['isdiscrete'] and \
                                 self._params[p]['included'] ])

        def get_exp_dparams (self, dtype):

                ''' Obtain a list of discrete parameters for dtype that has exponential function

                    :type  dtype: a string
                    :param dtype: name of a data type

                    :return: a list of names of active discrete parameters
                '''

                dparams = self.get_active_dparams ()
                return sorted ([ p for p in dparams if 'exp' in self._params[p][dtype[:2]+'_func'] ])

        def get_all_params (self):

                ''' Obtain a list of parameters

                    :return: a list of names of all available (float or fixed) parameters
                '''
                
                return sorted ([ p for p in self._params ])

        def extract_params (self, key, osconly=False):

                ''' extract parameters into a dictionary

                    :type       key: a string
                    :param      key: the key of the dictioary
                                     seed_mc / injected_data / included / etc

                    :type   osconly: a boolean
                    :param  osconly: if True, return only oscillation parameters

                    :return  params: a dictionary
                             params: parameter values
                '''
                
                params = {}
                for param in self.get_all_params ():
                        collect = False if osconly and not param in default_oscparams else True
                        if collect: params[param] = self.__call__ (param)[key]
                return params
