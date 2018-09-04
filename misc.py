#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contains miscellaneous classes/functions for ezfit:
####	- a get_sets function to return discrete set values
####    - an Error class for error exception
####	- a Map class for data structure
####    - an Info class to read/store/check user inputs
####    - a Toolbox class to perform array manipilation
####    - a cleverPickler class for pickling complex instances
####
######################################################################

from __future__ import print_function
import numpy as np
import os, types

###########################################################################
##### define all default constants
###########################################################################
## greco data effective livetime
greco_nyears = 2.27

## for histogram binning
eedges           = 10**np.linspace(0.75, 1.75, 9)
zedges           = np.arccos(np.linspace(-1.,1.,11))[::-1]
pedges           = np.array ([0., 50., 1000.])
default_edges = {'e':eedges, 'z':zedges, 'p':pedges}

default_ranges = {'e':(10**0.75, 10**1.75),
                  'z':(0, np.pi),
                  'p':(0, 1000) }

## for hyperplanes
default_nu_sysvalues = {'domeff':1.0, 'holeice':25, 'forward':0,
                        'absorption':1.0, 'scattering':1.0, 'coin':0}
default_mu_sysvalues = {'domeff':0.99, 'holeice':25, 'forward':0,
                        'absorption':1.0, 'scattering':1.0, 'oversizing':1}

global_sysvalues = {'domeff'    :1.0, 'holeice'   :25.,
                    'forward'   :0. , 'absorption':1.0,
                    'scattering':1.0, 'coin'      :0. ,
                    'oversizing':1.                   }

global_sysexps = {'domeff_exp'    :-7., 'holeice_exp'   :0. ,
                  'forward_exp'   :0. , 'absorption_exp':3. ,
                  'scattering_exp':0. , 'coin_exp'      :0. ,
                  'oversizing_exp':0.                   }

## all possible valid data types
datatypes = sorted (['numucc', 'numunc', 'nuecc', 'nuenc', 'nunc',
                     'nutaucc', 'nutaunc', 'noise', 'muon', 'data'])

discrete_parameters = sorted (np.unique ([p \
                                          for keys in [default_nu_sysvalues, default_mu_sysvalues] for \
                                          p in keys.keys () ]))

seconds_per_year = 3600.*24.*365.24

###########################################################################
##### a get_sets function to return discrete set values
###########################################################################
neutrino_domeff     = [0.88, 0.94, 0.97, 1.0, 1.03, 1.06, 1.12]
muon_domeff         = [0.69, 0.79, 0.99]
neutrino_holeice    = [15, 20, 25, 30, 35]
muon_holeice        = [15, 25, 30]
neutrino_forward    = [-5, -3, -1, 0, +1, +2]
muon_forward        = [-2, -4, 0]
global_neutrino_absorption = [1.0, 1.1]
neutrino_scattering = [1.0, 1.1]
global_muon_absorption = [0.8, 1.0, 1.1]
muon_scattering     = [0.8, 1.0, 1.1]
neutrino_coin       = [0, 1]
muon_oversizing     = [1, 3]

def get_sets (dtype, sys, has_bulkice=True):

        ''' A function to return what discrete sets to be loaded/included in the analysis.

        Parameters
        ----------
        :dtype       :string                                                                        
                     :numucc / nuecc / nutaucc / numunc / nuenc / nutaunc / muon
        :sys         :string 
                     :domeff / holeice / forward / coin / absorption / scattering
        :has_bulkice :boolean
                     :If True (default): include bulk ice off axis points

        Returns
        -------
        :sets :array of float
              :the discrete values included for the data type and systematic

        Notes
        -----
        Hyperplane is assumed here.

        Examples
        --------
        In[0]: from misc import get_sets
        In[1]: sets = get_sets ('numucc', 'domeff', has_bulkice=True)
        In[2]: print sets
        array ([0.88, 0.94, 0.97, 1.0, 1.03, 1.06, 1.12])

        '''
        ### IF both abs and sca are included, add off axis points to absorption sets
        if has_bulkice and sys=='absorption':
                muon_absorption = sorted (global_muon_absorption + [0.929, 1.142])
                neutrino_absorption = sorted (global_neutrino_absorption + [0.929])
        ### return sets
        try:
                pid = 'muon' if 'muon' in dtype else 'neutrino'
                return np.array (eval (pid+'_'+sys))
        except:
                message = 'misc:get_sets :: ' + pid + \
                          ' (' + dtype + ') has no systematic name of ' + sys + '.'
                raise InvalidArguments (message)

###########################################################################
##### exception / error classes
###########################################################################
class Error (Exception):
        pass

class InvalidArguments (Error):
        def __init__ (self, message):
                super (Error, self).__init__ (message)
                
###########################################################################
#### a Map class to map dictionary key to attribute.
###########################################################################
class Map (dict):

    ''' A class to map dictionary key to attribute.

        Example
        -------
        In[0]: from misc import Map
        In[1]: m = Map({'e':[1,2,3]})
        In[2]: m.e
        [1,2,3]
    '''

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems(): self[k] = v
        if kwargs:
            for k, v in kwargs.iteritems(): self[k] = v

    def __getattr__(self, attr): 
        if attr.startswith('__') and attr.endswith('__'):
            return super(Map, self).__getattr__(attr)
        return self.__getitem__(attr)

    def __setattr__(self, key, value): 
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item): 
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self): 
        return self.__dict__

    def __setstate__(self, d): 
        self.update(d)
        self.__dict__.update(d)

###########################################################################
##### an Info class to read/store/check user's inputs
###########################################################################
arguments = ['fit_data' , 'neutrinos'       , 'backgrounds',
             'eedges'   , 'zedges'          , 'pedges'     ,
             'pdictpath', 'nuparam_textfile', 'outfile'    ,
             'matter'   , 'oscnc'           , 'inverted'   ,
             'verbose'  ]
 
class Info (object):

        ''' A class to read/store/check/print Info given by
            user's input arguments
            
            Used by templates class (see templates.py)

            Note: this class need to be modified when new
                  parameters are implemented
        '''

        def __init__ (self, **kwargs):
                
                ''' Initialize a dictionary given by the user
                    arguements in templates ()

                    :type   fit_data: boolean
                    :param  fit_data: if True, use real data for data histogram

                    :type   neutrinos: a 1D numpy of string
                    :param  neutrinos: an array of neutrino data types
                 
                    :type   backgrounds: a 1D numpy of string
                    :param  backgrounds: an array of background data types
                
                    :type   eedges: a 1D numpy array
                    :param  eedges: an array of energy edges for template (in GeV)

                    :type   zedges: a 1D numpy array
                    :param  zedges: an array of zenith edges for template (in radian)

                    :type   pedges: a 1D numpy array
                    :param  pedges: an array of pid edges for template (in meters)

                    :type   pdictpath: a string
                    :param  pdictpath: path to pickled files

                    :type   nuparam_textfile: a string
                    :param  nuparam_textfile: path/address of nuparam textfile

                    :type   outfile: a string
                    :param  outfile: path/address of output file
                
                    :type   matter: boolean
                    :param  matter: if True, matter effect is taken into account

                    :type   oscnc: boolean
                    :param  oscnc: if True, oscillate neutral current events

                    :type   inverted: boolean
                    :param  inverted: if True, inverted hierarchy is assumed

                    :type   verbose: int
                    :param  verbose: If 0, no print out
                                     If 1, minimal print out
                                     If 2, detailed print out
                '''

                self._inputs = self._convert (**kwargs)
                self._check_info ()

        def __getstate__ (self):

                ''' get state for pickling '''

                return self.__dict__

        def __setstate__ (self, d):

                ''' set state for pickling '''

                self.__dict__ = d
                
        def __call__ (self, arg):
                
                ''' Return user's input of a given arg

                    :type   arg: a string
                    :param  arg: name of user's input

                    :return: the values of user's input
                '''
                
                if arg not in self._inputs:
                        message = 'misc:Info :: '+arg+' not registered as users inputs'
                        raise InvalidArgements (message)
                return self._inputs[arg]

        def __str__ (self):

                ''' Print out user's inputs '''
                
                print ('#### master dictionary info:')
                print ('####')
                for arg, value in sorted (self._inputs.iteritems ()):
                        line = '####'+' '*11+arg.center (15)+': '+'{0}'
                        form = str (value) if arg in ['fit_data', 'matter', 'oscnc', 'inverted'] \
                               else value
                        print (line.format (form))
                return ('####')

        def _convert (self, **kwargs):

                ''' Convert keyword arguments into a dictionary '''

                idict = Map ({})
                for key, value in kwargs.iteritems ():
                        idict[key] = value
                return idict

        def get_ranges (self):

                ''' get ranges from input edges '''

                ranges = {}
                for edge in ['eedges', 'zedges', 'pedges']:
                        ranges[edge[0]] = (self._inputs[edge][0], self._inputs[edge][-1])
                return ranges

        def get_edges (self):

                ''' get edges from input edges '''

                edges = {}
                for edge in ['eedges', 'zedges', 'pedges']:
                        edges[edge[0]] = self._inputs[edge]
                return edges

        def get_datatypes (self):

                ''' get datatypes from input neutrinos / backgrounds '''

                return np.concatenate ((self._inputs ['neutrinos'],
                                        self._inputs ['backgrounds']))
        
        def _check_info (self):

                ''' Check user's inputs '''
                
                ## check inputs
                if not sorted (self._inputs) == sorted (arguments):
                        message = 'misc:Info :: unexpected arguments'
                        raise InvalidArguments (message)
                
                ## check booleans
                for arg in ['fit_data', 'matter', 'oscnc', 'inverted']:
                        if not type (self._inputs[arg]) == bool:
                                message = 'misc:Info :: unexpected argument type ' + \
                                          arg + ' (' + type (self._inputs[arg]) + ')'
                                raise InvalidArguments (message)

                ## check paths
                for arg in ['pdictpath']:
                        if not os.path.exists (self._inputs[arg]):
                                message = 'misc:Info :: path does not exist; ' + \
                                          arg + ': ' + self._inputs[arg]
                                raise InvalidArguments (message)

                ## check nufiles

                ## check outdir

                return

###########################################################################
#### a Toolbox class to do array stuff
###########################################################################
class Toolbox (object):

        def __init__ (self):

                ''' Initialize a toolbox '''

        def __getstate__ (self):

                ''' get state for pickling '''

                return self.__dict__

        def __setstate__ (self, d):

                ''' set state for pickling '''

                self.__dict__ = d

        def check_path (self, path):

                ''' check if a path exist '''
                
                if not os.path.exists (path):
                        message = 'misc:toolbox :: path does not exist; ' + path
                        raise InvalidArguments (message)

        def check_file (self, filename):

                ''' check if a file exist '''
                
                if not os.path.isfile (filename):
                        message = 'misc:toolbox :: file does not exist' + \
                                  filename
                        raise InvalidArguments (message)
                
        def is_number (self, var):

                ''' check if var is a float / int '''

                return isinstance (var, float) or isinstance (var, int)
                
        def is_array (self, var):

                ''' check if var is a list / array '''

                return isinstance (var, list) or isinstance (var, np.ndarray)

        def is_dict (self, var):

                ''' check if var is a Map / dict '''

                return isinstance (var, Map) or isinstance (var, dict)

        def chop (self, array, cut):

                ''' apply one cut to an array

                    :type    array: either a list or a numpy array of float / int
                    :param   array: array of an event's variable

                    :type      cut: an array of boolean
                    :param     cut: boolean to select events

                    :return carray: a numpy array
                            carray: array of selected events
                '''
                
                carray = np.array ([])
                if self.is_array (array) and len (array) > 0:
                        carray = np.array (array)[np.array (cut)]
                return carray

        def merge (self, arrays):

                ''' merge the given arrays 

                    :type     args: arrays of arrays
                    :param    args: arrays from multiple dictionaries

                    :return carray: a numpy array
                            carray: an array concatenated from
                                    multiple dictionaries
                '''

                carray = np.array (arrays[0])
                if len (arrays[0]) > 1:
                        for i in np.arange (len (arrays)-1):
                                if self.is_array (arrays[i+1]):
                                        carray = np.concatenate ((carray, np.array (arrays[i+1])))
                return carray

        
        def concat_dicts (self, d1, d2, d3=None):

            ''' concatenate dictionaries (up to three) '''

            cdict = Map({})
            for key in d1.keys():
                if self.is_array (d1[key]):
                    cdict[key] = self.merge ([ d[key] for d in [d1, d2, d3] if d ])
                    continue
                if self.is_dict (d1[key]):
                    cdict[key] = Map({})
                    for skey in d1[key].keys():
                        cdict[key][skey] = self.merge ([ d[key][skey]
                                                         for d in [d1, d2, d3]
                                                         if d ])
            return cdict

