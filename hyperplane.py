#!/usr/bin/python

####
#### By Elim Cheung (07/24/2018)
#### Originally written by Michael Larson
####
#### This script contain a HyperPlane class for a given data type 
####
###################################################################

from __future__ import print_function
from copy import deepcopy
import numpy as np

from scipy.optimize import minimize, curve_fit
from misc import Toolbox, InvalidArguments

####################################################################
#### constants needed
####################################################################
from misc import datatypes, global_sysvalues, global_sysexps
toolbox = Toolbox ()

## a simple function to convert fitted arrays
f = lambda x : np.array ([ x.T[i].T for i in np.arange (len (x.T))])

####################################################################
#### Hyperplane class
####################################################################
class HyperPlane (object):

    ''' A class for a given data type to
              - perform a multi-dimensional fit for each bin

        Example to define a hyperplane object through library
        ----------------------------------------------------
        In [0]: import library, nuparams
        In [1]: nufile = 'nuisance_textfiles/nuparams_template.txt'
        In [2]: nuParams = nuparams.Nuparams (nufile, isinverted=False)
        In [3]: params = nuParams.extract_params ('seeded_mc')
        In [4]: dparams = nuParams.get_active_dparams ()
        In [5]: expparams = nuParams.get_exp_dparams ('numucc')
        In [6]: lib = library.Library (['numucc'], 'pickled_files/')
        In [7]: numucc_hplane = lib.get_hplanes ('numucc', params, dparams, expparams)

        Example to apply factors using the hyperplane object
        ----------------------------------------------------
        In [9]: factors = numucc_hplane.apply (params)
    '''
    
    def __init__ (self, dtype, histos, expfuncs, refvalues, verbose=0):

        ''' initialize a hyperplane for a given data type

            :type   dtype: a string
            :param  dtype: name of the data type

            :type   histos: a dictionary
            :param  histos: contain histograms of all discrete
                            sets for this data type

            :type   expfuncs: list of string
            :param  expfuncs: discrete parameter names with exp functions

            :type   refvalues: a dictionary
            :param  refvalues: reference values of discrete systematics

            :type   verbose: an int
            :param  verbose: If 0, no printout
                             If 1, print out basic info
                             If 2, print out info within chi2 fit
        '''

        self._dtype = dtype
        self._histos = histos
        self._expfuncs = expfuncs
        self._refvalues = refvalues
        self._verbose = verbose
        self._check_args ()

        ## set parameters
        self._dparams = sorted (self._refvalues)
        self._refvalues = [ self._refvalues[dp] for i, dp in enumerate (self._dparams) ]
        self._fitparams, self._fitseeds = self._set_params ()
        ## print info if verbose
        if self._verbose > 1: self._print_params ()

        ## fit hyperplane (the meat)
        self._ndof, self._coeffs, self._chi2_before, self._chi2_after = self._fit_plane ()
        
    def _check_args (self):

        ''' check user's inputs '''

        ## check data type
        if not self._dtype in datatypes:
            message = 'Member:check_args :: '+self._dtype+' not registed as data types.'
            raise InvalidArguments (message)

        ## check histos / refvalues
        for arg in ['histos', 'refvalues']:
            if not toolbox.is_dict (eval ('self._'+arg)):
                message = 'Member:check_args :: '+arg+' must be a dictionary / Map.'
                raise InvalidArguments (message)

        ## check nuparams
        if not toolbox.is_array (self._expfuncs):
            message = 'Member:check_args :: expfuncs must be a list or array.'
            raise InvalidArguments (message)
        
    def _check_params (self, params):

        ''' check if params is dict and have all available keys '''

        ## check if its a dictionary
        if not toolbox.is_dict (params):
            message = 'Hyperplane:check_params :: params must be a dictionary'
            raise InvalidArguments (message)
        
        ## check if all parameters are available
        for param in self._dparams:
            if param not in params.keys ():
                message = 'Hyperplane:check_params :: '+param+' must be in params'
                raise InvalidArguments (message)

    @staticmethod
    def _print_xvalues (xvalues):

        ''' print brief info of data points for hyperplane fit

            :return xvalues: a multi-dimensional np array 
                    xvalues: values of discrete parameters ( #parameters x #sets )
        '''
        
        print ('#### number of discrete parameters included: {0} '.format (xvalues.shape[0]))
        print ('#### number of discrete sets included: {0} '.format (xvalues.shape[1]))
        print ('####')
    
    def _print_params (self):

        ''' print hyperplane fit parameters information '''
        
        print ('#### ###############################################')
        print ('#### ############# {0:7} HyperPlane ##############'.format (self._dtype))
        print ('####')
        print ('#### A list of {0:2} fit parameters for this plane:'.format (len (self._fitparams)))
        print ('#### ----------------------------------------------')
        print ('#### {0:16} | {1:12} | {2:12}'.format ('parameter'.center (16),
                                                       'seed slope'.center (12),
                                                       'ref value'.center (12)))
        print ('#### ----------------------------------------------')
        for i, fp in enumerate (self._fitparams):
            parameter, seed = fp, np.round (self._fitseeds[i], 4)
            value = '--' if fp not in self._dparams else np.round (self._refvalues[i], 4)
            print ('#### {0:16} | {1:12} | {2:12}'.format (parameter, seed, value))
        print ('####')

    def _print_fit_plane (self, coeffs, hshape=None, nbins=None,
                          nbin=None, chi2b=None, chi2a=None):
        
        ''' print the progress of hyperplane fit
          
            :type   coeffs: list
            :param  coeffs: values of coefficients

            :type   hshape: tuple
            :param  hshape: histogram shape. If provided, print header
                            (about to start fitting)

            :type   nbins: int
            :param  nbins: number of bins in histogram. If provided, print header

            :type    nbin: int
            :param   nbin: current bin

            :type   chi2b: float
            :param  chi2b: total chi2 before fit of this bin

            :type   chi2a: float
            :param  chi2a: total chi2 after fit of this bin
        '''

        if hshape and nbins:
            ## about to start fit.. print header!
            print ('#### {0} HyperPlane fit starts with'.format (self._dtype))
            print ('#### histo shape {0}; {1} total number of bins'.format (hshape, nbins))
            print ('#### initial coeffs (seed values):')
            print ('####     {0:10} | {1:10} | {2:10} '.format ('param name'.center (10),
                                                                'slope'.center (10),
                                                                'exp'.center (10) ))
            print ('####     {0}'.format ('-'*(30+6)))

        ## print values in progress
        for index, param in enumerate (list (self._dparams) + ['constant']):
            refvalue, fitvalue, expvalue = self._find_values (coeffs, param, index)
            expvalue = np.round (expvalue, 4) if expvalue else '-----'.center (10)
            print ('####     {0:10} | {1:10} | {2:10} '.format (param.center (10),
                                                                np.round (fitvalue, 4),
                                                                expvalue))

        if nbin and chi2b and chi2a:
            ## print chi2 info for the current bin 
            print ('####     {0}th bin chi2 before and after: {1}, {2}'.format (nbin,
                                                                                np.round (chi2b,4),
                                                                                np.round (chi2a)  ))
        print ('####     {0}'.format ('-'*(30+6)))
        
    def _set_params (self):

        ''' set hyperplane fit parameters

            :return fitparams: a list
                    fitparams: names of fit parameters
                               systematic names + constant + extra exp factors

            :type  fitseeds: a list
            :param fitseeds: seed values for each fit parameters
        '''
        
        ## define hyperplane fit parameters (systematic + extra exps + constant)
        ## add exp parameter for any exponential set by user
        exparams = sorted ([ dparam+'_exp' for dparam in self._expfuncs ])
        fitparams = list (self._dparams) + exparams + ['constant']
        ## seeds for each fit parameters
        fitseeds = [ self._get_seed (fp) for fp in fitparams ]
        return fitparams, fitseeds
        
    def _get_seed (self, param):

        ''' a static method to return seed values of a given fit parameter.

            :type  param: a string
            :param param: name of the parameter of interest

            :return  seed: a float
                     seed: seed value for the parameter
        '''
        
        if param == 'constant':
            seed = 1.0
        elif param in self._dparams:
            seed = 0.0 ## seeds for flat space
        elif param in global_sysexps:
            seed = global_sysexps [param]
        else:
            message = 'HyperPlane:get_seed :: '+ param+ ' does not appear anywhere ...'
            raise InvalidArguments (message)
        return seed

    def _get_refid (self):

        ''' get the ID of the reference values
          
            :return setid: a string
                    setid: systematic values separated by '_' as the id of this set
        '''
        
        refid = ''
        for i, dp in enumerate (self._dparams):
            refid += str (float (self._refvalues[i]))
            if not i==len (self._dparams)-1: refid += '_'
        return refid

    def _get_values (self):

        ''' get the values from which hyperplane is fit 

            :return xvalues: a multi-dimensional np array 
                    xvalues: values of discrete parameters ( #parameters x #sets )

            :return yvalues: a multi-dimensional np array
                    yvalues: ratios of each histogram (weight) to reference histogram
                             ( #sets x histogram shape )

            :return variances: a multi-dimensional np array
                    variances: ratios of each histogram (weight**2) to reference histogram
                               ( #sets x histogram shape )
        '''

        xvalues, yvalues, variances = [], [], []

        refid = self._get_refid ()
        ref_w  = self._histos [refid]['H']
        ref_w2 = self._histos [refid]['H2']

        for setid in self._histos.keys ():

            ## xvalue = discrete set values
            string = setid.split ('_')
            xvalue = np.array ([ float (v) for v in string ])

            ## yvalue = ratio of H to ref histogram
            yvalue = np.divide (self._histos[setid]['H'], ref_w)
            yvalue [~np.isfinite (yvalue)] = 0.0
            yvalue [yvalue > 10000] = 1.0
            ## variance = ratio of H2 to ref histogram H2
            variance = np.divide (self._histos[setid]['H2'], ref_w2)
            variance [~np.isfinite (variance)] = 0.0

            ## append
            xvalues.append   (xvalue)
            yvalues.append   (yvalue)
            variances.append (variance)

        return np.array (xvalues).T, np.array (yvalues), np.array (variances)
    
    def _get_holder (self, xvalues, yvalues):

        ''' define a set of information holder for hyperplane fit
      
            :type  xvalues: a multi-dimensional np array 
            :param xvalues: values of discrete parameters ( #parameters x #sets )

            :type  yvalues: a multi-dimensional np array
            :param yvalues: ratios of each histogram (weight) to reference histogram
                            ( #sets x histogram shape )

            :return  shape: tuple
                     shape: shape of histogram

            :return  ndof: int
                     ndof: initial number of degree of freedoms = 0.

            :return  nfree: int
                     nfree: number of discrete parameters 

            :return  coeffs: a multi-dimensional np array 
                     coeffs: initial fitted coefficients (empty)

            :return chi2_after: a multi-dimensional np array 
                    chi2_after: initial chi2 after hyperplane fit (empty)

            :return chi2_before: a multi-dimensional np array 
                    chi2_before: initial chi2 before hyperplane fit (empty)
        '''
        
        ## histogram shape
        hshape = yvalues.shape[1:]
        ## a histogram shape for each fit parameters to store fit values
        coeffshape  = (hshape[0], hshape[1], hshape[2], len (self._fitparams))
        ## a histogram shape for each discrete parameter to store chi2 values
        chi2shape = (hshape[0], hshape[1], hshape[2], len (yvalues))

        ## in case some discrete parameters are not included
        nfree = np.sum ([ 1 for i in range (xvalues.shape[0]) if len (np.unique (xvalues[i])) > 1 ])
        ## define info holders
        return 0, nfree, np.zeros (coeffshape), np.zeros (chi2shape), np.zeros (chi2shape)

    def _find_values (self, coeffs, param, index):

        ''' determine the reference value, coeff (slope) value,
            and exp coeff value (if available) based on the
            given parameter.

            :type   coeffs: a list
            :param  coeffs: list of coefficient values

            :type   param: a string
            :param  param: name of parameter of interest
          
            :type   index: an int
            :param  index: index of param in self._dparam
        '''
        
        ## get ref value
        refvalue = '--' if param=='constant' else self._refvalues [index]
        ## get fit coeff value
        param_index = np.where (np.array (self._fitparams)==param)[0]
        fitvalue = coeffs [param_index[0]]
        ## get exp value
        exp_index = np.where (np.array (self._fitparams)==param+'_exp')[0]
        expvalue = None if len (exp_index) == 0 else coeffs [exp_index[0]]
        return refvalue, fitvalue, expvalue

    def _fit_plane (self):

        ''' perform fit for all bins 

            :return ndof: float
                    ndof: effective number degrees of freedom

            :return fitted_coeffs: a multi-dimensional np array
                    fitted_coeffs: fitted coefficients for all bins and all fit parameters
                                   (histogram shape x # fit params)

            :return chi2_before: a multi-dimensional np array
                    chi2_before: chi2 map before hyperplane fit for all bins and all discrete sets
                                 (histogram shape x # discrete sets)

            :return chi2_after: a multi-dimensional np array
                    chi2_after: chi2 map after hyperplane fit for all bins and all discrete sets
                                (histogram shape x # discrete sets)
        '''
        
        ## get data points to be fitted
        xvalues, yvalues, variances = self._get_values ()
        if self._verbose > 1: self._print_xvalues (xvalues)

        ## get info handler
        ndof, nfree, fitted_coeffs, chi2_after, chi2_before = self._get_holder (xvalues, yvalues)
        
        ## get ready to loop through bins
        ahist = self._histos [self._histos.keys ()[0]]['H']
        nbins, hshape = len (ahist.flatten ()), ahist.shape
        coeffs = deepcopy (self._fitseeds)
        if self._verbose > 1: self._print_fit_plane (coeffs, hshape=hshape, nbins=nbins)

        ## loop through bins
        for nbin in np.arange (nbins):

            ## get data points for this bin
            i, j, k = np.unravel_index (nbin, hshape)
            y = np.array ([ yvalues[m,i,j,k] for m in np.arange (len (yvalues)) ])
            v = np.array ([ variances[m,i,j,k] for m in np.arange (len (yvalues)) ])

            ## add to degrees of freedom
            if np.any(y > 0): ndof += len(y) - 1

            ## fit this bin
            fcoeffs, chi2b, chi2a = self._fit_bin (coeffs, xvalues, y, v)
            fitted_coeffs[i,j,k] = fcoeffs
            chi2_before[i,j,k]   = chi2b
            chi2_after[i,j,k]    = chi2a

            ## print info
            if self._verbose > 1: self._print_fit_plane (fcoeffs, nbin=nbin,
                                                         chi2b=np.sum (chi2b),
                                                         chi2a=np.sum (chi2a))

        return ndof-nfree, f (fitted_coeffs), f (chi2_before), f(chi2_after)

    def _fit_bin (self, coeffs, x, y, w2):

        ''' perform fit in a bin
        
            :type  coeffs: list
            :param coeffs: coefficients of hyperplane

            :type   x: a multi-dimensional numpy array
            :param  x: discrete set values (# discrete parameters x # discrete sets)

            :type   y: a multi-dimensional numpy array
            :param  y: one bin of histograms from all discrete sets
                       (# discrete sets)

            :type  w2: a multi-dimensional numpy array
            :param w2: one bin of histograms (weight**2) from all discrete sets
                       (# discrete sets)

            :return fitted_coeffs: list
                    fitted_coeffs: fitted coefficient 

            :return chi2_before: list
                    chi2_before: chi2 value 
         
        '''
        
        ## chi2 before hyperplane fit
        chi2_before = self._chi2 (coeffs, x=x, y=y, w2=w2, summed=False)

        ## perform chi2 hyperplane fit
        result = minimize (self._chi2, coeffs,
                           args = (x, y, w2, True),
                           options={'disp':False},
                           method = 'BFGS',
                           tol = 1e-16 )

        ## store coeff and chi2 values after fit
        fitted_coeffs = result.x
        chi2_after = self._chi2 (result.x, x=x, y=y, w2=w2, summed=False)
        
        return fitted_coeffs, chi2_before, chi2_after
    
    def _chi2 (self, coeffs, x=None, y=None, w2=None, summed=False):

        ''' chi2 function to be minimized 

            :type  coeffs: list
            :param coeffs: coefficients of hyperplane

            :type   x: a multi-dimensional numpy array
            :param  x: discrete set values (# discrete parameters x # discrete sets)

            :type   y: a multi-dimensional numpy array
            :param  y: histograms from all discrete sets
                       (# discrete sets x histogram shape)

            :type   w2: a multi-dimensional numpy array
            :param  w2: histograms (weight**2) from all discrete sets
                        (# discrete sets x histogram shape)

            :type   summed: a boolean
            :param  summed: If True, return summed chi2

            :return chi2: list
                    chi2: chi2 from all data points
        '''

        ## estimated y value from given coeffs
        yfit = self._eval (x, coeffs)

        ## chi2 of the fitted y values
        chi2s = (y-yfit)**2/w2
        chi2s [w2==0] = 0

        if summed: return np.sum (chi2s)
        return chi2s

    def _eval (self, xvalues, coeffs):

        ''' evaluate hyper plane when 
            1) fitting coeffs
            2) evaluating calculations from all bin

            :type  xvalues: list
            :param xvalues: when fitting coeffs - list of values of a discrete
                                                  parameter from its sets
                                                  (# discrete parameters)
                            when applying - list of discrete parameter values

            :type  coeffs: list
            :param coeffs: when fitting coeffs - list of coefficients
                                                 (# fit parameters)
                           when applying - fitted coefficients

            :return yest: a number
                    yest: total estimated y value
        '''
        
        ## constant is the last 
        yest = coeffs [-1]
        
        ## loop through each discrete parameter
        for index, param in enumerate (self._dparams):
            
            ## get values
            setvalue = xvalues[index]
            refvalue, fitvalue, expvalue = self._find_values (coeffs, param, index)
            
            ## get estimated y contribution
            has_exp = (type (expvalue) in [float, np.float64] and expvalue) or \
                      (type (expvalue) == np.ndarray and expvalue.any ())
            y = fitvalue * np.exp (expvalue * (setvalue-refvalue)) - fitvalue \
                if has_exp else fitvalue * (setvalue-refvalue)
            yest += y
            
        return yest
    
    def apply (self, params):

        ''' apply hyperplane with the given parameter values
            Note: oversizing is always evaluated at 1.0

            :type  params: a dictionary
            :param params: values of all floating
                           (including discrete) parameters

            :return result: a multi-dimensional numpy array
                    result: factors for each bin to be multiplied
                            given the parameter values
        '''

        ## add oversizing if muon
        if 'muon' in self._dtype:
            params ['oversizing'] = 1.0

        ## check params
        self._check_params (params)

        ## massage parameter values
        xvalues = np.array ([ params[p] for p in self._dparams ])
        ## evalulate
        result = self._eval (xvalues, self._coeffs)
        result [result<0.] = 0.

        ## print factors
        if self._verbose > 1: self._print_factors (result)
        return result

    def _print_factors (self, factors):

        ''' print factors
          
            :type  factors: multi-dimensional array
            :param factors: factors to be multiplied to histogram
                            (histogram shape)
        '''
        
        print ('#### ====================================================')
        print ('#### ================== {0:7} factors ================='.format (self._dtype))
        print ('#### {0} cascade bin:'.format (self._dtype))
        print ('#### {0}'.format (factors[:,:,0]))
        print ('#### {0} track bin:'.format (self._dtype))
        print ('#### {0}'.format (factors[:,:,1]))
        print ('####')
