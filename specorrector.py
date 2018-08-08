#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contain SPEcorrector class
####
####################################################################

from __future__ import print_function
import numpy as np
from scipy.interpolate import RectBivariateSpline
from misc import InvalidArguments, Toolbox 

toolbox = Toolbox ()

class SPEPeakCorrector (object):

            ''' A class to deal with SPE correction
                originally written by Michael Larson
                modified by Elim Cheung

                Example
                -------
                In [0]: from weightcalculator import SPEPeakCorrector
                In [1]: charge_per_nchannel = np.divide (events.hits.qTotal, events.hits.SRT_nch)
                In [2]: SPECorr = SPEPeakCorrector (charge_per_nchannel, events.w)
                In [3]: reweight_factor = SPECorr (0.04)
            '''
            
            def __init__ (self, q_nch, w,
                          bins=np.linspace(0., 4.0, 50.),
                          order='linear'):

                        ''' Initialize a SPE correction class.
                        
                            :type    q_nch: a 1D array
                            :param   q_nch: charge per number of channels

                            :type    w: a 1D array
                            :param   w: weights

                            :type    bins: a 1D array
                            :param   bins: charge / nchannels bin edges

                            :type    order: a string
                            :param   order: linear is recommanded

                        '''
                        
                        self._q_nch = np.nan_to_num (q_nch)
                        self._w     = w
                        self._bins  = bins
                        self._check_args ()

                        ## perform spline fit
                        self._spline2D = self._spline ()

            def __call__ (self, percent_shift):

                        ''' get SPE correction factor

                            :type  percent_shift: a float
                            :param percent_shift: SPE percentage shift

                            :return factor: a 1D numpy array
                                    factor: reweighting factor due to SPE shift
                        '''

                        if not isinstance (percent_shift, float):
                                    message = 'SPECorr:call :: percent_shift must be a float'
                                    raise InvalidArguments (message)

                        factor = self._spline2D.ev ([percent_shift]*len(self._q_nch), self._q_nch)
                        return np.nan_to_num (factor)
                        
            def _check_args (self):
                        
                        ''' check user's input arguments '''
                        
                        for arg in ['q_nch', 'w', 'bins']:
                                    if not toolbox.is_array (eval ('self._'+arg)):
                                                message = 'SPECorr:check_args :: '+arg+' must be a 1D array.'
                                                raise InvalidArguments (message)
                        
            def _spline (self):

                        ''' spline fit correction vs xvalues '''
                        
                        x_values = np.linspace (-0.15, 0.15, 11)
                        bin_cens = (np.array (self._bins[1:]) + np.array (self._bins[:-1]))/2.
                        y_values = np.zeros ([ len (x_values), len (self._bins)-1 ])
                        for t_i, t_x in enumerate (x_values):
                                    y_values[t_i] = self._get_corr (percent_shift=t_x)
                        return RectBivariateSpline (x_values, bin_cens, y_values, kx=1, ky=1, s=0.0) 
                        
            def _get_corr (self, percent_shift=0):

                        ''' get SPE correction 

                            :type  percent_shift: a float
                            :param percent_shift: SPE percentage shift

                            :return ratio: a 1D numpy array
                                    ratio: ratio of shifted histogram to no shift histogram
                        '''
                        
                        uncorr_hist, bins = np.histogram (self._q_nch, bins=self._bins, weights = self._w)
                        corr_hist, bins = np.histogram (self._q_nch * (1+percent_shift), bins=self._bins, weights=self._w)
                        corr_hist *= np.divide (np.sum (uncorr_hist), np.sum (corr_hist))
                        
                        ratio = corr_hist / uncorr_hist
                        ratio[np.isnan (ratio)] = 1.
                        ratio[np.isinf (ratio)] = 1.
                        return ratio
