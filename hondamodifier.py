#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contain the HondaModifier for
#### neutrino flux reweighting factor.
####
####################################################################

from __future__ import print_function
import numpy as np

class HondaModifier (object):

            ''' A class to deal with flux modification
                originally written by Juan Pablo Yanez
                modified by Elim Cheung

                Used internally by NeutrinoWeighter
            '''
            
            def __init__ (self):

                        ''' Initialize a Honda Flux object. '''

                        ### parameters from fits to Barr paper (Fig.7)
                        self._e1max_mu, self._e2max_mu =  3., 43
                        self._e1max_e , self._e2max_e  = 2.5, 10
                        self._e1max_mu_e, self._e2max_mu_e = 0.62, 11.45
                        ## Evaluated at
                        self._x1e, self._x2e = 0.5, 3.

                        ### parameters from fits to Barr paper (Fig.9)
                        self._z1max_mu, self._z2max_mu = 0.6, 5.
                        self._z1max_e , self._z2max_e  = 0.3, 5.
                        self._nue_cutoff, self._numu_cutoff = 650., 1000.
                        # Evaluated at
                        self._x1z, self._x2z = 0.5, 2.

                        ### parameters for Honda correction
                        hc_coeff = [0.02812, -0.05848, 1.045]
                        self._honda_fcn = np.poly1d (hc_coeff)
                        self._ecritical = -hc_coeff[1] / (hc_coeff[0]*2)
                        self._flux_renorm = self._honda_fcn (self._ecritical)-1.

            def __getstate__ (self):

                        ''' get state for pickling '''

                        return self.__dict__

            def __setstate__ (self, d):

                        ''' set state for pickling '''
                        
                        self.__dict__ = d
                        
            def _norm_fcn (self, x, A, sigma=0.3):

                        ''' define a norm Gaussian function '''
                        
                        return A/np.sqrt(2*np.pi*sigma**2) * np.exp(-x**2/(2*sigma**2))

            def _log_log_param (self, energy=1.,
                                y1=1., y2=1., x1=0.5, x2=3.,
                                cutoff_value = False):

                        ''' define a log log function '''

                        nu_nubar = np.sign (y2)
                        if nu_nubar == 0.0: nu_nubar = 1.
                        y1 = np.sign (y1) * np.log10 (np.abs (y1)+0.0001)
                        y2 = np.log10 (np.abs (y2+0.0001))
                        modification = nu_nubar*10**( ((y2-y1)/(x2-x1)) * (np.log10 (energy)-x1) + y1 - 2. )
                        if cutoff_value: modification *= np.exp (-1.*energy/cutoff_value)
                        return modification

            def _mod_NuMu_flux (self, energy, czenith,
                                e1=1., e2=1., z1=1., z2=1.):

                        ''' return modified numu flux based on Barr paper

                            :type    energy: a 1D numpy array
                            :param   energy: MC truth energy (GeV)

                            :type   czenith: a 1D numpy array
                            :param  czenith: MC truth cos zenith angle

                            :return  output: a 1D numpy array
                                     output: modified factor to numu flux
                        '''

                        ## A_ave = average change in a given energy when
                        ##         integrating over zenith angle
                        A_ave = self._log_log_param(energy=energy, 
                                                    y1=self._e1max_mu*e1, 
                                                    y2=self._e2max_mu*e2,
                                                    x1=self._x1e, x2=self._x2e)
                        ## A_shape = zenith angle information
                        A_shape = 2.5 * self._log_log_param(energy=energy, 
                                                            y1=self._z1max_mu*z1, 
                                                            y2=self._z2max_mu*z2,
                                                            x1=self._x1z, x2=self._x2z, 
                                                            cutoff_value = self._numu_cutoff)
                        return A_ave - (self._norm_fcn (czenith, A_shape, 0.32) - 0.75*A_shape)

            def _mod_NuE_flux (self, energy, czenith,
                               e1mu=1., e2mu=1., z1mu=1., z2mu=1.,
                               e1e=1., e2e=1., z1e=1., z2e=1.):

                        ''' return modified nue flux based on Barr paper
                            Modification factor is parameterized with numu flux.
                            Uncertainties of numu and nue fluxes are assumed
                            to be correlated.

                            :type    energy: a 1D numpy array
                            :param   energy: MC truth energy (GeV)

                            :type   czenith: a 1D numpy array
                            :param  czenith: MC truth cos zenith angle

                            :return  output: a 1D numpy array
                                     output: modified factor to nue flux
                        '''
                        
                        A_ave = self._log_log_param(energy=energy, 
                                                    y1=self._e1max_mu*e1mu + self._e1max_e*e1e, 
                                                    y2=self._e2max_mu*e2mu + self._e2max_e*e2e,
                                                    x1=self._x1e, x2=self._x2e)
                        A_shape = 1.*self._log_log_param(energy=energy, 
                                                         y1=self._z1max_mu*z1mu + self._z1max_e*z1e, 
                                                         y2=self._z2max_mu*z2mu + self._z2max_e*z2e,
                                                         x1=self._x1z, x2=self._x2z,
                                                         cutoff_value = self._nue_cutoff)
                        return A_ave - (1.5*self._norm_fcn (czenith, A_shape, 0.4) - 0.7*A_shape)

            def mod_ratio_NuBar (self, energy, coszen, pdg, nu_nubar, nubar):

                        ''' return modified numu / numubar ratio
                        
                            :type    energy: a 1D numpy array
                            :param   energy: MC truth energy (GeV)

                            :type    coszen: a 1D numpy array
                            :param   coszen: MC truth cos zenith angle

                            :type       pdg: a 1D numpy array
                            :param      pdg: particle encoding (+/- 12/4/6)

                            :type  nu_nubar: a float
                            :param nu_nubar: a nuisance parameter (fixed to 1.0)

                            :type     nubar: a float
                            :param    nubar: the nu/nubar nuisance parameter

                            :return  output: a 1D numpy array
                                     output: modified factor to nu-nubar ratio
                        '''

                        isnumu    = np.abs (pdg[0]) == 14
                        modfactor = np.ones (len (energy))
                        func      =  self._mod_NuMu_flux (energy, coszen, 1.0, 1.0, 1.0, 1.0) if isnumu else \
                                     self._mod_NuE_flux  (energy, coszen, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                        modfactor = nubar * func
                        weightmod = 1. + modfactor*nu_nubar
                        antineutrinos = pdg<0
                        weightmod[antineutrinos] = 1./(1+(1-nu_nubar)*modfactor[antineutrinos])
                        return weightmod

            def mod_ratio_UpHor (self, energy, coszen, pdg, uphor):

                        ''' return modified up / hor ratio
                        
                            :type    energy: a 1D numpy array
                            :param   energy: MC truth energy (GeV)

                            :type    coszen: a 1D numpy array
                            :param   coszen: MC truth cos zenith angle

                            :type       pdg: a 1D numpy array
                            :param      pdg: particle encoding (+/- 12/4/6)

                            :type     uphor: a float
                            :param    uphor: the up/hor nuisance parameter

                            :return  output: a 1D numpy array
                                     output: modified factor to up-horizontal ratio
                        '''
                        ### Juan Pablo: numu up/hor modification shouldn't be anything other than 1. 
                        if np.abs (pdg[0]) == 14: return 1. 

                        A_shape = 1.*np.abs (uphor)*self._log_log_param(energy=energy, 
                                                                        y1=(self._z1max_e+self._z1max_mu),
                                                                        y2=(self._z2max_e+self._z2max_mu),
                                                                        x1=self._x1z, x2=self._x2z,
                                                                        cutoff_value = self._nue_cutoff)
                        ### Juan Pablo: should be 0.3, not 3.5
                        return 1-0.3*np.sign (uphor)*self._norm_fcn (coszen, A_shape, 0.35)
            
            def honda_correction (self, loge):

                        ''' calculat honda correction factor
                        
                            :type  loge: a 1D numpy array
                            :param loge: log10 MC truth energy

                            :return output: a 1D array
                                    output: correction factor to honda flux
                        '''
                        
                        ebool  = loge > self._ecritical
                        output = np.ones_like (loge)
                        output[ebool]  = self._honda_fcn (loge[ebool]) - self._flux_renorm
                        return output
