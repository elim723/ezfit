#/usr/bin/python

####
#### Modified By Elim Cheung (07/24/2018)
#### Originally Written By Michael Larson
####
#### This script contain a ProbMap class to calculate
#### oscillation probability from finely binned probability map.
####
####################################################################

from __future__ import print_function
from scipy.interpolate import RectBivariateSpline
from misc import Toolbox, InvalidArguments
import numpy as np
import socket

###########################################################################
#### Prob3 essential
###########################################################################
from icecube import prob3 as BargerPropagator
datdir = '/opt/elims/software/tools/metaprojects/icerec/' if socket.gethostname()=='razor' else \
         '/data/condor_builds/users/elims/software/'
earth_model = datdir + 'ir041102/src/prob3/resources/oscillations/PREM_60layer.dat'
detector_depth = 2.
prop_height    = 20.
barger_prop = BargerPropagator.BargerPropagator (earth_model, detector_depth)
barger_prop.UseMassEigenstates(False)
barger_prop.SetOneMassScaleMode(False)
barger_prop.SetWarningSuppression(True)

###########################################################################
#### define constants
###########################################################################
default_params = {'dm21': 7.49e-5,
                  'dm31': 2.526e-3,
                  'theta23': np.arcsin (np.sqrt (0.440))  ,
                  'theta12': np.arcsin (np.sqrt (0.308))  ,
                  'theta13': np.arcsin (np.sqrt (0.02163)),
                  'deltacp': 0                             }

toolbox = Toolbox ()

###########################################################################
#### ProbMap class
###########################################################################
class ProbMap (object):

    ''' A class to create a finely-binned oscillation probability map.

        Example
        -------
        In [0]: from probmap import ProbMap
        In [1]: pmap = ProbMap (matter=True)
        In [2]: energy = 10**np.linspace (0.75, 1.75, 100)
        In [3]: coszen = np.linspace (-1, 1, 100)
        In [4]: pdg = 14*np.ones (100)
        In [5]: params = {'theta13':8.46*np.pi/180.,
                          'theta23':0.7365         ,
                          'dm31'   :2.457e-3       }
        In [6]: oscprob = pmap.get_prob (energy, coszen, pdg, params)
    '''
    
    def __init__ (self,
                  matter = True,
                  params = default_params,
                  ebins  = np.logspace(0,3,150), 
                  czbins = np.linspace(-1,1,150)):

        ''' Initialize a Prob3 object that calculates a oscillation
            probability map for both nu and nubar.

            :type  matter: boolean
            :param matter: if True, matter effect is taken into account

            :type  params: a dictionary
            :param params: values of oscillation parameters
        
            :type   ebins: a 1D numpy array
            :param  ebins: fine energy bins (GeV)

            :type  czbins: a 1D numpy array
            :param czbins: fine cosine zenith bins 
        '''

        self._matter = matter
        self._ebins  = ebins
        self._czbins = czbins
        self._params = self.extract_oscparams (params)
        
        ### set up variable for probability map
        self._nubar = np.array ([-1., 1.])
        self._emap, self._czmap, self._nu_nubar_map = np.meshgrid (self._ebins ,
                                                                   self._czbins,
                                                                   self._nubar,
                                                                   indexing = 'ij')

        ### get splined oscillation functions
        self.get_oscprob (self._params)

    @staticmethod
    def extract_oscparams (params):

        ''' extract oscillation parameters from full dictionary

            :type  params: a dictionary
            :param params: values of all floating parameters

            :return oscparams: a dictionary
                    oscparams: values of oscillation parameters
        '''
        
        ## extract osc params
        oscparams = {}
        for param in params:
            if param in default_params:
                oscparams[param] = params [param]
        return oscparams

    def __getstate__ (self):

        ''' get state for pickling '''

        return self.__dict__

    def __setstate__ (self, d):

        ''' set state for pickling '''

        self.__dict__ = d
    
    def _set_params (self, params):

        ''' set new parameters
            check for any new values if self._params exists

            :type  params: a dictionary
            :param params: values of oscillation parameters
        '''
        
        ## updated new params based on default_params
        updated = {}
        for key in default_params:
            ## if key not in params, refill default parameters (dm21 / deltacp / etc)
            value = default_params[key] if not key in params.keys () else params[key]
            updated[key] = value

        ## if self._params exist, check if anything different        
        if hasattr (self, '_params'):
            isdiff = False
            for key in self._params:
                if not updated[key] == self._params[key]:
                    isdiff = True; break
        else:
            isdiff = True
        return updated, isdiff

    def _get_probmap (self):

        ''' calculate oscillation probability in default bins.

            :return pmaps: tuple
                    pmaps: (prob from atm nue, prob from atm numu)
        '''
        
        pmap_from_e, pmap_from_mu = {}, {}
        shape = self._emap.shape
        for particle in [12, 14, 16]:
            pmap = self.calculate_prob3 (self._emap.flatten(), 
                                         self._czmap.flatten(), 
                                         self._nu_nubar_map.flatten() * particle,
                                         self._params)
                
            pmap_from_e[particle]  = np.reshape(pmap[:,0], shape)
            pmap_from_mu[particle] = np.reshape(pmap[:,1], shape)

        return pmap_from_e, pmap_from_mu

    def _spline_probmap (self):
        
        ''' spline probability maps

            :return  pspline: a dictionary
                     pspline: dictionary of 2D spline functions
        '''
        
        pspline = {-12:{}, 12:{}, -14:{}, 14:{}}
        for particle in [12, 14, 16]:
            for nubar in [-1,1]:
                i = 0 if nubar == -1 else 1
                nue_spline  = RectBivariateSpline (self._ebins, self._czbins, self._pmap_from_e[particle][:,:,i])
                numu_spline = RectBivariateSpline (self._ebins, self._czbins, self._pmap_from_mu[particle][:,:,i])
                pspline[nubar*12][nubar*particle] = nue_spline
                pspline[nubar*14][nubar*particle] = numu_spline
        return pspline
        
    def get_prop_distance (self, zenith):

        ''' calculate propagation distance (meter) from zenith
            angle based on Earth's geometry 
        
            :type  zenith: float or an array of float
            :param zenith: zenith angle(s)

            :return distance: float or an array of float 
                    distance: propagation distance(s) 
        '''
        
        ## based on Earth's geometry
        L1 = 19.; R = 6378.2 + L1
        phi = np.arcsin((1-L1/R)*np.sin(zenith))
        psi = zenith - phi
        return np.sqrt( (R-L1)**2 + R**2 - (2*(R-L1)*R*np.cos(psi)))

    def calculate_prob3 (self, e, cz, pdg, params):

        ''' calculate the oscillation probability

            :type  e: float or an array of float
            :param e: energy (GeV)

            :type  cz: float or an array of float
            :param cz: cos zenith angle

            :type  pdg: float or an array of float
            :param pdg: particle encoding (+/- 12/4/6)

            :type  params: a dictionary
            :param params: values of oscillation parameters

            :return  oscprob: an array with shape 2 x length of array 
                     oscprob: oscillation probabilities from atm nue and atm numu
        '''

        ## calculate oscillation probability
        NuE = 1 ; NuMu = 2 ; NuTau = 3
        sin_sq_theta12 = np.sin (params['theta12'])**2
        sin_sq_theta23 = np.sin (params['theta23'])**2
        sin_sq_theta13 = np.sin (params['theta13'])**2
        dm32 = params['dm31'] - params['dm21']
        kSquared = True
        kNuType = np.array (np.sign (pdg), dtype=np.int)
        out_nu = (NuE*(np.abs (pdg) == 12) + NuMu*(np.abs (pdg) == 14) + NuTau*(np.abs (pdg) == 16))
        osc_prob = np.zeros ([len(e), 2])
        for i in range (len (e)):
            if self._matter:
                barger_prop.SetMNS (sin_sq_theta12, sin_sq_theta13, sin_sq_theta23,
                                    params['dm21'], dm32, params['deltacp'],
                                    float (e[i]), kSquared, int (kNuType[i]))
                barger_prop.DefinePath (float (cz[i]), prop_height)
                barger_prop.propagate (int (kNuType[i]))
                osc_prob[i,:] = [barger_prop.GetProb ( NuE, int (out_nu[i])),
                                 barger_prop.GetProb (NuMu, int (out_nu[i]))]
            else:
                distance = self.get_prop_distance (np.arccos (float (cz[i])))
                osc_prob[i,:] = [barger_prop.GetVacuumProb (NuE, int (out_nu[i]),
                                                            float (e[i]), distance),
                                 barger_prop.GetVacuumProb (NuMu, int (out_nu[i]),
                                                            float (e[i]), distance)]
        return osc_prob
    
    def get_oscprob (self, params):

        ''' set new parameters; update prob maps; spline maps
            this is where spline function is defined

            :type  params: a dictionary
            :param params: values of oscillation parameters
        '''
        ### update params
        self._params, self._isdiff = self._set_params (params)
        ### do oscillation map if params are different or _pspline is abscent
        if self._isdiff or not hasattr (self, 'pspline'):
            ### calculate probability map
            self._pmap_from_e, self._pmap_from_mu = self._get_probmap ()
            ### spline probability map
            self.pspline = self._spline_probmap ()        

    def get_prob (self, e, cz, pdg, params):

        ''' obtain oscillation probability from splined prob

            :type  e: float or an array of float
            :param e: energy (GeV)

            :type  cz: float or an array of float
            :param cz: cos zenith angle

            :type  pdg: float or an array of float
            :param pdg: particle encoding (+/- 12/4/6)

            :type  params: a dictionary
            :param params: values of oscillation parameters

            :return  oscprob: an array with shape 2 x length of array 
                     oscprob: oscillation probabilities from atm nue and atm numu
        '''

        ## check if need to do oscillation spline
        self.get_oscprob (params)

        ## separate nu and nubar events
        isnubar = pdg < 0
        ## define particle encoding
        ptype = np.abs (pdg)[0]
        ## initialize prob container
        probs_e, probs_mu = np.ones_like(e), np.ones_like(e)

        ## oscillation probability from atm nue
        probs_e[isnubar] = self.pspline[-12][-1*ptype].ev (e[isnubar], cz[isnubar])
        probs_e[~isnubar] = self.pspline[12][ptype].ev (e[~isnubar], cz[~isnubar])
        ## oscillation probability from atm numu
        probs_mu[isnubar] = self.pspline[-14][-1*ptype].ev (e[isnubar], cz[isnubar])
        probs_mu[~isnubar] = self.pspline[14][ptype].ev (e[~isnubar], cz[~isnubar])
        
        return np.array ([probs_e, probs_mu]).T

