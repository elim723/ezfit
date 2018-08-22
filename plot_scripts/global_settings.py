#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This file defines the global variables for
#### various plots:
####
####   1. Common variables
####
####   2. Colors
####     -- data types
####     -- color maps
####
####   3. Hatches for data types
####
####   4. Labels
####     -- systematic
####     -- data types (members)
####     -- observables/event variables
####     -- analyses for GRECO and DRAGON
####
###############################################################

###################################
#### Common Variables
###################################
dm21 = 7.49E-5
seconds_per_year = 3600.*24.*365.24

###################################
#### Colors
###################################
Cdatatypes = {'numucc' : 'navy'       ,
              'nuecc'  : 'darkgreen'  ,
              'nutaucc': 'darkorange' ,
              'numunc' : 'deepskyblue',
              'nuenc'  : 'palegreen'  ,
              'nutaunc': 'khaki'      ,
              'nunc'   : 'pink'       ,
              'muon'   : 'lightgray'  ,
              'iccdata': 'lightgray'  ,
              'noise'  : 'plum'       ,
              'data'   : 'black'      ,
              'mc'     : 'red'        ,
              'bestfit': 'red'        ,
              'noosc'  : 'blue'        }

Cmaps = {'counts'   : 'Blues'   ,
         'variances': 'Reds'    ,
         'effects'  : 'Spectral' }

colors = {'dtypes': Cdatatypes,
          'maps'  : Cmaps      }

###################################
#### Hatches for data types only
###################################
hatches = {'numucc' : "x" ,
           'nuecc'  : "/" ,
           'nutaucc': '-' ,
           'numunc' : '+' ,
           'nuenc'  : '//',
           'nutaunc': '|' ,
           'nunc'   : '//',
           'muon'   : '+' ,
           'iccdata': '+' ,
           'noise'  : '.'  }

###################################
#### Labels
###################################
Lsystematics = { 'dm31'            : r'$\Delta$ m$^2_{32}$'          ,
                 'theta23'         : r'$\theta_{23}$'                ,
                 'theta13'         : r'$\theta_{13}$'                ,
                 'muon_flux'       : r'$\gamma_{\mu}$'               ,
                 'gamma'           : r'$\gamma_{\nu}$'               ,
                 'nue_numu_ratio'  : r'N$_{\nu_e}$ / N$_{\nu_{\mu}}$',
                 'barr_uphor_ratio': r'up / horizontal'              ,
                 'barr_nubar_ratio': r'$\nu$ / $\bar{\nu}$ '         ,
                 'axm_res'         : r'axial res'                    ,
                 'axm_qe'          : r'axial qe'                     ,
                 'DISa_nu'         : r'DIS $\nu$'                    ,
                 'DISa_nubar'      : r'DIS $\bar{\nu}$'              ,
                 'spe_corr'        : r'SPE correction'               ,
                 'nyears'          : r'global normalization'         ,
                 'norm_atmmu'      : r'N$_{\mu}$'                    ,
                 'norm_noise'      : r'N$_{\text{noise}}$'           ,
                 'norm_numu'       : r'N$_{\nu_{\mu}}$'              ,
                 'norm_nutau'      : r'N$_{\nu_{\tau}}$'             ,
                 'norm_nc'         : r'N$_{\text{NC}}$'              ,
                 'norm_nugen'      : r'N$_{\text{nugen}}$'           ,
                 'norm_nugenHE'    : r'N$_{\text{nugenHE}}$'         ,
                 'norm_corsika'    : r'N$_{\text{corsika}}$'         ,
                 'domeff'          : r'DOM efficiency'               ,
                 'holeice'         : r'hole ice'                     ,
                 'forward'         : r'hole ice forward'             ,
                 'coin'            : r'coincident fraction'          ,
                 'absorption'      : r'absorption scaling'           ,
                 'scattering'      : r'scattering scaling'            }

Ldatatypes = {'numucc' : r'$\nu_{\mu} + \bar{\nu}_{\mu}$ CC'  ,
              'nuecc'  : r'$\nu_e + \bar{\nu}_e$ CC'          ,
              'nutaucc': r'$\nu_{\tau} + \bar{\nu}_{\tau}$ CC',
              'numunc' : r'$\nu_{\mu} + \bar{\nu}_{\mu}$ NC'  ,
              'nuenc'  : r'$\nu_e + \bar{\nu}_e$ NC'          ,
              'nutaunc': r'$\nu_{\tau} + \bar{\nu}_{\tau}$ NC',
              'nunc'   : r'$\nu + \bar{\nu}$ NC'              ,
              'muon'   : r'atm $\mu$'                         ,
              'iccdata': r'inverted data'                     ,
              'noise'  : r'noise'                             ,
              'data'   : r'data'                              ,
              'mc'     : r'total MC'                          ,
              'bestfit': r'best fit'                          ,
              'noosc'  : r'null hypothesis'                    }

Lobservables = {'reco_e'     : r'reconstructed energy (GeV)'           ,
                'reco_loge'  : r'log$_{10}$ reconstructed energy (GeV)',
                'reco_z'     : r'reconstructed zenith (radian)'        ,
                'reco_zdeg'  : r'reconstructed zenith (degree)'        ,
                'reco_cz'    : r'reconstructed cos zenith'             ,
                'reco_L/E'   : r'reconstructed L/E (km / GeV)'         ,
                'reco_length': r'reconstructed track length (m)'       ,
                'reco_dllh'  : r'$\Delta$ log $L_{\text{reco}}$'       ,
                'mc_e'       : r'MC truth energy (GeV)'                ,
                'mc_loge'    : r'log$_{10}$ MC truth energy (GeV)'     ,
                'mc_z'       : r'MC truth zenith (radian)'             ,
                'mc_zdeg'    : r'MC truth zenith (degree)'             ,
                'mc_cz'      : r'MC truth cos zenith'                   }

Lvariables  = {'nabove200'            : r'charges (PE) above z = 200m'                             ,
               'c2qr6'                : r'charge ratio of first 600ns to total without 2 hits'     ,
               'qr6'                  : r'charge ratio of first 600ns to total'                    ,
               'vertexguessZ'         : r'vertical position (m) of the first hit in time'          ,
               'toi_eigenvalueratio'  : r'tensor of inertia eigenvalue ratio'                      ,
               'ilinefit_Speed'       : r'speed (c) from improved lineFit'                         ,
               'timeTo75'             : r'time (ns) to accumulate 75$\%$ of charges'               ,
               'vertexguessrho'       : r'radial position $\rho_{DOM}$ (m) of the earliest HLC DOM',
               'cog_separation'       : r'distance (m) between CoGs of 1st and 4th quartiles'      ,
               'ztravel'              : r'z travel (m)'                                            ,
               'spe_cz'               : r'SPE 11 cos zenith'                                       ,
               'vetocausalhits'       : r'charges (PE) from causal hits in veto region'            ,
               'fillratio'            : r'fill ratio'                                              ,
               'finiterecorho'        : r'vertex radial position $\rho$ (m) from FiniteReco'       ,
               'finiterecoz'          : r'vertex z position (m) from FiniteReco'                   ,
               'corridornch'          : r'number of channels along corridors'                      ,
               'nchannel'             : r'number of cleaned hit DOMs'                              ,
               'charge_rms_normalized': r'normalized RMS of total charges (PE)'                    ,
               't_rms'                : r'RMS event time (ns)'                                     ,
               'dcfiducialpe'         : r'chargs (PE) in DeepCore fiducial volume'                 ,
               'microcountpe'         : r'MicroCountPE (PE)'                                       ,
               'microcounthits'       : r'MicroCountHits'                                          ,
               'qtotal'               : r'total charges (PE)'                                      ,
               'rtvetoseries250pe'    : r'RTVetoSeries250PE (PE)'                                  ,
               'dcfilterpulses_vetope': r'DCFilterPulses VetoPE (PE)'                              ,
               'geV_per_channel'      : r'reconstructed energy per channel (GeV)'                  ,
               'bdtscore'             : r'BDT score'                                               ,
               'reco_x'               : r'reconstructed vertex x position (m)'                     ,
               'reco_y'               : r'reconstructed vertex y position (m)'                     ,
               'reco_z'               : r'reconstructed vertex z position (m)'                     ,
               'reco_rho'             : r'reconstructed vertex radial $\rho$ position (m)'          }

Lanalyses = {'dragon' :r'Analysis $\mathcal{B}$'     ,
             'greco'  :r'Analysis $\mathcal{A}$'     ,
             'chi2'   :r'$\chi^2$'                   ,
             'modchi2':r'modified $\chi^2$'          ,
             'poisson':r'2 $\times$ Poisson LLH'     ,
             'barlow' :r'2 $\times$ Barlow LLH'      ,
             'trials' :r'number of trials'           ,
             'counts' :r'number of counts in 3 years' }

labels = {'sys'   : Lsystematics,
          'dtypes': Ldatatypes  ,
          'obs'   : Lobservables,
          'var'   : Lvariables  ,
          'ana'   : Lanalyses    }
