#!/usr/bin/env python

####
#### Originally by Michael Larson
#### Modified by Elim Cheung (07/24/2018)
####
#### This script contains the likelihood class. One can
#### choose regular / modified chi2 or poisson / barlow
#### LLH. 
####
#### More info on BarlowLLH:
####    http://lss.fnal.gov/archive/other/man-hep-93-1.pdf
####
####################################################################

from __future__ import print_function
from scipy.optimize import minimize, fsolve
import numpy as np

from misc import Map, Toolbox, InvalidArguments

####################################################################
#### constants needed
####################################################################
bins_line = 4 ## need to change _print_line if bins_line is changed
toolbox = Toolbox ()

####################################################################
#### Likelihood class
####################################################################
class Likelihood (object):

    ''' Likelihood object is a class that perform likelihood
        calculation. One can pick between regular / modified
        chi2 or poisson / barlow LLH. '''
    
    def __init__ (self, histos, dhisto, method, verbose=1):

        ''' initialize likelihood object
        
            :type  histos: dictionary
            :param histos: {'numucc': {'H':[], 'H2':[]},
                            'nuecc' : {'H':[], 'H2':[]}, ...}

            :type  dhisto: dictionary
            :param dhisto: data histogram {'H':[], 'H2':[]}

            :type  method: string
            :param method: 'barlow', 'poisson', 'chi2', 'modchi2'

            :type  verbose: int
            :param verbose: If 0, no print out
                            If 1, basic print out
                            If 2, detailed print out
        '''
        
        self._histos = histos
        self._dhisto = dhisto
        self._method = method
        self._verbose = verbose
        self._check_args ()

        self._shape, self._nbins, self._dtypes = self._set_params ()
        ## change self._histos from dictionary to flattened numpy array
        self._ohistos, self._vhistos = self._order_histos (self._histos)
        self._dhisto = Map ({ 'H':self._dhisto.H.flatten (),
                              'H2':self._dhisto.H2.flatten () })

    def _check_args (self):

        ''' check user's input '''

        head = 'Likelihood:check_args :: '
        ## histos and dhisto are dictionaries
        for arg in ['histos', 'dhisto']:
            if not toolbox.is_dict (eval ('self._'+arg)):
                message = head+arg+' must be a dictionary.'
                raise InvalidArguments (message)

        ## verbose must be int
        if not isinstance (self._verbose, int):
            message = head+'verbose must be an integer.'
            raise InvalidArguments (message)

        ## method
        if not isinstance (self._method, str):
            message = head+'method must be a string.'
            raise InvalidArguments (message)
        if not self._method in ['chi2', 'modchi2', 'poisson', 'barlow']:
            ## if barlow
            warning = head+'You turned on Barlow; make sure you set unweighted info.'
            message = head+'method must be either chi2 / modchi2 / poisson / barlow.'
            raise InvalidArguments (message)

    def _check_barlow_args (self, unhistos, norms):

        ''' check user's barlow inputs '''
        
        head = 'Likelihood:check_barlow_args :: '
        ## unhistos: unweighted histos dictionary
        ## norms: normalization dictoinary
        for arg in ['unhistos', 'norms']:
            if not toolbox.is_dict (eval (arg)):
                message = head+arg+' must be a dictionary.'
                raise InvalidArguments (message)
        
    def _get_bininfo (self, index, totalmc, shape):
        
        ''' obtain informations from every four bins_line

            :type  index: int
            :param index: the bin index in the flattened histogram

            :type  totalmc: a dictionary
            :param totalmc: total mc dictionary {'H':[],'H2':[]}

            :type  shape: tuple
            :param shape: histogram shape

            :return  info: a dictionary
                     info: information of the bins in bins_line
        '''

        # get information about these four bins
        info = {'nbin':[], 'index':[], 'mc':[], 'data':[], 'chi2':[]}
        nbins = np.linspace (index*bins_line, index*bins_line+bins_line-1, bins_line)
        for nbin in nbins:
            index = np.unravel_index (nbin, shape)
            mc, data = totalmc['H'][nbin], self._dhisto['H'][nbin]
            mc2, data2 = totalmc['H2'][nbin], self._dhisto['H2'][nbin]
            chi2 = (mc - data)**2 / (mc2 + data2)
            info['nbin'].append (nbin)
            info['index'].append (index)
            info['mc'].append (np.round (mc, 1))
            info['data'].append (np.round (data, 1))
            info['chi2'].append (np.round (chi2, 2))
        return info
        
    def _print_header (self, shape, nbins, dtypes):
        
        ''' print header
        
            :type  shape: tuple
            :param shape: histogram shape

            :type  nbins: int
            :param nbins: total number of bins of histogram

            :type  ndtypes: int
            :param ndtypes: number of data types 
        '''

        print ('#### ####################################################')
        print ('#### ################ Likelihood Set up #################')
        print ('#### histogram info : {0} ({1})'.format (shape, nbins))
        print ('#### data types included ({0}): {1}'.format (len (dtypes), dtypes))
        if self._verbose > 1: self._print_info (self._histos, shape)
        
    def _print_info (self, histos, shape):

        ''' print bin information

            :type  histos: dictionary
            :param histos: {'numucc': {'H':[], 'H2':[]},
                            'nuecc' : {'H':[], 'H2':[]}, ...}

            :type  shape: tuple
            :param shape: histogram shape
        '''
        
        print ('#### MC / data content:')
        print ('#### {0}'.format ('-'*80))
        totalmc = self._get_totalmc (histos)
        ### print every four bins per print out line
        for i in np.arange (len (hist)/bins_line):
            info = self._get_bininfo (i, totalmc, shape)
            # print each four bins
            self._print_lines (info)
            print ('#### {0}'.format ('-'*80))
        print ('####')

    def _print_bin (self, info):

        ''' print the bin information 

            :type  info: a dictionary
            :param info: information of the bins in bins_line
        '''
            
        line1, line2, line3 = '#### ', '#### ', '#### '
        for i in np.arange (bins_line):
            line1 += '{'+str(i*2)+'}th {'+str(i*2+1)+'}' + ' '*5 + '|'
            line2 += 'mc: {'+str(i*2)+':1f}; d: {'+str(i*2+1)+':1f}' + ' '*3 + '|'
            line3 += 'chi2: {'+str(i)+':2f}' + ' '*6 + '|'
        print (line1.format (info['nbin'][0], info['index'][0], info['nbin'][1], info['index'][1],
                             info['nbin'][2], info['index'][2], info['nbin'][3], info['index'][3] ))
        print (line2.format (info['mc'][0], info['data'][0], info['mc'][1], info['data'][1],
                             info['mc'][2], info['data'][2], info['mc'][3], info['data'][3] ))
        print (line3.format (info['chi2'][0], info['chi2'][1], info['chi2'][2], info['chi2'][3]))

    def _print_barlow (self):
        
        ''' print initial barlow setting '''
        
        print ('#### init ps: {0}'.format(self._ps))
        print ('#### init norms: {0}'.format(self._norms))
        if self._verbose > 1:
            line = '#### N{0}, init weighted, init meanw * A {0}: {1}, {2}, {3}'
            for j in np.arange (len (self._dtypes)):
                print (line.format(j, self._Nj[j], np.sum (self._ohistos[j]),
                                   np.sum (self._meanw[j] * self._unohistos[j]) ))
        
    def _set_params (self):

        ''' set internal parameters and print info
        
            :return shape: tuple
                    shape: histogram shape
            
            :return nbins: int
                    nbins: total number of bins of histogram

            :return ndtypes: int
                    ndtypes: number of data types
        '''
        
        shape = self._dhisto.H.shape
        nbins = len (self._dhisto.H.flatten())
        dtypes = sorted ([ dtype for dtype in self._histos ])
        if self._verbose > 0: self._print_header (shape, nbins, dtypes)
        return shape, nbins, dtypes

    def _get_meanw (self):

        ''' get averaged mean weights per bin
            see Barlow paper for why it is needed
        '''

        ## weighted / unweighted histograms (excluding normalization factors)
        weighted = np.array ([ self._ohistos[j] / self._norms[j]
                               for j in np.arange (self._ndtypes) ])
        weighted_unweighted = np.nan_to_num (weighted/self._unohistos)

        ## massage empty bins
        for j in np.arange (self._ndtypes):
            ## check if any bin in each dtype have ratio = 0
            indicies = weighted_unweighted[j]==0
            ## special treatment if this dtype has no events in every bin
            if not np.sum(indicies)==0:
                m = weighted_unweighted[j][np.logical_not (indicies)]
                # Weird case: every bin has a total weight of 0. This
                # can happen if you have 0 of one of the neutrino
                # types [eg, taus]
                if np.sum(m)==0: continue
                weighted_unweighted[j][indicies] = np.min (m)
        return weighted_unweighted

    def set_barlow (self, unhistos, norms):

        ''' set unweighted histograms for barlow

            :type  unhistos: dictionary
            :param unhistos: unweighted histograms {'numucc': {'H':[], 'H2':[]},
                                                    'nuecc' : {'H':[], 'H2':[]}, ...}

            :type   norm: a dictionary
            :param  norm: normalization factors {'numucc':x, 'nuecc':x, 'nutaucc':x, ...}
        '''

        self._check_barlow_args (unhistos, norms)

        ## change self._unhistos/self._norms from dictionary to numpy array
        self._unohistos, self._unvhistos = self._order_histos (unhistos)
        self._norms = np.array ([ norms[dtype] for dtype in self._dtypes ])

        ## these variables are based on barlow paper
        self._Nj = np.array ([ np.sum (self._ohistos[j] / self._norms[j])
                               for j in np.arange (self._ndtypes) ]).astype (float)
        self._ps = np.array ([ self._norms[j] * np.sum (self._ohistos[j]) / self._Nj[j]
                               for j in np.arange (self._ndtypes) ])
        self._meanw = self._get_meanw ()

        ## print init info
        if self._verbose > 0: self._print_barlow ()
        return

    def _get_totalmc (self, histos):

        ''' get total mc from histograms of all data types

            :type  histos: dictionary
            :param histos: {'numucc': {'H':[], 'H2':[]},
                            'nuecc' : {'H':[], 'H2':[]}, ...}

            :return  totalmc: a dictionary
                     totalmc: total MC {'H':[], 'H2':[]}
        '''
            
        mc = Map ({'H':np.zeros (self._shape), 'H2':np.zeros (self._shape)})
        for i, dtype in enumerate (self._dtypes):
            mc['H']  += histos[dtype]['H']
            mc['H2'] += histos[dtype]['H2']
        return mc
                        
    def _order_histos (self, histos):

        ''' order histograms in order of self._dtypes

            :type  histos: dictionary
            :param histos: {'numucc': {'H':[], 'H2':[]},
                            'nuecc' : {'H':[], 'H2':[]}, ...}

            :return  ohistos: a multi-dimensional array
                     ohistos: flattened histograms in order of self._dtypes

            :return  vhistos: a multi-dimensional array
                     vhistos: flattened variances in order of self._dtypes
        '''

        ohistos, vhistos = [], []
        for dtype in self._dtypes:
            ohistos.append (histos[dtype]['H'].flatten ())
            vhistos.append (histos[dtype]['H2'].flatten ())
        return np.array (ohistos), np.array (vhistos)
    
    def get_ts (self):

        ''' calculate ts values from all bins
            test statistics (TS) is either chi2 / 2. or LLH value

            :return totalTS: float
                    totalTS: total ts from all bins

            :return binTS: a multi-dimensional array
                    binTS: ts value per bin (histogram shape)

            :return As: multi-dimensional array
                    As: fitted barlow llh value
        '''

        ## set up variables
        isbarlow = True if 'barlow' in self._method else False
        As = None 
        if isbarlow:
            if any ([ self._ps[j]<0 for j in np.arange (len (self._dtypes)) ]): return 1e10
            As = np.empty (self._unohistos) 

        ## loop through each bin
        binTS, totalTS = [], 0
        for nbin in np.arange (self._nbins):
            ts, An = self.get_binTS (nbin)
            binTS.append (ts)
            totalTS += ts
            if isbarlow: As[:,nbin] = An
        
        return totalTS, np.array (binTS).reshape (self._shape), As

    def get_binTS (self, nbin):

        ''' get the test statistics for a given bin
            ts = chi2 / 2. or llh value

            :type  nbin: int
            :param nbin: index of the flattened histogram

            :return ts: float
                    ts: ts of this bin

            :return An: 1D array
                    An: fitted barlow llh value for all data types
        '''
        
        ## info of this bin
        index = np.unravel_index (nbin, self._shape)
        di = self._dhisto.H[nbin]
        fi = np.sum ([ self._ohistos[j][nbin] for j in np.arange (len (self._dtypes)) ])

        ## print info
        if self._verbose > 1: 
            print ('#### +----------- {0}th bin ({1}) -----------+'.format(nbin, index))
            print ('#### +---- di, mci: {0}, {1}'.format(di, fi))
        
        ## determine TS
        if 'chi2' in self._method:
            ## determine chi2
            ts = self._calculate_chi2 (nbin, di, fi) ## chi2 / 2.
            An = None
        else:
            ## determine likelihood
            ts, An = self._calculate_llh (nbin, index, di, fi)
        
        ## print info
        if self._verbose > 1: 
            print ('#### +---- {0}: {1}'.format (self._method, 2*ts))

        return ts, An
    
    def _calculate_chi2 (self, nbin, di, fi):

        ''' get the regular / modified chi2 * 0.5 for a given bin

            :type  nbin: int
            :param nbin: index of the flattened histogram

            :type  di: float
            :param di: data count in the nth bin

            :type  fi: float
            :param fi: total MC count in the nth bin

            :return chi2: float
                    chi2: chi2 / 2. value
        '''

        ## if empty bins: chi2 = 0.
        if fi==0: return 0.0
        if 'modchi2' in self._method:
            ## collect total variance from all data types
            vfi = np.sum ([ self._vhistos[j][nbin] for j in np.arange (len (self._dtypes)) ])
            return 0.5 * (fi - di)**2 / (fi + vfi) 
        return 0.5 * (fi - di)**2 / fi 

    def _calculate_llh (self, nbin, di, fi):

        ''' get the poisson / barlow LLH value for a given bin

            :type  nbin: int
            :param nbin: index of the flattened histogram

            :type  di: float
            :param di: data count in the nth bin

            :type  fi: float
            :param fi: total MC count in the nth bin

            :return llh: float
                    llh: llh value

            :return An: a 1D array
                    An: fitted value for Barlow LLH (length = n datatypes)
        '''
        
        if 'barlow' in self._method:
            return self._calculate_barlow (nbin, di, fi)

        llh = 0.
        if fi > 0: llh += di * np.log (fi) - fi
        if di > 0: llh -= di * np.log (di) - di
        return -llh, None

    def _calculate_barlow (self, nbin, di, fi):

        ''' get the barlow LLH value for a given nth or ith bin
               -- solve for ti (Eq. 26 in Barlow's paper)
               -- solve for Ai (Eq. 25 in Barlow's paper)

            :type  nbin: int
            :param nbin: index of the flattened histogram

            :type  di: float
            :param di: data count in the nth bin

            :type  fi: float
            :param fi: total MC count in the nth bin

            :return llh: float
                    llh: llh value

            :return An: a 1D array
                    An: fitted value for Barlow LLH (length = n datatypes)
        '''

        ## ai = unweighted counts in this bins from all data types
        ## wi = mean weights in this bins from all data types
        ai = np.array ([ self._unohistos[j][nbin] for j in np.arange (self._dtypes) ])
        wi = np.array ([ self._meanw[j][nbin] for j in np.arange (self._dtypes) ])

        ## solve for ti (a scalar)
        ti = self._barlow_solve_ti (ai, wi, di)

        ## solve for Aji (an array of N data types)
        ## ti may be modified if special case
        ti, Ai = self._barlow_solve_Ai (ai, wi, ti)

        ## solve for fi for this ith bin (fi = a scalar)
        fi = np.sum ([ self._ps[j]*wi[j]*Ai[j] for j in np.arange (len (self._dtypes)) ])

        ## evaluate barlow LLH
        llh = 0
        # poisson part
        if fi > 0: llh += di * np.log (fi) - fi
        if di > 0: llh -= di * np.log (di) - di
        # mc uncertainty penalty part
        for j in np.arange (len (self._dtypes)):
            if Ai[j] > 0:
                llh += ai[j] * np.log (Ai[j]) - Ai[j]
                llh -= ai[j] * np.log (ai[j]) - ai[j]

        # print penalty
        if self._verbose > 1: self._print_barlow_penalty (ai, Ai)
                
        return -llh, Ai

    def _print_barlow_penalty (self, ai, Ai):

        ''' print penalty

            :type  ai: 1D numpy array
            :param ai: unweighted counts in this bin from all data types

            :type  Ai: 1D numpy array
            :param Ai: fitted unweighted counts in this bin from all data types           
        '''

        print ('#### ++++++++++++++++++++++++++++++++++++++++++++++++')
        print ('#### +---- Penalty due to MC uncertainty')
        print ('#### +---- -----------------------------------------+')
        print ('#### +----  dtype |    ai    |    Ai    |  penalty  |')
        for j in np.arange (len (self._dtypes)):
            line = '#### +---- {0:6} | {1:8} | {2:8} | {3:9} |'
            penalty = (ai[j]*np.log (Ai[j]) - Ai[j]) - (ai[j] * np.log (ai[j]) - ai[j])
            print (line.format (j, np.round (ai[j],4), np.round (Ai[j], 4),
                                np.round (penalty, 2) ))
    
    def _barlow_solve_ti (self, ai, wi, di):

        ''' solve for ti of nth bin from Eq. 26 in Barlow's paper
            ti is a scaler representing the difference bewteen fi and di

            :type  di: float
            :param di: data count in the nth or ith bin
        
            :type  wi: 1D numpy array
            :param wi: mean weights in this bin from all data types

            :type  ai: 1D numpy array
            :param ai: unweighted counts in this bin from all data types

            :return ti: float
                    ti: checked / modified value of ti
        '''

        ## solve ti
        ti_func = lambda ti: di / (1-ti) - np.sum ([ self._ps[j]*ai[j]*wi[j] / (1+self._ps[j]*wi[j]*ti)
                                                     for j in np.arange (len (self._dtypes)) ])
        ti = fsolve (ti_func, 0) [0]

        ## print info
        if self._verbose > 1:
            print ('#### +---- current ps       : {0}'.format (self._ps))
            print ('#### +---- current norms    : {0}'.format (self._norms))
            w = np.array ([ self._ohistos[j][nbin] for j in np.arange (len (self._dtypes)) ])
            print ('#### +---- weighted counts  : {0}'.format (w))
            print ('#### +---- unweighted counts: {0}'.format (ai))
            print ('#### +---- mean weight      : {0}'.format (wi))
            print ('#### +---- ti               : {0} ({1})'.format (ti, ti_func (ti)))

        ## check value of ti 
        return self._barlow_check_ti (ti, wi)
        
    def _barlow_check_ti (self, ti, wi):
        
        ''' check value of ti not to be smaller
            than the lowest possible value

            :type  ti: float
            :param ti: value of ti from Eq. 26 in Barlow's paper

            :type  wi: 1D numpy array
            :param wi: mean weights in this bin from all data types

            :return ti: float
                    ti: checked / modified value of ti
        '''
        
        ## apply boundary conditions to ti according to the paper
        max_pw = max ([ self._ps[j]*wi[j] for j in np.arange (len (self._dtypes)) ])
        lowest_ti = -1. / max_pw
        if ti < lowest_ti:
            ## replace ti to the lowest possible value
            ti = lowest_ti
            ## print new info
            if self._verbose > 1:
                print ('#### +---- ** max p*w       : {0}'.format (max_pw))
                print ('#### +---- ** max lowest ti : {0}'.format (lowest_ti))
                print ('#### +---- ** new ti        : {0} ({1})'.format (ti, ti_func (ti)))
        return ti

    def _barlow_solve_Ai (self, ai, wi, ti):

        ''' solve for Ai (Eq. 25 in Barlow's paper)

            :type  ai: 1D numpy array
            :param ai: unweighted counts in this bin from all data types

            :type  wi: 1D numpy array
            :param wi: mean weights in this bin from all data types

            :type  ti: a float
            :param ti: value of ti from Eq. 26 in Barlow's paper

            :return ti: a float
                    ti: updated value if special case is met

            :return Ai: a numpy array
                    Ai: fitted unweighted counts for all data types
        '''

        ## Eq. 20 in Barlow's paper
        Ai = np.array ([ ai[j] / (1+self._ps[j]*wi[j]*ti) for j in np.arange (len (self._dtypes)) ])
        if self._verbose > 1: print ('#### +---- Ai               : {0}'.format (Ai))
        ## look for special case (Eq. 20 in Barlow's paper)
        ks = [ j for j in np.arange (len (self._dtypes)) if ai[j]==0. and Ai[j]>0. ]
        ## check for special case
        ## both Ai and ti are modified if Ai[j] > 0 and ai[j] == 0
        if len (ks) > 0:
            ti, Ai = self._barlow_check_Ai (ks, wi, ai, ti, Ai)                
        ## for any Ai < 0, LLH max happens at A_i = 0
        Ai[Ai<0] = 0.0
        return ti, Ai
    
    def _barlow_check_Ai (self, ks, wi, ai, ti, Ai):

        ''' check value of Ai
            special treatment when Ai[j]>0 and ai[j]==0
        
            :type  ks: a list
            :param ks: indices of special case

            :type  wi: 1D numpy array
            :param wi: mean weights in this bin from all data types

            :type  ai: 1D numpy array
            :param ai: unweighted counts in this bin from all data types

            :type  ti: a float
            :param ti: value of ti from Eq. 26 in Barlow's paper

            :type  Ai: a numpy array
            :param Ai: fitted unweighted counts for all data types

            :return ti: a float
                    ti: updated value of ti 

            :return Ai: a numpy array
                    Ai: updated value of Ai (Eq. 20 in Barlow's paper)
        '''
        # max p in the special cases
        pk = np.max ([ self._ps[k]*wi[k] for k in ks ])
        # index of max p*w from all data types
        maxk = np.argmax (self._ps*wi)
        # update ti
        ti = -1./pk
        # print info
        if self._verbose > 1:
            print ('#### +---- ** SPECIAL Ai[j] > 0 && ai[j] == 0 CASE !')
            print ('#### +---- ** ks            : {0}'.format (ks))
            print ('#### +---- ** pk, maxk      : {0}, {1}'.format (pk, maxk))
            print ('#### +---- ** updated ti    : {0}'.format (ti))
            print ('#### +++++++++++++++++++++++++++++++++++++')
            print ('#### +---- Update Ai')
            print ('#### +---- ------------------------------+')
            print ('#### +----  dtype | before Ai | after Ai |')
        # update Ai with updated ti (Eq. 22 in Barlow's paper)
        for j in np.arange (len (self._dtypes)):
            if j == maxk:
                newA = di / (1+pk) - np.sum ([ self._ps[m]*wi[m]*ai[m] / (pk-self._ps[m]*wi[m])
                                               for m in np.arange (len (self._dtypes)) if not m==j ])
            else:
                newA = ai[j] / (1+self._ps[j]*wi[j]*ti)
            if self._verbose > 1: print ('#### +----  {0:5} |  {1:9} | {2:8} |'.format (j, Ai[j], newA))
            Ai[j] = newA
        if self._verbose > 1:
            print ('#### +++++++++++++++++++++++++++++++++++++')
            print ('#### +---- ** updated Ai    : {0}'.format (Ai))
        return ti, Ai
