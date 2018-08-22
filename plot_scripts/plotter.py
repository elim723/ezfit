#!/usr/bin/env python

####
#### By Elim Cheung (07/24/2018)
####
#### This script contains the plotter class that makes
#### plotting easier for a numu disappearance analysis.
####
#### The actual plotting scripts to run are located in
#### plot_scripts folder.
####
####################################################################

from __future__ import print_function
from copy import deepcopy
import numpy as np

######################################
#### matplotlib tools
######################################
import matplotlib
### no interactive plots
matplotlib.use('Agg')
### import coloar map
from matplotlib import cm
### import plot-tools
import matplotlib.pyplot as plt
### import gridspec for multi plots
import matplotlib.gridspec as gridspec
### import axes 3d for 3D contours
from mpl_toolkits.mplot3d import axes3d
### import AnchoredText for text in plots
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
### set up text styles for published plots
plt.rc ('text', usetex=True)
plt.rc ('font', family='sans-serif')
plt.rc ('font', serif='Computer Modern Roman')
plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
### set up global axis ticks
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['xtick.major.pad']='5'
plt.rcParams['ytick.major.pad']='5'

######################################
#### plot settings
######################################
from global_settings import dm21, seconds_per_year
from global_settings import colors, hatches, labels

import sys
sys.path.append ('../')
from misc import default_edges as edges

######################################
#### standard plotting canvas class
######################################
class Canvas (object):

    ''' a class that define canvas for various
         plots '''

    def __init__ (self, figsize, dimensions, 
                  width_ratios=None,
                  height_ratios=None,
                  wspace=0.1, hspace=0.1):

        ''' initialize canvas by its dimensions /
            spacing / ratios

            :type  figsize: tuple
            :param figsize: canvas width / height in inches

            :type  dimensions: tuple
            :param dimensions: number of subplots in row and column

            :type  width_ratios: list
            :param width_ratios: ratio of the width of subplots

            :type  height_ratios: list
            :param height_ratios: ratio of the height of subplots

            :type  wspace: float
            :param wspace: spacing in width between plots

            :type  hspace: float
            :param hspace: spacing in height between plots
        '''
        
        self.h  = plt.figure (figsize=figsize)
        self.gs = gridspec.GridSpec (dimensions[0], dimensions[1],
                                     width_ratios=width_ratios,
                                     height_ratios=height_ratios )
        self.gs.update (wspace=wspace, hspace=hspace)

    def set_style (self, **kwargs):

        for key, value in kwargs.iteritems ():
            setattr (self, key, value)

        ## must have the following keys
        for key in ['edges', 'labelsize', 'ticksize']:
            value = getattr (self, key, None)
            if value is None:
                value = 12 if 'label' in key else \
                        9  if 'tick'  in key else \
                        {'x':edges['e'], 'y':edges['z']}
                setattr (self, key, value)
        
    @staticmethod
    def get_range (*contents):

        ''' obtain the range from at least one content

            :type  content: a list of 1/2/multi- dimensional arrays
            :param content: content of a 1D / 2D histogram
        '''

        xmin, xmax = np.inf, -np.inf
        for content in contents:
            array = np.nan_to_num (content).flatten ()
            if xmin > np.min (array):
                xmin = np.min (array)
            if np.max (array) > xmax:
                xmax = np.max (array)
        return xmin, xmax

    def plot_2D (self, axis, content, vrange):

        ''' plot 2D content

            :type  axis: a matplotlib.axes object
            :param axis: current axis for plotting

            :type  content: a multi-dimensional array
            :param content: content per bin

            :type  vrange: tuple
            :param vrange: color bar ranges
        '''
        
        ### plot 2d
        mH = np.ma.masked_array (content).T
        mH.mask = np.logical_not (mH)
        plt.pcolormesh (self.edges['x'], self.edges['y'], mH,
                        vmin=vrange[0], vmax=vrange[1],
                        cmap=self.cmap, alpha=0.7)

        ### print bin content if asked
        if self.print2D: self.print_2D (content, vrange)

    def print_2D (self, content, vrange):

        ''' print 2D content

            :type  content: a multi-dimensional array
            :param content: content per bin
        '''

        rounding = self.rounding if hasattr (self, 'rounding') else 1.0
        contentsize = self.contentsize if hasattr (self, 'contentsize') else 3
        ### loop through each bin
        for edex, eedge in enumerate (self.edges['x'][:-1]):
            for zdex, zedge in enumerate (self.edges['y'][:-1]):
                ## rounding the value in this bin
                value = round (content[edex][zdex], rounding)
                ## set text and its color
                text = '-' if not np.isfinite (value) else str (value)
                darker = (value>vrange[1]*0.85 or value<vrange[0]*0.85)
                tcolor = 'white' if text=='-' or darker else 'black'
                ## set the coordinate of the printout
                x = self.edges['x'][edex] + (self.edges['x'][edex+1]-self.edges['x'][edex])/8.
                y = self.edges['y'][zdex] + (self.edges['y'][zdex+1]-self.edges['y'][zdex])/3.
                ## actually print
                plt.annotate (text, xy=(x, y), xycoords='data', color=tcolor, fontsize=contentsize)

    def get_power10s (self, power10x=False, power10y=False):

        ''' print tick labels as power of 10 

            :type  power10x: boolean
            :param power10x: if True, print 10**self.edges['x'] instead

            :type  power10y: boolean
            :param power10y: if True, print 10**self.edges['y'] instead

            :return eticks: a dictionary 
                    eticks: tick labels to be printed
                            {'x':[], 'y':[]}
        '''
        
        eticks = deepcopy (self.edges)
        for a in ['x', 'y']:
            edge = np.power (10, self.edges[a]) if eval ('power10'+a) else self.edges[a]
            fmt = '%.1f' if eval ('power10'+a) else '%.2f'
            eticks[a] = [fmt % x for x in edge]
        return eticks
                
    def format_2D (self, axis, title=None, power10x=False, power10y=False):

        ''' format 2D axis

            :type  axis: a Matplotlib.Axes object
            :param axis: axis to be formatted

            :type  title: a string
            :param title: title of the subplot
        '''
        
        #### set x, y labels and title
        axis.set_xlabel (self.labels['x'], fontsize=self.labelsize)
        axis.set_ylabel (self.labels['y'], fontsize=self.labelsize)
        if title: axis.set_title (title, fontsize=self.labelsize)

        #### set x, y ticks
        eticks = self.get_power10s (power10x=power10x, power10y=power10y)
        plt.xticks (self.edges['x'], eticks['x'])
        plt.yticks (self.edges['y'], eticks['y'])
        axis.tick_params (axis='x', labelsize=self.ticksize)
        axis.tick_params (axis='y', labelsize=self.ticksize)

        #### set grid lines
        for xmaj in self.edges['x']:
            axis.axvline (x=xmaj,ls=':', color='grey', alpha=0.2)
        for ymaj in self.edges['y']:
            axis.axhline (y=ymaj,ls=':', color='grey', alpha=0.2)

        #### set x, y limits
        axis.set_xlim (self.edges['x'][0], self.edges['x'][-1])
        axis.set_ylim (self.edges['y'][0], self.edges['y'][-1])
                
    def plot_colorbar (self, axis, (vmin, vmax), label):

        ''' plot colorbar

            :type  axis: a matplotlib.axes object
            :param axis: the axis things to be plotted

            :type  vmin/max: float
            :param vmin/max: colorbar range

            :type  label: string
            :param label: colorbar label
        '''
        
        norm = matplotlib.colors.BoundaryNorm (np.linspace (vmin, vmax, 1000), self.cmap.N)
        cb = matplotlib.colorbar.ColorbarBase (axis, cmap=self.cmap, norm=norm,
                                               format=self.fmt, alpha=0.7,
                                               orientation='vertical',
                                               ticks=np.linspace (vmin, vmax, 10),
                                               spacing='uniform')
        cb.ax.tick_params (labelsize=self.ticksize, length=2)
        cb.set_label (label, fontsize=self.labelsize)

    def save (self, name, title=None):

        ''' save current canvas to name
        
            :type  name: string
            :param name: full path (/address/name.png or .pdf)

            :type  title: string
            :param title: super title of the canvas
        '''

        if title: plt.suptitle (title, fontsize=self.labelsize)        
        self.h.savefig (name)
        plt.close('all')
        
######################################
#### histoeffect class
######################################
class HistoEffect (Canvas):

    def __init__ (self, **kwargs):

        ### default canvas / plot arragements
        super (HistoEffect, self).__init__ ((7.5, 5.5), (2, 3),
                                            width_ratios=[28, 28, 1],
                                            height_ratios=[1, 1],
                                            wspace=0.4, hspace=0.4)

        ### set style
        self.set_style (**kwargs)
        
    def sub (self, pid, lower, upper):

        ''' plot a sub plot

            :type  pid: string
            :param pid: cascade or track

            :type  lower: a 2D numpy array
            :param lower: chi / % between lower sigma
                          and baseline per bin

            :type  upper: a 2D numpy array
            :param upper: chi / % between upper sigma
                          and baseline per bin
        '''

        gsindices = [0, 1] if pid=='cascade' else [3, 4]
        vrange = self.get_range (lower, upper)

        for index, gsindex in enumerate (gsindices):
            axis = self.h.add_subplot (self.gs[gsindex])
            content = lower if index==0 else upper
            self.plot_2D (axis, content, vrange)
            stitle = '; lower' if index==0 else '; upper'
            title = self.title + '; ' + pid + stitle
            self.format_2D (axis, title, power10x=True)
        return vrange
            
    def plot (self, param, lower, upper):

        ''' plot systematic effects on histogram

            :type  param: string
            :param param: name of the systematic to be plotted                                                                                                                                                          
            :type  lower: a dictionary
            :param lower: comparison between lower sigma and baseline
                          {'cascade':[], 'track':[]} }

            :type  upper: a dictionary
            :param upper: comparison between upper sigma and baseline
                          {'cascade':[], 'track':[]} }
        '''

        ### plot 2D content
        for pid in ['cascade', 'track']:
            vrange = self.sub (pid, lower[pid], upper[pid])
            ## plot color bar
            index = 2 if pid == 'cascade' else 5
            axis = self.h.add_subplot (self.gs[index])
            self.plot_colorbar (axis, vrange, self.cbarlabel)

    
            
        
#### define contour colors for one single experiment (i.e. no multiple trials)
#cf90c, profilec, cf68c = [ np.append(np.array(Dark2_3.mpl_colors[index]), 1) for index in np.arange(len(Dark2_3.mpl_colors)) ] 

########################################################
#### Functions to assign colors
########################################################
def get_color_gradients(color, steps):
    
    ''' get a gradient of the given color '''

    gradcolors = {}
    for s in np.arange(steps):
        gradcolors[str(s)] = np.append (color[:3], (s+1)/float(steps) )
    return gradcolors

def get_colors(dictionary):
    
    ''' get a range of colors from red to blue based on 
        number of key/value in the given dictionary '''
    
    array = (np.arange( len(dictionary) )).astype(float)
    scaled_scores = (array - array.min()) / array.ptp()
    color_array = {}; colors = RdYlBu_9.mpl_colormap(scaled_scores)
    for index, key in enumerate( dictionary.keys() ): 
        color_array.update({key:colors[index]})
    return color_array

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):

    ''' shift the given color map based on midpoint '''

    cdict = { 'red': [], 'green': [], 'blue': [], 'alpha': [] }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack( [ np.linspace(0.0, midpoint, 128, endpoint=False), 
                               np.linspace(midpoint, 1.0, 129, endpoint=True)  ] )

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    return newcmap

###################################################
#### Functions to calculate sigmas
###################################################
def statistics(values, w):

    ''' calculate median/sigmas from all trials '''

    stat={}; sample = np.array(values)[np.isfinite(values)]
    if len(sample) > 0 :
        #stadard deviation
        indices = np.argsort(sample) ### indices of sorted array
        sample = np.array( [sample[i] for i in indices] )
        w = np.array( [w[i] for i in indices] )
        # have them sorted. now find middle 68%?
        cumulative = np.cumsum(w)/np.sum(w)
        stat['lower_90cl']   = sample[cumulative>0.05][0] # lower part is at 50%-45% = 5%        
        stat['lower_1sigma'] = sample[cumulative>0.16][0] # lower part is at 50%-34% = 16%
        stat['median']       = sample[cumulative>0.50][0] # median
        stat['upper_1sigma'] = sample[cumulative>0.84][0] # upper is 50%+34% = 84%
        stat['upper_90cl']   = sample[cumulative>0.95][0] # upper is 50%+45% = 95%
    return stat

###################################################
#### Functions to sort files based on numerical 
#### values in the file basenames
###################################################
def natural_sort (filenames): 

    ''' sort filenames by the first values of their names '''

    convert = lambda text: int(text)/float(np.power(10,len(text))) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(filenames, key = alphanum_key)

###################################################
#### Functions to format different axies
###################################################
def axis_dllh_along_bf_theta23 (axis, sin_sq_theta23s, llh_lim):

    ''' format -2dllh along fixed best fit dm31 axis '''

    axis.set_ylabel(r'-2$\Delta$LLH', fontsize=12)
    axis.set_xlim(sin_sq_theta23s[0], sin_sq_theta23s[-1])
    axis.set_ylim(0, llh_lim)
    axis.yaxis.set_ticks(np.arange(0.0, 4.0, 1))
    for xmaj in np.linspace(sin_sq_theta23s[0], sin_sq_theta23s[-1], 11): #HERE: 11 if 0.25-0.75; 17 if 0.1, 0.9
        axvline(x=xmaj,ls=':', color='gray', alpha =0.3)
    for ymaj in axis.yaxis.get_majorticklocs():
        axhline(y=ymaj,ls=':', color='gray', alpha =0.3)
    yticks = axis.yaxis.get_major_ticks()
    yticks[0].set_visible(False)
    axis.tick_params(axis='y', labelsize=8)
    axis.get_xaxis().set_ticklabels([])
    return

def axis_dllh_along_bf_dm31 (axis, dm31s, llh_lim):

    ''' format -2dllh along fixed best fit theta23 axis '''

    axis.set_ylim(dm31s[0], dm31s[-1]) #HERE
    #axis.set_ylim(2.0, 4.0)
    axis.set_xlim(0, llh_lim)
    axis.get_yaxis().set_ticks([])
    axis.xaxis.set_ticks_position("top")
    axis.set_xlabel(r'-2$\Delta$LLH', fontsize=12)
    axis.tick_params(axis='x', labelsize=8)
    axis.xaxis.set_ticks(np.arange(0.0, 4.0, 1))
    for xmaj in axis.xaxis.get_majorticklocs():
        axvline(x=xmaj,ls=':', color='gray', alpha =0.3)
    for ymaj in np.linspace(dm31s[0], dm31s[-1], 11): #HERE: 9 if 2.0-4.0; 11 if 2.0-3.0
        #for ymaj in np.linspace(2.0, 4.0, 9):
        axhline(y=ymaj,ls=':', color='gray', alpha =0.3)
    xticks = axis.xaxis.get_major_ticks()
    xticks[0].set_visible(False)
    axis.xaxis.set_label_coords(0.5, 1.13) 
    return

def axis_contour (axis, sin_sq_theta23s, dm31s):

    ''' format contour axes '''

    for xmaj in np.linspace(sin_sq_theta23s[0], sin_sq_theta23s[-1], 11): #HERE: 11 if 0.25-0.75; 17 if 0.1, 0.9
        axvline(x=xmaj,ls=':', color='gray', alpha =0.3)
    #axis.set_ylim(2.0, 4.0) # HERE
    axis.set_ylim(dm31s[0], dm31s[-1])
    #for ymaj in np.linspace(2.0, 4.0, 9 ):
    for ymaj in np.linspace(dm31s[0], dm31s[-1], 11 ): #HERE: 9 if 2.0-4.0; 11 if 2.0-3.0
        axhline(y=ymaj,ls=':', color='gray', alpha =0.3)
    axis.set_xlim(sin_sq_theta23s[0], sin_sq_theta23s[-1])
    axis.set_xlabel(r'sin$^2$ $\theta_{23}$', fontsize=15)
    axis.set_ylabel(r'$\Delta$ m$_{31}^2$ ($10^{-3}$ eV$^2$)', fontsize=15)
    axis.tick_params(axis='x', labelsize=10)
    axis.tick_params(axis='y', labelsize=10)
    return

def axis_contour3d (axis, sin_sq_theta23s, dm31s, zlabel, vmin, vmax):
                       
    ''' format contour3d axes '''
    
    label_fontsize, tick_fontsize = 12, 7
    axis.set_ylim3d(dm31s[0], dm31s[-1])
    axis.set_xlim3d(sin_sq_theta23s[0], sin_sq_theta23s[-1])
    axis.set_zlim3d(vmin, vmax)    
    axis.set_xlabel(r'sin$^2$ $\theta_{23}$', fontsize=label_fontsize)
    axis.set_ylabel(r'$\Delta$ m$_{31}^2$ ($10^{-3}$ eV$^2$)', fontsize=label_fontsize)
    axis.set_zlabel(zlabel, fontsize=label_fontsize)
    axis.tick_params(axis='x', labelsize=tick_fontsize)
    axis.tick_params(axis='y', labelsize=tick_fontsize)
    axis.tick_params(axis='z', labelsize=tick_fontsize)    
    axis.xaxis.labelpad = 20
    axis.yaxis.labelpad = 20
    axis.zaxis.labelpad = 20
    return
    
def axis_sys_behavior (axis, sin_sq_theta23s, dm31s, vmin, vmax, index):
                       
    ''' format systematic behavior axes '''
    
    label_fontsize, tick_fontsize = 7, 5
    axis.set_ylim(dm31s[0], dm31s[-1])
    axis.set_xlim(sin_sq_theta23s[0], sin_sq_theta23s[-1])
    if index>7:    axis.set_xlabel(r'sin$^2$ $\theta_{23}$', fontsize=label_fontsize)
    if index%4==0: axis.set_ylabel(r'$\Delta$ m$_{31}^2$ ($10^{-3}$ eV$^2$)', fontsize=label_fontsize)
    axis.tick_params(axis='x', labelsize=tick_fontsize, length=2)
    axis.tick_params(axis='y', labelsize=tick_fontsize, length=2)
    return

def axis_sys_fitness (axis, xticklabel, xtickvalue, ymin, ymax):
                       
    ''' format systematic fit axis '''
    
    label_fontsize, tick_fontsize = 12, 8
    axis.set_xticklabels(xticklabel, fontsize=label_fontsize, rotation=90)
    axis.set_ylabel(r'(bestfit - prior mean) / $\sigma$', fontsize=label_fontsize)
    axis.set_xticks(xtickvalue)
    axis.set_xlim(0, len(xticklabel)+1)
    axis.set_ylim(ymin, ymax)
    axis.tick_params(axis='x', labelsize=tick_fontsize, length=2)
    axis.tick_params(axis='y', labelsize=tick_fontsize, length=2)
    for xmaj in axis.xaxis.get_majorticklocs():
        axvline(x=xmaj,ls=':', color='gray', alpha =0.2)
    for ymaj in axis.yaxis.get_majorticklocs():
        axhline(y=ymaj,ls=':', color='gray', alpha =0.2)
    return

###################################################
#### Functions to build plots
###################################################
def build_dllh_along_theta23 ( axis, sin_sq_theta23s, profile, bf_theta23, color='black', median=False ):

    ''' plot -2dllh profile along changing theta23  '''

    if median:
        ### fill the region where 68% and 90% of trials lie
        gcolors = get_color_gradients (color, 3)
        axis.fill_between( sin_sq_theta23s, profile['lower_90cl'], profile['upper_90cl'], facecolor=gcolors['0'], alpha=0.15, linewidth=0 )
        axis.fill_between( sin_sq_theta23s, profile['lower_1sigma'], profile['upper_1sigma'], facecolor=gcolors['1'] , alpha=0.3, linewidth=0 )
        color, llhprofile = gcolors['2'], profile['median']
    else:
        ### add the best fit point in LLH profile
        bf_sin2theta23 = np.sin(bf_theta23*np.pi/180.)**2
        sin_sq_theta23s = np.array(sorted( np.append(sin_sq_theta23s, bf_sin2theta23) ))
        index =  np.where(sin_sq_theta23s==bf_sin2theta23)[0][0]
        llhprofile = np.insert(profile, index, 0.0)

    p, = axis.plot( sin_sq_theta23s, llhprofile, color=color, alpha=1, linewidth=2 )
    return

def build_dllh_along_dm31 ( axis, dm31s, profile, bf_dm31, color='black', median=False ):

    ''' plot -2dllh profile along changing dm31 '''

    if median:
        ### fill the region where 68% and 90% of trials lie
        gcolors = get_color_gradients (color, 3)
        axis.fill_betweenx( dm31s, profile['lower_90cl'], profile['upper_90cl'], facecolor=gcolors['0'], alpha=0.15, linewidth=0 )
        axis.fill_betweenx( dm31s, profile['lower_1sigma'], profile['upper_1sigma'], facecolor=gcolors['1'], alpha=0.3, linewidth=0 )
        color, llhprofile = gcolors['2'], profile['median']
    else:
        ### add the best fit point in LLH profile
        dm31s = np.array(sorted(np.append(dm31s, bf_dm31)))
        index =  np.where(dm31s==bf_dm31)[0][0]
        llhprofile = np.insert(profile, index, 0.0)

    #### HERE
    p, = axis.plot( llhprofile, dm31s, color=color, alpha=1, linewidth=2 )
    return

def build_contour ( axis, llh, sin_sq_theta23s, dm31s, lines, labels, 
                    color='red', only90=False, padlabel='', median=False ):

    ''' plot contour '''

    levels = [4.605] if only90 else [2.288, 4.605]
    llh = llh['median'] if median else llh
    CS = plt.contour (sin_sq_theta23s, dm31s, llh, levels=levels)
    if only90:
        #### HERE
        CS.collections[0].set_color(color)
        CS.collections[0].set_linewidth(2)
        CS.collections[0].set_alpha(0.7)
        
        lines.append(CS.collections[0])
        labels.append(r'90$\%$ '+padlabel)
    else:
        CS.collections[0].set_color(cf68c); CS.collections[0].set_linewidth(2)
        CS.collections[1].set_color(cf90c); CS.collections[1].set_linewidth(2)
        lines.extend((CS.collections[0], CS.collections[1]))
        labels.extend((r'1 $\sigma$ '+padlabel, r'90$\%$ '+padlabel))
    return 

###################################################
#### Main function called to plot
###################################################
def plot ( plotname, outdir, param_settings, llhinfo, plotstyle, sin2theta23s, dm31s, vmax, param_bfs={}, 
           contour2d=False, contour3d=False, sys_behavior=False, sys_fitness=False,
           multicontours2d=False, median_contour2d=False, median_multicontours2d=False ):

    ''' main function to be called to plot different types of plots '''

    ptitle = ''
    for seg in plotname.split ('_'): ptitle += seg + ' '

    ### special cases (only plots contour2D / sys_behavior) : median multiple contours
    if median_multicontours2d:
        bf_theta23s = np.array([ llhinfo[key]['bestfits']['theta23'] for key in llhinfo.keys() ])
        bf_dm31s = np.array([ llhinfo[key]['bestfits']['dm31'] for key in llhinfo.keys() ])
        ### 2D contour
        injected_sin2theta23s = np.array([ np.sin(plotstyle['theta23']['injected'][key])**2 for key in llhinfo.keys() ])
        injected_dm31s = np.array([ plotstyle['dm31']['injected'][key] for key in llhinfo.keys() ])
        contour2d_plot_template ( plotname, ptitle, outdir, llhinfo, sin2theta23s, dm31s, 
                                  injected_sin2theta23s, injected_dm31s, bf_theta23s, bf_dm31s, vmax, 
                                  multiple=True, median=True )
        ### sys_fitness
        if sys_fitness:
            plot_sys_fitness ( plotname, outdir, param_settings, llhinfo, plotstyle, median=True, multiple=True )
        return

    injected_sin2theta23, injected_dm31 = np.sin(plotstyle['theta23']['injected'])**2, plotstyle['dm31']['injected']
    ### special cases (only plots contour2D) : multiple contours
    if multicontours2d:
        bf_theta23s = np.array([ llhinfo[key]['bestfits']['theta23'] for key in llhinfo.keys() ])
        bf_dm31s = np.array([ llhinfo[key]['bestfits']['dm31'] for key in llhinfo.keys() ])
        contour2d_plot_template ( plotname, ptitle, outdir, llhinfo, sin2theta23s, dm31s, 
                                  injected_sin2theta23, injected_dm31, bf_theta23s, bf_dm31s, vmax, 
                                  legend=False, multiple=True )
        if sys_fitness:
            plot_sys_fitness ( plotname, outdir, param_settings, llhinfo, plotstyle, median=False, multiple=True )
        return

    ### simple single contour from 1 experiment (trial) or many trials
    bf_theta23, bf_dm31 = llhinfo['bestfits']['theta23'], llhinfo['bestfits']['dm31']
    llhs = { 'n2dllh':llhinfo['n2dllh'], 'dm31_profile':llhinfo['dllh_profile']['dm31'],
             'sin2theta23_profile':llhinfo['dllh_profile']['theta23'] }

    if contour2d or median_contour2d:
        contour2d_plot_template ( plotname, ptitle, outdir, llhs, sin2theta23s, dm31s, injected_sin2theta23, 
                                  injected_dm31, bf_theta23, bf_dm31, vmax, median=median_contour2d )
    if contour3d: 
        ### bf_theta is always a value if plot contour3D
        plot_contour3d ( plotname, outdir, llhs, sin2theta23s, dm31s, injected_sin2theta23, injected_dm31, 
                         np.sin(bf_theta23)**2, bf_dm31, vmax )
    if sys_fitness:
        plot_sys_fitness ( plotname, ptitle, outdir, param_settings, llhinfo, plotstyle, median=median_contour2d )
    if sys_behavior:
        ### only medians of bestfits are present in the plot
        plot_sys_behavior ( plotname, ptitle, outdir, param_settings, param_bfs, llhinfo, plotstyle, sin2theta23s, dm31s, 
                            injected_sin2theta23, injected_dm31, bf_theta23, bf_dm31, median_contour2d )
    return

###################################################
#### Functions to plot contours
####  -- one 2D contour
####  -- multiple 2D contours
####  -- one 3D contour
###################################################
def contour2d_plot_template ( plotname, ptitle, outdir, llhs, sin2theta23s, dm31s, 
                              injected_sin2theta23, injected_dm31, bf_theta23s, bf_dm31s, vmax, 
                              legend=True, multiple=False, median=False ):

    ''' A template to plot contour 2D'''

    f_contour = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1,3], width_ratios=[4,1])
    gs.update(wspace=0, hspace=0)
    meshed_sin2theta23s, meshed_dm31s = np.meshgrid ( sin2theta23s, dm31s )
    if multiple: colors = get_colors(llhs) 

    #################### Top left: dllh along changing sin2theta23
    ax1 = f_contour.add_subplot(gs[0])
    if multiple:
        plot_multicontour2d ( ax1, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, colors=colors, sin2theta23_profile=True, median=median )
    else:
        plot_contour2d ( ax1, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, sin2theta23_profile=True, median=median )
    axis_dllh_along_bf_theta23 ( ax1, sin2theta23s, vmax )
    #################### Top right: empty

    #################### Bottom left: contour
    ax2 = f_contour.add_subplot(gs[2])
    lines, labels = [], []
    injected_colors = colors.values() if multiple else 'black'
    if plotname not in ['exp', 'data'] and 'burnsample' not in plotname:
        injected = plt.scatter ( injected_sin2theta23, injected_dm31, edgecolors=injected_colors, marker='o', facecolors='none', s=30 )
    if multiple: 
        plot_multicontour2d ( ax2, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, lines=lines, 
                              labels=labels, colors=colors, contour=True, median=median )
    else:
        if plotname not in ['exp', 'data'] and 'burnsample' not in plotname :
            lines.append(injected); labels.append(r'injected')
        plot_contour2d ( ax2, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, 
                         lines=lines, labels=labels, contour=True, median=median )
    
    #### HERE
    if legend: plt.legend(lines, labels, scatterpoints=1, fontsize=10)
    axis_contour (ax2, sin2theta23s, dm31s)

    #################### Bottom right: -2dllh along changing dm31
    ax3 = f_contour.add_subplot(gs[3])
    if multiple: 
        plot_multicontour2d ( ax3, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, colors=colors, dm31_profile=True, median=median )
    else:
        plot_contour2d ( ax3, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, dm31_profile=True, median=median )
    axis_dllh_along_bf_dm31 ( ax3, dm31s, vmax )

    plt.suptitle(r''+ptitle, fontsize=15)
    extension = '_multicontour2D' if multiple else '_contour2D'
    savefig(outdir + plotname + extension + '.pdf')
    plt.close('all')
    return

def plot_contour2d ( ax, llhs, sin2theta23s, dm31s, bf_theta23, bf_dm31, lines=[], labels=[], 
                     sin2theta23_profile=False, contour=False, dm31_profile=False, median=False ):

    ''' plot 2D contour & -2 Delta LLh profiles '''

    if sin2theta23_profile:
        build_dllh_along_theta23 ( ax, sin2theta23s, llhs['sin2theta23_profile'], bf_theta23, 
                                   color=profilec, median=median )
    if contour:
        if median: bf_theta23, bf_dm31 = bf_theta23['median'], bf_dm31['median']
        #### HERE
        bestfit  = plt.scatter ( np.sin(bf_theta23)**2, bf_dm31, color=profilec, marker='*', label=r'best fit', s=30 )
        #bestfit  = plt.scatter ( np.sin(bf_theta23)**2, bf_dm31, color='black', marker='*', label=r'best fit', s=100, alpha=0.5 )
        lines.append(bestfit); labels.append(r'best fit')
        build_contour ( ax, llhs['n2dllh'], sin2theta23s, dm31s, lines, labels, only90=False, median=median )

    if dm31_profile:
        build_dllh_along_dm31 ( ax, dm31s, llhs['dm31_profile'], bf_dm31, color=profilec, median=median )
    return

def plot_multicontour2d ( ax, llhs, sin2theta23s, dm31s, bf_theta23s, bf_dm31s, lines=[], labels=[], 
                          colors=colors, sin2theta23_profile=False, contour=False, dm31_profile=False, median=False ):

    ''' plot multiple 2D contours & -2 Delta LLh profiles '''

    if sin2theta23_profile:
        for index, key in enumerate(llhs.keys()):
            build_dllh_along_theta23 ( ax, sin2theta23s, llhs[key]['dllh_profile']['theta23'], 
                                       bf_theta23s[index], color=colors[key], median=median )
    if contour:
        for index, key in enumerate(llhs.keys()):
            bf_theta23 = bf_theta23s[index]['median'] if median else bf_theta23s[index]
            bf_dm31    = bf_dm31s[index]['median'] if median else bf_dm31s[index]
            bestfit = plt.scatter ( np.sin(bf_theta23)**2, bf_dm31, color=colors[key], marker='*', label=r''+key, alpha=0.7, s=30 )
            name=''; segs=[x.strip() + ' ' for x in key.split('_')]
            for seg in segs: name += seg
            lines.append(bestfit); labels.append(r'best fit ' + name)
            build_contour ( ax, llhs[key]['n2dllh'], sin2theta23s, dm31s, lines, labels, 
                            padlabel=name, color=colors[key], only90=True, median=median )
    if dm31_profile:
        for index, key in enumerate(llhs.keys()):
            build_dllh_along_dm31 ( ax, dm31s, llhs[key]['dllh_profile']['dm31'], bf_dm31s[index], color=colors[key], median=median )
    return

def plot_contour3d ( plotname, outdir, llhs, sin2theta23s, dm31s, injected_sin2theta23, 
                     injected_dm31, bf_sin2theta23, bf_dm31, vmax ):

    ''' plot 3D contour & -2 Delta LLh profiles '''

    h = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[15,1])
    gs.update(wspace=0.3, hspace=0.3)

    X, Y = np.meshgrid ( sin2theta23s, dm31s )
    Z = np.copy(llhs['n2dllh'])
    Z[Z<0], Z[Z>vmax] = np.nan, np.nan

    #################### left: 3d LLH profile
    ax0 = h.add_subplot( gs[0], projection='3d' )
    surface = ax0.plot_surface( X, Y, Z, rstride=1, cstride=1, alpha=0.3, cmap=cm.BuPu, linewidth=0, antialiased=False, vmin=0, vmax=vmax )
    bestfit  = ax0.scatter3D ( bf_sin2theta23, bf_dm31, color='magenta', marker='*', label=r'best fit', alpha = 0.7 )
    if plotname in ['data', 'exp', 'burnsample']:
        lines = [bestfit]; labels = [r'best fit']
    else:
        injected = ax0.scatter3D ( injected_sin2theta23, injected_dm31, color='black', marker='o', label=r'injected', alpha=0.7 )
        lines = [injected, bestfit]; labels = [r'injected', r'best fit']
    cset = ax0.contour (X, Y, Z, levels=[2.288, 4.605], linewidth=1.5, zdir='z', offset=0, cmap=cm.coolwarm)

    #### dllh_profiles
    for param in ['dm31', 'sin2theta23']:
        profile = llhs[param+'_profile'][llhs[param+'_profile']<vmax]
        xs = dm31s if param=='dm31' else sin2theta23s
        zdir = 'x' if param=='dm31' else 'y'
        zs = sin2theta23s[0] if param=='dm31' else dm31s[-1]
        xs = xs[llhs[param+'_profile']<vmax]
        p, = ax0.plot3D( xs, profile, color='green', alpha=1, linewidth=1.5, zs=zs, zdir=zdir )

    axis_contour3d ( ax0, X[0], Y[:,0], r'-2 $\Delta$LLH', 0, vmax )

    #################### right: colorbar
    ax1 = h.add_subplot (gs[1])
    cb = plt.colorbar (surface, cax=ax1)
    cb.ax.tick_params (labelsize=7)
    cb.set_label (r'-2 $\Delta$ LLH values', fontsize=12)

    plt.savefig (outdir + plotname + '_contour3D.pdf')
    plt.close('all')
    return

###################################################
#### Functions to plot systematics-related plots
###################################################
def plot_sys_behavior ( plotname, ptitle, outdir, param_settings, param_bfs, llhinfo, plotstyle, sin2theta23s, 
                        dm31s, injected_sin2theta23, injected_dm31, bf_theta23, bf_dm31, median=False ):

    ''' plot systematic behavior in dm31/theta23 space '''

    bf_sin2theta23 = np.sin(bf_theta23['median'])**2 if median else np.sin(bf_theta23)**2
    bf_dm31 = bf_dm31['median'] if median else bf_dm31

    h = plt.figure(figsize=(23, 10))
    gs = gridspec.GridSpec(4, 6)
    gs.update(wspace=0.6, hspace=0.3)
    
    meshed_sin2theta23s, meshed_dm31s = np.meshgrid ( sin2theta23s, dm31s )
    paramkeys = plotstyle.keys()
    paramkeys.remove('dm31'); paramkeys.remove('theta23')

    for index, var in enumerate(sorted(paramkeys)):
        
        ### values = bestfit values at each grid point 
        ###  -- None if systematic not in nusiance_pararms_textfile or not included as nuisance parameters
        ###  -- values * 1000. if systematic is dm31
        ###  -- values['median'] if median is True (plotting only the median values from many trials)
        values = None if var not in param_settings.keys() or not param_settings[var].included else \
                 param_bfs[var]['median']*1000. if median and 'dm' in var else \
                 np.copy(param_bfs[var]['median']) if median else param_bfs[var]*1000. if 'dm' in var else np.copy(param_bfs[var])

        Z = (values - param_settings[var].prior) / float(param_settings[var].penalty) \
            if values is not None and hasattr(param_settings[var], 'prior') else values 
        zmin = 0. if Z is None or var=='barr_nu_nubar' else -1 if hasattr(param_settings[var], 'prior') else param_settings[var].value*0.75
        zmax = 1. if Z is None or var=='barr_nu_nubar' else 1 if hasattr(param_settings[var], 'prior') else param_settings[var].value*1.25

        subgs = gridspec.GridSpecFromSubplotSpec ( 1, 2, subplot_spec=gs[index], wspace=0.1, width_ratios=[16,1] )
        ax0 = h.add_subplot( subgs[0] )
        axis_sys_behavior ( ax0, meshed_sin2theta23s[0], meshed_dm31s[:,0], zmin, zmax, index )
        ########### if Z is None, print 'Systematic Off'
        if Z is None:
            at = AnchoredText('Systematic Off', prop=dict(size=8), frameon=True, loc=10 )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.5")
            ax0.add_artist(at)
            continue
        ########### contour
        #f = interp2d(sin2theta23s, dm31s, Z, kind='quintic')
        #xnew = np.arange(sin2theta23s[0], sin2theta23s[-1], .1)
        #ynew = np.arange(dm31s[0], dm31s[-1], .1)
        #data = f(xnew,ynew)
        #Xn, Yn = np.meshgrid(xnew, ynew)
        #ax0.pcolormesh(Xn, Yn, data, cmap=PRGn_11.mpl_colormap, vmin=zmin, vmax=zmax)
        cs = ax0.contourf (meshed_sin2theta23s, meshed_dm31s, Z, 200, linewidth=0, cmap=PRGn_11.mpl_colormap, vmin=zmin, vmax=zmax)
        ax0.text ( 0.93, 0.93, plotstyle[var]['ylabel'], horizontalalignment='right', 
                   verticalalignment='top', transform=ax0.transAxes, fontsize=12 )
        ########### right: colorbar
        ax1 = h.add_subplot (subgs[1])
        norm = matplotlib.colors.BoundaryNorm(np.linspace(zmin,zmax,1000), PRGn_11.mpl_colormap.N)
        cb = matplotlib.colorbar.ColorbarBase(ax1, cmap=PRGn_11.mpl_colormap, norm=norm, format='%1.2f', 
                                              orientation='vertical', ticks=np.linspace(zmin, zmax, 8), spacing='uniform')
        cb.ax.tick_params (labelsize=5, length=2)
        zlabel = r'(bestfit - prior mean) / $\sigma$' if hasattr(param_settings[var], 'prior') else r'bestfit'
        cb.set_label (zlabel, fontsize=8)

    plt.suptitle(r''+ptitle, fontsize=12)
    plt.savefig (outdir + plotname + '_sys_behavior.pdf')
    plt.close('all')
    return

def get_sys_fitness ( paramkeys, plotstyle, param_settings, llhinfo_bfs, key='',  median=False ) :
    
    ''' return 5 arrays : bestfit/injected, prior (if any), param indices, param labels '''
    
    yvalues, xvalues, xlabels = [], [], []
    index = 0
    for var in paramkeys:

        if var not in param_settings.keys() or not param_settings[var].included or \
           not hasattr(param_settings[var], 'prior') : 
            continue

        xvalues.append (index)
        xlabels.append (plotstyle[var]['ylabel'])
        bestfit_value = llhinfo_bfs[var]['median'] if median else llhinfo_bfs[var]
        pull = (bestfit_value - param_settings[var].prior) / param_settings[var].penalty
        yvalues.append (pull)
        index+=1

    return np.array(yvalues), np.array(xvalues), np.array(xlabels)

def plot_sys_fitness ( plotname, ptitle, outdir, param_settings, llhinfo, plotstyle, median=False, multiple=False ):

    ''' plot fitness of each systematic (the besfit/injected values) '''

    paramkeys = sorted(plotstyle.keys())
    paramkeys.remove('dm31'); paramkeys.remove('theta23')

    h = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1,1)
    ax = h.add_subplot(gs[0])

    if multiple:
        colors = get_colors(llhinfo)
        for key in llhinfo.keys():
            name=''; segs=[x.strip() + ' ' for x in key.split('_')]
            for seg in segs: name += seg
            ### need to be changed at some point ...
            params = param_settings[key] if median else param_settings
            #####################
            yvalues, xvalues, xlabels = get_sys_fitness ( paramkeys, plotstyle, params, 
                                                          llhinfo[key]['bestfits'], key=key, median=median )
            ax.scatter ( xvalues+1, yvalues, color=colors[key], marker='o', s=12, label=name, alpha=0.7 )
            if median: plt.legend(scatterpoints=1, fontsize=7)
    else:
        yvalues, xvalues, xlabels = get_sys_fitness ( paramkeys, plotstyle, param_settings, llhinfo['bestfits'], median=median )
        ax.scatter ( xvalues+1, yvalues, color='black', marker='o', s=12)

    ax.fill_between(np.arange(len(xvalues)+2), -1 , 1  , facecolor='yellow', alpha=0.2, label=r'1 $\sigma$')
    axis_sys_fitness (ax, xlabels, np.array(xvalues)+1, -2, 2)

    h.subplots_adjust(bottom=0.4)
    plt.suptitle(r''+ptitle, fontsize=12)
    plt.savefig (outdir + plotname + '_sys_fitness.pdf')
    plt.close('all')
    return
