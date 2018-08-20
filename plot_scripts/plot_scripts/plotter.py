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
