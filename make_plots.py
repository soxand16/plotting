#!/usr/bin/env python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
from scipy.interpolate import make_interp_spline, BSpline

directory = '/Users/Sam/research/PnCs/power_splitters/'
filename = 'data.csv'

# =============================================================================
# Label Lines Code
# =============================================================================


from math import atan2,degrees

#Label line with line2D label data
def labelLine(line,x,y_dis,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y+y_dis,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,0,label,align,**kwargs)

if __name__ == '__main__':
    # =============================================================================
    # Read in Data
    # =============================================================================
    data = np.loadtxt(directory+filename, delimiter=',', skiprows=1)
    
    frequency_experiment, perfect_t, defected_t, defected_y, frequency_fem, fem_t, fem_y = data.T
    
    # =============================================================================
    # Plot Parameters
    # =============================================================================
    data_smoothing = True
    mpl.rcParams['figure.figsize'] = (8.2, 4.8)
    mpl.rcParams['font.size'] = 11.0
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 'small'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['ytick.labelsize'] = 'small'
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fontsize']= 10.0
    
    # =============================================================================
    # Data Smoothing
    # =============================================================================
    if data_smoothing : 
        
        frequency_experiment_new = np.linspace(frequency_experiment.min(), frequency_experiment.max(), 300)
        
        # Perfect T 
        spline = make_interp_spline(frequency_experiment, perfect_t, k=3)
        spline_perfect_t = spline(frequency_experiment_new)
        # Defected T
        spline = make_interp_spline(frequency_experiment, defected_t, k=2)
        spline_defected_t = spline(frequency_experiment_new)
        # Defected Y
        spline = make_interp_spline(frequency_experiment, defected_y, k=3)
        spline_defected_y = spline(frequency_experiment_new)
           
        frequency_fem_new = np.linspace(frequency_fem.min(), frequency_fem.max(), 300)
        
        # FEM T
        spline = make_interp_spline(frequency_fem, fem_t, k=3)
        spline_fem_t = spline(frequency_fem_new)
        # FEM Y
        spline = make_interp_spline(frequency_fem, fem_y, k=2)
        spline_fem_y = spline(frequency_fem_new)
        
        spline_frequency_experiment = frequency_experiment_new
        spline_frequency_fem = frequency_fem_new
        
    
    
    # =============================================================================
    # T-Splitter: Perfect vs. Defected
    # =============================================================================
    
    fig0 = plt.figure(0)
    ax0 = fig0.add_subplot(111)
    
    # Axis Labels
    ax0.set_xlabel('Normalized Frequency')
    ax0.set_ylabel('Normalized Transmission')
    
    # Axis Limits
    ax0.set_ylim(-34, 5)
    ax0.set_xlim(0.2,0.4)
    
    # Plot Data
    ax0.plot(spline_frequency_experiment, spline_perfect_t, '-r', label='Perfect T-Splitter')
    ax0.plot(spline_frequency_experiment, spline_defected_t, '-b', label='Defected T-Splitter')
    
    # Add Legend
#    ax0.legend(loc=4)
    
    # Lable Lines    
    lines = plt.gca().get_lines()
    labelLine(lines[0], 0.295, -4, label=lines[0].get_label(), align=False)
    labelLine(lines[1], 0.285, 2.5, label=lines[1].get_label(), align=False)
    
    # Save fig
    plt.savefig(directory+'expiremental_t.png', dpi=1200)
    
    # =============================================================================
    # T-Splitter: Expiremental vs. FEM
    # =============================================================================
    
    fig0 = plt.figure(1)
    ax0 = fig0.add_subplot(111)
    
    # Axis Labels
    ax0.set_xlabel('Normalized Frequency')
    ax0.set_ylabel('Normalized Transmission')
    
    # Axis Limits
    ax0.set_ylim(-34, 5)
    ax0.set_xlim(0.2,0.4)
    
    # Plot Data
    ax0.plot(spline_frequency_experiment, spline_defected_t, '-b', label='Defected T-Splitter')
    ax0.plot(spline_frequency_fem, spline_fem_t, '-k', label='FEM')
    
    # Add Legend
#    ax0.legend(loc=4)
    
    # Lable Lines    
    lines = plt.gca().get_lines()
    labelLine(lines[0], 0.31, -13, label=lines[0].get_label(), align=False)
    labelLine(lines[1], 0.265, -6, label=lines[1].get_label(), align=False)
    
    # Save fig
    plt.savefig(directory+'fem_t.png', dpi=1200)
    
    # =============================================================================
    # Perfect T-Splitter vs. Defected Y-Splitter
    # =============================================================================
    
    fig0 = plt.figure(2)
    ax0 = fig0.add_subplot(111)
    
    # Axis Labels
    ax0.set_xlabel('Normalized Frequency')
    ax0.set_ylabel('Normalized Transmission')
    
    # Axis Limits
    ax0.set_ylim(-34, 5)
    ax0.set_xlim(0.2,0.4)
    
    # Plot Data
    ax0.plot(spline_frequency_experiment, spline_perfect_t, '-r', label='Perfect T-Splitter')
    ax0.plot(spline_frequency_experiment, spline_defected_y, '-g', label='Defected Y-Splitter')
    
    # Add Legend
#    ax0.legend(loc=4)
    
    # Lable Lines    
    lines = plt.gca().get_lines()
    labelLine(lines[0], 0.295, -5, label=lines[0].get_label(), align=False)
    labelLine(lines[1], 0.312, 4, label=lines[1].get_label(), align=False)
    
    # Save fig
    plt.savefig(directory+'expiremental_y.png', dpi=1200)
    
    # =============================================================================
    # Perfect T-Splitter vs. Defected Y-Splitter
    # =============================================================================
    
    fig0 = plt.figure(3)
    ax0 = fig0.add_subplot(111)
    
    # Axis Labels
    ax0.set_xlabel('Normalized Frequency')
    ax0.set_ylabel('Normalized Transmission')
    
    # Axis Limits
    ax0.set_ylim(-34, 5)
    ax0.set_xlim(0.2,0.4)
    
    # Plot Data
    ax0.plot(spline_frequency_experiment, spline_defected_y, '-g', label='Defected Y-Splitter')
    ax0.plot(spline_frequency_fem, spline_fem_y, '-m', label='FEM')
    
    # Add Legend
#    ax0.legend(loc=4)
    
    # Lable Lines    
    lines = plt.gca().get_lines()
    labelLine(lines[0], 0.295, -6, label=lines[0].get_label(), align=False)
    labelLine(lines[1], 0.285, 2.5, label=lines[1].get_label(), align=False)
    
    # Save fig
    plt.savefig(directory+'fem_y.png', dpi=1200)
    


