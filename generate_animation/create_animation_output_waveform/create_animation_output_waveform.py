"""
Created on Tue Jul 19 14:26:47 2016

@author: jack

DESCRIPTION
======
This script creates an animation that shows the shape of the output value at 
every point along the waveform as the measured waveform shifts relative to the 
reference waveform.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import tkMessageBox
PI = np.pi


def sBOS(yRef,yMeas):
    # take gradient of the image
    yGradRef = np.gradient(yRef, axis=0)
    yGradRef = yGradRef/np.max(yGradRef)
    yGradMeas = np.gradient(yMeas, axis=0)
    yGradMeas = yGradMeas/np.max(yGradMeas)
    yGrad = (yGradRef + yGradMeas)/2
    yGrad = yGrad - np.mean(yGrad.ravel())
    
    # find difference between image and ref image
    yDiff = yMeas-yRef
    #yDiff = yDiff - np.mean(yDiff.ravel())
        
    # output
    yOut = yGrad*yDiff
    return(yOut)


#------------------------------------------------------------
# create x, yRef
x = np.linspace(0, 2*PI, 128)
yRef = np.sin(x)


#------------------------------------------------------------
# set up figure and animation

fig = plt.figure()
ax1 = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-PI, PI), ylim=(-1, 1))

xAxLine, = ax1.plot([], [], '--k', lw=1)
refLine, = ax1.plot([], [], '-k', lw=1)
measLine, = ax1.plot([], [], '-b', lw=1)
outputLine, = ax1.plot([], [], '-r', lw=2)
phaseShift = ax1.text(0.02, 1.2, '', transform=ax1.transAxes)


#------------------------------------------------------------
# create function init, which is used in the animation
def init():
    """initialize animation"""
    xAxLine.set_data([], [])
    refLine.set_data([],[])
    measLine.set_data([],[])
    outputLine.set_data([],[])
    phaseShift.set_text('')
    return xAxLine, refLine, measLine, outputLine, phaseShift
    

#------------------------------------------------------------
# create function animate, which is called for each frame
def animate(i):
    """perform animation step"""
    PI = np.pi
    x = np.linspace(-2*PI, 2*PI, 128)
    yRef = np.sin(x)
    yMeas = np.sin(x-0.02*i)
    yOut = sBOS(yRef,yMeas)
        
    xAxLine.set_data([-2*PI,2*PI],[0,0])
    refLine.set_data(x,yRef)
    measLine.set_data(x,yMeas)
    outputLine.set_data(x,yOut)
    
    phaseShift.set_text('angular location = %.1f' % i)
    return xAxLine, refLine, measLine, outputLine, phaseShift


#------------------------------------------------------------
# call FuncAnimation and save it
ani = animation.FuncAnimation(fig, animate, frames=314, interval=30, blit=True, init_func=init)


#------------------------------------------------------------
# ask user if they would like to save files
filename = 'reference measured and output waveforms 001.mp4'
saveChoice = tkMessageBox.askyesno('Save animation?','Would you like to save the animation as ' + filename + '?')
if saveChoice:
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    print('results have been saved as ' + filename)
else:
    print('You have chosen not to save the animation')    








