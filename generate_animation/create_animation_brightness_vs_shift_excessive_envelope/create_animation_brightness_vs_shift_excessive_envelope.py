"""
Created on Tue Jul 19 14:26:47 2016

@author: jack

DESCRIPTION
======
This script creates an animation that shows the brightness vs shift curve at 
every location along the background pattern.  This one is slightly different 
than the others.  It has an envelope that was later realized to be excessive.

INSTRUCTIONS TO USE
======
When the code is run it will generate the animation.  It will also 
automatically append the phase vs brightness curve to a txt file at each time 
step.  It will then ask the user if they want to save the result.
"""

import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import animation
import tkMessageBox
pi = np.pi


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
x = np.linspace(-2*pi, 2*pi, 128)
yRef = np.sin(x)
locat1 = 64-1
locat2 = 72-1


#------------------------------------------------------------
# find the envelope for the max brightness at each angular location
shift = np.linspace(-pi, pi, 64)
maxBright = x*0
minBright = x*0
trace1 = shift*0
for angLoc in np.arange(len(x)):
    for s in np.arange(len(shift)):
        yMeas = np.sin(x+shift[s])
        yOut = sBOS(yRef,yMeas)
        if yOut[angLoc] >= maxBright[angLoc]:
            maxBright[angLoc]=yOut[angLoc]
        if yOut[angLoc] <= minBright[angLoc]:
            minBright[angLoc]=yOut[angLoc]


shift = np.linspace(0, 2*pi, 314)
trace1 = np.zeros([314,1])
for i in np.arange(314):
    x = np.linspace(-2*pi, 2*pi, 128)
    yRef = np.sin(x)
    yMeas = np.sin(x-0.02*i)
    yOut = sBOS(yRef,yMeas)
    trace1[i] = yOut[locat1]


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
#ax3 = plt.subplot(224)

ax1.axis([-pi,pi,-1,1])
xAxLine, = ax1.plot([], [], '--k', lw=1)
refLine, = ax1.plot([], [], '-k', lw=1)
measLine, = ax1.plot([], [], '-b', lw=1)
outputLine, = ax1.plot([], [], '-r', lw=2)
Irange1, = ax1.plot([], [], '-k', lw=1)
Idot1, = ax1.plot([], [], 'ob', lw=1)
Irange2, = ax1.plot([], [], '-k', lw=1)
Idot2, = ax1.plot([], [], 'og', lw=1)
envMax, = ax1.plot([], [], '-y', lw=1)
envMin, = ax1.plot([], [], '-y', lw=1)
#phaseShift = ax1.text(0.02, 1.2, '', transform=ax.transAxes)
ax2.axis([-pi/2,pi/2,-1,1])
#ax2.xlabel('Normalized intensity')
#ax2.ylabel('displacement [px]')
xAxLine2, = ax2.plot([], [], '--k', lw=1)
bright_vs_phase1, = ax2.plot([], [], 'o-b', lw=2)
bright_vs_phase2, = ax2.plot([], [], 'o-g', lw=2)
traceLine1, = ax2.plot([], [], '.b', markersize=1)
traceLine2, = ax2.plot([], [], '.g', markersize=1)


#------------------------------------------------------------
# create function init, which is used in the animation
def init():
    """initialize animation"""
    xAxLine.set_data([], [])
    refLine.set_data([],[])
    measLine.set_data([],[])
    outputLine.set_data([],[])
    Irange1.set_data([],[])
    Idot1.set_data([],[])
    Irange2.set_data([],[])
    Idot2.set_data([],[])
    #phaseShift.set_text('')
    envMax.set_data([],[])
    envMin.set_data([],[])
    xAxLine2.set_data([], [])
    bright_vs_phase1.set_data([],[])
    bright_vs_phase2.set_data([],[])
    traceLine1.set_data([],[])
    traceLine2.set_data([],[])
    return (xAxLine, refLine, measLine, outputLine, Irange1, Idot1, Irange2, 
            Idot2, envMax, envMin, bright_vs_phase1, bright_vs_phase2, 
            xAxLine2, traceLine1, traceLine2)
    

#------------------------------------------------------------
# create function animate, which is called for each frame
def animate(i):
    """perform animation step"""
    x = np.linspace(-2*pi, 2*pi, 128)
    yRef = np.sin(x)
    yMeas = np.sin(x-0.02*i)
    yOut = sBOS(yRef,yMeas)
        
    xAxLine.set_data([-2*pi,2*pi],[0,0])
    refLine.set_data(x,yRef)
    measLine.set_data(x,yMeas)
    outputLine.set_data(x,yOut)
    Irange1.set_data([x[locat1],x[locat1]],[-1,1])
    Idot1.set_data(x[locat1],yOut[locat1])
    
    Irange2.set_data([x[locat2],x[locat2]],[-1,1])
    Idot2.set_data(x[locat2],yOut[locat2])
    envMax.set_data(x,maxBright)
    envMin.set_data(x,minBright)
    #phaseShift.set_text('angular location = %.1f' % i)
    AL = (0.02*i+pi/2)%pi-pi/2      # goes from 0 to 2*pi
    # add brightness vs displacement here.  write to a csv and read each time this runs
    
    #traceLine2.set_data(shift,trace2)
    bright_vs_phase1.set_data(AL,yOut[locat1])
    bright_vs_phase2.set_data(AL,yOut[locat2])
    
#    Trace1 = open('trace1.txt','a')        # THIS BLOCK OF CODE WRITES trace1.txt
#    writer = csv.writer(Trace1)
#    writer.writerow( (AL, yOut[locat1]) )
#    Trace1.close()
#    Trace2 = open('trace2.txt','a')
#    writer = csv.writer(Trace2)
#    writer.writerow( (AL, yOut[locat2]) )
#    Trace2.close()
    
    Trace1 = open('trace1.txt','r')         # THIS BLOCK OF CODE READS trace1.txt
    reader = csv.reader(Trace1)
    trace1 = np.array(list(reader))
    traceLine1.set_data(trace1[:,0],trace1[:,1])
    Trace2 = open('trace2.txt','r')
    reader = csv.reader(Trace2)
    trace2 = np.array(list(reader))
    traceLine2.set_data(trace2[:,0],trace2[:,1])
    
    xAxLine2.set_data([-2*pi,2*pi],[0,0])
    return (xAxLine, refLine, measLine, outputLine, Irange1, Idot1, Irange2, 
            Idot2, envMax, envMin, bright_vs_phase1, bright_vs_phase2, 
            xAxLine2, traceLine1, traceLine2)
            



#------------------------------------------------------------
# call FuncAnimation and save it
ani = animation.FuncAnimation(fig, animate, frames=314, interval=30, blit=True, init_func=init)


#------------------------------------------------------------
# ask user if they would like to save files
filename = 'brightness vs shift with excessive envelope.mp4'
saveChoice = tkMessageBox.askyesno('Save animation?','Would you like to save the animation as ' + filename + '?')
if saveChoice:
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    print('results have been saved as ' + filename)
else:
    print('You have chosen not to save the animation')
    







