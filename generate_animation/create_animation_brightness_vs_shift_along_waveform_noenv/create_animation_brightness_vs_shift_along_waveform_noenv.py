"""
Created on Tue Jul 19 14:26:47 2016

@author: jack

DESCRIPTION
======
This script creates an animation that shows the brightness vs shift curve at 
every location along the background pattern, but unlike 
"create_animation_brightness_vs_shift_along_waveform.py" it does not include 
the envelope.

INSTRUCTIONS TO USE
======
When the code is run it will generate the animation and ask the user if they 
want to save the result.
"""

import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import animation
pi = np.pi


#------------------------------------------------------------
# function to perform S-BOS
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
x = np.linspace(-pi, pi, 256)
yRef = np.sin(x)
shift = np.linspace(-pi/2, pi/2, 64)
trace = np.zeros([len(x),len(shift)])
traceMono = np.zeros([len(x),len(shift)])
maxBright = x*0
minBright = x*0
threshold = 0.01
for angLoc in np.arange(len(x)):
    for s in np.arange(len(shift)):
        yMeas = np.sin(x+shift[s])
        yOut = sBOS(yRef,yMeas)
        trace[angLoc,s] = yOut[angLoc]
#        if yOut[angLoc] >= maxBright[angLoc]:
#            maxBright[angLoc]=yOut[angLoc]
#        if yOut[angLoc] <= minBright[angLoc]:
#            minBright[angLoc]=yOut[angLoc]
    gradTrace = np.gradient(trace[angLoc,:])
    i = int(round(len(shift)/2))
    while gradTrace[i] > threshold and i<len(gradTrace)-1:
        traceMono[angLoc,i] = trace[angLoc,i]
        i = i+1
    i = int(round(len(shift)/2))
    while gradTrace[i] > threshold and i>-1:
        traceMono[angLoc,i] = trace[angLoc,i]
        i = i-1
    maxBright[angLoc]=np.max(traceMono[angLoc,:])
    minBright[angLoc]=np.min(traceMono[angLoc,:])
    
traceMono=np.array(traceMono)

#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.axis([-pi,pi,-1,1])
xAxLine, = ax1.plot([], [], '--k', lw=1)
refLine, = ax1.plot([], [], '-k', lw=1)

Irange1, = ax1.plot([], [], 'o-k', lw=1)


ax2.axis([-pi/2,pi/2,-1,1])
#ax2.xlabel('Normalized intensity')
#ax2.ylabel('displacement [px]')
xAxLine2, = ax2.plot([], [], '--k', lw=1)
traceLine, = ax2.plot([], [], '-g', lw=1)




#------------------------------------------------------------
# create function init, which is used in the animation
def init():
    """initialize animation"""
    xAxLine.set_data([], [])
    refLine.set_data([],[])
    envMax.set_data([],[])
    envMin.set_data([],[])
    Irange1.set_data([],[])
    #Idot1.set_data([],[])

    xAxLine2.set_data([], [])
    traceLine.set_data([],[])


    return (xAxLine, refLine, Irange1, envMax, envMin, xAxLine2, traceLine)
     

#------------------------------------------------------------
# create function animate, which is called for each frame
def animate(i):
    """perform animation step"""
    xAxLine.set_data([-pi,pi],[0,0])
    refLine.set_data(x,yRef)
    envMax.set_data(x,maxBright)
    envMin.set_data(x,minBright)
    Irange1.set_data([x[i],x[i]],[-1,1])

    
    xAxLine2.set_data([-pi,pi],[0,0])
    traceLine.set_data(shift,trace[i,:])
  
    return (xAxLine, refLine, Irange1, envMax, envMin, xAxLine2, traceLine)


#------------------------------------------------------------
# call FuncAnimation and save it
ani = animation.FuncAnimation(fig, animate, frames=256, interval=30, blit=True, init_func=init)


#------------------------------------------------------------
# ask user if they would like to save files
filename = 'brightness vs shift curve no envelope.mp4'
saveChoice = tkMessageBox.askyesno('Save animation?','Would you like to save the animation as ' + filename + '?')
if saveChoice:
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    print('results have been saved as ' + filename)
else:
    print('You have chosen not to save the animation')    








