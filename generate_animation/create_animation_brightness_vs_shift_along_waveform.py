"""
Created on Tue Jul 19 14:26:47 2016

@author: jack

DESCRIPTION
======
This script creates an animation that shows the brightness vs shift curve at 
every location along the background pattern.

INSTRUCTIONS TO USE
======
When the code is run it will generate the animation.  It will also 
automatically append the phase vs brightness curve to a txt file at each time 
step.  It will then ask the user if they want to save the result.

Note that it will open a window as part of the process of creating the 
animation, but it will not display anything to the window.  This is normal.  
Also, it will take several seconds to generate and save the animation.
"""

import numpy as np
import csv
from matplotlib import pyplot as plt
from matplotlib import animation
import tkMessageBox
PI = np.pi


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
    yOut = yDiff
    return(yOut)


#------------------------------------------------------------
# create x, yRef
x = np.linspace(-PI, PI, 256)
yRef = np.sin(x)
shift = np.linspace(-PI/2, PI/2, 64)
trace = np.zeros([len(x),len(shift)])
traceMono = np.zeros([len(x),len(shift)])
maxBright = x*0
minBright = x*0
threshold = 0.001
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

ax1.axis([-PI,PI,-1,1])
xAxLine, = ax1.plot([], [], '--k', lw=1)
refLine, = ax1.plot([], [], '-k', lw=1)
envMax, = ax1.plot([], [], '-y', lw=1)
envMin, = ax1.plot([], [], '-y', lw=1)
Irange1, = ax1.plot([], [], 'o-k', lw=1)
TopLine1, = ax1.plot([], [], '--k', lw=1)
BottomLine1, = ax1.plot([], [], '--k', lw=1)

ax2.axis([-PI/2,PI/2,-1,1])
#ax2.xlabel('Normalized intensity')
#ax2.ylabel('displacement [px]')
xAxLine2, = ax2.plot([], [], '--k', lw=1)
traceLine, = ax2.plot([], [], '--b', lw=1)
traceMonoLine, = ax2.plot([], [], '-g',lw=2)
TopLine2, = ax2.plot([], [], '--k', lw=1)
BottomLine2, = ax2.plot([], [], '--k', lw=1)
LeftLine, = ax2.plot([], [], 'o--k', lw=1)
RightLine, = ax2.plot([], [], 'o--k', lw=1)


#------------------------------------------------------------
# create function init, which is used in the animation
def init():
    """initialize animation"""
    xAxLine.set_data([], [])
    refLine.set_data([],[])
    envMax.set_data([],[])
    envMin.set_data([],[])
    Irange1.set_data([],[])
    TopLine1.set_data([],[])
    BottomLine1.set_data([],[])

    xAxLine2.set_data([], [])
    traceLine.set_data([],[])
    traceMonoLine.set_data([],[])
    TopLine2.set_data([],[])
    BottomLine2.set_data([],[])
    LeftLine.set_data([],[])
    RightLine.set_data([],[])
    return (xAxLine, refLine, Irange1, envMax, envMin, xAxLine2, traceLine, 
            traceMonoLine, TopLine1, BottomLine1, TopLine2, BottomLine2, 
            LeftLine, RightLine)
     

#------------------------------------------------------------
# create function animate, which is called for each frame
def animate(i):
    """perform animation step"""
    xAxLine.set_data([-PI,PI],[0,0])
    refLine.set_data(x,yRef)
    envMax.set_data(x,maxBright)
    envMin.set_data(x,minBright)
    Irange1.set_data([x[i],x[i]],[maxBright[i],minBright[i]])

    xAxLine2.set_data([-PI,PI],[0,0])
    traceLine.set_data(shift,trace[i,:])
    tM = traceMono[i,traceMono[i,:]!=0]
    sh = shift[traceMono[i,:]!=0]
    traceMonoLine.set_data(sh,tM)
    if sh.any():
        coeff = np.polyfit(tM, sh, 3, rcond=None, full=False, w=None, cov=False)
        co = open('Coefficients.txt','a')
        writer = csv.writer(co)
        writer.writerow((x[i],coeff[0],coeff[1],coeff[2],coeff[3]))
        co.close()
        LeftLine.set_data([sh[0],sh[0]],[minBright[i],maxBright[i]])
        RightLine.set_data([sh[-1],sh[-1]],[minBright[i],maxBright[i]])
    
    return (xAxLine, refLine, Irange1, envMax, envMin, xAxLine2, traceLine, 
            traceMonoLine, TopLine1, BottomLine1, TopLine2, BottomLine2, 
            LeftLine, RightLine)


#------------------------------------------------------------
# call FuncAnimation
ani = animation.FuncAnimation(fig, animate, frames=256, interval=30, blit=True, init_func=init)


#------------------------------------------------------------
# ask user if they would like to save files
filename = 'just diff.mp4'
saveChoice = tkMessageBox.askyesno('Save animation?','Would you like to save the animation as ' + filename + '?')
if saveChoice:
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
    print('results have been saved as ' + filename)
else:
    print('You have chosen not to save the animation')    







