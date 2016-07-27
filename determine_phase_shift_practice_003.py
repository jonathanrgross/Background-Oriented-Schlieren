"""
Created on Tue Jul 19 14:26:47 2016

@author: jack
This script is for experimenting with how to convert the sBOS output value to 
phase shift.
This code doesn't seem to be working.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import tkMessageBox
import time


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
pi = np.pi
x = np.linspace(0, 2*pi, 128)
yRef = np.sin(x)

shift = np.linspace(-pi/2, pi/2, 32)
I= shift*0                               # preallocate I


#------------------------------------------------------------
# set up figure and animation
fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.axis([-2.0, 2.0, -1.0, 1.0])
ax2.axis([0, 2*pi, -1.2, 1.2])
#fig = plt.figure()
#ax = fig.subplot(2, aspect='equal', autoscale_on=False, xlim=(-2, 2), ylim=(-1, 1))
line, = ax1.plot([], [], '-', lw=2)
V1, = ax1.plot([], [], '-k', lw=1)
V2, = ax1.plot([], [], '-k', lw=1)
xAxLine, = ax2.plot([], [], '--k', lw=1)
refLine, = ax2.plot([], [], '-k', lw=1)
measLine, = ax2.plot([], [], '-b', lw=1)
outputLine, = ax2.plot([], [], '-r', lw=2)
angular_location = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)


#------------------------------------------------------------
# create function init, which is used in the animation
def init():
    """initialize animation"""
    line.set_data([], [])
    V1.set_data([], [])
    V2.set_data([], [])
    xAxLine.set_data([], [])
    refLine.set_data([],[])
    measLine.set_data([],[])
    outputLine.set_data([],[])
    time_text.set_text('')
    return line, time_text, V1, V2, xAxLine, refLine, measLine, outputLine
    

#------------------------------------------------------------
# create function animate, which is called for each frame
def animate(i):
    """perform animation step"""
    shift = np.linspace(-pi/2, pi/2, 32)
    count = 0
    for s in shift:
        yMeas = np.sin(x+s)
        yOut = sBOS(yRef,yMeas)
        
        I[count] = yOut[i]
        count = count+1
    
    line.set_data(shift,I)
    gradI = np.gradient(I)
    I2=I[gradI>0]
    shift2=shift[gradI>0]
    V1.set_data([shift2[0],shift2[0]],[-1,1])
    V2.set_data([shift2[-1],shift2[-1]],[-1,1])
    xAxLine.set_data([0,2*pi],[0,0])
    refLine.set_data(x,yRef)
    measLine.set_data(x,yMeas)
    outputLine.set_data(x,yOut)
    
    d = i/64
    angular_location.set_text('angular location = %.1f' % d)
    return line, time_text, V1,  V2, xAxLine, refLine, measLine, outputLine



#------------------------------------------------------------
# call FuncAnimation and save it
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the animation?')
if saveChoice:
    outputFilename = 'video_' + time.strftime("%Y-%m-%d") +'.mp4'
    ani = animation.FuncAnimation(fig, animate, frames=64, interval=5000, blit=True, init_func=init)
    ani.save(outputFilename, fps=30, extra_args=['-vcodec', 'libx264'])
    print('saved animation as ' + outputFilename)


#------------------------------------------------------------
# fit a polynomial
gradI = np.gradient(I)
I2=I[gradI>0]
shift2=shift[gradI>0]
coeff =np.polyfit(I2, shift2, 3, rcond=None, full=False, w=None, cov=False)
x=np.linspace(-0.8, 0.8, 30)
y=coeff[3] + coeff[2]*x+ coeff[1]*x**2+ coeff[0]*x**3

# plot the polynomial fit
fig=plt.figure()
plt.plot(I,shift)
plt.plot(I2,shift2)
plt.plot(x,y)










