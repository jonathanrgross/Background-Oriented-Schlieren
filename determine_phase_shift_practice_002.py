# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:26:47 2016

@author: jack
This script is for experimenting with how to convert the sBOS output value to 
phase shift.
"""

import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import animation

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



pi = np.pi
x = np.linspace(0, 2*pi, 128)
yRef = np.sin(x)

shift = np.linspace(-pi/2, pi/2, 32)
I= np.linspace(-pi/2, pi/2, 32)*0       # preallocate I


for angLoc in np.linspace(pi, 1.73*pi, 6):
    count = 0
    for s in shift:
        yMeas = np.sin(x+s)
        yOut = sBOS(yRef,yMeas)
        
        I[count] = yOut[angLoc]
        count = count+1
        
    #plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-1, 1))
    legEntry = str(angLoc)
    plt.plot(shift,I, label=legEntry)
    #time.sleep(1.0)

plt.legend(loc='lower right')
#    plt.plot((0,10),(0,0),'k', linewidth=0.5)
#    plt.plot(x,yRef,'k--')
#    plt.plot(x,yMeas,'k-')
#    plt.plot(x,yGrad,'c:')
#    plt.plot(x,yDiff,'c-.')
#    plt.plot(x,yOut,'r-',linewidth=2.0)
#    plt.title('sine wave')      # ADD LEGEND


#fig2 = plt.figure()

#%%
#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-1, 1))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')
    return line, time_text, energy_text

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
    time_text.set_text('time = %.1f')
    energy_text.set_text('energy = %.3f J')
    return line, time_text, energy_text

# choose the interval based on dt and the time to animate one step


ani = animation.FuncAnimation(fig, animate, frames=64,
                              interval=50, blit=True, init_func=init)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
ani.save('ooutput_video_name.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()