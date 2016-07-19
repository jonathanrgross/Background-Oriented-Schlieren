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
x = np.linspace(0, 2*pi, 64)

shift = np.linspace(-pi/2, pi/2, 32)
I= np.linspace(-pi/2, pi/2, 32)*0


for loc in np.linspace(0, 2*pi, 16):
    count = 0
    for s in shift:
        yMeas = np.sin(x+s)
        yOut = sBOS(yRef,yMeas)
        
        I[count] = yOut[loc]
        print(loc)
        count = count+1
        
    #plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-1, 1))
    plt.plot(shift,I)
    #time.sleep(1.0)

#    plt.plot((0,10),(0,0),'k', linewidth=0.5)
#    plt.plot(x,yRef,'k--')
#    plt.plot(x,yMeas,'k-')
#    plt.plot(x,yGrad,'c:')
#    plt.plot(x,yDiff,'c-.')
#    plt.plot(x,yOut,'r-',linewidth=2.0)
#    plt.title('sine wave')      # ADD LEGEND


#fig2 = plt.figure()
