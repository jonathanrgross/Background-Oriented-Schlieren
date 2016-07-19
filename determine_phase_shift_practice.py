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


fig1 = plt.figure()
pi = np.pi
x = np.linspace(0, 10, 100)
count = 0
I=np.zeros((10,), dtype=np.float)

yRef = np.sin(x)
for shift in np.linspace(-pi/2, pi/2, 10):
    #shift = np.pi/4
    yMeas = np.sin(x+shift)
        
    # take horizontal gradient of the image
    yGradRef = np.gradient(yRef, axis=0)
    yGradMeas = np.gradient(yMeas, axis=0)
    yGrad = yGradRef + yGradMeas
    yGrad = yGrad - np.mean(yGrad.ravel())
    
    # find difference between image and ref image
    yDiff = yMeas-yRef
    yDiff = yDiff -  np.mean(yDiff.ravel())
    
    
    # output
    yOut = yGrad*yDiff
    
    I[count] = yOut[20]
    count = count+1
    print('test')

#    plt.plot((0,10),(0,0),'k', linewidth=0.5)
#    plt.plot(x,yRef,'k--')
#    plt.plot(x,yMeas,'k-')
#    plt.plot(x,yGrad,'c:')
#    plt.plot(x,yDiff,'c-.')
#    plt.plot(x,yOut,'r-',linewidth=2.0)
#    plt.title('sine wave')      # ADD LEGEND


#fig2 = plt.figure()
plt.plot(np.linspace(-pi/2, pi/2, 10),I)
