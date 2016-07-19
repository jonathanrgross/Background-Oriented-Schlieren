# -*- coding: utf-8 -*-
"""
This script is to generate backgrounds
Created on Wed Jul 13 13:27:57 2016
@author: jack
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc

from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from scipy.signal import kaiserord, lfilter, firwin, freqz


width = 800
height = 600
N=50
# Simple data to display in various forms
x = np.linspace(0, N*2*np.pi, height)
y = np.sin(x)
fig1=plt.figure()
plt.plot(x,y)

#%% add code to enable a square wave
from scipy import signal
y = signal.square(x)
fig1=plt.figure()
plt.plot(x,y)


#%%
Y = np.expand_dims(y, 1)
#Y =np.resize(Y, (height,width)


while Y.size < height*width:
    Y = np.append(Y, Y, axis=1)
Y2 = Y[1:height, 1:width]

Y2rot=np.rot90(Y2, k=1)

#%%
fig1=plt.figure()
plt.imshow(Y2rot, cmap='gray',clim=(-1.0, 1.0))

# save image
filename = 'sBOS_' + str(N) + '.jpg'
scipy.misc.imsave(filename, Y2rot)