# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:27:57 2016
@author: jack
This script generates dot patterns to use as backgrounds for BOS
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc

from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from scipy.signal import kaiserord, lfilter, firwin, freqz


# define dimensions
width=1920                                                                      # select width
height=1080  

y = np.random.rand(height,width)
# select percent of the space to have dots
percentCoverage = 2
shift = (percentCoverage-50.)/100.
print(shift)
Y = np.around(y+shift)

# disply random dot pattern
fig1=plt.figure()
plt.imshow(Y, cmap='gray')
plt.title('dot pattern for BOS.')


#%% use dialate function to change size of dots
dotSize = 2
kernel = np.ones((dotSize,dotSize),np.uint8)
Ye = cv2.dilate(Y,kernel,iterations = 1)
fig2=plt.figure()
plt.imshow(Ye, cmap='gray')
plt.title('dot pattern for BOS.  dot size = '+str(dotSize)+' px, percent coverage = '+str(percentCoverage)+'%')


#%% save image
filename = 'BOS_' + str(width) + 'x'+ str(height) + '_'+str(-shift*2) +'.jpg'
scipy.misc.imsave(filename, Ye)