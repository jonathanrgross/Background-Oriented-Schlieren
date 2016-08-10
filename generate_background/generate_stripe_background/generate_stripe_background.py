"""
Created on Wed Jul 13 13:27:57 2016
@author: jack

DESCRIPTION
-----
This creates 1-D grayscale background patterns for S-BOS.

INSTRUCTIONS TO USE
-----
Input parameters are chosen by editing the variables listed under "select input 
parameters".  The user can select the dimension of the image in pixels, the 
wavelength in pixels, the waveform ('sq' for square and 'sin' for sine), and 
the orientation of the periodic pattern ('V' for vertical and 'H' for 
horizontal).  The user is shown the background and then asked if they would 
like to save it.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.misc
import tkMessageBox
import time
from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilename
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import signal


#------------------------------------------------------------
# function to generate a background image
def generateBackgroundImage(width,height,N,waveform,orientation,peak):
    # orientation
    if orientation == 'vertical' or orientation == 'v' or orientation == 'V':
        W=width
        width = height
        height = W
    
    # waveform
    N = height/wavelength
    x = np.linspace(0, N*2*np.pi, height)
    if waveform == 'square' or waveform == 'sq' or waveform == 'SQ':
        y = signal.square(x)
    elif waveform == 'triangle' or waveform == 't' or waveform == 'T':
        y = scipy.signal.sawtooth(x, width=peak)
    else:
        y = np.sin(x)
    
    # extend the vector to make it an array
    Y = np.resize(y,(width,height))
    
    if orientation == 'horizontal' or orientation == 'h' or orientation == 'H':
        Y = np.rot90(Y, k=1)
    
    return Y


#------------------------------------------------------------
# select input parameters
width=1920
height=1080
wavelength = 8 # wavelength in px
waveform = 'T'
orientation = 'V'
peak = 0.0


#------------------------------------------------------------
# call function to generate the background image
Y = generateBackgroundImage(width,height,wavelength,waveform,orientation,peak)


#------------------------------------------------------------
# display image
fig1=plt.figure()
plt.imshow(Y, cmap='gray',clim=(-1.0, 1.0))


#------------------------------------------------------------
# ask user if they would like to save files
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the background?')
if saveChoice:
    outputFilename = 'BG_' +  waveform + '_' + orientation + '_' + str(int(wavelength)) + 'px_'  + time.strftime("%Y-%m-%d") +'.jpg'
    scipy.misc.imsave(outputFilename, Y)
    print('saved image as ' + outputFilename)
else:
    print('You have chosen not to save the image')    

plt.close()


