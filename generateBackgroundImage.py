# -*- coding: utf-8 -*-
"""
This script is to generate backgrounds
Created on Wed Jul 13 13:27:57 2016
@author: jack
"""


def generateBackgroundImage(width,height,N,waveform,orientation):
    import numpy as np
    import cv2
    from scipy.signal import kaiserord, lfilter, firwin, freqz, square
    from scipy import signal
    
    if orientation == 'vertical' or orientation == 'v' or orientation == 'V':
        W=width
        width = height
        height = W
    x = np.linspace(0, N*2*np.pi, height)
    if waveform == 'square' or waveform == 'sq' or waveform == 'SQ':
        y = signal.square(x)
    else:
        y = np.sin(x)
    
    Y = np.expand_dims(y, 1)
    #Y =np.resize(Y, (height,width)
    
    while Y.size < height*width:
        Y = np.append(Y, Y, axis=1)
        
    Y2 = Y[1:height, 1:width]
    
    if orientation == 'vertical' or orientation == 'v' or orientation == 'V':
        Y2=np.rot90(Y2, k=1)
    
    return Y2