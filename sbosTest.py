# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 17:36:04 2016
@author: jack
This is just for testing some stuff out.  It isn't for actually processing
S-BOS images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import pyplot as plt
from Tkinter import Tk                                # tkinter is for GUIs
from tkFileDialog import askopenfilename
from scipy.signal import kaiserord, lfilter, firwin, freqz

# create a sine wave
pi = np.pi
x = np.arange(0, 12*pi)
s = np.sin(x*2*pi/2.6)+2
sg = np.gradient(s, axis=0)

sample_rate = 1
nyq_rate = sample_rate / 2.0
width = .20/nyq_rate    # width of the transition from pass to stop
ripple_db = 60.0        # The desired attenuation in the stop band, in dB
N, beta = kaiserord(ripple_db, width) # order and Kaiser parameter for the FIR filter
cutoff_hz = .45        # The cutoff frequency of the filter
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta)) # create a lowpass FIR filter
f = lfilter(taps, 1.0, s) # Use lfilter to filter x with the FIR filter 


plt.plot(x,s)
plt.plot(x,sg,'r')
plt.plot(x,f,':k')
plt.title('comparison')
