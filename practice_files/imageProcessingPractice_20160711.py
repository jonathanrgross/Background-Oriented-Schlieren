# -*- coding: utf-8 -*-
"""
author: Jonathan Gross
This script is to practice using various functions
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib import pyplot as plt
from Tkinter import Tk                                # tkinter is for GUIs
from tkFileDialog import askopenfilename
from scipy.signal import kaiserord, lfilter, firwin, freqz

# open a dialog box and select an image or video
Tk().withdraw()                                 # we don't want a full GUI, so keep the root window from appearing
file_opt = options = {}
options['initialdir'] = '/home/jack/Pictures'
options['initialfile'] = 'clara.jpg'
options['title'] = 'Select an image to process'
filename = askopenfilename(**file_opt)          # show an "Open" dialog box and return the path to the selected file
fileExtention = filename[-4:]


#%% check if it is an image or video
if fileExtention == '.bmp' or 'jpeg' or '.jpg' or '.png' or 'tiff':
    print('The selected file is an image')
    img = cv2.imread(filename,1)
    Iimg = img[:,:,0]
elif fileExtention == '.avi':
    print('The selected file is a video')
    nFrames = 200
    vidFrames = makeVideoArray(filename,nFrames)
    Iimg = vidFrames[:,:,30]
    Iref = np.median(vidFrames, axis=2)
    # is it possible for the user to select multiple files?
else:
    print('The file extention of the selected filed was not recognised')

# display image using matplotlib
plt.imshow(Iimg, cmap='gray')
#plt.xticks([]), plt.yticks([])                  # to hide tick values on X and Y axis
#plt.show()              # what does this do exactly?

#%% plot histogram of brightness
Iimg = img[:,:,0]
Iimg = Iimg - np.mean(Iimg.ravel())
histFig = plt.figure()
plt.hist(Iimg.ravel(), bins=256, range=(min(Iimg.ravel()), max(Iimg.ravel())), fc='k', ec='k')



#%% take horizontal gradient of the image
Igrad = np.gradient(Iimg, axis=1)
gradFig=plt.figure()
plt.imshow(Igrad, cmap='gray')


#%%


plt.figure()
plt.plot(Iimg[1000,:],'-x')
plt.plot(Igrad[1000,:],'-.')
plt.plot(np.gradient(Iimg[1000,:]),'-.')

#%% have user select the reference image 
print('Would you like to select a reference image?  [y/n]')
refImageInput = raw_input()
if refImageInput == 'y' or refImageInput == 'Y':
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    file_opt = options = {}
    options['filetypes'] = [('all files', '.*'), ('portable neytwork graphics', '.png'), 
    ('dont know what jpg is short for ', '.jpg'),  ('audio video interleave(?)', '.avi')]
    options['initialdir'] = '/home/jack/Pictures'
    options['initialfile'] = 'clara.jpg'
    options['title'] = 'This is a title'
    refImgFilename = askopenfilename(**file_opt) 
    refImg = cv2.imread(refImgFilename,1)
    # display image using matplotlib
    refFig=plt.figure()
    Iref = refImg[:,:,0]
    Iref = Iref - np.mean(Iref.ravel())
    plt.imshow(Iref, cmap='gray')
else:
    print('you have chosen not to select a reference image')


#%% take horizontal gradient of the reference image
Irefgrad = np.gradient(Iref, axis=1)
gradFig=plt.figure()
plt.imshow(Igrad, cmap='gray')

#%% average the gradients?
Iavggrad = Igrad + Irefgrad
Iavggrad = Iavggrad - np.mean(Iavggrad.ravel())
gradFig=plt.figure()
plt.imshow(Iavggrad, cmap='gray')

#%% find difference between image and ref image
Idiff = Iimg-Iref
Idiff = Idiff - np.mean(Idiff.ravel())
diffFig=plt.figure()
plt.imshow(Idiff, cmap='gray')


#%% output
Iout = Iavggrad*Idiff
outFig=plt.figure()
#plt.imshow(Iout, cmap='gray')
#plt.imshow(Iout, cmap='gray', vmin=min(Iout.ravel()), vmax=max(Iout.ravel()))
plt.imshow(Iout, cmap='gray', vmin=min(Iout.ravel()), vmax=max(Iout.ravel()))
title('output of S-BOS method, no filtering')
fig800 = plt.figure()
plt.plot(Iout[800,0:500])


#%% histogram of output
histFig = plt.figure()
plt.hist(Iout.ravel(), bins=256, range=(min(Iout.ravel()), max(Iout.ravel())), fc='k', ec='k')
title('histogram plot of output of S-BOS method, no filtering')

#%% gaussian filter
import scipy
sigma = 3
gIout = scipy.ndimage.filters.gaussian_filter(Iout, sigma, order=0)

#plt.figure()
gausFig = figure()
plt.imshow(gIout, cmap='gray', vmin=min(gIout.ravel()), vmax=max(gIout.ravel()))
title('output of S-BOS method with gaussian filter')

#%% Apply FIR filter

## The Nyquist rate of the signal
#sample_rate = 1
#nyq_rate = sample_rate / 2.0
#width = .20/nyq_rate    # width of the transition from pass to stop
#ripple_db = 60.0        # The desired attenuation in the stop band, in dB
#N, beta = kaiserord(ripple_db, width) # order and Kaiser parameter for the FIR filter
#cutoff_hz = .1        # The cutoff frequency of the filter
#taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta)) # create a lowpass FIR filter

FIRout = lfilter(taps, 1.0, Iout) # Use lfilter to filter x with the FIR filter

filtOutFig=plt.figure()
plt.imshow(FIRout, cmap='gray', vmin=min(FIRout.ravel()), vmax=max(FIRout.ravel()))
plt.title('output of S-BOS method with FIR filter')

fig800 = plt.figure()
plt.plot(Iout[800,150:350],)
plt.plot(FIRout[800,150:350],'r')




