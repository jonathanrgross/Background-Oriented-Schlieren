# -*- coding: utf-8 -*-
"""
author: Jonathan Gross
This is modified from processSBOSimages.py.  This script will be used with 
color S-BOS images.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilenames, askopenfilename
import tkMessageBox

from scipy.signal import kaiserord, lfilter, firwin, freqz


def performBOS(image, Iref, orientation):
    img = cv2.imread(image)
    Iimg = img[:,:,2]
    if orientation == 'H' or orientation == 'h':
        orient = 0
    else:
        orient = 1
    
    # take horizontal gradient of the image
    Igrad = np.gradient(Iimg, axis=orient)
    # take horizontal gradient of the reference image
    Irefgrad = np.gradient(Iref, axis=orient)
    
    # average the gradients?
    Iavggrad = Igrad + Irefgrad
    Iavggrad = Iavggrad - np.mean(Iavggrad.ravel())
    
    # find difference between image and ref image
    Idiff = Iimg-Iref
    Idiff = Idiff - np.mean(Idiff.ravel())
    # output
    Iout = Iavggrad*Idiff
    # add something to save image
    return Iout


# open a dialog box and select an image
Tk().withdraw()
file_opt = options = {}
options['initialdir'] = '/home/jack/Pictures'
options['title'] = 'Select an image to process'
ImgFilename = askopenfilenames(**file_opt)


# open a dialog box and select reference image
Tk().withdraw()
file_opt = options = {}
options['filetypes'] = [('all files', '.*'), ('portable neytwork graphics', '.png'), 
('JPG image', '.jpg')]
options['initialdir'] = '/home/jack/Pictures'
options['title'] = 'Select reference image'
refImgFilename = askopenfilename(**file_opt) 
refImg = cv2.imread(refImgFilename,1)


#%% find empty channel
meanBrightness = [np.mean(refImg[:,:,0]), np.mean(refImg[:,:,1]),np.mean(refImg[:,:,2])]

#ch1H = refImg[:,round(height/2),1]
#ch0V = refImg[round(width/2),:,1]
#ch1H = refImg[:,round(height/2),1]
#ch1V = refImg[round(width/2),:,1]
#ch2H = refImg[:,round(height/2),1]
#ch2V = refImg[round(width/2),:,1]
#ch0H = np.fft.fft(refImg[:,round(height/2),1], n=None, axis=-1, norm=None)
#ch0V = np.fft.fft(refImg[round(width/2),:,1], n=None, axis=-1, norm=None)
#ch1H = np.fft.fft(refImg[:,round(height/2),1], n=None, axis=-1, norm=None)
#ch1V = np.fft.fft(refImg[round(width/2),:,1], n=None, axis=-1, norm=None)
#ch2H = np.fft.fft(refImg[:,round(height/2),1], n=None, axis=-1, norm=None)
#ch2V = np.fft.fft(refImg[round(width/2),:,1], n=None, axis=-1, norm=None)
#ch0Hs = max(ch0H[2:])
#
## find wavelength of background pattern
#BGwaveform = refImg[400:2400,round(height/2),1]
#fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)
#
#N = len(BGwaveform)
#xf = np.linspace(0.0, 1.0/(2.0), N/2)
#yf = 2.0/N*np.abs(fftOut[0:N/2])
#
#maxVal = 1
#for i in range(2,len(yf)):
#    if yf[i] > maxVal:
#        maxIndex = i
#
#BGpatternWavelength = 1/xf[maxIndex]
#
#
# display image using matplotlib
# Three subplots sharing both x/y axes
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.imshow(refImg[:,:,0], cmap='gray')
ax2.imshow(refImg[:,:,1], cmap='gray')
ax3.imshow(refImg[:,:,2], cmap='gray')


# select orientation of background pattern
orientation = 'H'


#%% add a while loop to go through each image
height,width,channels = refImg.shape
nFrames = len(ImgFilename)
appendedResults = np.zeros([height,width,nFrames])
for x in range(nFrames):
    print('processing image ' + str(x))
    #Iout = performBOS(ImgFilename[x], refImg[:,:,0], 'H')
    Iout = performBOS(ImgFilename[x], refImg[:,:,2], 'V')
    appendedResults[:,:,x] = Iout
    #plt.figure()
    #plt.imshow(Iout, cmap='gray')    


#%% find wavelength of background pattern
BGwaveform = refImg[400:2400,round(height/2),1]
fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)

N = len(BGwaveform)
xf = np.linspace(0.0, 1.0/(2.0), N/2)
yf = 2.0/N*np.abs(fftOut[0:N/2])

maxVal = 1
for i in range(2,len(yf)):
    if yf[i] > maxVal:
        maxIndex = i

BGpatternWavelength = 1/xf[maxIndex]

#%% apply gaussian filter based on wavelength of background pattern
import scipy
sigma = BGpatternWavelength/1.5
gIout = np.zeros([height,width,nFrames])
for x in range(nFrames):
    print('Applying gaussian filter to remove background pattern artifact')
    gIout[:,:,x] = scipy.ndimage.filters.gaussian_filter(appendedResults[:,:,x], sigma, order=0)

#plt.figure()
gausFig = plt.figure()
plt.imshow(gIout[:,:,1], cmap='gray', vmin=min(gIout[:,:,1].ravel()), vmax=max(gIout[:,:,1].ravel()))
plt.title('output of S-BOS method with gaussian filter')


#%% ask user if they would like to save files
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the images?')
if saveChoice:
    for x in range(nFrames):
        outputFilename = 'sBOS_results_' + time.strftime("%Y-%m-%d") + '_' + str(x) +'V.jpg'
        scipy.misc.imsave(outputFilename, appendedResults[:,:,x])
        print('saved image as ' + outputFilename)
        




