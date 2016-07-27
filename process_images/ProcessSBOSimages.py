"""
author: Jonathan Gross
This script is to practice using various functions
If it doesn't work, try using the files in '2016-07-14 0.5mm periodic pattern/'
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
import scipy
from scipy.signal import kaiserord, lfilter, firwin, freqz


#------------------------------------------------------------
# function to perform BOS
def performBOS(image, Iref, orientation):
    img = cv2.imread(image)
    Iimg = img[top:bottom,left:right,0]
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
    return Iout, Iavggrad, Idiff


#------------------------------------------------------------
# open a dialog box and select an image
Tk().withdraw()
file_opt = options = {}
options['initialdir'] = '/home/jack/Pictures'
options['title'] = 'Select an image to process'
ImgFilename = askopenfilenames(**file_opt)


#------------------------------------------------------------
# open a dialog box and select reference image
Tk().withdraw()
file_opt = options = {}
options['filetypes'] = [('all files', '.*'), ('portable neytwork graphics', '.png'), 
('JPG image', '.jpg')]
options['initialdir'] = '/home/jack/Pictures'
options['title'] = 'Select reference image'
refImgFilename = askopenfilename(**file_opt) 
refImg = cv2.imread(refImgFilename,1)

# display image using matplotlib
plt.imshow(refImg[:,:,0], cmap='gray')

# select orientation of background pattern
orientation = 'H'
# input values to crop it
left = 240
right =4600
top = 300
bottom = 2300

#------------------------------------------------------------
# a for loop is used to process each image
height,width,channels = refImg.shape
nFrames = len(ImgFilename)
#appendedResults = np.zeros([height,width,nFrames])
appendedResults = np.zeros([bottom-top,right-left,nFrames])
for x in range(nFrames):
    print('processing image ' + str(x))
    Iout, Iavggrad, Idiff = performBOS(ImgFilename[x], refImg[top:bottom,left:right,0], orientation)
    appendedResults[:,:,x] = Iout


#------------------------------------------------------------
# find wavelength of background pattern
BGwaveform = refImg[400:2400,round(height/2),1]
fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)

N = len(BGwaveform)
xf = np.linspace(0.0, 1.0/(2.0), N/2)
yf = 2.0/N*np.abs(fftOut[0:N/2])
maxIndex = np.argmax(yf[2:])
BGpatternWavelength = 1/xf[maxIndex]


#------------------------------------------------------------
# apply gaussian filter based on wavelength of background pattern
sigma = BGpatternWavelength/1.5
#gIout = np.zeros([height,width,nFrames])
gIout = np.zeros([bottom-top,right-left,nFrames])
for x in range(nFrames):
    print('Applying gaussian filter to remove background pattern artifact')
    gIout[:,:,x] = scipy.ndimage.filters.gaussian_filter(appendedResults[:,:,x], sigma, order=0)

gausFig = plt.figure()
plt.imshow(gIout[:,:,0], cmap='gray', vmin=min(gIout[:,:,0].ravel()), vmax=max(gIout[:,:,0].ravel()))
plt.title('output of S-BOS method with gaussian filter')


#------------------------------------------------------------
# ask user if they would like to save files
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the images?')
if saveChoice:
    for x in range(nFrames):
        outputFilename = 'sBOS_results_' + time.strftime("%Y-%m-%d") + '_' + str(x) +'.jpg'
        scipy.misc.imsave(outputFilename, appendedResults[:,:,x])
        print('saved image as ' + outputFilename)
else:
    print('You have chosen not to save the image')    




