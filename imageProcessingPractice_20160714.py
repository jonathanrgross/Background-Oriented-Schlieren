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
from tkFileDialog import askopenfilenames
from scipy.signal import kaiserord, lfilter, firwin, freqz

def performBOS(image, refImage):
    img = cv2.imread(image)
    Iimg = img[:,:,0]
    # take horizontal gradient of the image
    Igrad = np.gradient(Iimg, axis=1)
    # take horizontal gradient of the reference image
    Irefgrad = np.gradient(Iref, axis=1)
    
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

# open a dialog box and select an image or video
Tk().withdraw()                                 # we don't want a full GUI, so keep the root window from appearing
file_opt = options = {}
options['initialdir'] = '/home/jack/Pictures'
options['initialfile'] = 'clara.jpg'
options['title'] = 'Select an image to process'
filename = askopenfilenames(**file_opt)          # show an "Open" dialog box and return the path to the selected file


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


#%% check if it is an image or video
image = filename[0]
fileExtention = image[-4:]
if fileExtention == '.bmp' or 'jpeg' or '.jpg' or '.png' or 'tiff':
    print('The selected file is an image')
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




#%% add a while loop to go through each image
                                                                                # I have an error here that I still need to fix.
for x in range(len(filename)):
    print('processing image ' + str(x))
    print(filename[x])
    Iout = performBOS(filename[x], Iref)
    plt.figure()
    plt.imshow(Iout, cmap='gray')    


#%% gaussian filter
import scipy
sigma = 3
gIout = scipy.ndimage.filters.gaussian_filter(Iout, sigma, order=0)

#plt.figure()
gausFig = plt.figure()
plt.imshow(gIout, cmap='gray', vmin=min(gIout.ravel()), vmax=max(gIout.ravel()))
plt.title('output of S-BOS method with gaussian filter')





