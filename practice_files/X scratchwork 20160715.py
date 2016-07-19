# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 20:59:41 2016

@author: jack
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

import cv2

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
plt.figure()
plt.imshow(refImg[:,:,0], cmap='gray')

plt.figure()
plt.imshow(refImg[:,:,1], cmap='gray')

plt.figure()
plt.imshow(refImg[:,:,2], cmap='gray')

#%%

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(refImg[:,:,2])
plt.figure()
plt.imshow(cl, cmap='gray')


#%% plot histogram of brightness
Iimg = refImg[:,:,2]
histFig = plt.figure()
plt.hist(Iimg.ravel(), bins=256, range=(min(Iimg.ravel()), max(Iimg.ravel())), fc='k', ec='k')

Iimg = cl
histFig = plt.figure()
plt.hist(Iimg.ravel(), bins=256, range=(min(Iimg.ravel()), max(Iimg.ravel())), fc='k', ec='k')

#%% apply a threshold
np.max(refImg)
Iimg = refImg[:,:,2]/np.max(refImg[:,:,2])

shift = .5
I = np.around(Iimg+shift)
plt.figure()
plt.imshow(I*100, cmap='gray')
