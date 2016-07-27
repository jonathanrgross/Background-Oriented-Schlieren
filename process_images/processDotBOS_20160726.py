# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:17:33 2016
@author: jack
This is based on an openpiv tutorial I found at:
http://www.openpiv.net/openpiv-python/src/tutorial.html
On July 26th I branched this from "processDotBOS20160726.py"
If this doesn't work, go back to that one.
"""

import openpiv.tools
import openpiv.process
import openpiv.scaling
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import timeit
from Tkinter import Tk
from tkFileDialog import askopenfilenames, askopenfilename
import tkMessageBox
from tkFileDialog import askopenfilename


#------------------------------------------------------------
# open image
def openImage():
    Tk().withdraw()
    file_opt = options = {}
    options['initialdir'] = '/home/jack/Pictures'
    options['title'] = 'Select an image to process'
    imgFilename = askopenfilenames(**file_opt)
    fo = 1
    filepath = str(imgFilename)
    filepath = filepath[:-4]
    while fo:
        filepath = filepath[:-1]
        if filepath[-1] == '/':
            fo = 0
    refImgFilename = filepath + "reference.JPG')"
    imgFilename = str(imgFilename)[2:-3]
    refImgFilename = str(refImgFilename[2:-2])
    print(imgFilename)
    print(refImgFilename)
    return(imgFilename, refImgFilename)




# OBSOLETE CODE FOR OPENING IMAGES
##%% open a dialog box and select an image or video
#Tk().withdraw()
#file_opt = options = {}
#options['initialdir'] = '/home/jack/Pictures'
#options['title'] = 'Select an image to process'
#frame_a_filename = askopenfilename(**file_opt)
##frame_a  = openpiv.tools.imread( frame_a_filename,0)
#frame_a = cv2.imread(frame_a_filename,0)
#
## open a dialog box and select an image or video
#Tk().withdraw()
#file_opt = options = {}
#options['initialdir'] = '/home/jack/Pictures'
#options['title'] = 'Select reference image'
#frame_b_filename = askopenfilename(**file_opt)
##frame_b  = openpiv.tools.imread( frame_b_filename )
#frame_b = cv2.imread(frame_b_filename,0)


#------------------------------------------------------------
# perform PIV
def performPIV(image1, image2, window_size, overlap, search_area_size):
    start = timeit.default_timer()
    print('calculating displacement vectors.  This could take a minute.')
    u, v, sig2noise = openpiv.process.extended_search_area_piv( image1, image2, 
              window_size=window_size, overlap=overlap, dt=0.02, 
              search_area_size=search_area_size, sig2noise_method='peak2peak' )
    x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )
    stop = timeit.default_timer()
    print('time to perform BOS: ' + str(stop-start) + ' s.')
    return(u, v, x, y)


#%%
imgFilename, refImgFilename = openImage()
frame_a = cv2.imread(refImgFilename,0)
frame_b = cv2.imread(imgFilename,0)

#%% crop images
# ADD SOMETHING INTERACTIVE TO LET THE USER CROP THE IMAGE
xl=0
xu=-1
yl=0
yu=-1
fac = frame_a[yl:yu, xl:xu]*256/np.max(frame_a)
fbc = frame_b[yl:yu, xl:xu]*256/np.max(frame_a)


#%% improve contrast with contrast limited adaptive histogram equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cla = clahe.apply(fac)
clb = clahe.apply(fbc)
plt.figure()
plt.imshow(fac, cmap='gray')
plt.figure()
plt.imshow(cla, cmap='gray')


##%% reset brightness on brightest pixels
#frame_b[frame_b > 200] = 0
#
#
## display preprocessed results
#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
#ax1.imshow(fac, cmap='gray')
#ax2.imshow(cla, cmap='gray')
#ax3.hist(fac.ravel(), bins=256, fc='k', ec='k')
#ax4.hist(cla.ravel(), bins=256, fc='k', ec='k')


#%% perform PIV
fac2=np.array(cla)
image1=fac2.astype(np.int32)
fbc2=np.array(clb)
image2=fbc2.astype(np.int32)
    
window_size=24
overlap=12
search_area_size=64
u,v,x,y = performPIV(image1, image2, window_size, overlap, search_area_size)


# plot the vector field
plt.figure()
plt.quiver( x, y, u, v )

#%%
# invalid vectors are set to NaN
u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )

# this interpolates the missing vectors
u, v = openpiv.filters.replace_outliers( u, v, method='localmean', kernel_size=2)

# scale the results so they're in correct units
x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

# plot the vector field
plt.figure()
plt.quiver( x, y, u, v )

#%% save the output vectors
filename = 'BOS_output_'
np.savetxt(filename+'u.csv', u, delimiter=",")
np.savetxt(filename+'v.csv', v, delimiter=",")
np.savetxt(filename+'x.csv', x, delimiter=",")
np.savetxt(filename+'y.csv', y, delimiter=",")
