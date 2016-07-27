# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:17:33 2016
@author: jack
This is a tutorial for openpiv.  I downloaded it from here:
http://www.openpiv.net/openpiv-python/src/tutorial.html
This is branched from "openPIV_demo.py", so if this doesn't work, try that 
file.
"""

import openpiv.tools
import openpiv.process
import openpiv.scaling
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Tkinter import Tk
from tkFileDialog import askopenfilenames, askopenfilename
import tkMessageBox


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


#------------------------------------------------------------
# display image
def displayImage(image):
    plt.figure()
    X = [ (2,1,1), (2,1,2) ]
    for nrows, ncols, plot_number in X:
        plt.subplot(nrows, ncols, plot_number)
        if plot_number == 1:
            imgMean = np.mean(image.ravel())
            imgStd = np.std(image.ravel())
            clim = (imgMean-2*imgStd,imgMean+2*imgStd)
            print(clim)
            plt.imshow(image, cmap='gray', clim=clim)
            plt.title('image')
        if plot_number == 2:
            plt.hist(image.ravel(), 256)
            plt.title('histogram')
        plt.show()


#%%
## read images in.  add something to let the user select the images from a dialog
#frame_a  = openpiv.tools.imread( '/home/jack/Pictures/dots1.jpg' )
#frame_b  = openpiv.tools.imread( '/home/jack/Pictures/dots1rot.jpg' )
imgFilename, refImgFilename = openImage()
img = cv2.imread(imgFilename,0)
refimg = cv2.imread(refImgFilename,0)
displayImage(img)


# OBSOLETE DUE TO CLAHE
##%% use logical indexing to set upper limit on brightness
#Img[Img > 200] = 200
#refImg[refImg > 200] = 200
#displayImage(Img)


#%% use contrast limited adaptive histogram equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)
displayImage(img)


#%% perform PIV
#A=frame_a.astype(np.int32)
#B=frame_b.astype(np.int32)
#u, v, sig2noise = openpiv.process.extended_search_area_piv( A.astype(np.int32), B.astype(np.int32), window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )

u, v, sig2noise = openpiv.process.extended_search_area_piv( refimg.astype(np.int32), img.astype(np.int32), window_size=24, overlap=12, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )

x, y = openpiv.process.get_coordinates( image_size=frame_a.shape, window_size=24, overlap=12 )

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

# save the output vectors
#openpiv.tools.save(x, y, u, v, filename='dots_output_20160712.txt' , fmt='%6.3f', delimiter='       ')
np.savetxt("dots_output_20160712_u.csv", u, delimiter=",")
np.savetxt("dots_output_20160712_v.csv", v, delimiter=",")
np.savetxt("dots_output_20160712_x.csv", x, delimiter=",")
np.savetxt("dots_output_20160712_y.csv", y, delimiter=",")
