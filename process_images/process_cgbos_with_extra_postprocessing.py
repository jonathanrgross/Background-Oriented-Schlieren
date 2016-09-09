"""
@author: jack

DESCRIPTION
======
This performs Simplified Background Oriented Schlieren on a pair of images.  
It also preprocesses the images, and post-processes the result.

INSTRUCTIONS
======
When the scipt is run it will prompt the user with a dialog to select an S-BOS 
file.   The orientation is set in the code.  If the user needs to change it, 
it must be changed in the section where the open_image function is called.

Right now I think I'm editing this on a new branch called "findwavelength".
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import timeit
import csv
from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilenames, askopenfilename
import tkMessageBox
import scipy
from scipy.signal import kaiserord, lfilter, firwin, freqz
import openpiv.tools
import openpiv.process
#closeall = plt.close('all')


#------------------------------------------------------------
# open image
def open_image():
    Tk().withdraw()
    file_opt = options = {}
    #options['initialdir'] = '/home/jack/Pictures'
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
# determine the orientation of the periodic pattern
def determine_orientation(image):
    Hwave = image[:,int(height/2)]
    Vwave = image[int(width/2),:]
    # take fft
    Hfft = np.fft.fft(Hwave, n=None, axis=-1, norm=None)
    Vfft = np.fft.fft(Vwave, n=None, axis=-1, norm=None)
    
    if max(Hfft) > max(Vfft):
        orientation = 'H'
    else:
        orientation = 'V'
    return(orientation)


#------------------------------------------------------------
# allow user to select where to crop the image
#def crop_image(image):
    
    

#------------------------------------------------------------
# display image
def display_image(image):
    plt.figure()
    X = [ (1,2,1), (2,2,2), (2,2,4) ]
    for nrows, ncols, plot_number in X:
        plt.subplot(nrows, ncols, plot_number)
        if plot_number == 1:
            imgMean = np.mean(image.ravel())
            imgStd = np.std(image.ravel())
            clim = (imgMean-2*imgStd,imgMean+2*imgStd)
            print(clim)
            plt.imshow(image, cmap='gray', clim=clim, interpolation='none')
            plt.title('image')
        if plot_number == 2:
            plt.hist(image.ravel(), 256)
            plt.title('histogram')
        if plot_number == 4:
                if orientation == 'H' or orientation == 'h':
                    plt.plot(image[:,int(height/2)])
                else:
                    plt.plot(image[int(width/2),:])
                plt.title('waveform')
        plt.show()



#------------------------------------------------------------
# blur to remove noise
def remove_noise(image,sigma):
    start = timeit.default_timer()
    smoothedImage = scipy.ndimage.filters.gaussian_filter(image, sigma, order=0)
    stop = timeit.default_timer()
    print('time to smooth image: ' + str(stop-start) + ' s.')
    return(smoothedImage)


#------------------------------------------------------------
# function to perform BOS
def perform_bos(Iimg, Iref, orientation):
    start = timeit.default_timer()
    Iimg = scale(np.float32(Iimg))
    Iref = scale(np.float32(Iref))
    
    # find difference between image and ref image
    Idiff = Iimg-Iref

    stop = timeit.default_timer()
    print('time to perform S-BOS: ' + str(stop-start) + ' s.')
    return Idiff


#------------------------------------------------------------
# adjust brightness so mean is zero
def scale(image):
    start = timeit.default_timer()
    image = image - np.amin(image)
    image = image/np.amax(image)
    image = 2*image - np.mean(2*image.ravel())
    stop = timeit.default_timer()
    print('time to scale image: ' + str(stop-start) + ' s.')
    return image



#%%------------------------------------------------------------
imgFilename, refImgFilename = open_image()
Img = cv2.imread(imgFilename,0)
refImg = cv2.imread(refImgFilename,0)
height,width = refImg.shape
orientation = determine_orientation(refImg)
Img = np.float32(Img)
refImg = np.float32(refImg)


#%% add a function to crop the image
left = 0
right =width
top = 0
bottom = height


#%% perform various preprocessing operations

Img = scale(Img)
refImg = scale(refImg)

sigma = 3
Img = remove_noise(Img, sigma)
refImg = remove_noise(refImg, sigma)

#display_image(Img)
#display_image(refImg)

Iout = perform_bos(Img, refImg, orientation)
#display_image(Iout)
plt.figure(), plt.imshow(Iout[500:2500,:],cmap='gray')
plt.show()


#%% determine where the S-BOS method cannot detect phase shift
if orientation == 'H' or orientation == 'h':
    v1 = np.amax(Iout, axis=1)
    v2 = np.amin(Iout, axis=1)
else:
    v1 = np.amax(Iout, axis=0)
    v2 = np.amin(Iout, axis=0)

rIo = v1-v2
brIo =scipy.ndimage.filters.gaussian_filter(rIo, sigma, order=0)
vtuft = rIo - brIo + 1
#plt.figure(), plt.plot(rIo),plt.plot(rIo-qw)

rIo3 = np.zeros(len(rIo))
rIo3[vtuft>1] = 1

plt.figure()
plt.plot(rIo)
plt.plot(brIo)
plt.plot(rIo3*max(rIo))


Mask = np.resize(rIo3,[height,width])
mIout = scIout*Mask
mIout[mIout==0] = np.nan


#%% this interpolates the missing vectors
kernel_size = 4
interpolatedIout, interpolatedIout = openpiv.filters.replace_outliers( mIout, mIout, method='localmean', kernel_size=kernel_size)
smIout = scipy.ndimage.filters.gaussian_filter(interpolatedIout, wavelength/2, order=0)
plt.figure(), 
plt.subplot(131)
plt.imshow(Iout, cmap='gray')
plt.title('before interpolation')
plt.subplot(132)
plt.imshow(interpolatedIout, cmap='gray')
plt.title('kernel size = ' + str(kernel_size))
plt.subplot(133)
plt.imshow(smIout, cmap='gray')
plt.title('with low pass filter')

# the interpolation step results in some pixels being nan.  This sets those 
# pixels to zero so that imsave will work.
smIout = np.nan_to_num(smIout)


# this plots the brightness at a single vector across the waveform to assess 
# the effectiveness of the interpolation at correcting for the S-BOS artifacts
plt.figure()
plt.plot(scIout[1000,:])
plt.plot(interpolatedIout[1000,:])
plt.title('comparison of waveform before and after interpolating')


#%%------------------------------------------------------------
# ask user if they would like to save files
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the images?')
if saveChoice:
    outputFilename = 'sBOS_results_' + time.strftime("%Y-%m-%d") +'.jpg'
    scipy.misc.imsave(outputFilename, smIout)
    print('saved image as ' + outputFilename)
else:
    print('You have chosen not to save the image')    




