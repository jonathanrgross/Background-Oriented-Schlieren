"""
@author: jack

DESCRIPTION
======
This one attempt to improve on the S-BOS method.  While it is able to 
process the images, it is largely unsucessful in the sense that for all the 
added complexity, the quality of the processed image does not appear to be a 
noticeable improvement over the original S-BOS method.

The difference between the original S-BOS method and this version is that the 
original S-BOS method produces intensity values that are not linear with 
pixel displacement, making them purely quantitative representations of the 
density gradient field.  This attempted to scale the intensity to correct for 
the nonlinear intensity vs displacement relationship.  It also attempted to 
remove the periodic artifact inherent to the S-BOS method using a more 
sophisticated approach.  While these artifacts are typically removed by 
blurring the image, this attempted to determine the locations of the pixels 
that needed to be corrected, and then interpolate them.  Ultimately this 
did not bear fruit within the time frame of the project.

INSTRUCTIONS
======
When the scipt is run it will prompt the user with a dialog to select an 
image to process.  For the reference image, it will automatically open 
the image in the directory that is titled 'reference.jpg'.

It will then ask the user if they would like to remove the 'non-periodic 
component' of the image.  If they select yes, the script will subtract 
low frequency content from the image.

After the image has been processed, the user is offered the option to save the 
results.
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
    Hwave = image[:,int(width/2)]
    Vwave = image[int(height/2),:]
    # take fft
    Hfft = np.fft.fft(Hwave, n=None, axis=-1, norm=None)
    Vfft = np.fft.fft(Vwave, n=None, axis=-1, norm=None)
    
    if max(Hfft[2:200]) > max(Vfft[2:200]):
        orientation = 'H'
    else:
        orientation = 'V'
    return(orientation)


#------------------------------------------------------------
# allow user to select where to crop the image
#def crop_image(image):
    # NOTE: THIS WAS A PLACEHOLDER FOR A FUNCTION I DIDN'T GET AROUND TO
    
    

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
                    plt.plot(image[:,int(width/2)])
                else:
                    plt.plot(image[int(height/2),:])
                plt.title('waveform')
        plt.show()


#------------------------------------------------------------
# use fft to find the wavelength of the background pattern
def find_wavelength(image,orientation):
    start = timeit.default_timer()
    if orientation == 'H' or orientation == 'h':
        BGwaveform = image[:,int(width/2)]
    else:
        BGwaveform = image[int(height/2),:]
    # take fft
    fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)
    
    N = len(BGwaveform)
    xf = np.linspace(0.0, 1.0/(2.0), N/2)
    yf = 2.0/N*np.abs(fftOut[0:N/2])
    maxIndex = np.argmax(yf[2:])
    wavelength = 1/xf[maxIndex+2]
    #plt.figure()
    #plt.plot(xf,yf,'x-')
    stop = timeit.default_timer()
    print('the wavelength of the pattern is ' + str(wavelength) + ' px.')
    print('time to refine wavelength: ' + str(stop-start) + ' s.')
    # add the ability  to find the phase
    return(wavelength)


#------------------------------------------------------------
# use correlate to get a more precise wavelength
def refine_wavelength(image, wavelength, orientation):
    start = timeit.default_timer()
    if orientation == 'H' or orientation == 'h':
        v1 = image[:,int(width/2)]
    else:
        v1 = image[int(height/2),:]
    
    x = np.array(range(int(wavelength*100)))
    count = 0
    shift = np.linspace(-wavelength/3, wavelength/3, 100)
    maxcor = np.zeros(len(shift))
    
    for i in shift:
        WL = wavelength+i
        v2 = np.sin(x*(2*np.pi)/WL)
    
        cor = np.correlate(v1, v2, mode='full')
    
        maxcor[count] = np.max(cor)
        count = count + 1
    
    #plt.figure()
    #plt.plot(shift,maxcor)
    newwavelength = wavelength + shift[np.argmax(maxcor)]
    print('the wavelength is ' + str(newwavelength))
    
    stop = timeit.default_timer()
    print('time to find wavelength: ' + str(stop-start) + ' s.')
    return(newwavelength)


#------------------------------------------------------------
# use correlation to find the phase
def find_phase(image, wavelength, orientation):
    if orientation == 'H' or orientation == 'h':
        v1 = image[:,int(width/2)]
    else:
        v1 = image[int(height/2),:]
    
    x = np.array(range(int(6*wavelength)))
    y = np.sin((x)*(2*np.pi)/wavelength)
    
    cor = np.correlate(y[0:int(2*wavelength)], v1, mode='valid')
    
    phaseoffset = wavelength - (len(cor) - np.argmax(cor*range(len(cor))) - 1)
    print('the phase shift is ' + str(phaseoffset))
    return(phaseoffset)


#------------------------------------------------------------
# apply a filter to find the nonperiodic component of the background
def find_nonperiodic_componant(image,wavelength, orientation):
    start = timeit.default_timer()
    sigma = wavelength*2
    nonPeriodicComp = scipy.ndimage.filters.gaussian_filter(image, sigma, order=0)
    stop = timeit.default_timer()
    print('time to find nonperiodic componant: ' + str(stop-start) + ' s.')
    return(nonPeriodicComp)


#------------------------------------------------------------
# blur to remove noise
def remove_noise(image,wavelength,sigmaPerWavelength):
    start = timeit.default_timer()
    sigma = wavelength*sigmaPerWavelength
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
    
    if orientation == 'H' or orientation == 'h':
        orient = 0
    else:
        orient = 1
    # take gradient of the image
    Igrad = np.gradient(Iimg, axis=orient)
    # take gradient of the reference image
    Irefgrad = np.gradient(Iref, axis=orient)
    # average the gradients
    Iavggrad = (Igrad + Irefgrad)/2
    # find difference between image and ref image
    Idiff = Iimg-Iref
    # output
    Iout = Iavggrad*Idiff
    stop = timeit.default_timer()
    print('time to perform S-BOS: ' + str(stop-start) + ' s.')
    return Iout, Iavggrad, Idiff


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


#------------------------------------------------------------
# read in the coefficients that are used to find the scaling functions
def get_coefficients():
    # open data with coefficients
    co = open('coeff.txt','r')
    reader = csv.reader(co)
    coeff = np.array(list(reader))
    coeff = coeff.astype(np.float)
    # perform curvefits to the coefficients
    c3 = np.polyfit(coeff[:,0], coeff[:,1], 3)
    c2 = np.polyfit(coeff[:,0], coeff[:,2], 3)
    c1 = np.polyfit(coeff[:,0], coeff[:,3], 3)
    c0 = np.polyfit(coeff[:,0], coeff[:,4], 3)
    return(c0, c1, c2, c3)


#------------------------------------------------------------
# use the polynomial functions to get the constants for the scaling functions
def get_polynomial(theta, c0,c1,c2,c3):
    y0 = c0[3] + c0[2]*theta + c0[1]*theta**2 + c0[0]*theta**3
    y1 = c1[3] + c1[2]*theta + c1[1]*theta**2 + c1[0]*theta**3
    y2 = c2[3] + c2[2]*theta + c2[1]*theta**2 + c2[0]*theta**3
    y3 = c3[3] + c3[2]*theta + c3[1]*theta**2 + c3[0]*theta**3
    return([y0, y1, y2, y3])



#------------------------------------------------------------
imgFilename, refImgFilename = open_image()
Img = cv2.imread(imgFilename,0)
refImg = cv2.imread(refImgFilename,0)
height,width = refImg.shape
orientation = determine_orientation(refImg)
Img = np.float32(Img)
refImg = np.float32(refImg)


#------------------------------------------------------------
# add a function to crop the image
left = 0
right =width
top = 0
bottom = height
Img = Img[top:bottom,left:right]
refImg = refImg[top:bottom,left:right]
height,width = refImg.shape
plt.figure()
plt.imshow(refImg,cmap='gray')
orientation = determine_orientation(refImg)
print(orientation)


#------------------------------------------------------------
# find the nonperiodic component of the image

# ask user if they would like to save files
wavelength = find_wavelength(refImg,orientation)
wavelength = refine_wavelength(refImg, wavelength, orientation)
msg = ('Would you like to subtract the non-periodic component of the images?')
npcChoice = tkMessageBox.askyesno('subtract NPC?',msg)
if npcChoice:
    NPC = find_nonperiodic_componant(refImg,wavelength,orientation)
    display_image(NPC)
    Img = Img - NPC
    refImg = refImg - NPC
else:
    print('You have chosen not to subtract the nonperiodic component')   


#------------------------------------------------------------
# perform various preprocessing operations
wavelength = find_wavelength(refImg,orientation)

#display_image(Img)
#display_image(refImg)

Img = scale(Img)
refImg = scale(refImg)

sigmaPerWavelength = 1.0/16.0
Img = remove_noise(Img, wavelength, sigmaPerWavelength)
refImg = remove_noise(refImg, wavelength, sigmaPerWavelength)

#display_image(Img)
#display_image(refImg)

Iout, Iavggrad, Idiff = perform_bos(Img, refImg, orientation)

#display_image(Iout)
#plt.show()


#------------------------------------------------------------
# adjust brightness based on intensity vs shift curve
theta = 0
c0, c1, c2, c3 = get_coefficients()
y = get_polynomial(theta, c0,c1,c2,c3)
scIout = y[1]*Iout + y[2]*Iout**2 + y[3]*Iout**3

display_image(scIout)
if orientation == 'H' or orientation == 'h':
    mn = np.mean(Iout, axis=1)
    mns = np.mean(scIout, axis=1)
else:
    mn = np.mean(Iout, axis=0)
    mns = np.mean(scIout, axis=0)

#plt.figure(2)
#plt.plot(mn)
#plt.plot(mns)


#------------------------------------------------------------
# determine where the S-BOS method cannot detect phase shift
if orientation == 'H' or orientation == 'h':
    v1 = np.amax(Iout, axis=1)
    v2 = np.amin(Iout, axis=1)
else:
    v1 = np.amax(Iout, axis=0)
    v2 = np.amin(Iout, axis=0)

rIo = v1-v2
brIo =scipy.ndimage.filters.gaussian_filter(rIo, wavelength, order=0)
vtuft = rIo - brIo + 1
#plt.figure()
#plt.plot(rIo)
#plt.plot(rIo-qw)

rIo3 = np.zeros(len(rIo))
rIo3[vtuft>1] = 1

#plt.figure()
#plt.plot(rIo)
#plt.plot(brIo)
#plt.plot(rIo3*max(rIo))

Mask = np.resize(rIo3,[height,width])
mIout = scIout*Mask
mIout[mIout==0] = np.nan


#------------------------------------------------------------
# this interpolates the missing vectors
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


#------------------------------------------------------------
# ask user if they would like to save files
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the images?')
if saveChoice:
    outputFilename = 'sBOS_results_' + time.strftime("%Y-%m-%d") +'.jpg'
    scipy.misc.imsave(outputFilename, smIout)
    print('saved image as ' + outputFilename)
else:
    print('You have chosen not to save the image')    




