"""
author: Jonathan Gross
This is for learning how to apply scaling and masking to S-BOS images
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
closeall = plt.close('all')


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
    X = [ (1,2,1), (2,2,2), (2,2,4) ]
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
        if plot_number == 4:
                if orientation == 'H' or orientation == 'h':
                    plt.plot(image[:,round(height/2)])
                else:
                    plt.plot(image[round(width/2),:])
                plt.title('waveform')
        plt.show()


#------------------------------------------------------------
# use fft to find the wavelength of the background pattern
def findWavelength(image,orientation):
    start = timeit.default_timer()
    if orientation == 'H' or orientation == 'h':
        BGwaveform = image[:,round(height/2)]
    else:
        BGwaveform = image[round(width/2),:]
    # take fft
    fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)
    
    N = len(BGwaveform)
    xf = np.linspace(0.0, 1.0/(2.0), N/2)
    yf = 2.0/N*np.abs(fftOut[0:N/2])
    maxIndex = np.argmax(yf[2:])
    wavelength = 1/xf[maxIndex]
    #plt.figure()
    #plt.plot(xf,yf)
    stop = timeit.default_timer()
    print('the wavelength of the pattern is ' + str(wavelength) + ' px.')
    print('time to find wavelength: ' + str(stop-start) + ' s.')
    return(wavelength)


#------------------------------------------------------------
# apply a filter to find the nonperiodic component of the background
def findNonPeriodicComponant(image,orientation):
    start = timeit.default_timer()
    wavelength = findWavelength(image,orientation)
    sigma = wavelength*2
    nonPeriodicComp = scipy.ndimage.filters.gaussian_filter(image, sigma, order=0)
    stop = timeit.default_timer()
    print('time to find nonperiodic componant: ' + str(stop-start) + ' s.')
    return(nonPeriodicComp)


#------------------------------------------------------------
# blur to remove noise
def removeNoise(image,sigmaPerWavelength):
    start = timeit.default_timer()
    wavelength = findWavelength(image,orientation)
    sigma = wavelength*sigmaPerWavelength
    smoothedImage = scipy.ndimage.filters.gaussian_filter(image, sigma, order=0)
    stop = timeit.default_timer()
    print('time to smooth image: ' + str(stop-start) + ' s.')
    return(smoothedImage)


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
# function to perform BOS
def performBOS(Iimg, Iref, orientation):
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
# read in the coefficients that are used to find the scaling functions
def getCoefficients():
    # open data with coefficients
    co = open('/home/jack/Documents/Environments/sBOS_virtual_environment/sBOS/coeff.txt','r')
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
def getPolynomial(theta, c0,c1,c2,c3):
    y0 = c0[3] + c0[2]*theta + c0[1]*theta**2 + c0[0]*theta**3
    y1 = c1[3] + c1[2]*theta + c1[1]*theta**2 + c1[0]*theta**3
    y2 = c2[3] + c2[2]*theta + c2[1]*theta**2 + c2[0]*theta**3
    y3 = c3[3] + c3[2]*theta + c3[1]*theta**2 + c3[0]*theta**3
    return([y0, y1, y2, y3])


#%%------------------------------------------------------------
# open images
#Img = cv2.imread('/home/jack/Pictures/2016-07-14 0.5mm periodic pattern/cropped/0001_cropped.JPG',1)
#refImg = cv2.imread('/home/jack/Pictures/2016-07-14 0.5mm periodic pattern/cropped/Reference_cropped.JPG',1)
#orientation = 'H'

imgFilename, refImgFilename = openImage()
Img = cv2.imread(imgFilename,0)
refImg = cv2.imread(refImgFilename,0)
orientation = 'V'
#%%
Img = np.float32(Img)
refImg = np.float32(refImg)

height,width = refImg.shape
left = 0
right =width
top = 0
bottom = height


#%% find the nonperiodic component of the image
NPC = findNonPeriodicComponant(refImg,orientation)
displayImage(NPC)


#%% perform various preprocessing operations
wavelength = findWavelength(refImg,orientation)

displayImage(Img)
displayImage(refImg)

Img = Img - NPC
refImg = refImg - NPC

Img = scale(Img)
refImg = scale(refImg)

sigmaPerWavelength = 1.0/16.0
Img = removeNoise(Img, sigmaPerWavelength)
refImg = removeNoise(refImg, sigmaPerWavelength)

displayImage(Img)
displayImage(refImg)

Iout, Iavggrad, Idiff = performBOS(Img, refImg, orientation)
displayImage(Iout)


#%%------------------------------------------------------------
# scale brightness to get phase shift
theta = 0
c0, c1, c2, c3 = getCoefficients()
y = getPolynomial(theta, c0,c1,c2,c3)
#sIout = y[0] + y[1]*Iout + y[2]*Iout**2 + y[3]*Iout**3
sIout = y[1]*Iout + y[2]*Iout**2 + y[3]*Iout**3

displayImage(sIout)
if orientation == 'H' or orientation == 'h':
    mn = np.mean(Iout, axis=1)
    mns = np.mean(sIout, axis=1)
else:
    mn = np.mean(Iout, axis=0)
    mns = np.mean(sIout, axis=0)


plt.figure(2)
plt.plot(mn)
plt.plot(mns)


#%% determine where the S-BOS method cannot detect phase shift
if orientation == 'H' or orientation == 'h':
    v1 = np.amax(Iout, axis=1)
    v2 = np.amin(Iout, axis=1)
else:
    v1 = np.amax(Iout, axis=0)
    v2 = np.amin(Iout, axis=0)

rIo = v1-v2
rIo2 = rIo[:]*1
rIo2[rIo2>0.0011] = 0.0011
rIo3 = np.zeros(len(rIo))
rIo3[rIo>0.0011] = 0.0011
plt.figure()
plt.plot(rIo)
plt.plot(rIo3)


#%% blur to remove noise
sigmaPerWavelength = 1/16
Iout2 = removeNoise(Iout, sigmaPerWavelength)


#%% set non maxima to zero
Imask=Iout[:]*0.0

IoutFD = Iout[:,2:] - Iout[:,1:-1]
IoutBD = Iout[:,1:-1] - Iout[:,:-2]
xtremes = IoutFD*IoutBD
Imask[IoutFD<0 & IoutBD>0] = 1
IoutBD
IoutX = numpy.r_[True, Iout[1:] < Iout[:-1]] & numpy.r_[Iout[:-1] < Iout[1:], True]

Imas[IoutFD*IoutBD>0] = 1

#%% interpolate locations that are at zero


#%% create a mask for the sections that cannot be detected
xmv = np.arange(height)
duty = 0.5

mask = signal.square(xmv*2*2*pi/wavelength +pi/2, duty=duty)/2 + 0.5

sV = np.sin(xmv*2*pi/wavelength + pi/2)
plt.figure(), plt.plot(xmv, mask)
plt.axis([0, 100, -2, 2])


#%% turn the mask vector into an array
Mask = np.transpose(np.resize(mask,(width,height)))
plt.figure(), plt.imshow(Mask, cmap='gray')

maskTestArray = Iout
maskTestArray[:,0:int(width/2)] = Mask[:,0:int(width/2)]*maskTestArray[:,0:int(width/2)]-Mask[:,0:int(width/2)]
plt.figure(), plt.imshow(maskTestArray, cmap='gray', clim=(-0.2,0.2))
plt.title('image with mask')


#%%------------------------------------------------------------
# ask user if they would like to save files
saveChoice = tkMessageBox.askyesno('Save results?','Would you like to save the images?')
if saveChoice:
    outputFilename = 'sBOS_results_' + time.strftime("%Y-%m-%d") +'.jpg'
    scipy.misc.imsave(outputFilename, Iout)
    print('saved image as ' + outputFilename)
else:
    print('You have chosen not to save the image')    




