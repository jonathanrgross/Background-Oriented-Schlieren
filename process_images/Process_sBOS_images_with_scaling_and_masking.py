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
from matplotlib import pyplot as plt
from Tkinter import Tk
from tkFileDialog import askopenfilenames, askopenfilename
import tkMessageBox
import scipy
from scipy.signal import kaiserord, lfilter, firwin, freqz
closeall = plt.close('all')


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
    sigma = wavelength*3
    nonPeriodicComp = scipy.ndimage.filters.gaussian_filter(image, sigma, order=0)
    stop = timeit.default_timer()
    print('time to find nonperiodic componant: ' + str(stop-start) + ' s.')
    return(nonPeriodicComp)


#------------------------------------------------------------
# blur to remove noise
def removeNoise(image):
    start = timeit.default_timer()
    wavelength = findWavelength(image,orientation)
    sigma = wavelength/32
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


#%%------------------------------------------------------------
# open images
Img = cv2.imread('/home/jack/Pictures/2016-07-14 0.5mm periodic pattern/cropped/0001_cropped.JPG',1)
refImg = cv2.imread('/home/jack/Pictures/2016-07-14 0.5mm periodic pattern/cropped/Reference_cropped.JPG',1)
orientation = 'H'

#Img = cv2.imread('/home/jack/Pictures/flame_20160726/const_BG_size/12/DSC_0253.JPG',1)
#refImg = cv2.imread('/home/jack/Pictures/flame_20160726/const_BG_size/12/reference.JPG',1)
#orientation = 'V'


Img = np.float32(Img[:,:,0])
refImg = np.float32(refImg[:,:,0])

height,width = refImg.shape
left = 0
right =width
top = 0
bottom = height

wavelength = findWavelength(refImg,orientation)

displayImage(Img)
displayImage(refImg)


#%%
NPC = findNonPeriodicComponant(refImg,orientation)
displayImage(NPC)


#%%
Img = Img - NPC
refImg = refImg - NPC

Img = scale(Img)
refImg = scale(refImg)

Img = removeNoise(Img)
refImg = removeNoise(refImg)

displayImage(Img)
displayImage(refImg)

Iout, Iavggrad, Idiff = performBOS(Img, refImg, orientation)
displayImage(Iout)

#%%------------------------------------------------------------
# find wavelength of background pattern
BGwaveform = refImg[400:2400,round(height/2),1]
fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)

N = len(BGwaveform)
xf = np.linspace(0.0, 1.0/(2.0), N/2)
yf = 2.0/N*np.abs(fftOut[0:N/2])
maxIndex = np.argmax(yf[2:])
wavelength = 1/xf[maxIndex]


#%% apply a low pass filter and subtract
plt.plot(refImg[1000,:,0])
sigma = BGpatternWavelength*3
fRef = np.zeros([bottom-top,right-left])
rI = np.float32(refImg[:,:,0])
fRef = scipy.ndimage.filters.gaussian_filter(rI, sigma, order=0)

gausFig = plt.figure()
plt.imshow(fRef, cmap='gray', vmin=min(fRef[:,:].ravel()), vmax=max(fRef[:,:].ravel()))
plt.title('filtered reference image')

refImg = refImg[:,:,0]-fRef
Img = Img[:,:,0]-fRef

plt.figure()
plt.plot(refImg[1000,:])
plt.plot(Img[1000,:])

#%%------------------------------------------------------------
# process image
start = timeit.default_timer()
Iout, Iavggrad, Idiff = performBOS(Img, refImg, orientation)
stop = timeit.default_timer()




#%% plot reference image
plt.figure()
plt.hist(refImg.ravel(), 256 )
plt.figure()
plt.imshow(refImg, cmap='gray')
mn = np.mean(refImg, axis=1)
plt.figure()
plt.plot(mn)

#%% plot image
plt.figure()
plt.hist(Img.ravel(), 256 )
plt.figure()
plt.imshow(Img, cmap='gray')
mn = np.mean(Img, axis=1)
plt.figure()
plt.plot(mn)

#%% plot results
plt.figure()
plt.hist(Iout.ravel(), 256 )
plt.figure()
plt.imshow(Iout, cmap='gray')
mn = np.mean(Iout, axis=1)
plt.figure()
plt.plot(mn)


#------------------------------------------------------------
# find wavelength of background pattern
BGwaveform = refImg[400:2400,round(height/2),1]
fftOut = np.fft.fft(BGwaveform, n=None, axis=-1, norm=None)

N = len(BGwaveform)
xf = np.linspace(0.0, 1.0/(2.0), N/2)
yf = 2.0/N*np.abs(fftOut[0:N/2])
maxIndex = np.argmax(yf[2:])
BGpatternWavelength = 1/xf[maxIndex]


#%%------------------------------------------------------------
# scale brightness based to get phase shift
y0 = c0[3] + c0[2]*0 + c0[1]*0**2 + c0[0]*0**3
y1 = c1[3] + c1[2]*0 + c1[1]*0**2 + c1[0]*0**3
y2 = c2[3] + c2[2]*0 + c2[1]*0**2 + c2[0]*0**3
y3 = c3[3] + c3[2]*0 + c3[1]*0**2 + c3[0]*0**3
sIout = y0 + y1*Iout + y2*Iout**2 + y3*Iout**3

mn = np.mean(Iout, axis=1)
mns = np.mean(sIout, axis=1)
# compare results
plt.figure(1)
plt.subplot(221)
plt.imshow(Iout,cmap='gray', clim=(-0.3,0.3))
plt.title('image')
plt.subplot(222)
plt.imshow(sIout,cmap='gray', clim=(-0.3,0.3))
plt.title('image with scaling')
plt.subplot(223)
plt.hist(Iout.ravel(),256, range=(-0.3,0.3))
plt.title('histogram of image')
plt.subplot(224)
plt.hist(sIout.ravel(),256, range=(-0.3,0.3))
plt.title('histogram of image with scaling')
plt.show()

plt.figure(2)
plt.plot(mn)
plt.plot(mns)

#%% create a mask for the sections that cannot be detected
xmv = np.arange(height)
duty = 0.5

wavelength = BGpatternWavelength
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




