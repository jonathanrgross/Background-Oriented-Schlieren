# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:16:59 2016

@author: jack
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import cv2
from scipy import signal
pi = np.pi
from Tkinter import Tk
from tkFileDialog import askopenfilenames, askopenfilename
import tkMessageBox


#------------------------------------------------------------
# open file with coeeficients
co = open('/home/jack/Documents/Environments/sBOS_virtual_environment/sBOS/coeff.txt','r')
reader = csv.reader(co)
coeff = np.array(list(reader))
coeff = coeff.astype(np.float)
# plot coeeficients
plt.figure()
plt.plot(coeff[:,0],coeff[:,1], '.', label='3rd order term')
plt.plot(coeff[:,0],coeff[:,2], '.', label='2nd order term')
plt.plot(coeff[:,0],coeff[:,3], '.', label='1st order term')
plt.plot(coeff[:,0],coeff[:,4], '.', label='0th order term')
plt.xlabel('angular locaton [rad]')
plt.ylabel('polynomial coefficients')
plt.legend(loc='lower right')


#------------------------------------------------------------
# perform curvefits to the coefficients to get the coefficients as a function
# of angular location
c3 = np.polyfit(coeff[:,0], coeff[:,1], 3)
c2 = np.polyfit(coeff[:,0], coeff[:,2], 3)
c1 = np.polyfit(coeff[:,0], coeff[:,3], 3)
c0 = np.polyfit(coeff[:,0], coeff[:,4], 3)

x = np.linspace(-pi/2, pi/2, 128)
y0 = c0[3] + c0[2]*x + c0[1]*x**2 + c0[0]*x**3
y1 = c1[3] + c1[2]*x + c1[1]*x**2 + c1[0]*x**3
y2 = c2[3] + c2[2]*x + c2[1]*x**2 + c2[0]*x**3
y3 = c3[3] + c3[2]*x + c3[1]*x**2 + c3[0]*x**3

# plot the fits, see if they match
plt.plot(x,y0, label='0th fit')
plt.plot(x,y1, label='1st fit')
plt.plot(x,y2, label='2nd fit')
plt.plot(x,y3, label='3rd fit')
plt.legend(loc='lower right')


#%%------------------------------------------------------------
# open a dialog box and select reference image
Tk().withdraw()
file_opt = options = {}
options['filetypes'] = [('all files', '.*'), ('portable neytwork graphics', '.png'), 
('JPG image', '.jpg')]
options['initialdir'] = '/home/jack/Pictures'
options['title'] = 'Select reference image'
refImgFilename = askopenfilename(**file_opt) 
refImg = cv2.imread(refImgFilename,1)
refImg = refImg[:,:,0]

# display image using matplotlib
fig1 = plt.figure()
plt.imshow(refImg, cmap='gray')


#%%------------------------------------------------------------
# find wavelength of background pattern using fft
refImgB = cv2.GaussianBlur(refImg,(27,1),0)
mn = np.mean(refImgB, axis=1)
plt.figure()
plt.plot(mn)

mnf = np.fft.fft(mn, n=None, norm=None)
N = len(mn)
xf = np.linspace(0.0, 1.0/(2.0), N/2)
yf = 2.0/N*np.abs(mnf[0:N/2])

plt.figure()
plt.plot(xf,yf)
maxIndex = np.argmax(yf[2:]) + 2
wavelength = 1/xf[maxIndex]
print('the wavelength is ' + str(wavelength))


#%%------------------------------------------------------------
# Find the offset distance from the top edge that corresponds to the peak
height,width = refImg.shape
cycles = np.floor(height/wavelength) - 1
peakTest = np.linspace(0, cycles*wavelength, cycles+1)

findPhase = np.zeros(30,)
pS = np.linspace(0, wavelength, 30)

for i in range(len(pS)):
    e = peakTest.astype(int)
    findPhase[i] = sum(mn[e])
    peakTest = peakTest + pS[1]-pS[0]

fig4=plt.figure()
plt.plot(pS,findPhase)
plt.title('this maximum of this curve is the location of the first peak')
plt.xlabel('D = distance from edge of image')
plt.ylabel('sum of points at x*wavelength + D')

firstPeakLocation = pS[np.argmax(findPhase)]


#%% check if the wavelength and first peak location are correct
mnN = (mn-np.min(mn))/np.max(mn)
xmv = np.arange(height)
testWave = np.sin(xmv*2*pi/wavelength+pi/2 -firstPeakLocation+pi/2)
plt.figure(), plt.plot(mnN), plt.plot((testWave+1)/4)


#%% determine the wavelength using a method besides fft
wv=[]
ccc=0
for cc in range(len(mnN)-1):
    if mnN[i]>mnN[cc-1] and mnN[i]>mnN[cc+1]:
        wv.extend([cc])

plt.figure(), plt.plot(wv)
WV = np.array(wv)
di1 = WV[0:-1]
di2 = WV[1:]
DI = di2 - di1
plt.figure(), plt.plot(DI)
np.mean(DI)
np.median(DI)



#%%------------------------------------------------------------
# create a mask for the sections that cannot be detected
xmv = np.arange(height)
duty = 0.5

mask = signal.square(xmv*2*2*pi/wavelength +pi/2, duty=duty)/2 + 0.5

sV = np.sin(xmv*2*pi/wavelength + pi/2)
plt.figure(), plt.plot(xmv, mask), plt.plot(xmv, sV)
plt.axis([0, 100, -2, 2])

#%% turn the mask vector into an array
Mask = np.transpose(np.resize(mask,(width,height)))
plt.figure(), plt.imshow(Mask, cmap='gray')

plt.figure(), plt.imshow(Mask*refImg, cmap='gray')
plt.title('image with mask')



#%%
# now that the wavelength and location of the first peak are known, at every 
# point in the image the amount by which the brightness should be scaled can 
# be found


# I need to construct arrays where each element has the correct value
#V1 = 
# scaling function at theta = 0:
theta = 0
y0 = c0[3] + c0[2]*theta + c0[1]*theta**2 + c0[0]*theta**3
y1 = c1[3] + c1[2]*theta + c1[1]*theta**2 + c1[0]*theta**3
y2 = c2[3] + c2[2]*theta + c2[1]*theta**2 + c2[0]*theta**3
y3 = c3[3] + c3[2]*theta + c3[1]*theta**2 + c3[0]*theta**3

IoutScaled = Iout + y1*Iout + y2*Iout**2 + y3*Iout**3

# scale brightness
#IarrayScaled = Iarray + Iarray*scaleArray1 + Iarray*scaleArray2**2 + Iarray*scaleArray3**3



"""
# BELOW HERE IS CODE USED WHILE WRITING THE SCRIPT THAT IS NOT NEEDED

# remove noise
refImgB = cv2.blur(img,(15,1))
fig2 = plt.figure()
plt.subplot(211),plt.imshow(refImg, cmap='gray'),plt.title('Original')
plt.subplot(212),plt.imshow(refImgB, cmap='gray'),plt.title('Blurred')

fig2b = plt.figure()
plt.plot(refImg[800:1000,1000])
plt.plot(refImg[800:1000,1001])
plt.plot(refImg[800:1000,1002])
plt.plot(refImg[800:1000,1003])


#------------------------------------------------------------
# identify the location in the waveform to find the scaling function
# find mean, min, max.  take gradient.  somehow make those values function inputs for scaling
mn = np.mean(refImgB, axis=0)
mn=np.resize(mn,(refImg.shape[0],refImg.shape[1]))
plt.imshow(mn)
refImgB = refImgB-
imgGrad1 = np.gradient(refImgB, axis=0)
plt.imshow(imgGrad1)
fig3 = plt.figure()
plt.plot(refImgB[800:1000,1000])
plt.plot(imgGrad[800:1000,1000])

#------------------------------------------------------------

# blur image to make it more sinusoidal
refImgB = cv2.GaussianBlur(refImg,(27,1),0)
# plot results of blurring
fig2 = plt.figure()
plt.subplot(211),plt.imshow(refImg[800:1000,1400:1600], cmap='gray'),plt.title('Original')
plt.subplot(212),plt.imshow(refImgB[800:1000,1400:1600], cmap='gray'),plt.title('Blurred')

# find mean of all the columns
mn = np.mean(refImgB, axis=1)
mng = np.gradient(mn)
mngb = cv2.GaussianBlur(mng,(1,27),0)

plt.figure()
plt.plot(mn)
plt.plot(mng)
plt.plot(mngb)
"""








